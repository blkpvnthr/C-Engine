"""Probability calibration + uncertainty evaluation (Milestone 6, §20).

This module scores and *recalibrates* the probabilistic outputs of the supervised
learning models (``cengine/learning/supervised``) so that a reported probability of,
say, 0.7 actually corresponds to a ~70% empirical frequency. It provides:

  * scalar scores — Brier score and log loss (thin wrappers over the shared
    ``base.classification_metrics`` so the metric math is defined in exactly one place);
  * reliability-curve binning + Expected / Maximum Calibration Error (ECE / MCE);
  * two post-hoc calibrators — isotonic regression and temperature scaling — that are
    **FIT ON A HELD-OUT VALIDATION SPLIT ONLY** (never on train, never on test);
  * prediction-interval and per-quantile coverage checks for the quantile-regression
    uncertainty heads;
  * an abstention helper that converts low-confidence predictions into abstentions and
    reports whether doing so is *justified* by a measured selective-accuracy gain.

Device
------
Everything here runs on the **CPU** in pure numpy / scikit-learn. This is an explicit,
documented placement — not a silent GPU claim. These are small, serial, latency-
insensitive post-processing steps over a validation set (a few thousand rows); isotonic
regression is a CPU-only scikit-learn estimator and temperature scaling is a scalar 1-D
line search. There is no MLX/Metal path to fall back *from*, so the DevicePolicy GPU-first
rule does not apply to calibration; ``CALIBRATION_DEVICE`` records the fixed placement.

Two kinds of "confidence" — do not conflate them
------------------------------------------------
``confidence_from_proba`` yields **model confidence**: the probability mass a *calibrated
classifier* assigns to its predicted class (``max(p, 1 - p)`` in ``[0.5, 1]``). It is a
probability and is the *only* quantity these tools calibrate or abstain on.

This is DISTINCT from the engine's **heuristic ranking confidence** (e.g. the market
scanner's ordinal rank/priority score). That ranking signal is a preference ordering, not a
probability — it is not frequency-calibrated, must not be fed into ``IsotonicCalibrator`` /
``TemperatureScaler`` / ``expected_calibration_error`` as if it were a probability, and is
not what ``apply_abstention`` gates on. Keep the two separate.

Invariants
----------
  * Recommendation-only research infrastructure. Nothing here submits orders, touches a
    broker, or interacts with the mandatory -0.5% protective exit (that exit is independent
    environment dynamics, never a calibrated model output).
  * No fabricated labels or probabilities: every function consumes *observed* outcomes and
    *produced* model scores only.
  * Insufficient validation data still works but yields a ``research_only`` calibrator
    (a flagged, non-promotable result), never a silently promoted one.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from cengine.learning.supervised.base import classification_metrics

CALIBRATION_DEVICE = "cpu"  # fixed, documented placement (see module docstring)

_EPS = 1e-7

__all__ = [
    "CALIBRATION_DEVICE",
    "AbstentionResult",
    "IntervalCoverageReport",
    "IsotonicCalibrator",
    "QuantileCoverageReport",
    "ReliabilityCurve",
    "TemperatureScaler",
    "apply_abstention",
    "brier_score",
    "calibration_report",
    "confidence_from_proba",
    "expected_calibration_error",
    "log_loss",
    "maximum_calibration_error",
    "prediction_interval_coverage",
    "quantile_coverage",
    "reliability_curve",
]


# --------------------------------------------------------------------------- helpers
def _as_prob(p: np.ndarray) -> np.ndarray:
    """Flatten to 1-D float64 and clip into the open unit interval."""
    return np.clip(np.asarray(p, dtype=np.float64).ravel(), 0.0, 1.0)


def _as_binary(y: np.ndarray) -> np.ndarray:
    return np.asarray(y, dtype=np.float64).ravel()


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, _EPS, 1.0 - _EPS)
    return np.log(p / (1.0 - p))


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


# --------------------------------------------------------------------- scalar scores
def brier_score(y_true: np.ndarray, p_pred: np.ndarray) -> float:
    """Brier score (mean squared error of probabilities). Lower is better.

    Delegates to the shared ``base.classification_metrics`` so the definition is not
    duplicated here.
    """
    return float(classification_metrics(y_true, p_pred)["brier"])


def log_loss(y_true: np.ndarray, p_pred: np.ndarray) -> float:
    """Binary log loss / negative log-likelihood. Lower is better.

    Delegates to the shared ``base.classification_metrics``.
    """
    return float(classification_metrics(y_true, p_pred)["log_loss"])


# --------------------------------------------------------------- reliability + ECE/MCE
@dataclass
class ReliabilityCurve:
    """Binned reliability diagram comparing predicted confidence to observed frequency.

    ``bin_confidence`` is the mean predicted probability in each bin; ``bin_accuracy`` is
    the empirical positive rate. A perfectly calibrated model has ``accuracy == confidence``
    in every populated bin. Empty bins are ``nan`` and excluded from ECE / MCE.
    """

    bin_edges: np.ndarray
    bin_counts: np.ndarray
    bin_confidence: np.ndarray
    bin_accuracy: np.ndarray
    strategy: str
    n: int

    def ece(self) -> float:
        """Expected Calibration Error: count-weighted mean |accuracy - confidence|."""
        mask = self.bin_counts > 0
        total = int(self.bin_counts.sum())
        if total == 0 or not mask.any():
            return 0.0
        gap = np.abs(self.bin_accuracy[mask] - self.bin_confidence[mask])
        weights = self.bin_counts[mask] / total
        return float(np.sum(weights * gap))

    def mce(self) -> float:
        """Maximum Calibration Error: worst-bin |accuracy - confidence|."""
        mask = self.bin_counts > 0
        if not mask.any():
            return 0.0
        return float(np.max(np.abs(self.bin_accuracy[mask] - self.bin_confidence[mask])))

    def to_dict(self) -> dict:
        return {
            "strategy": self.strategy,
            "n": self.n,
            "bin_edges": self.bin_edges.tolist(),
            "bin_counts": self.bin_counts.tolist(),
            "bin_confidence": self.bin_confidence.tolist(),
            "bin_accuracy": self.bin_accuracy.tolist(),
            "ece": self.ece(),
            "mce": self.mce(),
        }


def _bin_edges(p: np.ndarray, n_bins: int, strategy: str) -> np.ndarray:
    if strategy == "quantile":
        edges = np.unique(np.quantile(p, np.linspace(0.0, 1.0, n_bins + 1)))
        if edges.size < 2:  # degenerate (all equal) -> single full-range bin
            edges = np.array([0.0, 1.0])
        edges[0], edges[-1] = 0.0, 1.0
        return edges
    if strategy != "uniform":
        raise ValueError(f"unknown binning strategy: {strategy!r}")
    return np.linspace(0.0, 1.0, n_bins + 1)


def reliability_curve(
    y_true: np.ndarray,
    p_pred: np.ndarray,
    *,
    n_bins: int = 10,
    strategy: str = "uniform",
) -> ReliabilityCurve:
    """Build a reliability curve.

    strategy="uniform" uses equal-width probability bins; "quantile" uses equal-count
    bins (robust when predictions cluster). Empty bins are reported as ``nan``.
    """
    p = _as_prob(p_pred)
    y = _as_binary(y_true)
    if p.shape != y.shape:
        raise ValueError(f"y_true {y.shape} and p_pred {p.shape} must match")
    n = int(p.size)
    edges = _bin_edges(p, max(1, int(n_bins)), strategy)
    nb = edges.size - 1
    if n == 0:
        empty = np.full(nb, np.nan)
        return ReliabilityCurve(edges, np.zeros(nb, dtype=int), empty, empty, strategy, 0)
    idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, nb - 1)
    counts = np.bincount(idx, minlength=nb).astype(int)
    conf_sum = np.bincount(idx, weights=p, minlength=nb)
    acc_sum = np.bincount(idx, weights=y, minlength=nb)
    with np.errstate(invalid="ignore", divide="ignore"):
        safe = counts.astype(np.float64)
        conf = np.where(counts > 0, conf_sum / np.where(safe == 0, 1.0, safe), np.nan)
        acc = np.where(counts > 0, acc_sum / np.where(safe == 0, 1.0, safe), np.nan)
    return ReliabilityCurve(edges, counts, conf, acc, strategy, n)


def expected_calibration_error(
    y_true: np.ndarray,
    p_pred: np.ndarray,
    *,
    n_bins: int = 10,
    strategy: str = "uniform",
) -> float:
    """Expected Calibration Error (ECE) over ``n_bins`` reliability bins."""
    return reliability_curve(y_true, p_pred, n_bins=n_bins, strategy=strategy).ece()


def maximum_calibration_error(
    y_true: np.ndarray,
    p_pred: np.ndarray,
    *,
    n_bins: int = 10,
    strategy: str = "uniform",
) -> float:
    """Maximum Calibration Error (MCE) over ``n_bins`` reliability bins."""
    return reliability_curve(y_true, p_pred, n_bins=n_bins, strategy=strategy).mce()


def calibration_report(
    y_true: np.ndarray,
    p_pred: np.ndarray,
    *,
    n_bins: int = 10,
    strategy: str = "uniform",
) -> dict:
    """Bundle classifier metrics + ECE/MCE for an uncertainty-evaluation summary."""
    base = classification_metrics(y_true, p_pred)
    curve = reliability_curve(y_true, p_pred, n_bins=n_bins, strategy=strategy)
    base.update({"ece": curve.ece(), "mce": curve.mce(),
                 "n_bins": int(curve.bin_edges.size - 1), "strategy": strategy,
                 "device": CALIBRATION_DEVICE})
    return base


# ------------------------------------------------------------------- calibrators
class IsotonicCalibrator:
    """Isotonic-regression probability calibrator (non-parametric, monotonic).

    FIT ON THE VALIDATION SPLIT ONLY — never on the training data the base model saw and
    never on the test data used to judge calibration. Fitting on train leaks; fitting on
    test invalidates the evaluation.

    Insufficient / degenerate validation data (fewer than two samples, a single observed
    class, or constant inputs) yields an identity (pass-through) calibrator flagged
    ``research_only=True`` — a usable but non-promotable result. A valid fit on fewer than
    ``min_samples`` points is also flagged ``research_only``.

    Runs on the CPU (scikit-learn ``IsotonicRegression``).
    """

    device = CALIBRATION_DEVICE

    def __init__(self, *, min_samples: int = 50) -> None:
        self.min_samples = int(min_samples)
        self._iso = None
        self.fitted = False
        self.research_only = False
        self.n_fit = 0

    def fit(self, p_val: np.ndarray, y_val: np.ndarray) -> "IsotonicCalibrator":
        """Fit on VALIDATION probabilities/labels only."""
        p = _as_prob(p_val)
        y = _as_binary(y_val)
        if p.shape != y.shape:
            raise ValueError(f"p_val {p.shape} and y_val {y.shape} must match")
        self.n_fit = int(p.size)
        self._iso = None
        self.fitted = True
        if self.n_fit < 2 or np.unique(y).size < 2 or np.unique(p).size < 2:
            self.research_only = True  # identity fallback; not enough signal to fit
            return self
        from sklearn.isotonic import IsotonicRegression

        self._iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        self._iso.fit(p, y)
        self.research_only = self.n_fit < self.min_samples
        return self

    def transform(self, p_pred: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("IsotonicCalibrator must be fit on VALIDATION data first")
        p = _as_prob(p_pred)
        if self._iso is None:
            return p
        return np.clip(self._iso.predict(p), 0.0, 1.0)

    def predict(self, p_pred: np.ndarray) -> np.ndarray:
        return self.transform(p_pred)


class TemperatureScaler:
    """Single-parameter temperature-scaling calibrator for binary scores.

    Divides the logit of each prediction by a learned temperature ``T > 0`` (``T > 1``
    softens overconfident probabilities, ``T < 1`` sharpens underconfident ones), fit by
    minimizing validation log loss with a bounded 1-D golden-section search (pure numpy;
    no scipy dependency).

    FIT ON THE VALIDATION SPLIT ONLY. Accepts probabilities by default, or raw logits with
    ``from_logits=True``. Degenerate / tiny validation data yields ``T = 1`` (identity) and
    ``research_only=True``; a valid fit on fewer than ``min_samples`` points is likewise
    flagged ``research_only``.

    Runs on the CPU (scalar line search over log loss).
    """

    device = CALIBRATION_DEVICE

    def __init__(self, *, min_samples: int = 50, from_logits: bool = False,
                 t_lo: float = 1e-2, t_hi: float = 1e2) -> None:
        self.min_samples = int(min_samples)
        self.from_logits = bool(from_logits)
        self.t_lo = float(t_lo)
        self.t_hi = float(t_hi)
        self.temperature = 1.0
        self.fitted = False
        self.research_only = False
        self.n_fit = 0

    def _to_logits(self, scores: np.ndarray) -> np.ndarray:
        s = np.asarray(scores, dtype=np.float64).ravel()
        return s if self.from_logits else _logit(_as_prob(s))

    def fit(self, scores_val: np.ndarray, y_val: np.ndarray) -> "TemperatureScaler":
        """Fit the temperature on VALIDATION scores/labels only."""
        logits = self._to_logits(scores_val)
        y = _as_binary(y_val)
        if logits.shape != y.shape:
            raise ValueError(f"scores_val {logits.shape} and y_val {y.shape} must match")
        self.n_fit = int(logits.size)
        self.fitted = True
        if self.n_fit < 2 or np.unique(y).size < 2:
            self.temperature = 1.0
            self.research_only = True
            return self
        self.temperature = self._search(logits, y)
        self.research_only = self.n_fit < self.min_samples
        return self

    def _nll(self, logits: np.ndarray, y: np.ndarray, t: float) -> float:
        p = _sigmoid(logits / t)
        return log_loss(y, p)

    def _search(self, logits: np.ndarray, y: np.ndarray, iters: int = 80) -> float:
        a, b = math.log(self.t_lo), math.log(self.t_hi)
        gr = (math.sqrt(5.0) - 1.0) / 2.0
        c, d = b - gr * (b - a), a + gr * (b - a)
        fc, fd = self._nll(logits, y, math.exp(c)), self._nll(logits, y, math.exp(d))
        for _ in range(iters):
            if fc < fd:
                b, d, fd = d, c, fc
                c = b - gr * (b - a)
                fc = self._nll(logits, y, math.exp(c))
            else:
                a, c, fc = c, d, fd
                d = a + gr * (b - a)
                fd = self._nll(logits, y, math.exp(d))
        return float(math.exp((a + b) / 2.0))

    def transform(self, scores: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("TemperatureScaler must be fit on VALIDATION data first")
        return _sigmoid(self._to_logits(scores) / self.temperature)

    def predict(self, scores: np.ndarray) -> np.ndarray:
        return self.transform(scores)


# ----------------------------------------------------------- interval / quantile coverage
@dataclass
class QuantileCoverageReport:
    """Empirical coverage of predicted quantiles vs their nominal levels.

    For a well-calibrated quantile head, ``P(y <= q_level)`` should equal ``level``.
    """

    quantiles: np.ndarray
    empirical_coverage: np.ndarray
    coverage_error: np.ndarray
    max_abs_error: float
    monotonic: bool
    n: int

    def to_dict(self) -> dict:
        return {
            "quantiles": self.quantiles.tolist(),
            "empirical_coverage": self.empirical_coverage.tolist(),
            "coverage_error": self.coverage_error.tolist(),
            "max_abs_error": self.max_abs_error,
            "monotonic": self.monotonic,
            "n": self.n,
        }


def quantile_coverage(
    y_true: np.ndarray,
    q_pred: np.ndarray,
    quantiles: np.ndarray,
    *,
    tol: float = 1e-9,
) -> QuantileCoverageReport:
    """Check per-quantile empirical coverage of a quantile-regression head.

    ``q_pred`` is ``(N, Q)`` (per-sample predicted quantile values) and ``quantiles`` the
    ``Q`` nominal levels in ``[0, 1]``. ``empirical_coverage[j] = mean(y <= q_pred[:, j])``.
    ``monotonic`` reports whether predicted quantiles are non-decreasing across levels for
    every sample (a quantile-crossing check).
    """
    y = np.asarray(y_true, dtype=np.float64).ravel()
    q = np.asarray(q_pred, dtype=np.float64)
    levels = np.asarray(quantiles, dtype=np.float64).ravel()
    if q.ndim == 1:
        q = q.reshape(-1, 1)
    if q.shape[1] != levels.size:
        raise ValueError(f"q_pred has {q.shape[1]} cols but {levels.size} quantile levels")
    if q.shape[0] != y.size:
        raise ValueError(f"q_pred has {q.shape[0]} rows but {y.size} targets")
    n = int(y.size)
    if n == 0:
        zeros = np.zeros(levels.size)
        return QuantileCoverageReport(levels, zeros, -levels, float("nan"), True, 0)
    emp = np.array([float(np.mean(y <= q[:, j])) for j in range(levels.size)])
    err = emp - levels
    monotonic = bool(np.all(np.diff(q, axis=1) >= -tol)) if levels.size > 1 else True
    return QuantileCoverageReport(levels, emp, err, float(np.max(np.abs(err))),
                                  monotonic, n)


@dataclass
class IntervalCoverageReport:
    """Prediction-interval coverage (PICP) and mean width (MPIW) vs a nominal level."""

    nominal: float
    picp: float
    mpiw: float
    coverage_error: float
    n: int

    def to_dict(self) -> dict:
        return {"nominal": self.nominal, "picp": self.picp, "mpiw": self.mpiw,
                "coverage_error": self.coverage_error, "n": self.n}


def prediction_interval_coverage(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    nominal: float,
) -> IntervalCoverageReport:
    """Coverage of prediction intervals ``[lower, upper]`` at a ``nominal`` level.

    PICP = fraction of targets inside the interval (should be ~``nominal``);
    MPIW = mean interval width; ``coverage_error = PICP - nominal``.
    """
    y = np.asarray(y_true, dtype=np.float64).ravel()
    lo = np.asarray(lower, dtype=np.float64).ravel()
    hi = np.asarray(upper, dtype=np.float64).ravel()
    if not (y.shape == lo.shape == hi.shape):
        raise ValueError("y_true, lower, upper must share shape")
    n = int(y.size)
    if n == 0:
        return IntervalCoverageReport(float(nominal), 0.0, 0.0, -float(nominal), 0)
    inside = (y >= np.minimum(lo, hi)) & (y <= np.maximum(lo, hi))
    picp = float(np.mean(inside))
    mpiw = float(np.mean(np.abs(hi - lo)))
    return IntervalCoverageReport(float(nominal), picp, mpiw, picp - float(nominal), n)


# ------------------------------------------------------------------- abstention
def confidence_from_proba(p_pred: np.ndarray) -> np.ndarray:
    """MODEL confidence from a binary probability: ``max(p, 1 - p)`` in ``[0.5, 1]``.

    This is a *calibratable probability*, NOT the engine's heuristic ranking/priority
    confidence (an ordinal preference score). Do not pass ranking scores here.
    """
    p = _as_prob(p_pred)
    return np.maximum(p, 1.0 - p)


@dataclass
class AbstentionResult:
    """Outcome of confidence-gated abstention on a set of binary predictions.

    ``justified`` is True only when abstaining measurably helps: selective accuracy on the
    retained (acted-upon) predictions exceeds full-set accuracy AND retained coverage is at
    least ``min_coverage``. Otherwise abstaining is not recommended for this set.
    """

    confidence_threshold: float
    abstained: np.ndarray
    n: int
    n_retained: int
    abstention_rate: float
    coverage: float
    full_accuracy: float
    selective_accuracy: float
    justified: bool
    extras: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"confidence_threshold": self.confidence_threshold, "n": self.n,
                "n_retained": self.n_retained, "abstention_rate": self.abstention_rate,
                "coverage": self.coverage, "full_accuracy": self.full_accuracy,
                "selective_accuracy": self.selective_accuracy,
                "justified": self.justified, **self.extras}


def apply_abstention(
    y_true: np.ndarray,
    p_pred: np.ndarray,
    *,
    confidence_threshold: float = 0.6,
    min_coverage: float = 0.1,
    decision_threshold: float = 0.5,
) -> AbstentionResult:
    """Treat low-confidence predictions as abstentions and report whether it is justified.

    A prediction abstains when its model confidence (``max(p, 1 - p)``) is below
    ``confidence_threshold``. Retained predictions use ``decision_threshold`` for the
    hard label. ``justified`` is True only if selective accuracy on the retained set beats
    full-set accuracy while retaining at least ``min_coverage`` of the data — i.e. the
    abstention is earned by measured performance, not assumed.

    Recommendation-only: callers decide what to do with abstentions; this never suppresses
    or defers any mandatory risk control.
    """
    p = _as_prob(p_pred)
    y = _as_binary(y_true)
    if p.shape != y.shape:
        raise ValueError(f"y_true {y.shape} and p_pred {p.shape} must match")
    n = int(p.size)
    conf = confidence_from_proba(p)
    abstained = conf < float(confidence_threshold)
    retained = ~abstained
    n_ret = int(retained.sum())
    coverage = float(n_ret / n) if n else 0.0
    abst_rate = float(1.0 - coverage)
    full_pred = (p >= float(decision_threshold)).astype(np.float64)
    full_acc = float(np.mean(full_pred == y)) if n else 0.0
    if n_ret > 0:
        sel_pred = full_pred[retained]
        sel_acc = float(np.mean(sel_pred == y[retained]))
    else:
        sel_acc = 0.0
    justified = bool(n_ret > 0 and coverage >= float(min_coverage) and sel_acc > full_acc)
    return AbstentionResult(
        confidence_threshold=float(confidence_threshold),
        abstained=abstained, n=n, n_retained=n_ret, abstention_rate=abst_rate,
        coverage=coverage, full_accuracy=full_acc, selective_accuracy=sel_acc,
        justified=justified, extras={"min_coverage": float(min_coverage),
                                     "decision_threshold": float(decision_threshold),
                                     "device": CALIBRATION_DEVICE})
