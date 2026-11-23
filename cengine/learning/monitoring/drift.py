"""Model drift monitoring (Milestone 6, §26).

Compares a LIVE inference batch against a REFERENCE distribution captured at TRAIN time and
maps the result onto explicit health states. The detectors implemented here are:

  1. Feature-distribution drift  — Population Stability Index (PSI) plus KL and
     Jensen-Shannon divergence, computed PER FEATURE against the train-time histogram.
  2. Missing-feature rate        — fraction of live rows whose feature value is non-finite.
  3. Prediction-distribution drift — PSI on the model's OWN output distribution (optional:
     requires a reference prediction histogram + live predictions).
  4. Calibration deterioration   — live Expected Calibration Error (ECE) minus the train-time
     ECE. Evaluated ONLY when real live outcome labels are supplied; labels are NEVER
     fabricated, so with no labels this detector is simply not evaluated.
  5. Out-of-distribution (OOD) input score — per-row RMS z-score against the train-time
     per-feature mean/std (a diagonal-covariance approximation); the OOD *rate* is the
     fraction of rows beyond an RMS-z threshold.

State machine (severity-ordered, the overall state is the most severe evaluated detector,
with two aggregate escalations below):

    HEALTHY < WARNING < ABSTAIN < DEGRADED < BLOCKED < ROLLBACK_REQUIRED

  * HEALTHY           — within train-time tolerances; nothing to do.
  * WARNING           — moderate drift; keep using but watch (advisory).
  * ABSTAIN           — the batch is too small to assert health (insufficient data). This is a
                        SAFE, NON-PROMOTING verdict: a too-small batch never certifies a model
                        as healthy. Distribution-estimation detectors are suppressed to ABSTAIN
                        here (their small-sample readings are noise), but data-integrity signals
                        (missing-feature, schema) remain authoritative and still surface.
  * DEGRADED          — a significant distribution/calibration shift in at least one detector.
  * BLOCKED           — features cannot be trusted at all this batch (missing-feature or OOD
                        rate past its block band, or a feature-count schema mismatch).
  * ROLLBACK_REQUIRED — an extreme single-detector shift, OR >= ``rollback_min_detectors``
                        detectors severe at once: advise reverting to the last approved model
                        or the deterministic baseline.

AUTHORITY INVARIANT (enforced + documented): drift monitoring is RECOMMENDATION-ONLY. A
``DriftReport`` NEVER grants trading authority to any model — ``grants_trading_authority`` is
hard-wired ``False``. Nothing here submits orders, promotes a model, opens a broker, or sits
on / defers the mandatory -0.5% protective exit (that exit is independent environment dynamics,
not a model action). The states are advisory signals for operators / the champion-challenger
and registry layers; those layers decide, under their own authorization, what to do.

COMPUTE: pure numpy/scipy on the CPU by design — these are histogram / divergence ops, not
MLX kernels. The device choice is recorded through ``DevicePolicy`` with ``gpu_supported=False``
so the CPU placement is OBSERVABLE and MEASURABLE (never a silent GPU claim), consistent with
the Milestone 6 hardware contract.

THRESHOLD DEFAULTS (independently justified, all configurable via ``DriftThresholds``):
  * PSI bands 0.10 / 0.25 — the de-facto industry standard: <0.10 stable, 0.10-0.25 moderate,
    >=0.25 significant. 0.50 is used as a severe single-feature rollback trigger.
  * Missing-feature 0.02 / 0.10 / 0.50 — a few percent missing is tolerable noise, >10% is a
    real pipeline problem, >=50% means the feature vector is essentially unusable (block).
  * OOD rate 0.05 / 0.20 / 0.50 at RMS-z 4.0 — under the reference the per-row RMS z-score is
    ~1, so a 4-sigma RMS row is a genuine outlier; >5% outliers warns, >20% degrades, >=50%
    means most inputs are off-manifold (block).
  * ECE delta 0.05 / 0.10 — a 5-point absolute rise in calibration error is a meaningful
    deterioration, 10 points is severe.
  * ``min_eval_samples`` 100 — below this a PSI estimate is too noisy to trust -> ABSTAIN.
  * ``rollback_min_detectors`` 2 — one severe detector is DEGRADED/BLOCKED; two or more
    simultaneous severe signals indicate structural failure warranting rollback.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np
from scipy.spatial.distance import jensenshannon
from scipy.stats import entropy

from cengine.learning.hardware import DevicePolicy

DRIFT_MONITOR_VERSION = "drift-monitor-1.0.0"


class DriftState(str, Enum):
    HEALTHY = "healthy"
    WARNING = "warning"
    ABSTAIN = "abstain"
    DEGRADED = "degraded"
    BLOCKED = "blocked"
    ROLLBACK_REQUIRED = "rollback_required"


# Severity ordering used to take the most-severe state across detectors.
_SEVERITY: dict = {
    DriftState.HEALTHY: 0,
    DriftState.WARNING: 1,
    DriftState.ABSTAIN: 2,
    DriftState.DEGRADED: 3,
    DriftState.BLOCKED: 4,
    DriftState.ROLLBACK_REQUIRED: 5,
}

# Advisory, recommendation-only guidance per state. None of these are actions taken here.
_ACTIONS: dict = {
    DriftState.HEALTHY: "no action (advisory): within train-time tolerances",
    DriftState.WARNING: "advisory: moderate drift — keep using, increase monitoring cadence",
    DriftState.ABSTAIN: "advisory: batch too small to assert health — do not certify on it",
    DriftState.DEGRADED: "advisory: significant drift — recommend demoting to shadow / "
                         "reducing reliance (operator decides; no automatic action)",
    DriftState.BLOCKED: "advisory: inputs untrustworthy this batch — recommend the service "
                        "abstain from model recommendations (no automatic action)",
    DriftState.ROLLBACK_REQUIRED: "advisory: recommend operator revert to the last approved "
                                  "model or the deterministic baseline (no automatic action)",
}

_DEGRADED_OR_WORSE = _SEVERITY[DriftState.DEGRADED]


def _max_state(states) -> DriftState:
    best = DriftState.HEALTHY
    for s in states:
        if _SEVERITY[s] > _SEVERITY[best]:
            best = s
    return best


def _band(value: float, warn: float, degrade: float, extreme: float,
          extreme_state: DriftState) -> DriftState:
    """Map a non-negative drift metric onto a state via three ascending thresholds."""
    if value >= extreme:
        return extreme_state
    if value >= degrade:
        return DriftState.DEGRADED
    if value >= warn:
        return DriftState.WARNING
    return DriftState.HEALTHY


# --------------------------------------------------------------------------- thresholds


@dataclass(frozen=True)
class DriftThresholds:
    """Configurable, independently-justified thresholds (see module docstring)."""

    psi_warn: float = 0.10
    psi_degrade: float = 0.25
    psi_rollback: float = 0.50
    missing_warn: float = 0.02
    missing_degrade: float = 0.10
    missing_block: float = 0.50
    pred_psi_warn: float = 0.10
    pred_psi_degrade: float = 0.25
    pred_psi_rollback: float = 0.50
    ood_z: float = 4.0
    ood_rate_warn: float = 0.05
    ood_rate_degrade: float = 0.20
    ood_rate_block: float = 0.50
    ece_delta_warn: float = 0.05
    ece_delta_degrade: float = 0.10
    min_eval_samples: int = 100
    rollback_min_detectors: int = 2
    n_bins: int = 10
    smoothing: float = 1e-6


# --------------------------------------------------------------------------- numerics


def _quantile_edges(values: np.ndarray, n_bins: int) -> np.ndarray:
    """Finite quantile bin edges from reference values (ties collapsed, constants handled)."""
    v = values[np.isfinite(values)]
    if v.size == 0:
        return np.array([0.0, 1.0])
    edges = np.unique(np.quantile(v, np.linspace(0.0, 1.0, n_bins + 1)))
    if edges.size < 2:
        c = float(edges[0])
        edges = np.array([c - 0.5, c + 0.5])
    return edges


def _hist_props(values: np.ndarray, edges: np.ndarray, smoothing: float) -> np.ndarray:
    """Smoothed, normalized histogram proportions. Out-of-range values clip into end bins."""
    v = values[np.isfinite(values)]
    if v.size:
        v = np.clip(v, edges[0], edges[-1])
    counts, _ = np.histogram(v, bins=edges)
    counts = counts.astype(np.float64) + smoothing
    total = counts.sum()
    return counts / total if total > 0 else np.full(counts.shape, 1.0 / counts.size)


def _psi(ref_p: np.ndarray, live_p: np.ndarray) -> float:
    """Population Stability Index: sum (live - ref) * ln(live / ref)."""
    return float(np.sum((live_p - ref_p) * np.log(live_p / ref_p)))


def _kl(live_p: np.ndarray, ref_p: np.ndarray) -> float:
    """KL(live || ref) in nats (info lost approximating the live dist by the reference)."""
    return float(entropy(live_p, ref_p))


def _js(live_p: np.ndarray, ref_p: np.ndarray) -> float:
    """Jensen-Shannon divergence in bits, symmetric and bounded in [0, 1]."""
    return float(jensenshannon(live_p, ref_p, base=2) ** 2)


def expected_calibration_error(y_true: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    """Equal-width-bin Expected Calibration Error for probabilistic predictions."""
    y_true = np.asarray(y_true, dtype=np.float64)
    p = np.clip(np.asarray(p, dtype=np.float64), 0.0, 1.0)
    n = p.size
    if n == 0:
        return 0.0
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        m = idx == b
        cnt = int(m.sum())
        if cnt == 0:
            continue
        ece += (cnt / n) * abs(float(y_true[m].mean()) - float(p[m].mean()))
    return float(ece)


# --------------------------------------------------------------------------- reference


@dataclass
class ReferenceDistribution:
    """Train-time reference captured ONCE, then compared against every live batch.

    Holds per-feature histogram edges + proportions (for PSI/KL/JS), per-feature mean/std
    (for the OOD z-score), and optionally a prediction histogram and a reference ECE.
    """

    feature_names: tuple
    bin_edges: list
    ref_props: list
    means: np.ndarray
    stds: np.ndarray
    pred_bin_edges: Optional[np.ndarray] = None
    pred_props: Optional[np.ndarray] = None
    ref_ece: Optional[float] = None
    n_ref: int = 0
    n_bins: int = 10
    smoothing: float = 1e-6

    @classmethod
    def from_training(cls, X: np.ndarray, feature_names, *,
                      predictions: Optional[np.ndarray] = None,
                      y_true: Optional[np.ndarray] = None,
                      n_bins: int = 10, smoothing: float = 1e-6) -> "ReferenceDistribution":
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        names = tuple(feature_names)
        if len(names) != X.shape[1]:
            raise ValueError(
                f"feature_names ({len(names)}) != X columns ({X.shape[1]})")
        bin_edges, ref_props = [], []
        for i in range(X.shape[1]):
            edges = _quantile_edges(X[:, i], n_bins)
            bin_edges.append(edges)
            ref_props.append(_hist_props(X[:, i], edges, smoothing))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            means = np.nanmean(X, axis=0)
            stds = np.nanstd(X, axis=0)
        means = np.where(np.isfinite(means), means, 0.0)
        stds = np.where(np.isfinite(stds) & (stds >= 1e-8), stds, 1.0)
        pred_edges = pred_props = ref_ece = None
        if predictions is not None:
            pv = np.asarray(predictions, dtype=np.float64).ravel()
            pred_edges = _quantile_edges(pv, n_bins)
            pred_props = _hist_props(pv, pred_edges, smoothing)
            if y_true is not None:
                ref_ece = expected_calibration_error(y_true, pv, n_bins)
        return cls(feature_names=names, bin_edges=bin_edges, ref_props=ref_props,
                   means=means, stds=stds, pred_bin_edges=pred_edges, pred_props=pred_props,
                   ref_ece=ref_ece, n_ref=int(X.shape[0]), n_bins=n_bins, smoothing=smoothing)

    def to_dict(self) -> dict:
        return {
            "feature_names": list(self.feature_names),
            "bin_edges": [e.tolist() for e in self.bin_edges],
            "ref_props": [p.tolist() for p in self.ref_props],
            "means": self.means.tolist(),
            "stds": self.stds.tolist(),
            "pred_bin_edges": None if self.pred_bin_edges is None else self.pred_bin_edges.tolist(),
            "pred_props": None if self.pred_props is None else self.pred_props.tolist(),
            "ref_ece": self.ref_ece,
            "n_ref": self.n_ref,
            "n_bins": self.n_bins,
            "smoothing": self.smoothing,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ReferenceDistribution":
        return cls(
            feature_names=tuple(d["feature_names"]),
            bin_edges=[np.asarray(e, dtype=np.float64) for e in d["bin_edges"]],
            ref_props=[np.asarray(p, dtype=np.float64) for p in d["ref_props"]],
            means=np.asarray(d["means"], dtype=np.float64),
            stds=np.asarray(d["stds"], dtype=np.float64),
            pred_bin_edges=(None if d.get("pred_bin_edges") is None
                            else np.asarray(d["pred_bin_edges"], dtype=np.float64)),
            pred_props=(None if d.get("pred_props") is None
                        else np.asarray(d["pred_props"], dtype=np.float64)),
            ref_ece=d.get("ref_ece"),
            n_ref=int(d.get("n_ref", 0)),
            n_bins=int(d.get("n_bins", 10)),
            smoothing=float(d.get("smoothing", 1e-6)),
        )


# --------------------------------------------------------------------------- results


@dataclass
class DetectorResult:
    name: str
    state: DriftState
    metric: float
    evaluated: bool = True
    # ``sample_sensitive`` detectors estimate a DISTRIBUTION (PSI/KL/JS/OOD-rate/ECE) and are
    # statistically unreliable on a tiny batch — small-sample noise alone can fake a severe
    # reading. When the batch is below ``min_eval_samples`` these are suppressed to at most
    # ABSTAIN. Data-integrity detectors (missing-feature, schema) are reliable at ANY size.
    sample_sensitive: bool = True
    detail: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"name": self.name, "state": self.state.value, "metric": self.metric,
                "evaluated": self.evaluated, "sample_sensitive": self.sample_sensitive,
                "detail": self.detail}


@dataclass
class DriftReport:
    overall_state: DriftState
    detectors: list
    feature_psi: dict
    feature_kl: dict
    feature_js: dict
    missing_rates: dict
    prediction_psi: Optional[float]
    calibration_delta: Optional[float]
    ood_rate: float
    ood_mean_score: float
    n_live: int
    insufficient_data: bool
    device: str
    device_reason: str
    recommended_action: str
    # Hard-wired invariant: drift monitoring is advisory and grants NO trading authority.
    grants_trading_authority: bool = False
    monitor_version: str = DRIFT_MONITOR_VERSION
    now_ns: int = 0

    @property
    def healthy(self) -> bool:
        return self.overall_state == DriftState.HEALTHY

    def to_dict(self) -> dict:
        return {
            "overall_state": self.overall_state.value,
            "detectors": [d.to_dict() for d in self.detectors],
            "feature_psi": self.feature_psi,
            "feature_kl": self.feature_kl,
            "feature_js": self.feature_js,
            "missing_rates": self.missing_rates,
            "prediction_psi": self.prediction_psi,
            "calibration_delta": self.calibration_delta,
            "ood_rate": self.ood_rate,
            "ood_mean_score": self.ood_mean_score,
            "n_live": self.n_live,
            "insufficient_data": self.insufficient_data,
            "device": self.device,
            "device_reason": self.device_reason,
            "recommended_action": self.recommended_action,
            "grants_trading_authority": self.grants_trading_authority,
            "monitor_version": self.monitor_version,
            "now_ns": self.now_ns,
        }


# --------------------------------------------------------------------------- monitor


class DriftMonitor:
    """Compares a live batch against a train-time ``ReferenceDistribution``.

    ADVISORY ONLY. ``evaluate`` returns a ``DriftReport`` with an explicit state and a
    recommendation; it takes no action, promotes no model, and grants no trading authority.
    Works on any batch size: an insufficient batch yields ABSTAIN rather than a false HEALTHY.
    """

    def __init__(self, reference: ReferenceDistribution,
                 thresholds: Optional[DriftThresholds] = None, *,
                 device_policy: Optional[DevicePolicy] = None) -> None:
        self.reference = reference
        self.thresholds = thresholds or DriftThresholds()
        self.policy = device_policy or DevicePolicy()

    def evaluate(self, X_live: np.ndarray, *,
                 predictions: Optional[np.ndarray] = None,
                 y_live: Optional[np.ndarray] = None, now_ns: int = 0) -> DriftReport:
        t = self.thresholds
        ref = self.reference
        X = np.asarray(X_live, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        n_live, n_feat = X.shape
        names = ref.feature_names

        # Device is CPU by design: histogram/divergence ops are not MLX kernels. Recording
        # the decision through DevicePolicy keeps the CPU placement observable/measurable.
        decision = self.policy.choose(
            "drift_monitor", n_elements=n_live * max(1, n_feat), gpu_supported=False)
        device, device_reason = decision.device, decision.reason

        # Feature-count schema mismatch -> inputs are not interpretable -> BLOCKED.
        if n_feat != len(names):
            det = DetectorResult(
                "feature_schema", DriftState.BLOCKED, float(n_feat),
                detail={"expected_features": len(names), "got_features": n_feat})
            return DriftReport(
                overall_state=DriftState.BLOCKED, detectors=[det], feature_psi={},
                feature_kl={}, feature_js={}, missing_rates={}, prediction_psi=None,
                calibration_delta=None, ood_rate=0.0, ood_mean_score=0.0, n_live=n_live,
                insufficient_data=n_live < t.min_eval_samples, device=device,
                device_reason=device_reason,
                recommended_action=_ACTIONS[DriftState.BLOCKED], now_ns=now_ns)

        # (1) Feature-distribution drift: PSI/KL/JS per feature.
        feature_psi, feature_kl, feature_js, feat_states = {}, {}, {}, []
        for i, name in enumerate(names):
            live_p = _hist_props(X[:, i], ref.bin_edges[i], t.smoothing)
            ref_p = ref.ref_props[i]
            psi = _psi(ref_p, live_p)
            feature_psi[name] = psi
            feature_kl[name] = _kl(live_p, ref_p)
            feature_js[name] = _js(live_p, ref_p)
            feat_states.append(
                _band(psi, t.psi_warn, t.psi_degrade, t.psi_rollback,
                      DriftState.ROLLBACK_REQUIRED))
        worst_feat_psi = max(feature_psi.values()) if feature_psi else 0.0
        worst_feature = max(feature_psi, key=feature_psi.get) if feature_psi else None
        feat_state = _max_state(feat_states) if feat_states else DriftState.HEALTHY

        # (2) Missing-feature rate.
        missing = (~np.isfinite(X)).mean(axis=0) if n_live else np.zeros(n_feat)
        missing_rates = {names[i]: float(missing[i]) for i in range(n_feat)}
        worst_missing = float(missing.max()) if n_feat else 0.0
        worst_missing_feature = (max(missing_rates, key=missing_rates.get)
                                 if missing_rates else None)
        missing_state = _band(worst_missing, t.missing_warn, t.missing_degrade,
                              t.missing_block, DriftState.BLOCKED)

        # (5) OOD input score: per-row RMS z-score vs train-time mean/std (diagonal approx).
        with warnings.catch_warnings(), np.errstate(invalid="ignore", divide="ignore"):
            warnings.simplefilter("ignore", category=RuntimeWarning)
            z = (X - ref.means) / ref.stds
            rms = np.sqrt(np.nanmean(z ** 2, axis=1))
        valid = np.isfinite(rms)
        if valid.any():
            scores = rms[valid]
            ood_rate = float(np.mean(scores > t.ood_z))
            ood_mean = float(np.mean(scores))
        else:
            ood_rate, ood_mean = 0.0, 0.0
        ood_state = _band(ood_rate, t.ood_rate_warn, t.ood_rate_degrade, t.ood_rate_block,
                          DriftState.BLOCKED)

        # (3) Prediction-distribution drift (optional).
        prediction_psi: Optional[float] = None
        pred_detector = DetectorResult(
            "prediction_distribution", DriftState.HEALTHY, 0.0, evaluated=False,
            detail={"reason": "no reference and/or live predictions"})
        if predictions is not None and ref.pred_props is not None:
            p_live = _hist_props(
                np.asarray(predictions, dtype=np.float64).ravel(), ref.pred_bin_edges,
                t.smoothing)
            prediction_psi = _psi(ref.pred_props, p_live)
            pred_state = _band(prediction_psi, t.pred_psi_warn, t.pred_psi_degrade,
                               t.pred_psi_rollback, DriftState.ROLLBACK_REQUIRED)
            pred_detector = DetectorResult(
                "prediction_distribution", pred_state, prediction_psi,
                detail={"psi": prediction_psi})

        # (4) Calibration deterioration (optional; NEVER fabricates labels).
        calibration_delta: Optional[float] = None
        cal_detector = DetectorResult(
            "calibration", DriftState.HEALTHY, 0.0, evaluated=False,
            detail={"reason": "no live labels (never fabricated)"})
        if y_live is not None and predictions is not None and ref.ref_ece is not None:
            live_ece = expected_calibration_error(
                y_live, np.asarray(predictions, dtype=np.float64).ravel(), t.n_bins)
            calibration_delta = float(live_ece - ref.ref_ece)
            d = max(0.0, calibration_delta)
            cal_state = (DriftState.DEGRADED if d >= t.ece_delta_degrade
                         else DriftState.WARNING if d >= t.ece_delta_warn
                         else DriftState.HEALTHY)
            cal_detector = DetectorResult(
                "calibration", cal_state, calibration_delta,
                detail={"live_ece": live_ece, "ref_ece": ref.ref_ece})

        detectors = [
            DetectorResult("feature_distribution", feat_state, worst_feat_psi,
                           detail={"worst_feature": worst_feature, "psi": feature_psi,
                                   "kl": feature_kl, "js": feature_js}),
            DetectorResult("missing_feature", missing_state, worst_missing,
                           sample_sensitive=False,
                           detail={"worst_feature": worst_missing_feature,
                                   "rates": missing_rates}),
            DetectorResult("out_of_distribution", ood_state, ood_rate,
                           detail={"mean_rms_z": ood_mean, "ood_z": t.ood_z}),
            pred_detector,
            cal_detector,
        ]

        # Aggregate. With insufficient data the distribution-estimation detectors are
        # unreliable (small-sample PSI noise can fake a severe reading), so they are
        # suppressed to at most ABSTAIN — a safe, non-promoting verdict that never certifies
        # health. Data-integrity detectors (missing-feature, schema) stay authoritative at any
        # size, so a genuinely broken feature vector still surfaces. With sufficient data the
        # overall state is the most-severe evaluated detector, escalated to ROLLBACK_REQUIRED
        # when >= ``rollback_min_detectors`` detectors are severe at once.
        insufficient = n_live < t.min_eval_samples
        evaluated = [d for d in detectors if d.evaluated]
        if insufficient:
            integrity = [d.state for d in evaluated if not d.sample_sensitive]
            overall = _max_state([DriftState.ABSTAIN, *integrity])
        else:
            overall = _max_state([d.state for d in evaluated]) if evaluated else DriftState.ABSTAIN
            severe = [d for d in evaluated if _SEVERITY[d.state] >= _DEGRADED_OR_WORSE]
            if len(severe) >= t.rollback_min_detectors:
                overall = _max_state([overall, DriftState.ROLLBACK_REQUIRED])

        return DriftReport(
            overall_state=overall, detectors=detectors, feature_psi=feature_psi,
            feature_kl=feature_kl, feature_js=feature_js, missing_rates=missing_rates,
            prediction_psi=prediction_psi, calibration_delta=calibration_delta,
            ood_rate=ood_rate, ood_mean_score=ood_mean, n_live=n_live,
            insufficient_data=insufficient, device=device, device_reason=device_reason,
            recommended_action=_ACTIONS[overall], now_ns=now_ns)
