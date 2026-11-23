"""Economic metrics with dependence-aware uncertainty (Milestone 6, §19).

Scores a series of *per-decision net returns* (realistic, cost-inclusive — gross metrics are
never reported here) with the economic statistics that matter for deployment: net return,
expectancy, profit factor, Sharpe, Sortino, max drawdown, average win/loss, win rate and
turnover.

Two honesty requirements are enforced structurally, not by convention:

* **Dependence-aware uncertainty.** Trading decisions are serially dependent — 24k bars are
  not 24k independent experiments. Naive i.i.d. confidence intervals understate the true
  uncertainty. Every uncertainty estimate here uses a *stationary* or *moving-block*
  bootstrap that preserves short-range dependence, and an honest *effective sample size*
  (ESS) is reported alongside the raw count via the integrated autocorrelation time.

* **Synthetic evidence is never a profitability claim.** Every output carries an explicit
  ``data_label`` of ``'synthetic'`` or ``'real'``. When the label is ``'synthetic'`` the
  result reports ``profitability_claim_allowed = False`` and its verdict text states plainly
  that the numbers are illustrative only. Profitability may be asserted ONLY from
  ``'real'``-labeled realized outcomes.

Pure numpy/scipy (CPU). This module is recommendation-only research infrastructure: it never
places orders, never touches the mandatory -0.5% protective-exit path, and never fabricates
returns. Insufficient data yields an explicitly-flagged, untrained/research-only result —
never a promoted one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

ECONOMIC_METRICS_VERSION = "economic-metrics-1.0.0"

# Accepted provenance labels. 'synthetic' results may NEVER be used to claim profitability.
DATA_LABEL_SYNTHETIC = "synthetic"
DATA_LABEL_REAL = "real"
_VALID_LABELS = (DATA_LABEL_SYNTHETIC, DATA_LABEL_REAL)

# Sample-size floors below which no verdict beyond "research-only" is honest. These gate the
# *verdict text*; the metrics themselves are always computed so the pipeline keeps working.
_MIN_DECISIONS_FOR_VERDICT = 30
_MIN_EFFECTIVE_SAMPLE_FOR_VERDICT = 20.0


def validate_data_label(data_label: str) -> str:
    """Return a normalized data label or raise if it is not 'synthetic' / 'real'.

    A missing/invalid label is a hard error: an unlabeled economic result could be mistaken
    for a real profitability claim, which this module exists to prevent.
    """
    label = str(data_label).strip().lower()
    if label not in _VALID_LABELS:
        raise ValueError(
            f"data_label must be one of {_VALID_LABELS!r}; got {data_label!r}. "
            "Economic results must declare provenance so synthetic evidence can never be "
            "mistaken for a real profitability claim."
        )
    return label


def _clean_returns(net_returns) -> np.ndarray:
    """Coerce to a 1-D float64 array of finite per-decision net returns."""
    arr = np.asarray(net_returns, dtype=np.float64).ravel()
    if arr.size == 0:
        return arr
    return arr[np.isfinite(arr)]


# --------------------------------------------------------------------------- metrics


@dataclass
class EconomicMetrics:
    """Economic summary of a per-decision net-return series.

    ``net_return`` is the simple aggregate (sum) of per-decision net returns;
    ``net_return_compounded`` is the geometrically-compounded total. ``avg_loss`` is reported
    as a *negative* number (the mean of losing returns). ``max_drawdown`` is a non-negative
    magnitude (the depth of the worst peak-to-trough decline on the equity curve).
    """

    version: str = ECONOMIC_METRICS_VERSION
    data_label: str = DATA_LABEL_SYNTHETIC
    n: int = 0
    net_return: float = 0.0
    net_return_compounded: float = 0.0
    net_expectancy: float = 0.0
    profit_factor: float = 0.0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    sharpe: float = 0.0
    sortino: float = 0.0
    max_drawdown: float = 0.0
    turnover: Optional[float] = None
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    n_wins: int = 0
    n_losses: int = 0
    drawdown_mode: str = "compound"
    periods_per_year: Optional[float] = None
    profitability_claim_allowed: bool = False
    note: str = ""

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


def _max_drawdown(returns: np.ndarray, mode: str) -> float:
    """Worst peak-to-trough decline as a non-negative magnitude.

    ``mode='compound'`` builds a multiplicative equity curve ``prod(1+r)`` and measures
    fractional drawdown relative to the running peak. ``mode='additive'`` builds a cumulative
    sum and measures absolute drawdown from the running peak.
    """
    if returns.size == 0:
        return 0.0
    if mode == "additive":
        equity = np.cumsum(returns)
        peak = np.maximum.accumulate(equity)
        dd = equity - peak  # <= 0
        return float(-dd.min())
    # compound (default): guard against total ruin so the curve stays well-defined.
    growth = np.clip(1.0 + returns, 1e-12, None)
    equity = np.cumprod(growth)
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak  # <= 0
    return float(-dd.min())


def _turnover(weights) -> Optional[float]:
    """Mean per-step turnover from a position/weight series, or None if not supplied.

    Accepts a 1-D series (single instrument weight over time) -> mean |Δw|, or a 2-D
    (n_steps, n_instruments) matrix -> mean over steps of the row-summed |Δw|.
    """
    if weights is None:
        return None
    w = np.asarray(weights, dtype=np.float64)
    if w.ndim == 1:
        if w.size < 2:
            return 0.0
        return float(np.mean(np.abs(np.diff(w))))
    if w.ndim == 2:
        if w.shape[0] < 2:
            return 0.0
        return float(np.mean(np.sum(np.abs(np.diff(w, axis=0)), axis=1)))
    raise ValueError("weights must be 1-D or 2-D")


def economic_metrics(
    net_returns,
    *,
    data_label: str,
    weights=None,
    risk_free: float = 0.0,
    periods_per_year: Optional[float] = None,
    drawdown_mode: str = "compound",
) -> EconomicMetrics:
    """Compute economic metrics over a series of per-decision NET returns.

    Parameters
    ----------
    net_returns:
        1-D fractional per-decision net returns (0.01 == +1%). Must already include
        commissions, slippage, spread and borrow — gross inputs are a caller error.
    data_label:
        ``'real'`` or ``'synthetic'``. Synthetic results can never claim profitability.
    weights:
        Optional position/weight series used only to compute turnover.
    risk_free:
        Per-period minimum-acceptable return (MAR) for Sharpe/Sortino. Default 0.
    periods_per_year:
        If given, Sharpe/Sortino are annualized by ``sqrt(periods_per_year)``; otherwise they
        are reported per-period (dimensionless).
    drawdown_mode:
        ``'compound'`` (default) or ``'additive'``.
    """
    label = validate_data_label(data_label)
    r = _clean_returns(net_returns)
    m = EconomicMetrics(
        data_label=label,
        drawdown_mode=drawdown_mode,
        periods_per_year=periods_per_year,
        profitability_claim_allowed=False,
    )
    m.turnover = _turnover(weights)
    if r.size == 0:
        m.note = "insufficient_data: empty return series (research-only, never promoted)"
        return m

    m.n = int(r.size)
    wins = r[r > 0.0]
    losses = r[r < 0.0]
    m.n_wins = int(wins.size)
    m.n_losses = int(losses.size)
    m.gross_profit = float(wins.sum())
    m.gross_loss = float(-losses.sum())  # non-negative magnitude
    m.net_return = float(r.sum())
    m.net_return_compounded = float(np.prod(np.clip(1.0 + r, 1e-12, None)) - 1.0)
    m.net_expectancy = float(r.mean())
    m.win_rate = float(wins.size / r.size)
    m.avg_win = float(wins.mean()) if wins.size else 0.0
    m.avg_loss = float(losses.mean()) if losses.size else 0.0  # negative or 0

    if m.gross_loss <= 0.0:
        # No losing decisions: profit factor is undefined/infinite. Report inf iff there is
        # realized profit, else 0 (flat series), and never silently coerce to a finite value.
        m.profit_factor = float("inf") if m.gross_profit > 0.0 else 0.0
    else:
        m.profit_factor = float(m.gross_profit / m.gross_loss)

    excess = r - risk_free
    ann = float(np.sqrt(periods_per_year)) if periods_per_year else 1.0
    sd = float(np.std(excess, ddof=1)) if excess.size >= 2 else 0.0
    m.sharpe = float(excess.mean() / sd * ann) if sd > 0.0 else 0.0
    # Sortino downside deviation: sqrt(mean(min(0, excess)^2)) over the FULL sample (standard
    # convention), so a series with no downside has zero downside deviation.
    downside = np.minimum(excess, 0.0)
    dd_dev = float(np.sqrt(np.mean(downside ** 2)))
    m.sortino = float(excess.mean() / dd_dev * ann) if dd_dev > 0.0 else 0.0
    m.max_drawdown = _max_drawdown(r, drawdown_mode)

    # Profitability may be asserted ONLY from real, adequately-sized evidence.
    m.profitability_claim_allowed = label == DATA_LABEL_REAL and m.n >= _MIN_DECISIONS_FOR_VERDICT
    if label == DATA_LABEL_SYNTHETIC:
        m.note = "synthetic evidence: illustrative only — NOT a profitability claim"
    elif m.n < _MIN_DECISIONS_FOR_VERDICT:
        m.note = (f"real but small (n={m.n}<{_MIN_DECISIONS_FOR_VERDICT}): "
                  "research-only, not a profitability claim")
    else:
        m.note = "real realized outcomes"
    return m


# ------------------------------------------------------- dependence-aware uncertainty


@dataclass
class SampleSizeReport:
    """Honest sample-size accounting for serially-dependent decision returns."""

    n: int
    effective_sample_size: float
    integrated_autocorr_time: float
    autocorr_lag1: float
    data_label: str
    warning: Optional[str] = None

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


def _autocorr(x: np.ndarray, max_lag: int) -> np.ndarray:
    """Biased sample autocorrelations rho[1..max_lag] of a 1-D series."""
    xc = x - x.mean()
    denom = float(np.dot(xc, xc))
    if denom <= 0.0:
        return np.zeros(max_lag, dtype=np.float64)
    out = np.empty(max_lag, dtype=np.float64)
    for k in range(1, max_lag + 1):
        out[k - 1] = float(np.dot(xc[:-k], xc[k:]) / denom)
    return out


def sample_size_report(
    values, *, data_label: str, max_lag: Optional[int] = None
) -> SampleSizeReport:
    """Report raw n and an effective sample size via the integrated autocorrelation time.

    ESS = n / IACT where IACT = 1 + 2 * sum_{k>=1} rho_k, truncated at the first non-positive
    autocorrelation (Geyer's initial-positive-sequence idea, simplified). Positive serial
    correlation (IACT > 1) shrinks the effective sample, which is the honest regime for
    trading-decision series.
    """
    label = validate_data_label(data_label)
    x = _clean_returns(values)
    n = int(x.size)
    if n < 3:
        return SampleSizeReport(
            n=n, effective_sample_size=float(max(n, 0)), integrated_autocorr_time=1.0,
            autocorr_lag1=0.0, data_label=label,
            warning="insufficient_data: need >=3 observations for autocorrelation",
        )
    cap = max_lag if max_lag is not None else min(n - 1, max(1, n // 4))
    cap = int(min(cap, n - 1))
    rho = _autocorr(x, cap)
    iact = 1.0
    for rk in rho:
        if rk <= 0.0:
            break
        iact += 2.0 * rk
    iact = max(iact, 1e-9)
    ess = float(np.clip(n / iact, 1.0, float(n)))
    warning = None
    if ess < _MIN_EFFECTIVE_SAMPLE_FOR_VERDICT:
        warning = (f"low effective sample size (ESS={ess:.1f} from n={n}); "
                   "treat uncertainty as wide — research-only")
    return SampleSizeReport(
        n=n, effective_sample_size=ess, integrated_autocorr_time=float(iact),
        autocorr_lag1=float(rho[0]), data_label=label, warning=warning,
    )


@dataclass
class BootstrapCI:
    """A dependence-aware bootstrap confidence interval for a scalar statistic."""

    statistic: str
    point_estimate: float
    ci_low: float
    ci_high: float
    confidence: float
    method: str
    n_boot: int
    expected_block_length: float
    n: int
    effective_sample_size: float
    data_label: str
    standard_error: float = 0.0

    def brackets(self, value: float) -> bool:
        """True iff ``value`` lies within [ci_low, ci_high]."""
        return self.ci_low <= value <= self.ci_high

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


def _default_block_length(n: int) -> float:
    """Heuristic expected block length ~ n**(1/3) (the known optimal rate order)."""
    return float(max(1.0, round(n ** (1.0 / 3.0))))


def stationary_bootstrap_indices(
    n: int, *, expected_block_length: float, n_boot: int, rng: np.random.Generator
) -> np.ndarray:
    """(n_boot, n) resample indices from the stationary bootstrap (Politis & Romano 1994).

    Blocks have random geometric length with mean ``expected_block_length`` and wrap around
    the series, so the resample preserves short-range serial dependence and is stationary.
    """
    if n <= 0:
        return np.empty((n_boot, 0), dtype=np.int64)
    p = 1.0 / max(1.0, float(expected_block_length))
    p = min(1.0, p)
    out = np.empty((n_boot, n), dtype=np.int64)
    for b in range(n_boot):
        idx = np.empty(n, dtype=np.int64)
        cur = int(rng.integers(0, n))
        for t in range(n):
            idx[t] = cur
            if rng.random() < p:
                cur = int(rng.integers(0, n))  # start a fresh block
            else:
                cur = (cur + 1) % n  # continue the current block
        out[b] = idx
    return out


def moving_block_bootstrap_indices(
    n: int, *, block_length: int, n_boot: int, rng: np.random.Generator
) -> np.ndarray:
    """(n_boot, n) resample indices from the circular moving-block bootstrap."""
    if n <= 0:
        return np.empty((n_boot, 0), dtype=np.int64)
    bl = int(max(1, min(block_length, n)))
    n_blocks = int(np.ceil(n / bl))
    out = np.empty((n_boot, n), dtype=np.int64)
    for b in range(n_boot):
        starts = rng.integers(0, n, size=n_blocks)
        offsets = np.arange(bl)
        block = (starts[:, None] + offsets[None, :]) % n  # circular
        out[b] = block.ravel()[:n]
    return out


def bootstrap_metric_ci(
    values,
    *,
    data_label: str,
    statistic_fn: Callable[[np.ndarray], float] = np.mean,
    statistic_name: str = "mean",
    method: str = "stationary",
    expected_block_length: Optional[float] = None,
    n_boot: int = 2000,
    confidence: float = 0.95,
    seed: int = 7,
) -> BootstrapCI:
    """Dependence-aware bootstrap CI for a scalar statistic of a return series.

    ``method`` is ``'stationary'`` (default, geometric random block length) or
    ``'moving_block'`` (fixed circular blocks). Both preserve short-range serial dependence,
    unlike an i.i.d. bootstrap, so the interval does not understate uncertainty on dependent
    trading returns. The percentile interval is returned together with the ESS so callers can
    see how much independent information actually backs it.
    """
    label = validate_data_label(data_label)
    x = _clean_returns(values)
    n = int(x.size)
    ess_report = sample_size_report(x, data_label=label) if n >= 1 else None
    ess = ess_report.effective_sample_size if ess_report else float(n)
    if n == 0:
        return BootstrapCI(
            statistic=statistic_name, point_estimate=float("nan"), ci_low=float("nan"),
            ci_high=float("nan"), confidence=confidence, method=method, n_boot=0,
            expected_block_length=0.0, n=0, effective_sample_size=0.0, data_label=label,
        )
    point = float(statistic_fn(x))
    if n == 1:
        return BootstrapCI(
            statistic=statistic_name, point_estimate=point, ci_low=point, ci_high=point,
            confidence=confidence, method=method, n_boot=0,
            expected_block_length=1.0, n=1, effective_sample_size=1.0, data_label=label,
        )
    bl = expected_block_length if expected_block_length is not None else _default_block_length(n)
    rng = np.random.default_rng(seed)
    if method == "moving_block":
        idx = moving_block_bootstrap_indices(
            n, block_length=int(round(bl)), n_boot=n_boot, rng=rng)
    elif method == "stationary":
        idx = stationary_bootstrap_indices(
            n, expected_block_length=bl, n_boot=n_boot, rng=rng)
    else:
        raise ValueError("method must be 'stationary' or 'moving_block'")
    samples = np.array([statistic_fn(x[row]) for row in idx], dtype=np.float64)
    samples = samples[np.isfinite(samples)]
    alpha = 1.0 - confidence
    lo = float(np.percentile(samples, 100.0 * alpha / 2.0))
    hi = float(np.percentile(samples, 100.0 * (1.0 - alpha / 2.0)))
    return BootstrapCI(
        statistic=statistic_name, point_estimate=point, ci_low=lo, ci_high=hi,
        confidence=confidence, method=method, n_boot=int(samples.size),
        expected_block_length=float(bl), n=n, effective_sample_size=ess,
        data_label=label, standard_error=float(np.std(samples, ddof=1)),
    )


# ---------------------------------------------------------------- bundled evaluation


@dataclass
class EconomicEvaluation:
    """Bundles economic metrics with dependence-aware uncertainty and an honest verdict."""

    version: str
    data_label: str
    metrics: EconomicMetrics
    expectancy_ci: BootstrapCI
    sample_size: SampleSizeReport
    profitability_claim_allowed: bool
    verdict: str
    sharpe_ci: Optional[BootstrapCI] = None
    warnings: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "data_label": self.data_label,
            "metrics": self.metrics.to_dict(),
            "expectancy_ci": self.expectancy_ci.to_dict(),
            "sharpe_ci": self.sharpe_ci.to_dict() if self.sharpe_ci else None,
            "sample_size": self.sample_size.to_dict(),
            "profitability_claim_allowed": self.profitability_claim_allowed,
            "verdict": self.verdict,
            "warnings": list(self.warnings),
        }


def evaluate_returns(
    net_returns,
    *,
    data_label: str,
    weights=None,
    risk_free: float = 0.0,
    periods_per_year: Optional[float] = None,
    drawdown_mode: str = "compound",
    method: str = "stationary",
    n_boot: int = 2000,
    confidence: float = 0.95,
    include_sharpe_ci: bool = False,
    seed: int = 7,
) -> EconomicEvaluation:
    """One-call economic evaluation: metrics + dependence-aware CIs + honest sample size.

    The returned verdict never asserts profitability from synthetic data or from a sample too
    small / too autocorrelated to support it; such cases are labeled research-only.
    """
    label = validate_data_label(data_label)
    metrics = economic_metrics(
        net_returns, data_label=label, weights=weights, risk_free=risk_free,
        periods_per_year=periods_per_year, drawdown_mode=drawdown_mode,
    )
    ss = sample_size_report(net_returns, data_label=label)
    exp_ci = bootstrap_metric_ci(
        net_returns, data_label=label, statistic_fn=np.mean, statistic_name="net_expectancy",
        method=method, n_boot=n_boot, confidence=confidence, seed=seed,
    )
    sharpe_ci = None
    if include_sharpe_ci:
        def _sharpe(a: np.ndarray) -> float:
            if a.size < 2:
                return 0.0
            s = float(np.std(a - risk_free, ddof=1))
            return float((a - risk_free).mean() / s) if s > 0 else 0.0

        sharpe_ci = bootstrap_metric_ci(
            net_returns, data_label=label, statistic_fn=_sharpe, statistic_name="sharpe",
            method=method, n_boot=n_boot, confidence=confidence, seed=seed + 1,
        )

    warnings: list = []
    if ss.warning:
        warnings.append(ss.warning)
    if metrics.n < _MIN_DECISIONS_FOR_VERDICT:
        warnings.append(f"small_sample: n={metrics.n}<{_MIN_DECISIONS_FOR_VERDICT}")

    claim_ok = (
        label == DATA_LABEL_REAL
        and metrics.n >= _MIN_DECISIONS_FOR_VERDICT
        and ss.effective_sample_size >= _MIN_EFFECTIVE_SAMPLE_FOR_VERDICT
    )
    if label == DATA_LABEL_SYNTHETIC:
        verdict = ("SYNTHETIC — illustrative only. No profitability may be inferred from "
                   "synthetic evidence.")
    elif not claim_ok:
        verdict = ("RESEARCH-ONLY — real but insufficient/too-dependent evidence "
                   f"(n={metrics.n}, ESS={ss.effective_sample_size:.1f}). Not a deployable "
                   "profitability claim.")
    elif exp_ci.ci_low > 0.0:
        verdict = ("POSITIVE-EXPECTANCY (real) — bootstrap CI for net expectancy excludes "
                   "zero on the lower side. Further falsification still required before "
                   "deployment.")
    else:
        verdict = ("INCONCLUSIVE (real) — net-expectancy CI includes zero; no significant "
                   "edge demonstrated.")
    return EconomicEvaluation(
        version=ECONOMIC_METRICS_VERSION, data_label=label, metrics=metrics,
        expectancy_ci=exp_ci, sample_size=ss, sharpe_ci=sharpe_ci,
        profitability_claim_allowed=claim_ok, verdict=verdict, warnings=warnings,
    )
