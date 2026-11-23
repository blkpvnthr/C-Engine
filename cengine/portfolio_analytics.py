"""GPU-accelerated portfolio analytics (Milestone 3, §5).

Operationally CONSUMES the existing Metal covariance kernels (``covariance_matrix``,
``ewma_covariance_matrix``, ``column_means`` via the native ``MetalRuntime``) to
compute the portfolio risk inputs, with a numpy CPU fallback that is numerically
equivalent. Correlation, portfolio volatility, risk contributions and scenario
losses are cheap linear algebra on the small assets x assets matrix and run on
CPU/native per the measured-crossover policy (do not dispatch tiny ops to the GPU).

Hard rules honored here:
  * Timestamp-aligned returns with EXPLICIT missing-data handling — unavailable
    observations are never replaced with fabricated zero returns (§5). Complete-case
    alignment drops timestamps not shared by all included symbols; symbols with
    insufficient coverage are excluded and reported.
  * Covariance is regularized (ridge + optional diagonal variance floor),
    symmetrized, and checked for PSD / numerical validity (§5).
  * GPU is never a prerequisite: a missing/failed runtime falls back to numpy.
  * These are optimizer INPUTS; they never authorize trades.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

try:  # native runtime is optional; numpy fallback is always available
    import _cengine_native as _native
except Exception:  # noqa: BLE001
    _native = None


COVARIANCE_VERSION = "portfolio-cov-1.0.0"


@dataclass(frozen=True)
class ReturnMatrix:
    """Timestamp-aligned returns, observations x assets (row-major), no fabricated data."""

    symbols: tuple[str, ...]
    timestamps_ns: tuple[int, ...]
    matrix: np.ndarray                        # shape (observations, assets), float
    excluded: dict = field(default_factory=dict)  # symbol -> reason (insufficient coverage)

    @property
    def observations(self) -> int:
        return int(self.matrix.shape[0]) if self.matrix.ndim == 2 else 0

    @property
    def assets(self) -> int:
        return len(self.symbols)


def build_return_matrix(returns_by_symbol: dict[str, dict[int, float]], *,
                        min_observations: int = 20) -> ReturnMatrix:
    """Align per-symbol ``{timestamp_ns: return}`` maps on shared timestamps.

    Complete-case: only timestamps present for EVERY retained symbol are used.
    Symbols whose own observation count is below ``min_observations`` are excluded
    up front (reported in ``excluded``); then the shared-timestamp intersection is
    taken across the retained symbols. Missing observations are NEVER zero-filled.
    """
    excluded: dict = {}
    retained: list[str] = []
    for sym, series in returns_by_symbol.items():
        clean = {int(ts): float(r) for ts, r in series.items()
                 if r is not None and np.isfinite(r)}
        if len(clean) < min_observations:
            excluded[sym] = f"insufficient_observations:{len(clean)}<{min_observations}"
        else:
            retained.append(sym)
    retained.sort()

    if not retained:
        return ReturnMatrix((), (), np.empty((0, 0)), excluded)

    common: Optional[set] = None
    cleaned_series: dict[str, dict[int, float]] = {}
    for sym in retained:
        clean = {int(ts): float(r) for ts, r in returns_by_symbol[sym].items()
                 if r is not None and np.isfinite(r)}
        cleaned_series[sym] = clean
        common = set(clean) if common is None else (common & set(clean))

    shared = sorted(common or set())
    if len(shared) < min_observations:
        # Not enough overlapping observations to estimate covariance honestly.
        for sym in retained:
            excluded.setdefault(sym, f"insufficient_shared_observations:{len(shared)}")
        return ReturnMatrix((), (), np.empty((0, 0)), excluded)

    matrix = np.array([[cleaned_series[sym][ts] for sym in retained] for ts in shared],
                      dtype=float)
    return ReturnMatrix(tuple(retained), tuple(shared), matrix, excluded)


@dataclass(frozen=True)
class CovarianceResult:
    covariance: np.ndarray                    # regularized assets x assets
    source: str                               # "metal" | "cpu"
    symmetric: bool
    positive_semidefinite: bool
    min_eigenvalue: float
    observations: int
    assets: int
    version: str = COVARIANCE_VERSION


class PortfolioAnalytics:
    """GPU-consuming covariance + portfolio-risk analytics.

    ``runtime`` is an optional ``_cengine_native.MetalRuntime``; when absent/unavailable
    the numpy fallback is used (``source == "cpu"``). ``ridge`` is a tiny diagonal
    regularizer for PSD stability; ``variance_floor`` optionally floors diagonal
    variances (regularization for poorly conditioned matrices).
    """

    def __init__(self, *, runtime=None, ridge: float = 1e-8,
                 variance_floor: float = 0.0) -> None:
        self._runtime = runtime
        self.ridge = float(ridge)
        self.variance_floor = float(variance_floor)

    @property
    def gpu_available(self) -> bool:
        return self._runtime is not None and getattr(self._runtime, "available", False)

    # -- covariance (GPU-consumed) -------------------------------------------
    def covariance(self, rm: ReturnMatrix, *, ewma_lambda: Optional[float] = None
                   ) -> CovarianceResult:
        """Sample (or EWMA) covariance, regularized + PSD-checked.

        Consumes the native Metal ``covariance_matrix`` / ``ewma_covariance_matrix``
        kernel when a runtime is available; otherwise numpy (ddof=1 / EWMA)."""
        if rm.assets == 0 or rm.observations < 2:
            raise ValueError("covariance requires >=2 observations and >=1 asset")
        obs, assets = rm.observations, rm.assets
        flat = rm.matrix.astype(np.float32).flatten().tolist()
        source = "cpu"
        cov: Optional[np.ndarray] = None

        if self.gpu_available:
            try:
                if ewma_lambda is None:
                    raw = self._runtime.covariance_matrix(flat, obs, assets)
                else:
                    raw = self._runtime.ewma_covariance_matrix(
                        flat, obs, assets, float(ewma_lambda), float(self.variance_floor))
                cov = np.asarray(raw, dtype=float).reshape(assets, assets)
                if not np.all(np.isfinite(cov)):
                    cov = None  # kernel flagged invalid cells -> fall back
                else:
                    source = "metal"
            except Exception:  # noqa: BLE001 — GPU failure must never block allocation
                cov = None
                source = "cpu"

        if cov is None:
            cov = self._cpu_covariance(rm.matrix, ewma_lambda)

        cov = self._regularize(cov)
        sym = bool(np.allclose(cov, cov.T, atol=1e-9))
        eigvals = np.linalg.eigvalsh(cov)
        min_eig = float(eigvals.min())
        return CovarianceResult(cov, source, sym, min_eig >= -1e-10, min_eig, obs, assets)

    def _cpu_covariance(self, matrix: np.ndarray, ewma_lambda: Optional[float]) -> np.ndarray:
        if ewma_lambda is None:
            return np.atleast_2d(np.cov(matrix, rowvar=False, ddof=1))
        # EWMA: newest weight 1, older lambda^k; matches the Metal kernel convention.
        obs = matrix.shape[0]
        weights = np.array([ewma_lambda ** (obs - 1 - t) for t in range(obs)], dtype=float)
        wsum = weights.sum()
        mean = (weights[:, None] * matrix).sum(axis=0) / wsum
        centered = matrix - mean
        cov = (weights[:, None] * centered).T @ centered / wsum
        if self.variance_floor > 0:
            np.fill_diagonal(cov, np.maximum(np.diag(cov), self.variance_floor))
        return np.atleast_2d(cov)

    def _regularize(self, cov: np.ndarray) -> np.ndarray:
        cov = np.atleast_2d(np.asarray(cov, dtype=float))
        cov = 0.5 * (cov + cov.T)                       # enforce symmetry
        if self.variance_floor > 0:
            np.fill_diagonal(cov, np.maximum(np.diag(cov), self.variance_floor))
        cov = cov + self.ridge * np.eye(cov.shape[0])   # ridge -> numerically PSD
        return cov

    # -- derived analytics (cheap; CPU/native) -------------------------------
    @staticmethod
    def correlation(cov: np.ndarray) -> np.ndarray:
        d = np.sqrt(np.clip(np.diag(cov), 0.0, None))
        denom = np.outer(d, d)
        with np.errstate(divide="ignore", invalid="ignore"):
            corr = np.where(denom > 0, cov / denom, 0.0)
        np.fill_diagonal(corr, 1.0)
        return corr

    def portfolio_volatility(self, weights: np.ndarray, cov: np.ndarray) -> float:
        """Portfolio volatility sqrt(wᵀΣw) via the native allocator when available."""
        w = np.asarray(weights, dtype=float)
        n = len(w)
        if _native is not None and hasattr(_native, "portfolio_variance"):
            var = _native.portfolio_variance(
                w.tolist(), cov.astype(np.float32).flatten().tolist(), n)
            return float(np.sqrt(max(0.0, var)))
        return float(np.sqrt(max(0.0, w @ cov @ w)))

    def risk_contributions(self, weights: np.ndarray, cov: np.ndarray) -> np.ndarray:
        w = np.asarray(weights, dtype=float)
        n = len(w)
        if _native is not None and hasattr(_native, "portfolio_risk_contributions"):
            rc = _native.portfolio_risk_contributions(
                w.tolist(), cov.astype(np.float32).flatten().tolist(), n)
            return np.asarray(rc, dtype=float)
        sigma_w = cov @ w
        vol = float(np.sqrt(max(0.0, w @ sigma_w)))
        return (w * sigma_w / vol) if vol > 0 else np.zeros_like(w)

    def scenario_losses(self, weights: np.ndarray, cov: np.ndarray, *,
                        scenarios: int = 4096, seed: int = 12345,
                        horizon_scale: float = 1.0) -> dict:
        """Deterministic Gaussian Monte-Carlo portfolio losses from the covariance.

        CPU/numpy (labeled): a seeded multivariate-normal draw of portfolio returns
        -> VaR/ES. The GPU monte-carlo kernels remain compiled-but-unbound; GPU
        scenario consumption is a tracked future increment (requires pybind bindings).
        """
        w = np.asarray(weights, dtype=float)
        rng = np.random.default_rng(seed)
        chol = np.linalg.cholesky(cov + 1e-12 * np.eye(cov.shape[0]))
        normals = rng.standard_normal(size=(scenarios, cov.shape[0]))
        sims = normals @ chol.T * np.sqrt(horizon_scale)
        port = sims @ w
        losses = -port
        var95 = float(np.quantile(losses, 0.95))
        var99 = float(np.quantile(losses, 0.99))
        es95 = float(losses[losses >= var95].mean()) if np.any(losses >= var95) else var95
        return {"scenarios": scenarios, "var_95": var95, "var_99": var99,
                "expected_shortfall_95": es95, "mean_loss": float(losses.mean()),
                "worst_loss": float(losses.max()), "source": "cpu_numpy"}
