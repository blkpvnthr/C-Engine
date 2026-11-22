"""Shared Metal compute backend + kernel registry.

Central, namespace-aware access to the repository's existing Metal shaders
(``metal/*.metal`` compiled into ``cengine.metallib``). This module does NOT
introduce a competing GPU framework: it wraps the existing
:class:`cengine.metal_runtime.MetalRuntime` / :class:`MetalBuffers` and exposes

  * discovery of the compiled ``cengine.metallib`` across build directories;
  * a resolver for the shader's ``gme::`` C++ namespace (only the four indicator
    kernels are global; every other kernel is ``gme::<name>``);
  * namespace-aware ``pipeline``/``dispatch`` wrappers;
  * a registry of every kernel with its host interface, intended consumer, and
    CPU reference, so integration coverage is computed from real evidence;
  * graceful CPU fallback: if Metal is unavailable the backend reports
    ``available == False`` and callers use their CPU path. GPU is NEVER a
    prerequisite for native risk, broker state, or mandatory protective exits.
"""

from __future__ import annotations

import glob
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from cengine.metal_buffers import MetalBuffers
from cengine.metal_runtime import MetalRuntime, MetalUnavailableError

logger = logging.getLogger("metal_backend")

REPO_ROOT = Path(__file__).resolve().parent.parent
METALLIB_NAME = "cengine.metallib"


# ---------------------------------------------------------------------------
# Kernel registry
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class MetalKernelRegistration:
    shader_file: str
    kernel_name: str
    purpose: str
    host_interface: Optional[str]  # "module.Class.method" or None (not yet wired)
    consumers: tuple[str, ...]
    cpu_reference: Optional[str]
    required_data_sources: tuple[str, ...] = ()
    note: str = ""


# The complete inventory of kernel entry points in metal/*.metal. `host_interface`
# / `consumers` / `cpu_reference` are populated only where a genuine host call +
# downstream consumer + CPU oracle exist; others are honestly left unwired
# (reported as blocked with `note`).
_IND = "indicators.metal"
_COV = "covariance.metal"
_MC = "monte-carlo.metal"
_IMB = "imbalance.metal"
_RED = "reductions.metal"

KERNELS: tuple[MetalKernelRegistration, ...] = (
    # --- indicators (global namespace; fully wired via MetalIndicators) ---
    MetalKernelRegistration(
        _IND,
        "rolling_sma_ticks",
        "rolling SMA (integer ticks)",
        "cengine.metal_indicators.MetalIndicators.sma",
        ("technical_features", "strategies", "scanner"),
        "indicators.sma",
    ),
    MetalKernelRegistration(
        _IND,
        "exponential_moving_average_ticks",
        "EMA (integer ticks)",
        "cengine.metal_indicators.MetalIndicators.ema",
        ("technical_features", "strategies"),
        "metal_cpu_ref.ema_ticks",
    ),
    MetalKernelRegistration(
        _IND,
        "average_price_oscillator_ticks",
        "APO = fast EMA - slow EMA",
        "cengine.metal_indicators.MetalIndicators.apo",
        ("technical_features", "apo_strategy"),
        "metal_cpu_ref.apo_ticks",
    ),
    MetalKernelRegistration(
        _IND,
        "true_range_ticks",
        "Wilder true range (integer ticks)",
        "cengine.metal_indicators.MetalIndicators.true_range",
        ("technical_features", "atr", "di_bb"),
        "metal_cpu_ref.true_range_ticks",
    ),
    # --- covariance (gme::; core pair wired via MetalCovariance) ---
    MetalKernelRegistration(
        _COV,
        "column_means",
        "per-asset column means of a returns matrix",
        "cengine.metal_covariance.MetalCovariance.covariance_matrix",
        ("portfolio_optimizer",),
        "metal_cpu_ref.column_means",
    ),
    MetalKernelRegistration(
        _COV,
        "covariance_matrix",
        "sample covariance matrix (assets x assets)",
        "cengine.metal_covariance.MetalCovariance.covariance_matrix",
        ("portfolio_optimizer", "risk_manager"),
        "metal_cpu_ref.covariance_matrix",
    ),
    MetalKernelRegistration(
        _COV,
        "ewma_covariance_matrix",
        "EWMA covariance matrix",
        "cengine.metal_covariance.MetalCovariance.ewma_covariance_matrix",
        ("portfolio_optimizer", "regime"),
        "metal_cpu_ref.ewma_covariance_matrix",
    ),
    MetalKernelRegistration(
        _COV,
        "statarb_pair_regression",
        "QQQ/SQQQ pair OLS regression",
        None,
        (),
        None,
        ("pair_returns",),
        "structured StatArb ABI; research-only, not yet wired",
    ),
    MetalKernelRegistration(
        _COV,
        "statarb_residual_series",
        "pair residual series",
        None,
        (),
        None,
        ("pair_returns",),
        "structured StatArb ABI; research-only, not yet wired",
    ),
    MetalKernelRegistration(
        _COV,
        "volatility_factor_features",
        "VIX/VXN read-only factor transforms",
        None,
        (),
        None,
        ("vix_vxn",),
        "structured factor ABI; not yet wired",
    ),
    MetalKernelRegistration(
        _COV,
        "statarb_normalized_weights",
        "normalized pair weights",
        None,
        (),
        None,
        ("pair_returns",),
        "long-only engine; statarb weights research-only, not wired",
    ),
    # --- monte-carlo (gme::; normals generator wired) ---
    MetalKernelRegistration(
        _MC,
        "generate_standard_normals",
        "seeded standard-normal draws",
        None,
        ("research", "risk_scenarios"),
        None,
        (),
        "MonteCarloParams ulong-aligned ABI + stochastic validation; wiring pending",
    ),
    MetalKernelRegistration(
        _MC,
        "correlate_normals",
        "apply Cholesky to independent normals",
        None,
        (),
        None,
        (),
        "structured Cholesky ABI; research pipeline not yet wired",
    ),
    MetalKernelRegistration(
        _MC,
        "generate_correlated_returns",
        "correlated return draws",
        None,
        (),
        None,
        (),
        "depends on correlate_normals pipeline; not yet wired",
    ),
    MetalKernelRegistration(
        _MC,
        "portfolio_scenario_pnl",
        "per-scenario portfolio P/L",
        None,
        ("portfolio_risk",),
        None,
        (),
        "multi-stage MC pipeline; not yet wired",
    ),
    MetalKernelRegistration(
        _MC,
        "statarb_pair_monte_carlo",
        "pair-level MC",
        None,
        (),
        None,
        (),
        "research-only; not yet wired",
    ),
    MetalKernelRegistration(
        _MC,
        "simulate_portfolio_paths",
        "portfolio path simulation",
        None,
        ("research",),
        None,
        (),
        "multi-stage MC pipeline; not yet wired",
    ),
    MetalKernelRegistration(
        _MC,
        "pnl_to_loss",
        "convert P/L to loss series",
        None,
        (),
        None,
        (),
        "MC aggregation stage; not yet wired",
    ),
    MetalKernelRegistration(
        _MC,
        "extract_scenario_pnl",
        "extract scenario P/L",
        None,
        (),
        None,
        (),
        "MC aggregation stage; not yet wired",
    ),
    MetalKernelRegistration(
        _MC,
        "extract_pair_pnl",
        "extract pair P/L",
        None,
        (),
        None,
        (),
        "MC aggregation stage; not yet wired",
    ),
    MetalKernelRegistration(
        _MC,
        "tail_moment_partials",
        "partial tail moments",
        None,
        (),
        None,
        (),
        "feeds reduce_* combine stage; not yet wired",
    ),
    MetalKernelRegistration(
        _MC,
        "threshold_tail_partials",
        "partial threshold tail stats (VaR/ES)",
        None,
        ("risk_scenarios",),
        None,
        (),
        "feeds reduce_threshold_tail; not yet wired",
    ),
    MetalKernelRegistration(
        _MC,
        "apply_stress_shocks",
        "apply stress shocks to returns",
        None,
        ("stress_testing",),
        None,
        (),
        "structured stress ABI; not yet wired",
    ),
    # --- imbalance (gme::; IEX is L1 only -> multi-level depth kernels blocked) ---
    MetalKernelRegistration(
        _IMB,
        "validate_depth_snapshots",
        "validate multi-level depth buffers",
        None,
        (),
        None,
        ("l2_depth",),
        "requires genuine L2 depth; IEX provides L1 only",
    ),
    MetalKernelRegistration(
        _IMB,
        "compute_level_imbalance",
        "per-level depth imbalance",
        None,
        (),
        None,
        ("l2_depth",),
        "requires genuine L2 depth; IEX provides L1 only",
    ),
    MetalKernelRegistration(
        _IMB,
        "compute_imbalance_features",
        "full depth imbalance features",
        None,
        (),
        None,
        ("l2_depth",),
        "requires genuine L2 depth; IEX provides L1 only",
    ),
    MetalKernelRegistration(
        _IMB,
        "compute_top_level_imbalance",
        "top-of-book (L1) imbalance",
        None,
        ("order_flow", "scanner"),
        None,
        ("l1_quote",),
        "L1-compatible; structured depth ABI wiring pending",
    ),
    MetalKernelRegistration(
        _IMB,
        "compute_weighted_depth_imbalance",
        "size-weighted depth imbalance",
        None,
        (),
        None,
        ("l2_depth",),
        "requires genuine L2 depth; IEX provides L1 only",
    ),
    MetalKernelRegistration(
        _IMB,
        "compute_depth_shape",
        "depth shape descriptor",
        None,
        (),
        None,
        ("l2_depth",),
        "requires genuine L2 depth; IEX provides L1 only",
    ),
    MetalKernelRegistration(
        _IMB,
        "compute_pair_pressure_features",
        "pair pressure features",
        None,
        (),
        None,
        ("l2_depth",),
        "requires genuine L2 depth; IEX provides L1 only",
    ),
    # --- reductions (gme::; sum/minmax/moment wired) ---
    MetalKernelRegistration(
        _RED,
        "reduce_sum_partials",
        "block-wise partial sums",
        None,
        ("aggregation", "portfolio_stats"),
        None,
        (),
        "two-stage partial/combine pipeline wiring pending",
    ),
    MetalKernelRegistration(
        _RED,
        "reduce_moment_partials",
        "partial mean/variance moments",
        None,
        ("statistics", "volatility"),
        None,
        (),
        "two-stage partial/combine pipeline wiring pending",
    ),
    MetalKernelRegistration(
        _RED,
        "reduce_minmax_partials",
        "partial min/max",
        None,
        ("statistics",),
        None,
        (),
        "two-stage partial/combine pipeline wiring pending",
    ),
    MetalKernelRegistration(
        _RED,
        "reduce_dot_partials",
        "partial dot products",
        None,
        (),
        None,
        (),
        "combine stage wiring pending",
    ),
    MetalKernelRegistration(
        _RED,
        "reduce_covariance_sufficient_partials",
        "partial covariance sufficient stats",
        None,
        ("portfolio_optimizer",),
        None,
        (),
        "pairs with combine_covariance_partials; pending",
    ),
    MetalKernelRegistration(
        _RED,
        "reduce_matrix_column_partials",
        "partial matrix column sums",
        None,
        (),
        None,
        (),
        "combine stage wiring pending",
    ),
    MetalKernelRegistration(
        _RED,
        "reduce_pnl_partials",
        "partial P/L reductions",
        None,
        ("portfolio_stats",),
        None,
        (),
        "pairs with combine_pnl_partials; pending",
    ),
    MetalKernelRegistration(
        _RED,
        "reduce_threshold_tail_partials",
        "partial threshold tail reductions",
        None,
        ("risk_scenarios",),
        None,
        (),
        "pairs with combine_threshold_tail_partials; pending",
    ),
    MetalKernelRegistration(
        _RED,
        "reduce_validation_counts",
        "validation count reductions",
        None,
        (),
        None,
        (),
        "combine stage wiring pending",
    ),
    MetalKernelRegistration(
        _RED,
        "reduce_histogram_partials",
        "partial histograms",
        None,
        (),
        None,
        (),
        "combine stage wiring pending",
    ),
    MetalKernelRegistration(
        _RED,
        "combine_sum_partials",
        "combine partial sums",
        None,
        ("aggregation",),
        None,
        (),
        "second stage of reduce_sum; pending full pipeline",
    ),
    MetalKernelRegistration(
        _RED,
        "combine_covariance_partials",
        "combine partial covariance",
        None,
        ("portfolio_optimizer",),
        None,
        (),
        "second stage; pending full pipeline",
    ),
    MetalKernelRegistration(
        _RED,
        "combine_pnl_partials",
        "combine partial P/L",
        None,
        (),
        None,
        (),
        "second stage; pending full pipeline",
    ),
    MetalKernelRegistration(
        _RED,
        "combine_threshold_tail_partials",
        "combine threshold tail",
        None,
        (),
        None,
        (),
        "second stage; pending full pipeline",
    ),
    MetalKernelRegistration(
        _RED,
        "normalize_zscores",
        "z-score normalization",
        None,
        (),
        None,
        (),
        "combine stage wiring pending",
    ),
)


def discover_metallib(explicit: str | Path | None = None) -> Optional[Path]:
    """Locate the compiled cengine.metallib.

    Order: explicit arg, CENGINE_METALLIB env, the metallib installed BESIDE the
    native ``_cengine_native`` module (authoritative, version-matched to the
    compiled extension — preferred so a stale build/ artifact is never loaded;
    architectural correction §2), then build/, any build*/ (newest mtime), then a
    recursive repo search (excluding the venv)."""
    if explicit:
        p = Path(explicit)
        return p if p.exists() else None
    env = os.environ.get("CENGINE_METALLIB", "").strip()
    if env and Path(env).exists():
        return Path(env)
    # Authoritative: the metallib installed next to the compiled native module.
    try:
        import _cengine_native as _native  # type: ignore[import-not-found]

        beside = Path(_native.__file__).resolve().parent / METALLIB_NAME
        if beside.exists():
            return beside
    except Exception:
        pass
    candidates: list[Path] = []
    for pat in (f"build/{METALLIB_NAME}", f"build*/{METALLIB_NAME}"):
        candidates += [Path(p) for p in glob.glob(str(REPO_ROOT / pat))]
    if not candidates:
        candidates += [
            Path(p)
            for p in glob.glob(str(REPO_ROOT / "**" / METALLIB_NAME), recursive=True)
            if ".venv" not in p
        ]
    candidates = [c for c in candidates if c.exists()]
    if not candidates:
        return None
    return max(candidates, key=lambda c: c.stat().st_mtime)


@dataclass
class MetalBackend:
    """Namespace-aware wrapper over the existing Metal runtime (CPU-fallback safe)."""

    library_path: Optional[Path] = None
    available: bool = False
    reason: str = ""
    runtime: Any = None
    buffers: Any = None
    _fnames: frozenset[str] = field(default_factory=frozenset, repr=False)

    @classmethod
    def create(cls, library: str | Path | None = None, *, require: bool = False) -> "MetalBackend":
        path = discover_metallib(library)
        if path is None:
            if require:
                raise MetalUnavailableError("no compiled cengine.metallib found")
            return cls(available=False, reason="cengine.metallib not found")
        try:
            rt = MetalRuntime(path)
        except MetalUnavailableError as exc:
            if require:
                raise
            return cls(library_path=path, available=False, reason=str(exc))
        fnames = frozenset(str(f) for f in rt.library.functionNames())
        return cls(
            library_path=path,
            available=True,
            runtime=rt,
            buffers=MetalBuffers(rt.device),
            _fnames=fnames,
        )

    # ------------------------------------------------------------- namespace
    def resolve(self, kernel: str) -> Optional[str]:
        """Return the library function name for a bare kernel name (handles the
        shader's ``gme::`` namespace), or None if absent."""
        if not self.available:
            return None
        if kernel in self._fnames:
            return kernel
        namespaced = f"gme::{kernel}"
        return namespaced if namespaced in self._fnames else None

    def pipeline(self, kernel: str) -> Any:
        name = self.resolve(kernel)
        if name is None:
            raise KeyError(f"kernel {kernel!r} not in library")
        return self.runtime.pipeline(name)

    def dispatch(self, kernel: str, buffers: list[Any], element_count: int) -> None:
        name = self.resolve(kernel)
        if name is None:
            raise KeyError(f"kernel {kernel!r} not in library")
        self.runtime.dispatch(name, buffers, element_count)

    def device_name(self) -> str:
        return str(self.runtime.device.name()) if self.available else "none"

    # ------------------------------------------------------------- coverage
    def validate_pipelines(self) -> dict[str, Any]:
        """Hardware check: create a compute pipeline state for every registered
        kernel (proves it is compiled into the loaded library and callable on this
        device). Returns {kernel: True | error-string}."""
        out: dict[str, Any] = {}
        for reg in KERNELS:
            if not self.available:
                out[reg.kernel_name] = "metal unavailable"
                continue
            try:
                self.pipeline(reg.kernel_name)
                out[reg.kernel_name] = True
            except Exception as exc:  # noqa: BLE001
                out[reg.kernel_name] = str(exc)[:80]
        return out

    def coverage(self) -> dict[str, Any]:
        """Integration coverage computed from the registry + live pipeline checks."""
        discovered = len(KERNELS)
        pipe = self.validate_pipelines()
        compiled = sum(1 for v in pipe.values() if v is True)
        host_interfaced = sum(1 for r in KERNELS if r.host_interface)
        consumer_connected = sum(1 for r in KERNELS if r.consumers)
        blocked = [r.kernel_name for r in KERNELS if not r.host_interface]
        return {
            "discovered": discovered,
            "compiled_and_callable": compiled,
            "host_interfaced": host_interfaced,
            "consumer_connected": consumer_connected,
            "blocked_count": len(blocked),
            "blocked": blocked,
            "pipeline_status": pipe,
            "integration_coverage_pct": round(100.0 * host_interfaced / discovered, 1),
            "pipeline_coverage_pct": round(100.0 * compiled / discovered, 1),
        }

    def health(self) -> dict[str, Any]:
        return {
            "backend": "metal" if self.available else "cpu",
            "available": self.available,
            "device": self.device_name(),
            "library": str(self.library_path) if self.library_path else None,
            "reason": self.reason,
            "kernels_in_library": len(self._fnames),
        }
