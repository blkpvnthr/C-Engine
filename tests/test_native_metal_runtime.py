"""Phase A — native Objective-C++ Metal runtime (architectural correction).

Validates the native ``trading::metal::MetalRuntime`` exposed via
``_cengine_native``: on-device execution of the 7 previously-wired kernels,
strict CPU parity, pipeline caching, persistent-buffer reuse, telemetry, the
native kernel registry, graceful unavailability, and — critically — numerical
parity against the legacy PyObjC path it replaces (the regression gate for the
migration). These require genuine Apple-Silicon hardware; they skip elsewhere.
"""

from __future__ import annotations

import platform
import random

import pytest

import _cengine_native as native
from cengine import metal_backend as mb
from cengine import metal_cpu_ref as ref

pytestmark = pytest.mark.skipif(
    platform.system() != "Darwin"
    or platform.machine() != "arm64"
    or not getattr(native, "metal_available", False),
    reason="native Metal runtime requires macOS on Apple Silicon with Metal built in",
)

PRICES = [100, 101, 102, 100, 99, 101, 103, 105, 104, 106, 108, 107, 109, 111, 110]


@pytest.fixture(scope="module")
def metallib() -> str:
    lib = mb.discover_metallib()
    assert lib is not None, "no cengine.metallib discovered"
    return str(lib)


@pytest.fixture(scope="module")
def runtime(metallib: str) -> "native.MetalRuntime":
    rt = native.MetalRuntime(metallib)
    assert rt.available, f"runtime unavailable: {rt.reason}"
    return rt


# --------------------------------------------------------------------------- init
def test_runtime_initializes_on_device(runtime):
    assert runtime.device_name  # non-empty device name
    assert runtime.reason == ""
    names = runtime.function_names()
    assert len(names) == 45, f"expected 45 entry points, got {len(names)}"


def test_namespace_aware_resolution(runtime):
    # Global indicator kernels resolve by bare name.
    assert runtime.has_kernel("rolling_sma_ticks")
    # gme:: kernels resolve via the namespace fallback.
    assert runtime.has_kernel("covariance_matrix")
    assert runtime.has_kernel("ewma_covariance_matrix")
    assert not runtime.has_kernel("this_kernel_does_not_exist")


def test_unavailable_runtime_reports_reason():
    rt = native.MetalRuntime("/nonexistent/path/cengine.metallib")
    assert rt.available is False
    assert "metallib" in rt.reason.lower() or "cannot load" in rt.reason.lower()


# ------------------------------------------------------------------- indicator parity
def test_sma_exact_cpu_parity(runtime):
    assert runtime.sma(PRICES, 5) == ref.sma_ticks(PRICES, 5)


def test_ema_exact_cpu_parity(runtime):
    assert runtime.ema(PRICES, 2, 10) == ref.ema_ticks(PRICES, 2, 10)


def test_apo_exact_cpu_parity(runtime):
    fast = runtime.ema(PRICES, 3, 10)
    slow = runtime.ema(PRICES, 1, 10)
    assert runtime.apo(fast, slow) == ref.apo_ticks(fast, slow)


def test_true_range_exact_cpu_parity(runtime):
    highs = [p + 2 for p in PRICES]
    lows = [p - 2 for p in PRICES]
    prev = [PRICES[0]] + PRICES[:-1]
    assert runtime.true_range(highs, lows, prev) == ref.true_range_ticks(highs, lows, prev)


def test_indicator_invalid_args_raise(runtime):
    with pytest.raises(Exception):
        runtime.sma([], 5)
    with pytest.raises(Exception):
        runtime.sma(PRICES, 0)
    with pytest.raises(Exception):
        runtime.ema(PRICES, 11, 10)  # numerator > denominator


# ------------------------------------------------------------------ covariance parity
def _random_matrix(obs: int, assets: int, seed: int = 7):
    random.seed(seed)
    return [[random.gauss(0.0, 1.0) for _ in range(assets)] for _ in range(obs)]


def test_covariance_cpu_parity(runtime):
    obs, assets = 24, 5
    mat = _random_matrix(obs, assets)
    flat = [x for row in mat for x in row]
    gpu = runtime.covariance_matrix(flat, obs, assets)
    cpu = [x for row in ref.covariance_matrix(mat) for x in row]
    assert max(abs(a - b) for a, b in zip(gpu, cpu)) < 1e-4


def test_ewma_covariance_cpu_parity(runtime):
    obs, assets = 24, 5
    mat = _random_matrix(obs, assets)
    flat = [x for row in mat for x in row]
    gpu = runtime.ewma_covariance_matrix(flat, obs, assets, 0.94, 0.0)
    cpu = [x for row in ref.ewma_covariance_matrix(mat, 0.94) for x in row]
    assert max(abs(a - b) for a, b in zip(gpu, cpu)) < 1e-4


# ---------------------------------------------------------- pipeline cache + telemetry
def test_pipeline_cache_and_buffer_reuse(runtime):
    before = runtime.telemetry()
    for _ in range(12):
        runtime.sma(PRICES, 5)
    after = runtime.telemetry()
    # Repeated dispatch of the same kernel must hit the cache, not rebuild.
    assert after.pipeline_cache_hits > before.pipeline_cache_hits
    assert after.dispatches_completed > before.dispatches_completed
    assert after.dispatch_failures == before.dispatch_failures
    # The persistent pool must recycle buffers rather than allocate per call.
    assert after.buffers_reused > before.buffers_reused


def test_telemetry_counts_completions(runtime):
    t = runtime.telemetry()
    assert t.dispatches_completed > 0
    assert t.dispatches_submitted >= t.dispatches_completed
    assert 0.0 <= t.pipeline_cache_hit_rate <= 1.0


# -------------------------------------------------------------------- kernel registry
def test_kernel_registry_reconciles_against_device(runtime):
    reg = native.KernelRegistry()
    reg.reconcile_with_device(runtime)
    cov = reg.coverage()
    assert cov.discovered == 45
    assert cov.compiled_callable == 45  # every kernel builds a pipeline on device
    assert cov.host_callable == 7  # the 7 wired kernels have a native host interface
    # JSON serialization is well-formed and reflects the live counts.
    import json

    doc = json.loads(reg.to_json())
    assert doc["discovered"] == 45
    assert len(doc["kernels"]) == 45


# ------------------------------------------------- regression parity vs legacy PyObjC
def test_native_matches_legacy_pyobjc_indicators(metallib):
    """The native runtime must agree with the PyObjC path it replaces."""
    legacy = mb.MetalBackend.create(require=False)
    if not legacy.available:
        pytest.skip("legacy PyObjC backend unavailable")
    from cengine.metal_indicators import MetalIndicators
    from cengine.metal_runtime import MetalRuntime as PyMetalRuntime

    py_rt = PyMetalRuntime(metallib)
    py_ind = MetalIndicators(py_rt)
    native_rt = native.MetalRuntime(metallib)
    assert native_rt.sma(PRICES, 5) == py_ind.sma(PRICES, 5)
    assert native_rt.true_range(
        [p + 2 for p in PRICES], [p - 2 for p in PRICES], [PRICES[0]] + PRICES[:-1]
    ) == py_ind.true_range(
        [p + 2 for p in PRICES], [p - 2 for p in PRICES], [PRICES[0]] + PRICES[:-1]
    )


def test_native_matches_legacy_pyobjc_covariance(metallib):
    legacy = mb.MetalBackend.create(require=False)
    if not legacy.available:
        pytest.skip("legacy PyObjC backend unavailable")
    from cengine.metal_covariance import MetalCovariance

    py_cov = MetalCovariance(legacy)
    obs, assets = 20, 4
    mat = _random_matrix(obs, assets, seed=11)
    flat = [x for row in mat for x in row]
    native_rt = native.MetalRuntime(metallib)
    gpu_native = native_rt.covariance_matrix(flat, obs, assets)
    gpu_legacy = [x for row in py_cov.covariance_matrix(mat) for x in row]
    assert max(abs(a - b) for a, b in zip(gpu_native, gpu_legacy)) < 1e-4
