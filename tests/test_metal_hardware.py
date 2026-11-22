"""Genuine Apple Silicon Metal kernel execution + CPU-parity tests.

These dispatch real kernels on the Metal device (no mocks). They skip cleanly on
hosts without a compiled library / Metal device so CI on other platforms stays
green, but on supported Apple Silicon they are the hardware-execution evidence.
"""

from __future__ import annotations

import pytest

from cengine import metal_cpu_ref as cpu
from cengine.metal_backend import KERNELS, MetalBackend

_backend = MetalBackend.create()
pytestmark = pytest.mark.skipif(
    not _backend.available, reason=f"Metal unavailable: {_backend.reason}"
)


# ------------------------------------------------------------------ backend
def test_all_registered_kernels_pipeline_create_on_hardware():
    cov = _backend.coverage()
    assert cov["discovered"] == len(KERNELS) == 45
    # Every registered kernel must be compiled into the loaded library and
    # produce a compute pipeline state on this device.
    failures = {k: v for k, v in cov["pipeline_status"].items() if v is not True}
    assert failures == {}, f"pipeline creation failed: {failures}"
    assert cov["compiled_and_callable"] == 45


# ----------------------------------------------------------------- indicators
def test_sma_gpu_cpu_parity():
    from cengine.metal_indicators import MetalIndicators
    ind = MetalIndicators(_backend.runtime)
    prices = [100, 101, 102, 103, 104, 103, 102, 105, 108, 110]
    gpu = ind.sma(prices, 3)
    assert gpu == cpu.sma_ticks(prices, 3)  # exact integer parity


def test_ema_gpu_cpu_parity():
    from cengine.metal_indicators import MetalIndicators
    ind = MetalIndicators(_backend.runtime)
    prices = [100, 110, 120, 115, 118, 130, 125, 140]
    gpu = ind.ema(prices, 2, 10)  # alpha = 0.2
    assert gpu == cpu.ema_ticks(prices, 2, 10)


def test_apo_gpu_cpu_parity():
    from cengine.metal_indicators import MetalIndicators
    ind = MetalIndicators(_backend.runtime)
    fast = [10, 12, 14, 16, 18]
    slow = [9, 11, 13, 15, 17]
    gpu = ind.apo(fast, slow)
    assert gpu == cpu.apo_ticks(fast, slow)


def test_true_range_gpu_cpu_parity():
    from cengine.metal_indicators import MetalIndicators
    ind = MetalIndicators(_backend.runtime)
    highs = [110, 112, 115, 113]
    lows = [105, 107, 109, 108]
    prev = [108, 110, 111, 114]
    gpu = ind.true_range(highs, lows, prev)
    assert gpu == cpu.true_range_ticks(highs, lows, prev)


# ----------------------------------------------------------------- covariance
_RETURNS = [
    [0.01, -0.02, 0.005],
    [0.02, -0.01, 0.004],
    [-0.015, 0.03, -0.002],
    [0.005, -0.005, 0.001],
    [0.012, -0.018, 0.006],
    [-0.008, 0.02, -0.003],
]


def _max_abs_diff(a, b):
    return max(abs(a[i][j] - b[i][j]) for i in range(len(a)) for j in range(len(a[0])))


def test_covariance_matrix_gpu_cpu_parity():
    from cengine.metal_covariance import MetalCovariance
    mc = MetalCovariance(_backend)
    gpu = mc.covariance_matrix(_RETURNS)
    ref = cpu.covariance_matrix(_RETURNS)
    assert len(gpu) == 3 and len(gpu[0]) == 3
    # float32 GPU vs float64 CPU: small returns -> tight absolute tolerance.
    assert _max_abs_diff(gpu, ref) < 1e-6
    # symmetry
    assert abs(gpu[0][1] - gpu[1][0]) < 1e-7


def test_ewma_covariance_gpu_cpu_parity():
    from cengine.metal_covariance import MetalCovariance
    mc = MetalCovariance(_backend)
    gpu = mc.ewma_covariance_matrix(_RETURNS, lam=0.94)
    ref = cpu.ewma_covariance_matrix(_RETURNS, lam=0.94)
    assert _max_abs_diff(gpu, ref) < 1e-6


def test_covariance_requires_available_backend():
    # Guard: the host interface refuses to run on an unavailable backend.
    import cengine.metal_covariance as m
    class _Dead:
        available = False
    with pytest.raises(RuntimeError):
        m.MetalCovariance(_Dead())
