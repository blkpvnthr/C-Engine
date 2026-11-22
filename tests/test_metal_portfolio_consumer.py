"""End-to-end consumer evidence: the portfolio optimizer consumes GPU covariance.

Traces returns -> Metal covariance kernel -> RegimeMarkowitzAllocator.optimize ->
AllocationAdvice, proving the covariance kernels have a real downstream consumer.
"""

from __future__ import annotations

import numpy as np
import pytest

from cengine.metal_backend import MetalBackend
from cengine.metal_portfolio import covariance_from_returns
from cengine.portfolio_optimization import AllocationConstraints, RegimeMarkowitzAllocator

_backend = MetalBackend.create()

_RETURNS = np.array(
    [
        [0.010, -0.020, 0.005],
        [0.020, -0.010, 0.004],
        [-0.015, 0.030, -0.002],
        [0.005, -0.005, 0.001],
        [0.012, -0.018, 0.006],
        [-0.008, 0.020, -0.003],
        [0.009, -0.012, 0.002],
        [0.004, -0.006, 0.0015],
    ]
)
_SYMBOLS = ("NVDA", "SQQQ", "GLD")


def test_cpu_fallback_source():
    cov, source = covariance_from_returns(_RETURNS, backend=None)
    assert source == "cpu"
    assert cov.shape == (3, 3)
    assert np.allclose(cov, cov.T)  # symmetric


@pytest.mark.skipif(not _backend.available, reason=f"Metal unavailable: {_backend.reason}")
def test_metal_source_and_cpu_parity():
    cov_gpu, source = covariance_from_returns(_RETURNS, backend=_backend)
    assert source == "metal"
    cov_cpu, _ = covariance_from_returns(_RETURNS, backend=None)
    # GPU (float32) vs CPU (float64) covariance agree on small returns.
    assert np.max(np.abs(cov_gpu - cov_cpu)) < 1e-6


@pytest.mark.skipif(not _backend.available, reason=f"Metal unavailable: {_backend.reason}")
def test_optimizer_consumes_gpu_covariance_end_to_end():
    cov_gpu, source = covariance_from_returns(_RETURNS, backend=_backend)
    assert source == "metal"
    allocator = RegimeMarkowitzAllocator(
        AllocationConstraints(gross_limit=0.72, per_asset_limit=0.12, risk_aversion=5.0)
    )
    expected_returns = np.array([0.012, 0.004, 0.002])
    advice = allocator.optimize(_SYMBOLS, expected_returns, cov_gpu, risk_on_probability=0.7)
    assert advice.symbols == _SYMBOLS
    assert len(advice.weights) == 3
    # Advice respects the gross constraint (optimizer consumed the GPU covariance).
    assert sum(abs(w) for w in advice.weights) <= 0.72 + 1e-6
