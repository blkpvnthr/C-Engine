"""GPU Monte-Carlo scenario kernel binding + consumption (Milestone 4, §19).

Validates the newly-bound native MetalRuntime.portfolio_scenario_returns (chaining
gme::correlate_normals + gme::portfolio_scenario_pnl) against an exact numpy CPU
oracle using HOST-seeded normals, and its consumption by PortfolioAnalytics.
"""

from __future__ import annotations

from pathlib import Path

import _cengine_native as native
import numpy as np
import pytest

from cengine.portfolio_analytics import PortfolioAnalytics


def _runtime():
    lib = Path(native.__file__).parent / "cengine.metallib"
    if not lib.exists():
        lib = Path("build-metal/cengine.metallib")
    if not lib.exists():
        return None
    rt = native.MetalRuntime(str(lib))
    return rt if rt.available else None


def _psd_cov(assets: int, seed: int = 11) -> np.ndarray:
    rng = np.random.default_rng(seed)
    a = rng.normal(0, 1, size=(assets, assets))
    return a @ a.T / assets + np.eye(assets) * 0.01


@pytest.mark.parametrize("scenarios,assets", [(256, 3), (1024, 4), (4096, 8)])
def test_portfolio_scenario_returns_parity(scenarios, assets):
    rt = _runtime()
    if rt is None:
        pytest.skip("no Metal runtime")
    rng = np.random.default_rng(scenarios + assets)
    cov = _psd_cov(assets)
    L = np.linalg.cholesky(cov)  # lower-triangular
    w = rng.random(assets)
    w /= w.sum()
    normals = rng.standard_normal(size=(scenarios, assets)).astype(np.float32)

    gpu = np.asarray(rt.portfolio_scenario_returns(
        normals.flatten().tolist(), L.astype(np.float32).flatten().tolist(),
        w.astype(np.float32).tolist(), scenarios, assets, 0))
    # exact CPU oracle: correlated = normals @ L.T ; portfolio return = correlated . w
    cpu = (normals.astype(np.float64) @ L.T) @ w
    assert gpu.shape == (scenarios,)
    assert float(np.max(np.abs(gpu - cpu))) < 1e-4


def test_lognormal_return_model_differs():
    rt = _runtime()
    if rt is None:
        pytest.skip("no Metal runtime")
    assets, scen = 4, 1024
    rng = np.random.default_rng(3)
    L = np.linalg.cholesky(_psd_cov(assets))
    w = np.full(assets, 1.0 / assets)
    normals = rng.standard_normal(size=(scen, assets)).astype(np.float32)
    nflat = normals.flatten().tolist()
    lflat = L.astype(np.float32).flatten().tolist()
    wl = w.astype(np.float32).tolist()
    linear = np.asarray(rt.portfolio_scenario_returns(nflat, lflat, wl, scen, assets, 0))
    lognormal = np.asarray(rt.portfolio_scenario_returns(nflat, lflat, wl, scen, assets, 1))
    assert float(np.max(np.abs(linear - lognormal))) > 1e-6


def test_scenario_losses_consumes_gpu_monte_carlo():
    rt = _runtime()
    if rt is None:
        pytest.skip("no Metal runtime")
    cov = _psd_cov(5)
    w = np.full(5, 0.2)
    an = PortfolioAnalytics(runtime=rt)
    out = an.scenario_losses(w, cov, scenarios=4096, seed=42)
    assert out["source"] == "metal_monte_carlo"
    assert out["var_95"] <= out["var_99"]
    assert out["expected_shortfall_95"] >= out["var_95"]
    # deterministic for a fixed seed
    out2 = an.scenario_losses(w, cov, scenarios=4096, seed=42)
    assert out2["var_95"] == out["var_95"]


def test_scenario_losses_cpu_fallback():
    cov = _psd_cov(5)
    w = np.full(5, 0.2)
    out = PortfolioAnalytics(runtime=None).scenario_losses(w, cov, scenarios=2048, seed=7)
    assert out["source"] == "cpu_numpy"
    assert out["var_95"] <= out["var_99"]
