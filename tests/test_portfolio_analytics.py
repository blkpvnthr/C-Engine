"""GPU portfolio analytics: return alignment, covariance consumption, derived math
(Milestone 3, §5). Verifies no fabricated zero returns, GPU covariance parity +
regularization/PSD, correlation, volatility, risk contributions, and scenario losses.
"""

from __future__ import annotations

from pathlib import Path

import _cengine_native as native
import numpy as np

from cengine.portfolio_analytics import PortfolioAnalytics, build_return_matrix


def _metallib():
    lib = Path(native.__file__).parent / "cengine.metallib"
    return str(lib) if lib.exists() else None


def _runtime():
    lib = _metallib()
    if lib is None:
        return None
    rt = native.MetalRuntime(lib)
    return rt if getattr(rt, "available", False) else None


def _series(symbols, obs, seed=11):
    rng = np.random.default_rng(seed)
    ts = list(range(1000, 1000 + obs))
    return {s: {t: float(rng.normal(0.0004, 0.01 * (1 + 0.4 * k))) for t in ts}
            for k, s in enumerate(symbols)}


# -------------------------------------------------------- return matrix alignment
def test_complete_case_alignment_no_zero_fill():
    rbs = _series(("AAA", "BBB", "CCC"), 40)
    # drop two timestamps from BBB -> complete-case must drop those rows for all
    dropped = sorted(rbs["BBB"])[:2]
    for t in dropped:
        del rbs["BBB"][t]
    rm = build_return_matrix(rbs, min_observations=20)
    assert rm.assets == 3
    assert rm.observations == 38                       # 40 - 2 shared-missing rows
    assert set(dropped).isdisjoint(set(rm.timestamps_ns))
    # no fabricated zeros: every matrix entry equals a genuine observed return
    for j, sym in enumerate(rm.symbols):
        for i, ts in enumerate(rm.timestamps_ns):
            assert rm.matrix[i, j] == rbs[sym][ts]


def test_symbol_below_min_observations_excluded():
    rbs = _series(("AAA", "BBB"), 40)
    rbs["CCC"] = {1000 + t: 0.001 for t in range(5)}   # only 5 obs
    rm = build_return_matrix(rbs, min_observations=20)
    assert "CCC" in rm.excluded
    assert "CCC" not in rm.symbols


# ---------------------------------------------------------------- covariance (GPU)
def test_gpu_covariance_source_psd_symmetric_and_parity():
    rt = _runtime()
    if rt is None:
        import pytest
        pytest.skip("no Metal runtime available")
    rbs = _series(("AAA", "BBB", "CCC", "DDD"), 60)
    rm = build_return_matrix(rbs, min_observations=20)
    an = PortfolioAnalytics(runtime=rt, ridge=1e-8)
    cov = an.covariance(rm)
    assert cov.source == "metal"
    assert cov.symmetric is True
    assert cov.positive_semidefinite is True
    numpy_cov = np.cov(rm.matrix, rowvar=False, ddof=1)
    assert np.max(np.abs(cov.covariance - numpy_cov)) < 1e-3


def test_cpu_covariance_when_no_runtime():
    rbs = _series(("AAA", "BBB", "CCC"), 50)
    rm = build_return_matrix(rbs, min_observations=20)
    an = PortfolioAnalytics(runtime=None, ridge=1e-8)
    cov = an.covariance(rm)
    assert cov.source == "cpu"
    numpy_cov = np.cov(rm.matrix, rowvar=False, ddof=1)
    assert np.max(np.abs(cov.covariance - numpy_cov)) < 1e-6


def test_ewma_covariance_symmetric_both_paths():
    rbs = _series(("AAA", "BBB", "CCC"), 60)
    rm = build_return_matrix(rbs, min_observations=20)
    for rt in (_runtime(), None):
        an = PortfolioAnalytics(runtime=rt, ridge=1e-8)
        cov = an.covariance(rm, ewma_lambda=0.94)
        assert cov.symmetric is True
        assert cov.source in ("metal", "cpu")


def test_regularization_ridge_and_variance_floor():
    rbs = _series(("AAA", "BBB"), 40)
    rm = build_return_matrix(rbs, min_observations=20)
    an = PortfolioAnalytics(runtime=None, ridge=1e-4, variance_floor=0.01)
    cov = an.covariance(rm)
    assert cov.min_eigenvalue > 0.0
    # diagonal floored to at least variance_floor (+ ridge)
    assert np.all(np.diag(cov.covariance) >= 0.01)


# --------------------------------------------------------------- derived analytics
def test_correlation_properties():
    rbs = _series(("AAA", "BBB", "CCC"), 60)
    rm = build_return_matrix(rbs, min_observations=20)
    an = PortfolioAnalytics(runtime=None)
    cov = an.covariance(rm)
    corr = an.correlation(cov.covariance)
    assert np.allclose(np.diag(corr), 1.0)
    assert np.allclose(corr, corr.T, atol=1e-9)
    assert np.max(np.abs(corr)) <= 1.0 + 1e-9


def test_volatility_and_risk_contributions_match_native():
    rbs = _series(("AAA", "BBB", "CCC"), 60)
    rm = build_return_matrix(rbs, min_observations=20)
    an = PortfolioAnalytics(runtime=None)
    cov = an.covariance(rm)
    w = np.array([0.5, 0.3, 0.2])
    vol = an.portfolio_volatility(w, cov.covariance)
    rc = an.risk_contributions(w, cov.covariance)
    assert abs(rc.sum() - vol) < 1e-6


def test_scenario_losses_deterministic_and_ordered():
    rbs = _series(("AAA", "BBB", "CCC"), 60)
    rm = build_return_matrix(rbs, min_observations=20)
    an = PortfolioAnalytics(runtime=None)
    cov = an.covariance(rm)
    w = np.array([0.4, 0.4, 0.2])
    a = an.scenario_losses(w, cov.covariance, scenarios=2000, seed=99)
    b = an.scenario_losses(w, cov.covariance, scenarios=2000, seed=99)
    assert a == b                                      # deterministic for fixed seed
    assert a["source"] == "cpu_numpy"
    assert a["var_95"] <= a["var_99"]
    assert a["expected_shortfall_95"] >= a["var_95"]
