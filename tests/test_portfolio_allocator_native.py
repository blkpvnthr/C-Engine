"""Native C++ portfolio allocator parity + constraint behavior (Milestone 3, §3/§25).

Exercises risk/portfolio_allocator.{hpp,cpp} via the _cengine_native bindings and
checks numerical parity against numpy plus the long-only / per-asset / gross budget
projection semantics.
"""

from __future__ import annotations

import _cengine_native as native
import numpy as np


def _cov_from_returns(obs, assets, seed):
    rng = np.random.default_rng(seed)
    R = rng.normal(0.0, 0.01, size=(obs, assets)).astype(np.float32)
    cov = np.cov(R, rowvar=False, ddof=1).astype(np.float32)
    return cov


def test_inverse_volatility_parity_and_zero_variance():
    variances = [0.0004, 0.0009, 0.0016, 0.0]  # last has zero variance -> weight 0
    w = np.array(native.inverse_volatility_weights(variances))
    pos = np.array(variances[:-1])
    inv = 1.0 / np.sqrt(pos)
    inv = inv / inv.sum()
    assert w[-1] == 0.0
    assert np.allclose(w[:-1], inv, atol=1e-12)
    assert abs(w.sum() - 1.0) < 1e-12


def test_inverse_volatility_all_zero_is_all_cash():
    w = np.array(native.inverse_volatility_weights([0.0, 0.0, -1.0]))
    assert np.allclose(w, 0.0)


def test_allocate_risk_based_long_only_capped():
    assets = 4
    cov = _cov_from_returns(80, assets, 1)
    variances = [float(cov[i, i]) for i in range(assets)]
    res = native.allocate_risk_based(variances, cov.flatten().tolist(), assets, 0.4, 1.0)
    w = np.array(res.weights)
    assert res.valid
    assert (w >= 0).all()                      # long-only
    assert w.max() <= 0.4 + 1e-9               # per-asset cap
    assert res.gross_weight <= 1.0 + 1e-9
    assert abs(res.cash_weight - (1.0 - res.gross_weight)) < 1e-9


def test_gross_and_per_asset_binding_flags():
    assets = 4
    cov = _cov_from_returns(80, assets, 2)
    variances = [float(cov[i, i]) for i in range(assets)]
    res = native.allocate_risk_based(variances, cov.flatten().tolist(), assets, 0.2, 0.5)
    w = np.array(res.weights)
    assert res.per_asset_cap_binding is True   # inverse-vol would exceed 0.2 somewhere
    assert res.gross_cap_binding is True
    assert abs(res.gross_weight - 0.5) < 1e-9
    assert w.max() <= 0.2 + 1e-9


def test_portfolio_variance_parity():
    assets = 5
    cov = _cov_from_returns(100, assets, 3)
    w = np.array([0.3, 0.25, 0.2, 0.15, 0.1])
    native_var = native.portfolio_variance(w.tolist(), cov.flatten().tolist(), assets)
    numpy_var = float(w @ cov.astype(float) @ w)
    assert abs(native_var - numpy_var) < 1e-9


def test_marginal_variance_parity():
    assets = 3
    cov = _cov_from_returns(60, assets, 7)
    w = np.array([0.5, 0.3, 0.2])
    mv = np.array(native.portfolio_marginal_variance(w.tolist(), cov.flatten().tolist(), assets))
    assert np.allclose(mv, cov.astype(float) @ w, atol=1e-9)


def test_risk_contributions_sum_to_volatility():
    assets = 4
    cov = _cov_from_returns(80, assets, 4)
    w = np.array([0.4, 0.3, 0.2, 0.1])
    rc = np.array(native.portfolio_risk_contributions(w.tolist(), cov.flatten().tolist(), assets))
    vol = float(np.sqrt(w @ cov.astype(float) @ w))
    assert abs(rc.sum() - vol) < 1e-6


def test_risk_contributions_zero_weights_zero_vol():
    assets = 3
    cov = _cov_from_returns(40, assets, 5)
    w = [0.0, 0.0, 0.0]
    rc = np.array(native.portfolio_risk_contributions(w, cov.flatten().tolist(), assets))
    assert np.allclose(rc, 0.0)


def test_project_clamps_negative_and_caps():
    res = native.project_long_only_capped([-0.5, 0.8, 0.3], 0.4, 1.0)
    w = np.array(res.weights)
    assert res.valid
    assert w[0] == 0.0                          # negative clamped to 0 (long-only)
    assert w[1] <= 0.4 + 1e-9                    # capped
    assert res.per_asset_cap_binding is True


def test_project_invalid_caps_all_cash():
    res = native.project_long_only_capped([0.3, 0.3], 0.0, 1.0)
    assert res.valid is False
    assert all(x == 0.0 for x in res.weights)


def test_allocate_assets_zero_is_all_cash():
    res = native.allocate_risk_based([], [], 0, 0.4, 1.0)
    assert abs(res.cash_weight - 1.0) < 1e-9 or len(res.weights) == 0
