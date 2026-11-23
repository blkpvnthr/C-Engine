"""Hard portfolio constraint enforcement (Milestone 3, §8)."""

from __future__ import annotations

import numpy as np

from cengine.portfolio_constraints import PortfolioConstraints, enforce


def _diag_cov(variances):
    return np.diag(np.asarray(variances, dtype=float))


def _corr_cov(n, var, corr):
    """assets x assets covariance: equal variance, equal off-diagonal correlation."""
    sd = var ** 0.5
    cov = np.full((n, n), corr * var, dtype=float)
    np.fill_diagonal(cov, var)
    return cov * (sd / sd)  # keep as covariance (var on diag)


def test_per_asset_cap_scales_down():
    c = PortfolioConstraints(max_position_weight=0.2, max_gross_exposure=1.0)
    rep = enforce({"A": 0.5, "B": 0.1}, constraints=c)
    assert rep.feasible
    assert rep.weights["A"] == 0.2
    assert any("max_position_weight" in b for b in rep.binding)


def test_gross_exposure_scaled_down():
    c = PortfolioConstraints(max_position_weight=1.0, max_gross_exposure=0.5)
    rep = enforce({"A": 0.4, "B": 0.4}, constraints=c)
    assert rep.feasible
    assert abs(sum(rep.weights.values()) - 0.5) < 1e-9
    assert "max_gross_exposure" in rep.binding


def test_net_exposure_scaled_down():
    c = PortfolioConstraints(max_position_weight=1.0, max_gross_exposure=10.0,
                             max_net_exposure=0.5)
    rep = enforce({"A": 0.6, "B": 0.4}, constraints=c)
    assert rep.feasible
    assert abs(sum(rep.weights.values()) - 0.5) < 1e-9
    assert "max_net_exposure" in rep.binding


def test_effective_leverage_scaled_down():
    c = PortfolioConstraints(max_position_weight=1.0, max_gross_exposure=10.0,
                             max_net_exposure=10.0, max_effective_leverage=1.0)
    rep = enforce({"A": 0.5, "B": 0.5}, constraints=c, effective_leverage=2.0)
    assert rep.feasible
    # whole book scaled by 1/2 -> gross 0.5
    assert abs(sum(rep.weights.values()) - 0.5) < 1e-9
    assert "max_effective_leverage" in rep.binding


def test_portfolio_volatility_scaled_down():
    syms = ("A", "B")
    cov = _diag_cov([0.04, 0.04])  # sd 0.2 each
    c = PortfolioConstraints(max_position_weight=1.0, max_gross_exposure=10.0,
                             max_net_exposure=10.0, max_portfolio_volatility=0.05)
    rep = enforce({"A": 0.5, "B": 0.5}, constraints=c, covariance=cov, symbols=syms)
    assert rep.feasible
    vol = float(np.sqrt(np.array([rep.weights.get(s, 0.0) for s in syms])
                        @ cov @ np.array([rep.weights.get(s, 0.0) for s in syms])))
    assert vol <= 0.05 + 1e-9
    assert "max_portfolio_volatility" in rep.binding


def test_correlated_cluster_scaled_down():
    syms = ("A", "B", "C")
    cov = _corr_cov(3, 0.04, 0.95)  # all highly correlated
    c = PortfolioConstraints(max_position_weight=1.0, max_gross_exposure=10.0,
                             max_net_exposure=10.0, max_portfolio_volatility=100.0,
                             max_correlated_exposure=0.5, correlation_threshold=0.8)
    rep = enforce({"A": 0.3, "B": 0.3, "C": 0.3}, constraints=c, covariance=cov,
                  symbols=syms)
    assert rep.feasible
    assert sum(abs(v) for v in rep.weights.values()) <= 0.5 + 1e-9
    assert "max_correlated_exposure" in rep.binding


def test_drawdown_circuit_breaker_rejects():
    c = PortfolioConstraints(max_drawdown=0.2)
    rep = enforce({"A": 0.1}, constraints=c, drawdown=0.25)
    assert rep.feasible is False
    assert rep.weights == {}
    assert "drawdown" in (rep.rejection_reason or "")
    assert "max_drawdown" in rep.violated


def test_turnover_cap_rejects():
    c = PortfolioConstraints(max_position_weight=1.0, max_gross_exposure=10.0,
                             max_net_exposure=10.0, max_turnover=0.3)
    rep = enforce({"A": 1.0}, constraints=c, current_weights={"A": 0.0})
    assert rep.feasible is False
    assert "turnover" in (rep.rejection_reason or "")
    assert "max_turnover" in rep.violated


def test_unauthorized_instrument_rejects():
    c = PortfolioConstraints()
    rep = enforce({"SQQQ": 0.1, "A": 0.1}, constraints=c,
                  authorized=lambda s: s != "SQQQ")
    assert rep.feasible is False
    assert rep.rejection_reason == "unauthorized_instrument"
    assert any("SQQQ" in v for v in rep.violated)


def test_family_and_sector_caps_scale_group():
    c = PortfolioConstraints(max_position_weight=1.0, max_gross_exposure=10.0,
                             max_net_exposure=10.0, max_family_weight=0.4,
                             max_sector_weight=0.4)
    rep = enforce({"A": 0.3, "B": 0.3}, constraints=c,
                  family_of=lambda s: "momentum", sector_of=lambda s: "tech")
    assert rep.feasible
    assert sum(rep.weights.values()) <= 0.4 + 1e-9
    assert any("max_family_weight" in b for b in rep.binding)


def test_feasible_allocation_passes_through():
    c = PortfolioConstraints(max_position_weight=0.5, max_gross_exposure=1.0,
                             max_net_exposure=1.0)
    rep = enforce({"A": 0.3, "B": 0.3}, constraints=c)
    assert rep.feasible
    assert rep.weights == {"A": 0.3, "B": 0.3}
    assert rep.violated == []
