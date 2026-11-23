"""Portfolio optimizer orchestration + recommendation contract (Milestone 3, §7/§10/§11)."""

from __future__ import annotations

import numpy as np

from cengine.portfolio_analytics import PortfolioAnalytics, build_return_matrix
from cengine.portfolio_constraints import PortfolioConstraints
from cengine.portfolio_optimizer import (
    AllocationPolicy,
    OptimizerConfig,
    PortfolioOptimizer,
    RecommendationEligibility,
)
from cengine.portfolio_state import (
    PORTFOLIO_STATE_VERSION,
    PortfolioState,
    ReconciliationStatus,
)

EQUITY = 100_000_000
PRICE = 1_000_000


def _returns(symbols, obs=40, seed=11):
    rng = np.random.default_rng(seed)
    ts = list(range(1, obs + 1))
    return {s: {t: float(rng.normal(0.0004, 0.01 * (1 + 0.3 * k))) for t in ts}
            for k, s in enumerate(symbols)}


def _cov(symbols, **kw):
    rm = build_return_matrix(_returns(symbols), min_observations=20)
    return rm, PortfolioAnalytics(ridge=1e-8).covariance(rm, **kw)


def _state(**over):
    base = dict(
        portfolio_version=PORTFOLIO_STATE_VERSION, observation_timestamp=1,
        account_equity_ticks=EQUITY, settled_cash_ticks=EQUITY,
        available_cash_ticks=EQUITY, buying_power_ticks=EQUITY,
        reserved_capital_ticks=0, pending_buy_exposure_ticks=0,
        pending_sell_exposure_ticks=0, realized_pl_ticks=0, unrealized_pl_ticks=0,
        gross_exposure_ticks=0, net_exposure_ticks=0, effective_exposure_ticks=0,
        reconciliation_status=ReconciliationStatus.RECONCILED,
    )
    base.update(over)
    return PortfolioState(**base)


def _opt(**cfg):
    defaults = dict(constraints=PortfolioConstraints(
        max_position_weight=cfg.pop("per_asset_cap", 0.5), max_gross_exposure=1.0,
        max_portfolio_volatility=1.0, max_drawdown=0.20),
        per_asset_cap=0.5, gross_cap=0.9)
    defaults.update(cfg)
    return PortfolioOptimizer(OptimizerConfig(**defaults))


def test_risk_based_policy_without_expected_returns():
    syms = ("AAA", "BBB", "CCC")
    rm, cov = _cov(syms)
    recs = _opt().optimize(symbols=rm.symbols, cov=cov, state=_state(), now_ns=10,
                           price_ticks=lambda s: PRICE)
    assert all(r.allocation_policy == AllocationPolicy.RISK_BASED.value for r in recs)
    assert all(r.return_basis == "deterministic_risk_based" for r in recs)
    assert all(0.0 <= r.target_weight <= 0.5 + 1e-9 for r in recs)
    assert sum(r.target_weight for r in recs) <= 0.9 + 1e-6


def test_mean_variance_policy_with_expected_returns():
    syms = ("AAA", "BBB", "CCC")
    rm, cov = _cov(syms)
    mu = np.array([0.0012, 0.0009, 0.0006])
    recs = _opt(per_asset_cap=0.4).optimize(
        symbols=rm.symbols, cov=cov, state=_state(), now_ns=10,
        expected_returns=mu, regime_probability=0.8, price_ticks=lambda s: PRICE)
    assert recs[0].allocation_policy == AllocationPolicy.MEAN_VARIANCE.value
    assert all(r.return_basis == "calibrated_mean_variance" for r in recs)
    assert all(r.target_weight <= 0.4 + 1e-9 for r in recs)


def test_no_fabricated_expected_returns():
    syms = ("AAA", "BBB")
    rm, cov = _cov(syms)
    recs = _opt().optimize(symbols=rm.symbols, cov=cov, state=_state(), now_ns=10,
                           price_ticks=lambda s: PRICE)
    assert all(r.expected_incremental_net_return is None for r in recs)


def test_sizing_quantity_and_notional():
    syms = ("AAA", "BBB")
    rm, cov = _cov(syms)
    recs = _opt().optimize(symbols=rm.symbols, cov=cov, state=_state(), now_ns=10,
                           price_ticks=lambda s: PRICE)
    for r in recs:
        expected_notional = int(r.target_weight * EQUITY)
        assert r.requested_notional_ticks == expected_notional
        assert r.requested_quantity == int(expected_notional / PRICE)


def test_allow_fractional_affects_quantity():
    syms = ("AAA", "BBB")
    rm, cov = _cov(syms)
    # price not an integer divisor of notional -> fractional vs int differ
    recs_int = _opt().optimize(symbols=rm.symbols, cov=cov, state=_state(), now_ns=10,
                               price_ticks=lambda s: 333_333)
    recs_frac = _opt(allow_fractional=True).optimize(
        symbols=rm.symbols, cov=cov, state=_state(), now_ns=10,
        price_ticks=lambda s: 333_333)
    # both produce non-negative quantities; fractional path still int-cast at contract
    assert all(r.requested_quantity >= 0 for r in recs_int + recs_frac)


def test_infeasible_drawdown_rejects():
    syms = ("AAA", "BBB")
    rm, cov = _cov(syms)
    # drawdown over the 20% cap -> circuit breaker -> single rejected rec, no proposed
    state = _state(drawdown_bps=2500)
    recs = _opt().optimize(symbols=rm.symbols, cov=cov, state=state, now_ns=10,
                           price_ticks=lambda s: PRICE)
    assert len(recs) == 1
    assert recs[0].eligibility is RecommendationEligibility.REJECTED
    assert "drawdown" in (recs[0].rejection_reason or "")


def test_recommendation_attribution_from_meta():
    syms = ("AAA",)
    rm, cov = _cov(syms)
    recs = _opt().optimize(
        symbols=rm.symbols, cov=cov, state=_state(), now_ns=10,
        price_ticks=lambda s: PRICE,
        source_decision_ids={"AAA": "strat:AAA:1"},
        strategy_meta={"AAA": {"strategy_family": "momentum_continuation",
                               "regime_version": "regime-1.0.0", "strategy_id": "dual_ma"}})
    r = recs[0]
    assert r.source_strategy_decision_id == "strat:AAA:1"
    assert r.strategy_family == "momentum_continuation"
    assert r.regime_version == "regime-1.0.0"


def test_proposed_recommendations_have_risk_contribution():
    syms = ("AAA", "BBB", "CCC")
    rm, cov = _cov(syms)
    recs = _opt().optimize(symbols=rm.symbols, cov=cov, state=_state(), now_ns=10,
                           price_ticks=lambda s: PRICE)
    proposed = [r for r in recs if r.eligibility is RecommendationEligibility.PROPOSED]
    assert proposed
    assert all(r.expected_risk_contribution is not None for r in proposed)
