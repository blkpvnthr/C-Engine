"""Intelligent execution planner + policies + limit prices (Milestone 4, §8/§9)."""

from __future__ import annotations

from cengine.execution_planner import (
    PENNY_TICKS,
    ExecutionPlan,
    ExecutionPlanner,
    ExecutionPolicy,
    PlannedExecutionPolicy,
    Urgency,
    compute_limit_price_ticks,
    select_policy,
)
from cengine.microstructure import MicrostructureFeatures
from order_manager import OrderType, Side


def _feat(symbol="AAA", *, bid=99.99, ask=100.01, mid=100.0, spread_bps=2.0,
          bid_size=500, ask_size=400, is_stale=False, crossed=False, kyle=None,
          intensity=2.0, mean_size=300.0):
    return MicrostructureFeatures(
        symbol=symbol, bid=bid, ask=ask, mid=mid, spread_bps=spread_bps,
        bid_size=bid_size, ask_size=ask_size, is_stale=is_stale,
        crossed_or_locked=crossed, kyle_lambda=kyle, trade_arrival_intensity=intensity,
        mean_trade_size=mean_size, confidence=0.9, quality="OK")


class _Cand:
    def __init__(self, symbol):
        self.symbol = symbol


# ------------------------------------------------------------------- planner
def test_tight_spread_normal_is_marketable_limit():
    p = ExecutionPlanner().plan(symbol="AAA", side=Side.BUY, quantity=100, now_ns=1_000,
                                features=_feat(), decision_price_ticks=1_000_000,
                                urgency=Urgency.NORMAL)
    assert p.execution_policy == ExecutionPolicy.MARKETABLE_LIMIT.value
    assert p.limit_price_ticks is not None and p.limit_price_ticks % PENNY_TICKS == 0
    assert p.limit_price_ticks <= 1_000_100 + PENNY_TICKS   # at/near the ask
    assert p.cost_components["basis"] == "microstructure"
    assert p.estimated_transaction_cost_bps is not None


def test_wide_spread_patient_is_passive_or_bounded():
    f = _feat(bid=99.0, ask=100.0, mid=99.5, spread_bps=100.0, bid_size=200, ask_size=200)
    p = ExecutionPlanner().plan(symbol="BBB", side=Side.BUY, quantity=10, now_ns=1,
                                features=f, decision_price_ticks=995_000,
                                urgency=Urgency.PATIENT)
    assert p.execution_policy in (ExecutionPolicy.PASSIVE_LIMIT.value,
                                  ExecutionPolicy.BOUNDED_REPRICE.value)
    assert p.limit_price_ticks == 990_000   # joins the bid


def test_stale_features_force_immediate_market_with_conservative_cost():
    p = ExecutionPlanner().plan(symbol="CCC", side=Side.BUY, quantity=10, now_ns=1,
                                features=_feat(is_stale=True), urgency=Urgency.NORMAL)
    assert p.execution_policy == ExecutionPolicy.IMMEDIATE_EXECUTION.value
    assert p.limit_price_ticks is None
    assert p.cost_components["basis"] == "conservative_fallback"


def test_urgent_and_mandatory_policies():
    pu = ExecutionPlanner().plan(symbol="AAA", side=Side.SELL, quantity=10, now_ns=1,
                                 features=_feat(), urgency=Urgency.URGENT)
    assert pu.execution_policy == ExecutionPolicy.URGENT_RISK_REDUCTION.value
    assert pu.limit_price_ticks is None
    pm = ExecutionPlanner().plan(symbol="AAA", side=Side.SELL, quantity=10, now_ns=1,
                                 features=_feat(), urgency=Urgency.MANDATORY)
    assert pm.execution_policy == ExecutionPolicy.MANDATORY_PROTECTIVE_EXIT.value


def test_large_order_vs_liquidity_is_time_sliced():
    f = _feat(ask_size=100)   # top-of-book 100 shares
    p = ExecutionPlanner().plan(symbol="AAA", side=Side.BUY, quantity=80, now_ns=1,
                                features=f, decision_price_ticks=1_000_000,
                                urgency=Urgency.NORMAL)
    assert p.execution_policy == ExecutionPolicy.TIME_SLICED_EXECUTION.value


def test_no_quote_downgrades_limit_policy_to_immediate_no_fabricated_price():
    f = _feat(bid=None, ask=None)  # spread known but no top-of-book prices
    p = ExecutionPlanner().plan(symbol="AAA", side=Side.BUY, quantity=10, now_ns=1,
                                features=f, decision_price_ticks=1_000_000,
                                urgency=Urgency.NORMAL)
    assert p.execution_policy == ExecutionPolicy.IMMEDIATE_EXECUTION.value
    assert p.limit_price_ticks is None


def test_contract_fields_present():
    p = ExecutionPlanner(execution_authorization="mock").plan(
        symbol="AAA", side=Side.BUY, quantity=10, now_ns=5_000, features=_feat(),
        decision_price_ticks=1_000_000, strategy_decision_id="s1",
        portfolio_decision_id="p1", capital_reservation_id="r1")
    assert p.strategy_decision_id == "s1" and p.portfolio_decision_id == "p1"
    assert p.capital_reservation_id == "r1"
    assert p.execution_authorization == "mock"
    assert p.expiration_timestamp > p.decision_timestamp
    assert p.observed_bid_ticks and p.observed_ask_ticks


# ----------------------------------------------------- compute_limit_price_ticks
def test_buy_limit_bounded_by_max_slippage():
    # ask far above decision -> limit bounded by decision*(1+slippage), not the ask
    lim = compute_limit_price_ticks(side=Side.BUY, policy=ExecutionPolicy.MARKETABLE_LIMIT,
                                    bid_ticks=99_900, ask_ticks=2_000_000,
                                    decision_price_ticks=100_000, max_slippage_bps=25.0)
    assert lim is not None and lim <= int(100_000 * 1.0025) + PENNY_TICKS
    assert lim < 2_000_000


def test_sell_limit_floored_by_max_slippage():
    lim = compute_limit_price_ticks(side=Side.SELL, policy=ExecutionPolicy.MARKETABLE_LIMIT,
                                    bid_ticks=10_000, ask_ticks=100_100,
                                    decision_price_ticks=100_000, max_slippage_bps=25.0)
    assert lim is not None and lim >= int(100_000 * 0.9975) - PENNY_TICKS


def test_immediate_policy_has_no_limit():
    assert compute_limit_price_ticks(
        side=Side.BUY, policy=ExecutionPolicy.IMMEDIATE_EXECUTION,
        bid_ticks=100, ask_ticks=200, decision_price_ticks=150) is None


def test_passive_buy_joins_bid():
    lim = compute_limit_price_ticks(side=Side.BUY, policy=ExecutionPolicy.PASSIVE_LIMIT,
                                    bid_ticks=99_900, ask_ticks=100_100,
                                    decision_price_ticks=100_000)
    assert lim == 99_900


# --------------------------------------------------- select_policy direct checks
def test_select_policy_matrix():
    assert select_policy(urgency=Urgency.MANDATORY, spread_bps=2.0, expected_cost_bps=None,
                         expiring_soon=False, momentum_up=False, size_vs_liquidity=0.0,
                         stale=False) is ExecutionPolicy.MANDATORY_PROTECTIVE_EXIT
    assert select_policy(urgency=Urgency.NORMAL, spread_bps=None, expected_cost_bps=None,
                         expiring_soon=False, momentum_up=False, size_vs_liquidity=0.0,
                         stale=False) is ExecutionPolicy.IMMEDIATE_EXECUTION
    assert select_policy(urgency=Urgency.NORMAL, spread_bps=10.0, expected_cost_bps=None,
                         expiring_soon=False, momentum_up=False, size_vs_liquidity=0.0,
                         stale=False) is ExecutionPolicy.PASSIVE_LIMIT


# ----------------------------------------------------- PlannedExecutionPolicy
def test_planned_policy_returns_instruction_and_fallback():
    planner = ExecutionPlanner()
    policy = PlannedExecutionPolicy()
    plan = planner.plan(symbol="AAA", side=Side.BUY, quantity=100, now_ns=1,
                        features=_feat(), decision_price_ticks=1_000_000)
    policy.set_plan(plan)
    instr = policy.instruction_for(_Cand("AAA"))
    assert instr.order_type is OrderType.LIMIT and instr.limit_price_ticks == plan.limit_price_ticks
    # unknown symbol -> MARKET fallback
    fallback = policy.instruction_for(_Cand("ZZZ"))
    assert fallback.order_type is OrderType.MARKET and fallback.limit_price_ticks is None


def test_planned_policy_market_for_immediate():
    planner = ExecutionPlanner()
    policy = PlannedExecutionPolicy()
    plan = planner.plan(symbol="CCC", side=Side.BUY, quantity=10, now_ns=1,
                        features=_feat(is_stale=True))
    policy.set_plan(plan)
    instr = policy.instruction_for(_Cand("CCC"))
    assert instr.order_type is OrderType.MARKET
    assert isinstance(plan, ExecutionPlan)
