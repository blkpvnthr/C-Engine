"""Intelligent-execution decision evidence (Milestone 4, §21).

Verifies the new M4 event types + emitters, journal round-trip, and plan->event
traceability.
"""

from __future__ import annotations

from cengine.decision_events import DecisionEventSink, DecisionEventType, events_from_rows
from cengine.execution_planner import ExecutionPlanner, Urgency
from cengine.microstructure import MicrostructureFeatures
from cengine.position_management import (
    AtrTrailingProtection,
    ExitPriority,
    PositionAction,
    PositionActionType,
)
from cengine.strategy_maturity import MaturityState
from cengine.transaction_cost import analyze_execution
from order_manager import (
    ManagedOrder,
    OrderIntent,
    OrderType,
    Side,
    TimeInForce,
)


def _features(symbol="AAA", stale=False):
    return MicrostructureFeatures(
        symbol=symbol, bid=99.99, ask=100.01, mid=100.0, spread_bps=2.0,
        bid_size=500, ask_size=400, kyle_lambda=1e-9, trade_arrival_intensity=2.0,
        mean_trade_size=300, is_stale=stale, confidence=0.9,
        quality=("STALE" if stale else "OK"))


def test_new_event_types_exist():
    for name in ("EXECUTION_PLAN", "EXECUTION_PLAN_REJECTED", "MICROSTRUCTURE_SNAPSHOT",
                 "TRANSACTION_COST_ANALYSIS", "POSITION_MANAGEMENT", "PROFIT_PROTECTION"):
        assert hasattr(DecisionEventType, name)


def test_record_execution_plan_emits_and_links_decision_ids():
    sink = DecisionEventSink()
    plan = ExecutionPlanner().plan(
        symbol="AAA", side=Side.BUY, quantity=100, now_ns=1_000, features=_features(),
        decision_price_ticks=1_000_000, urgency=Urgency.NORMAL,
        portfolio_decision_id="pf:AAA:1", strategy_decision_id="st:AAA:1")
    sink.record_execution_plan(plan, 2_000)
    (ev,) = sink.peek()
    assert ev.event_type is DecisionEventType.EXECUTION_PLAN
    assert ev.decision_id == plan.execution_plan_id
    assert ev.payload["portfolio_decision_id"] == "pf:AAA:1"
    assert ev.payload["execution_policy"] == plan.execution_policy
    assert ev.payload["limit_price_ticks"] == plan.limit_price_ticks


def test_record_execution_plan_rejected():
    sink = DecisionEventSink()
    plan = ExecutionPlanner().plan(
        symbol="AAA", side=Side.BUY, quantity=0, now_ns=1_000, features=_features(),
        decision_price_ticks=1_000_000)
    assert plan.rejection_reason is not None
    sink.record_execution_plan(plan, 2_000)
    assert sink.peek()[0].event_type is DecisionEventType.EXECUTION_PLAN_REJECTED


def test_record_microstructure():
    sink = DecisionEventSink()
    sink.record_microstructure(_features(), 2_000)
    ev = sink.peek()[0]
    assert ev.event_type is DecisionEventType.MICROSTRUCTURE_SNAPSHOT
    assert ev.payload["spread_bps"] == 2.0
    assert ev.payload["quality"] == "OK"


def _managed(symbol="AAA", qty=100, avg=1_000_200.0):
    intent = OrderIntent(symbol=symbol, side=Side.BUY, quantity=qty,
                         order_type=OrderType.MARKET, time_in_force=TimeInForce.DAY,
                         client_order_id="c1", created_ns=1_000)
    order = ManagedOrder(intent=intent, client_order_id="c1")
    order.approved_quantity = qty
    order.cumulative_filled_quantity = qty
    order.average_fill_price_ticks = avg
    return order


def test_record_transaction_cost_analysis():
    sink = DecisionEventSink()
    outcome = analyze_execution(
        _managed(), side=Side.BUY, decision_timestamp_ns=1_000,
        decision_price_ticks=1_000_000, arrival_price_ticks=1_000_100,
        submission_timestamp_ns=1_100, acknowledgement_timestamp_ns=1_200,
        final_fill_timestamp_ns=1_500, quoted_spread_bps=2.0, expected_slippage_bps=1.0)
    sink.record_transaction_cost_analysis(outcome, 2_000, decision_id="pf:AAA:1")
    ev = sink.peek()[0]
    assert ev.event_type is DecisionEventType.TRANSACTION_COST_ANALYSIS
    assert ev.payload["implementation_shortfall_bps"] == 2.0
    assert ev.payload["realized_slippage_bps"] == 0.9999


def test_record_position_management_profit_protection():
    sink = DecisionEventSink()
    action = PositionAction(
        symbol="AAA", priority=ExitPriority.PROFIT_PROTECTION,
        action_type=PositionActionType.EXIT_FULL, reason="atr_trailing",
        maturity=MaturityState.RESEARCH, shadow_only=True, policy="atr_trailing")
    sink.record_position_management(action, 2_000)
    ev = sink.peek()[0]
    assert ev.event_type is DecisionEventType.PROFIT_PROTECTION
    assert ev.payload["shadow_only"] is True
    assert ev.payload["policy"] == "atr_trailing"


def test_events_from_rows_round_trips_m4_events():
    sink = DecisionEventSink()
    sink.record_microstructure(_features(), 2_000)
    plan = ExecutionPlanner().plan(symbol="AAA", side=Side.BUY, quantity=10, now_ns=1,
                                   features=_features(), decision_price_ticks=1_000_000,
                                   portfolio_decision_id="pf:1")
    sink.record_execution_plan(plan, 3_000)
    rows = sink.drain()
    events = events_from_rows(rows)
    types = {e.event_type for e in events}
    assert DecisionEventType.MICROSTRUCTURE_SNAPSHOT in types
    assert DecisionEventType.EXECUTION_PLAN in types
    # payload survives the round-trip
    plan_ev = [e for e in events if e.event_type is DecisionEventType.EXECUTION_PLAN][0]
    assert plan_ev.payload["portfolio_decision_id"] == "pf:1"


def test_profit_protection_policy_shadow_by_default():
    # cross-check: a default-maturity policy yields a shadow_only action
    from cengine.position_management import PositionView
    from cengine.strategy_maturity import ExecutionAuthorization
    pol = AtrTrailingProtection()
    view = PositionView(symbol="AAA", quantity=100, unrealized_return=0.004,
                        max_favorable_excursion=0.02, atr_pct=0.005)
    action = pol.evaluate(view, authorization=ExecutionAuthorization.NONE)
    assert action is not None and action.shadow_only is True
