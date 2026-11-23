"""Decision-event integrity (directive §8).

Complements tests/test_decision_events.py with the execution-lifecycle emitters,
bounded-queue overflow accounting, journal persistence-failure resilience, and
missing-event recovery from the authoritative journal.
"""

from __future__ import annotations

from cengine.decision_events import (
    DecisionEvent,
    DecisionEventSink,
    DecisionEventType,
    events_from_rows,
)
from order_manager import (
    ExecutionUpdate,
    ManagedOrder,
    OrderIntent,
    OrderStatus,
    OrderType,
    Side,
    TimeInForce,
)


def _managed(symbol="AAPL", qty=10, corr="AAPL:1:regime_intelligence"):
    intent = OrderIntent(symbol=symbol, side=Side.BUY, quantity=qty,
                         order_type=OrderType.MARKET, time_in_force=TimeInForce.DAY,
                         strategy_id="regime_intelligence", correlation_id=corr,
                         client_order_id="c1", created_ns=1_000)
    order = ManagedOrder(intent=intent, client_order_id="c1")
    order.venue_order_id = "MOCK-1"
    order.last_event_ns = 2_000
    return order


# ------------------------------------------------------------- execution emitters
def test_order_intent_emitter_preserves_attribution():
    sink = DecisionEventSink()
    order = _managed()

    class _Rec:
        decision_id = "D1"
        symbol_id = "AAPL"
        strategy_family = type("F", (), {"value": "momentum_continuation"})()
        strategy_version = "v1"
        regime_version = "regime-1.0.0"
        feature_version = "features-1.0.0"

    sink.record_order_intent(_Rec(), order.intent, 2_000,
                             authorization="mock", reference_price_ticks=1_000_000)
    (ev,) = sink.peek()
    assert ev.event_type is DecisionEventType.ORDER_INTENT
    assert ev.decision_id == "D1"
    assert ev.payload["correlation_id"] == order.intent.correlation_id
    assert ev.payload["authorization"] == "mock"


def test_risk_and_fill_emitters():
    sink = DecisionEventSink()
    order = _managed()
    sink.record_risk_decision(decision_id="D1", symbol_id="AAPL",
                              correlation_id=order.intent.correlation_id,
                              approved=True, reason="ok", now_ns=2_000)
    sink.record_broker_ack(order, 2_000, decision_id="D1")
    order.cumulative_filled_quantity = 10
    order.average_fill_price_ticks = 1_000_000.0
    upd = ExecutionUpdate(client_order_id="c1", venue_order_id="MOCK-1",
                          status=OrderStatus.FILLED, event_ns=3_000,
                          cumulative_filled_quantity=10, last_fill_quantity=10,
                          last_fill_price_ticks=1_000_000)
    sink.record_fill(order, upd, 3_000, decision_id="D1")
    types = [e.event_type for e in sink.peek()]
    assert DecisionEventType.RISK_APPROVAL in types
    assert DecisionEventType.BROKER_ACK in types
    assert DecisionEventType.COMPLETE_FILL in types


def test_partial_vs_complete_fill_classification():
    sink = DecisionEventSink()
    order = _managed()
    partial = ExecutionUpdate(client_order_id="c1", venue_order_id="MOCK-1",
                              status=OrderStatus.PARTIALLY_FILLED, event_ns=3_000,
                              cumulative_filled_quantity=4, last_fill_quantity=4,
                              last_fill_price_ticks=1_000_000)
    sink.record_fill(order, partial, 3_000)
    assert sink.peek()[-1].event_type is DecisionEventType.PARTIAL_FILL


# ------------------------------------------------------------------- overflow (§8)
def test_bounded_overflow_accounts_dropped_without_raising():
    sink = DecisionEventSink(capacity=3)
    for i in range(10):
        sink.emit(DecisionEvent(event_type=DecisionEventType.ORDER_INTENT,
                                timestamp_ns=i, decision_id=str(i)))
    assert sink.size == 3
    assert sink.emitted == 10
    assert sink.dropped == 7
    # oldest evicted; newest retained
    assert [e.decision_id for e in sink.peek()] == ["7", "8", "9"]


# ------------------------------------------------- persistence-failure resilience
def test_journal_failure_never_blocks_emission():
    class _BadJournal:
        def append(self, *a, **k):
            raise RuntimeError("journal down")

    sink = DecisionEventSink(journal=_BadJournal())
    # must not raise despite the journal failing on every append
    sink.record_risk_decision(decision_id="D", symbol_id="AAPL",
                              correlation_id="x", approved=False, reason="r",
                              now_ns=1)
    assert sink.size == 1
    assert sink.emitted == 1


# ---------------------------------------------------------- missing-event recovery
def test_recovery_from_journal_rows():
    # Simulate the authoritative journal capturing rows, the in-memory buffer losing
    # them to overflow, then recovering them by rehydrating the journal rows.
    source = DecisionEventSink()
    source.record_risk_decision(decision_id="D1", symbol_id="AAPL",
                                correlation_id="AAPL:1:regime_intelligence",
                                approved=True, reason="ok", now_ns=10)
    rows = source.drain()  # these would have been persisted to the journal
    assert source.size == 0

    recovered = DecisionEventSink()
    added = recovered.rehydrate(rows)
    assert added == 1
    ev = recovered.peek()[0]
    assert ev.event_type is DecisionEventType.RISK_APPROVAL
    assert ev.decision_id == "D1"
    assert ev.payload["correlation_id"] == "AAPL:1:regime_intelligence"


def test_events_from_rows_skips_unknown_event_types():
    rows = [
        {"event_type": "order_intent", "timestamp_ns": 1, "decision_id": "D1"},
        {"event_type": "not_a_real_event", "timestamp_ns": 2},
        {"event_type": "complete_fill", "timestamp_ns": 3, "payload": {"x": 1}},
    ]
    events = events_from_rows(rows)
    assert [e.event_type for e in events] == [
        DecisionEventType.ORDER_INTENT, DecisionEventType.COMPLETE_FILL]
    assert events[1].payload == {"x": 1}


def test_realized_outcome_emitter_traceable():
    sink = DecisionEventSink()
    sink.record_realized_outcome(decision_id="D1", symbol_id="AAPL",
                                 correlation_id="AAPL:1:regime_intelligence",
                                 now_ns=9, realized_return=0.012, holding_ns=600,
                                 realized_pnl_ticks=1200)
    (ev,) = sink.peek()
    assert ev.event_type is DecisionEventType.REALIZED_OUTCOME
    assert ev.decision_id == "D1"
    assert ev.payload["realized_return"] == 0.012
