"""Deterministic mock broker (directive §9).

Verifies the isolated venue produces correct, deterministic acks and fill
schedules (full / partial-then-full / reject / no-fill) that reconcile cleanly
through the real OrderManager.
"""

from __future__ import annotations

import pytest

from cengine.mock_broker import FillPlan, SimulatedBroker
from order_manager import (
    OrderIntent,
    OrderManager,
    OrderStatus,
    OrderType,
    RiskDecision,
    Side,
    TimeInForce,
)


class _ApproveRisk:
    def evaluate(self, intent):
        return RiskDecision(True, "d", "v", intent.created_ns,
                            approved_quantity=intent.quantity)


def _intent(symbol="AAPL", qty=10, cid="c1"):
    return OrderIntent(symbol=symbol, side=Side.BUY, quantity=qty,
                       order_type=OrderType.MARKET, time_in_force=TimeInForce.DAY,
                       strategy_id="regime_intelligence", client_order_id=cid,
                       created_ns=1_000)


def test_submit_returns_deterministic_ack():
    broker = SimulatedBroker()
    ack1 = broker.submit(_intent(cid="a"))
    ack2 = broker.submit(_intent(cid="b"))
    assert ack1.venue_order_id == "MOCK-00000001"
    assert ack2.venue_order_id == "MOCK-00000002"
    assert ack1.accepted_ns > 0 and ack2.accepted_ns > ack1.accepted_ns


@pytest.mark.asyncio
async def test_full_fill_through_order_manager():
    broker = SimulatedBroker(prices_ticks={"AAPL": 1_000_000})
    mgr = OrderManager(risk_engine=_ApproveRisk(), venue=broker)
    order = await mgr.submit(_intent(qty=10))
    assert order.status is OrderStatus.SUBMITTED
    for upd in broker.drain_updates(order.venue_order_id):
        order = await mgr.reconcile(upd)
    assert order.status is OrderStatus.FILLED
    assert order.cumulative_filled_quantity == 10
    assert order.average_fill_price_ticks == 1_000_000


@pytest.mark.asyncio
async def test_partial_then_full():
    broker = SimulatedBroker(default_plan=FillPlan.PARTIAL_THEN_FULL,
                             prices_ticks={"AAPL": 500_000})
    mgr = OrderManager(risk_engine=_ApproveRisk(), venue=broker)
    order = await mgr.submit(_intent(qty=10))
    updates = broker.drain_updates(order.venue_order_id)
    assert len(updates) == 2
    order = await mgr.reconcile(updates[0])
    assert order.status is OrderStatus.PARTIALLY_FILLED
    assert order.cumulative_filled_quantity == 5
    order = await mgr.reconcile(updates[1])
    assert order.status is OrderStatus.FILLED
    assert order.cumulative_filled_quantity == 10


@pytest.mark.asyncio
async def test_reject_plan():
    broker = SimulatedBroker(default_plan=FillPlan.REJECT)
    mgr = OrderManager(risk_engine=_ApproveRisk(), venue=broker)
    order = await mgr.submit(_intent(qty=10))
    (upd,) = broker.drain_updates(order.venue_order_id)
    order = await mgr.reconcile(upd)
    assert order.status is OrderStatus.REJECTED
    assert order.cumulative_filled_quantity == 0
    assert order.rejection_reason == "mock_venue_rejected"


@pytest.mark.asyncio
async def test_nofill_plan_stays_resting():
    broker = SimulatedBroker(default_plan=FillPlan.NOFILL)
    mgr = OrderManager(risk_engine=_ApproveRisk(), venue=broker)
    order = await mgr.submit(_intent(qty=10))
    assert broker.drain_updates(order.venue_order_id) == []
    assert order.status is OrderStatus.SUBMITTED


def test_per_symbol_plan_and_price_overrides():
    broker = SimulatedBroker()
    broker.set_plan("SQQQ", FillPlan.REJECT)
    broker.set_price("AAPL", 1_234_500)
    ack = broker.submit(_intent(symbol="AAPL", cid="x"))
    (upd,) = broker.drain_updates(ack.venue_order_id)
    assert upd.last_fill_price_ticks == 1_234_500
    ack2 = broker.submit(_intent(symbol="SQQQ", cid="y"))
    (upd2,) = broker.drain_updates(ack2.venue_order_id)
    assert upd2.status is OrderStatus.REJECTED


def test_cancel_removes_schedule():
    broker = SimulatedBroker(default_plan=FillPlan.NOFILL)
    ack = broker.submit(_intent(cid="z"))
    broker.cancel(client_order_id="z", venue_order_id=ack.venue_order_id)
    assert broker.canceled == 1
    assert ack.venue_order_id not in broker.pending_order_ids()
