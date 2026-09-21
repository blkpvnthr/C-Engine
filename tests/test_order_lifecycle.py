import pytest

from order_manager import (
    ExecutionAck,
    ExecutionUpdate,
    OrderIntent,
    OrderManager,
    OrderStatus,
    OrderType,
    RiskDecision,
    Side,
    TimeInForce,
)


class Risk:
    def evaluate(self, intent):
        return RiskDecision(
            True, "d1", "test", intent.created_ns, approved_quantity=intent.quantity
        )


class PaperVenue:
    def submit(self, intent):
        return ExecutionAck(intent.client_order_id, "paper-1", intent.created_ns + 1)

    def cancel(self, **kwargs):
        return None


@pytest.mark.asyncio
async def test_ack_is_not_fill_and_fill_is_reconciled():
    manager = OrderManager(risk_engine=Risk(), venue=PaperVenue())
    intent = OrderIntent(
        "AAPL",
        Side.BUY,
        2,
        OrderType.MARKET,
        TimeInForce.DAY,
        client_order_id="client-1",
        created_ns=10,
    )
    submitted = await manager.submit(intent)
    assert submitted.status is OrderStatus.SUBMITTED
    filled = await manager.reconcile(
        ExecutionUpdate("client-1", "paper-1", OrderStatus.FILLED, 12, 2, 2, 100, venue_sequence=1)
    )
    assert filled.status is OrderStatus.FILLED
    assert filled.average_fill_price_ticks == 100
