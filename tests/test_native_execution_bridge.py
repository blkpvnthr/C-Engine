import pytest

from cengine.native_execution import NativeExecutionBridge
from order_manager import (
    ExecutionAck,
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
        return RiskDecision(True, "d", "test", intent.created_ns)


class Venue:
    def submit(self, intent):
        return ExecutionAck(intent.client_order_id, "native-venue", intent.created_ns + 1)

    def cancel(self, **kwargs):
        return None


class NativeExecution:
    maker_id = 1
    taker_id = 999
    quantity = 2
    price_ticks = 101


@pytest.mark.asyncio
async def test_native_execution_reaches_order_manager_reconcile():
    manager = OrderManager(risk_engine=Risk(), venue=Venue())
    order = await manager.submit(
        OrderIntent(
            "AAPL",
            Side.BUY,
            2,
            OrderType.MARKET,
            TimeInForce.DAY,
            client_order_id="client",
            created_ns=1,
        )
    )
    bridge = NativeExecutionBridge(manager.reconcile)
    bridge.register(1, order.client_order_id, order.venue_order_id, 2)
    await bridge.on_execution(NativeExecution())
    assert manager.snapshot("client").status is OrderStatus.FILLED
