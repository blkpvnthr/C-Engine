import pytest

from cengine.portfolio import AccountSnapshot, AccountState, PositionBook
from order_manager import (
    ExecutionUpdate,
    ManagedOrder,
    OrderIntent,
    OrderStatus,
    OrderType,
    Side,
    TimeInForce,
)


def order(status=OrderStatus.SUBMITTED):
    intent = OrderIntent(
        "AAPL",
        Side.BUY,
        3,
        OrderType.MARKET,
        TimeInForce.DAY,
        strategy_id="turtle",
        client_order_id="c1",
        created_ns=1,
    )
    return ManagedOrder(intent, "c1", status=status, approved_quantity=3, venue_order_id="v1")


def test_account_requires_authoritative_snapshot():
    state = AccountState()
    with pytest.raises(RuntimeError):
        state.snapshot()
    state.replace_from_broker(AccountSnapshot(10, 20, 30, 0, 1))
    assert state.snapshot().buying_power_ticks == 20


def test_fill_updates_strategy_owned_position():
    book = PositionBook()
    managed = order(OrderStatus.FILLED)
    update = ExecutionUpdate("c1", "v1", OrderStatus.FILLED, 2, 3, 3, 101, venue_sequence=1)
    book.apply_execution(managed, update)
    assert book.quantity("AAPL") == 3
    assert book.quantity("AAPL", "turtle") == 3
    assert len(book.fills()) == 1
