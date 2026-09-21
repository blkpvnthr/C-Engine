import pytest

from cengine.portfolio import BrokerOrder, BrokerPosition, PositionBook
from order_manager import Side


def test_unknown_broker_order_blocks_startup():
    book = PositionBook()
    book.replace_from_broker(
        (BrokerPosition("AAPL", 2, 100),),
        (BrokerOrder("unknown", "venue-1", "AAPL", Side.BUY, 2, 0, "new"),),
        1,
    )
    with pytest.raises(RuntimeError, match="unowned broker open orders"):
        book.assert_consistent()


def test_broker_position_divergence_is_rejected():
    book = PositionBook()
    book.replace_from_broker((BrokerPosition("AAPL", 2, 100),), (), 1)
    # Broker later loses the position while internal authoritative total is nonzero.
    with pytest.raises(RuntimeError, match="missing internal position"):
        book.replace_from_broker((), (), 2)
