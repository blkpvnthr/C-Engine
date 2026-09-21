from cengine.metrics import PortfolioMetricsCollector
from cengine.portfolio import AccountSnapshot, AccountState, BrokerPosition, PositionBook
from cengine.reservations import RiskReservation, RiskReservationBook
from order_manager import Side


class Prices:
    def price_ticks(self, symbol: str) -> int:
        assert symbol == "AAPL"
        return 110


def test_portfolio_metrics_include_exposure_reservations_and_drawdown():
    account = AccountState()
    account.replace_from_broker(AccountSnapshot(1_000, 2_000, 3_000, 25, 1))
    positions = PositionBook()
    positions.replace_from_broker((BrokerPosition("AAPL", 2, 100),), (), 1)
    reservations = RiskReservationBook()
    reservations.reserve(RiskReservation("c1", "AAPL", Side.BUY, 3, 105))
    collector = PortfolioMetricsCollector(account, positions, reservations, Prices())

    first = collector.snapshot(10)
    assert first.gross_exposure_ticks == 220
    assert first.net_exposure_ticks == 220
    assert first.reserved_notional_ticks == 315
    assert first.drawdown_bps == 0

    account.replace_from_broker(AccountSnapshot(900, 1_800, 2_700, 25, 2))
    assert collector.snapshot(11).drawdown_bps == 1_000
