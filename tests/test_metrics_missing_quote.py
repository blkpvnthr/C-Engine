"""Regression: the metrics snapshot must not crash when a held position lacks a
fresh authoritative quote.

Exposed once real fills produced held positions: the reconciliation loop's
metrics.snapshot() marked every position to market via price_ticks(), which RAISES
'authoritative quote unavailable' at startup (before a held symbol's first quote
arrives) or for a quiet symbol -> crashed the reconciliation loop and the engine.
The snapshot now marks such a position at cost basis instead of crashing.
"""

from __future__ import annotations

from cengine.metrics import PortfolioMetricsCollector
from cengine.portfolio import AccountSnapshot, AccountState, BrokerPosition, PositionBook
from cengine.reservations import RiskReservationBook


class _RaisingPrices:
    def price_ticks(self, symbol: str) -> int:
        raise RuntimeError(f"authoritative quote unavailable for {symbol}")


def test_snapshot_marks_at_cost_basis_when_quote_unavailable():
    account = AccountState()
    account.replace_from_broker(AccountSnapshot(1_000, 2_000, 3_000, 0, 1))
    positions = PositionBook()
    positions.replace_from_broker((BrokerPosition("DIA", 2, 100),), (), 1)  # cost 2*100
    collector = PortfolioMetricsCollector(
        account, positions, RiskReservationBook(), _RaisingPrices())

    m = collector.snapshot(10)  # must NOT raise despite the missing quote

    assert m.gross_exposure_ticks == 200  # cost-basis fallback (quantity * avg price)
    assert m.net_exposure_ticks == 200
    assert m.position_count == 1
