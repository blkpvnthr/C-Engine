"""Portfolio telemetry calculated from authoritative state and observed prices."""

from __future__ import annotations

import time
from dataclasses import dataclass
from threading import RLock
from typing import Protocol

from .portfolio import AccountState, PositionBook
from .reservations import RiskReservationBook


class PriceProvider(Protocol):
    def price_ticks(self, symbol: str) -> int: ...


@dataclass(frozen=True, slots=True)
class PortfolioMetrics:
    timestamp_ns: int
    cash_ticks: int
    equity_ticks: int
    buying_power_ticks: int
    realized_pnl_ticks: int
    gross_exposure_ticks: int
    net_exposure_ticks: int
    reserved_notional_ticks: int
    position_count: int
    open_order_count: int
    drawdown_bps: int


class PortfolioMetricsCollector:
    def __init__(
        self,
        account: AccountState,
        positions: PositionBook,
        reservations: RiskReservationBook,
        prices: PriceProvider,
    ) -> None:
        self.account = account
        self.positions = positions
        self.reservations = reservations
        self.prices = prices
        self._high_water_equity_ticks = 0
        self._lock = RLock()

    def snapshot(self, timestamp_ns: int | None = None) -> PortfolioMetrics:
        account = self.account.snapshot()
        gross = 0
        net = 0
        positions = self.positions.positions()
        for position in positions:
            value = position.quantity * self.prices.price_ticks(position.symbol)
            gross += abs(value)
            net += value
        with self._lock:
            self._high_water_equity_ticks = max(self._high_water_equity_ticks, account.equity_ticks)
            drawdown_bps = 0
            if self._high_water_equity_ticks > 0:
                drawdown_bps = max(
                    0,
                    (self._high_water_equity_ticks - account.equity_ticks)
                    * 10_000
                    // self._high_water_equity_ticks,
                )
        return PortfolioMetrics(
            timestamp_ns=timestamp_ns or time.time_ns(),
            cash_ticks=account.cash_ticks,
            equity_ticks=account.equity_ticks,
            buying_power_ticks=account.buying_power_ticks,
            realized_pnl_ticks=account.realized_pnl_ticks,
            gross_exposure_ticks=gross,
            net_exposure_ticks=net,
            reserved_notional_ticks=self.reservations.gross_notional_ticks(),
            position_count=sum(position.quantity != 0 for position in positions),
            open_order_count=len(self.positions.open_orders()),
            drawdown_bps=drawdown_bps,
        )
