"""Authoritative, versioned portfolio state (Milestone 3, §4).

Assembles a point-in-time :class:`PortfolioState` from the authoritative broker-
reconciled inputs (AccountState, PositionBook, RiskReservationBook, live prices)
and the exposure registry. Broker-reconciled state is the authoritative financial
input; submitted sell orders are NOT treated as completed liquidations and their
proceeds are NOT assumed available.

Elective allocation must fail closed when the state is stale or unreconciled:
:meth:`PortfolioState.allocatable` returns False in that case.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional

PORTFOLIO_STATE_VERSION = "portfolio-state-1.0.0"


class ReconciliationStatus(str, Enum):
    RECONCILED = "reconciled"
    PENDING = "pending"
    STALE = "stale"
    UNRECONCILED = "unreconciled"


@dataclass(frozen=True)
class PortfolioPosition:
    symbol: str
    quantity: int
    market_value_ticks: int
    effective_exposure_ticks: int             # signed, leverage-adjusted
    exposure_group: Optional[str]


@dataclass(frozen=True)
class PortfolioState:
    portfolio_version: str
    observation_timestamp: int
    account_equity_ticks: int
    settled_cash_ticks: int
    available_cash_ticks: int
    buying_power_ticks: int
    reserved_capital_ticks: int
    pending_buy_exposure_ticks: int
    pending_sell_exposure_ticks: int
    realized_pl_ticks: int
    unrealized_pl_ticks: int
    gross_exposure_ticks: int
    net_exposure_ticks: int
    effective_exposure_ticks: int             # signed, leverage-adjusted portfolio total
    positions: tuple = ()
    open_order_count: int = 0
    portfolio_high_watermark_ticks: int = 0
    drawdown_bps: int = 0
    reconciliation_status: ReconciliationStatus = ReconciliationStatus.UNRECONCILED
    staleness_ns: int = 0

    @property
    def drawdown_fraction(self) -> float:
        return self.drawdown_bps / 10_000.0

    def allocatable(self) -> tuple[bool, str]:
        """Elective allocation fail-closed gate (§4). Mandatory protective exits are
        NOT governed by this — they run independently regardless of state."""
        if self.reconciliation_status in (ReconciliationStatus.STALE,
                                          ReconciliationStatus.UNRECONCILED):
            return False, f"portfolio_state_{self.reconciliation_status.value}"
        if self.available_cash_ticks <= 0 and self.buying_power_ticks <= 0:
            return False, "no_available_capital"
        return True, "allocatable"


class PortfolioStateBuilder:
    """Builds versioned PortfolioState snapshots from authoritative inputs.

    Reuses the account/positions/reservations aggregates. ``exposure_info`` maps a
    symbol to an object exposing ``target_daily_leverage`` and
    ``underlying_exposure_group`` (the ExposureRegistry entry), or None for
    conventional equity (leverage +1). High-watermark is tracked across snapshots.
    """

    def __init__(self, *, account, positions, reservations,
                 price_ticks: Callable[[str], int],
                 exposure_info: Optional[Callable[[str], object]] = None,
                 max_staleness_ns: int = 30 * 1_000_000_000,
                 now_fn: Callable[[], int] = time.time_ns) -> None:
        self._account = account
        self._positions = positions
        self._reservations = reservations
        self._price_ticks = price_ticks
        self._exposure_info = exposure_info or (lambda s: None)
        self._max_staleness_ns = int(max_staleness_ns)
        self._now = now_fn
        self._high_water_ticks = 0

    def _leverage(self, symbol: str) -> float:
        info = self._exposure_info(symbol)
        lev = getattr(info, "target_daily_leverage", None) if info is not None else None
        return float(lev) if lev is not None else 1.0

    def _group(self, symbol: str) -> Optional[str]:
        info = self._exposure_info(symbol)
        return getattr(info, "underlying_exposure_group", None) if info is not None else None

    def build(self, *, now_ns: Optional[int] = None,
              account_observed_ns: Optional[int] = None) -> PortfolioState:
        now_ns = now_ns if now_ns is not None else self._now()
        try:
            account = self._account.snapshot()
        except Exception:  # noqa: BLE001 — no authoritative account -> unreconciled
            return PortfolioState(
                PORTFOLIO_STATE_VERSION, now_ns, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                reconciliation_status=ReconciliationStatus.UNRECONCILED)

        gross = 0
        net = 0
        effective = 0
        positions: list[PortfolioPosition] = []
        unrealized = 0
        for pos in self._positions.positions():
            if pos.quantity == 0:
                continue
            try:
                px = int(self._price_ticks(pos.symbol))
            except Exception:  # noqa: BLE001 — missing quote -> state not fully priced
                return PortfolioState(
                    PORTFOLIO_STATE_VERSION, now_ns, account.equity_ticks,
                    account.cash_ticks, 0, account.buying_power_ticks,
                    self._reservations.gross_notional_ticks(), 0, 0,
                    account.realized_pnl_ticks, 0, 0, 0, 0,
                    reconciliation_status=ReconciliationStatus.UNRECONCILED)
            mv = pos.quantity * px
            lev = self._leverage(pos.symbol)
            eff = int(mv * lev)
            gross += abs(mv)
            net += mv
            effective += eff
            unrealized += pos.quantity * px - pos.cost_ticks
            positions.append(PortfolioPosition(pos.symbol, pos.quantity, mv, eff,
                                              self._group(pos.symbol)))

        # Pending exposure from reservations (submitted, unfilled) — NOT realized.
        pending_buy = 0
        pending_sell = 0
        reserved = self._reservations.gross_notional_ticks()
        for r in self._reservations.snapshot():
            notional = r.quantity * r.price_ticks
            if r.signed_quantity > 0:
                pending_buy += notional
            else:
                pending_sell += notional

        # available cash is conservative: settled cash minus pending BUY reservations.
        # Pending SELL proceeds are NOT added (submitted sells are not completed).
        available_cash = account.cash_ticks - pending_buy

        self._high_water_ticks = max(self._high_water_ticks, account.equity_ticks)
        drawdown_bps = 0
        if self._high_water_ticks > 0:
            drawdown_bps = max(0, (self._high_water_ticks - account.equity_ticks)
                               * 10_000 // self._high_water_ticks)

        staleness = 0
        status = ReconciliationStatus.RECONCILED
        if account_observed_ns is not None:
            staleness = max(0, now_ns - int(account_observed_ns))
            if staleness > self._max_staleness_ns:
                status = ReconciliationStatus.STALE

        return PortfolioState(
            portfolio_version=PORTFOLIO_STATE_VERSION,
            observation_timestamp=now_ns,
            account_equity_ticks=account.equity_ticks,
            settled_cash_ticks=account.cash_ticks,
            available_cash_ticks=available_cash,
            buying_power_ticks=account.buying_power_ticks,
            reserved_capital_ticks=reserved,
            pending_buy_exposure_ticks=pending_buy,
            pending_sell_exposure_ticks=pending_sell,
            realized_pl_ticks=account.realized_pnl_ticks,
            unrealized_pl_ticks=unrealized,
            gross_exposure_ticks=gross,
            net_exposure_ticks=net,
            effective_exposure_ticks=effective,
            positions=tuple(positions),
            open_order_count=len(self._positions.open_orders()),
            portfolio_high_watermark_ticks=self._high_water_ticks,
            drawdown_bps=drawdown_bps,
            reconciliation_status=status,
            staleness_ns=staleness,
        )
