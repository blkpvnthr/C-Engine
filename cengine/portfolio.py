"""Authoritative account, position, fill, and open-order state."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from threading import RLock
from typing import Optional

from order_manager import ExecutionUpdate, ManagedOrder, OrderStatus, Side


@dataclass(frozen=True, slots=True)
class Fill:
    client_order_id: str
    venue_order_id: str
    strategy_id: str
    symbol: str
    side: Side
    quantity: int
    price_ticks: int
    event_ns: int


@dataclass(slots=True)
class Position:
    symbol: str
    quantity: int = 0
    cost_ticks: int = 0
    strategy_quantities: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class AccountSnapshot:
    cash_ticks: int
    buying_power_ticks: int
    equity_ticks: int
    realized_pnl_ticks: int
    updated_ns: int


class AccountState:
    def __init__(self) -> None:
        self._snapshot: Optional[AccountSnapshot] = None
        self._lock = RLock()

    def replace_from_broker(self, snapshot: AccountSnapshot) -> None:
        if snapshot.updated_ns <= 0:
            raise ValueError("broker account timestamp must be positive")
        with self._lock:
            if self._snapshot and snapshot.updated_ns < self._snapshot.updated_ns:
                raise ValueError("stale broker account snapshot")
            self._snapshot = snapshot

    def snapshot(self) -> AccountSnapshot:
        with self._lock:
            if self._snapshot is None:
                raise RuntimeError("authoritative broker account state unavailable")
            return replace(self._snapshot)


class PositionBook:
    """Fill-ledger-derived positions with strategy ownership."""

    def __init__(self) -> None:
        self._positions: dict[str, Position] = {}
        self._fills: list[Fill] = []
        self._open_orders: dict[str, ManagedOrder] = {}
        self._lock = RLock()

    def track_order(self, order: ManagedOrder) -> None:
        with self._lock:
            if order.status in {
                OrderStatus.FILLED,
                OrderStatus.CANCELED,
                OrderStatus.REJECTED,
                OrderStatus.EXPIRED,
                OrderStatus.RISK_REJECTED,
            }:
                self._open_orders.pop(order.client_order_id, None)
            else:
                self._open_orders[order.client_order_id] = replace(order)

    def apply_execution(self, order: ManagedOrder, update: ExecutionUpdate) -> None:
        if update.last_fill_quantity == 0:
            self.track_order(order)
            return
        if update.last_fill_price_ticks is None:
            raise ValueError("priced fill required")
        signed = (
            update.last_fill_quantity
            if order.intent.side is Side.BUY
            else -update.last_fill_quantity
        )
        strategy = order.intent.strategy_id
        symbol = order.intent.symbol
        fill = Fill(
            order.client_order_id,
            update.venue_order_id,
            strategy,
            symbol,
            order.intent.side,
            update.last_fill_quantity,
            update.last_fill_price_ticks,
            update.event_ns,
        )
        with self._lock:
            position = self._positions.setdefault(symbol, Position(symbol))
            position.quantity += signed
            position.cost_ticks += signed * update.last_fill_price_ticks
            position.strategy_quantities[strategy] = (
                position.strategy_quantities.get(strategy, 0) + signed
            )
            self._fills.append(fill)
            self.track_order(order)

    def quantity(self, symbol: str, strategy_id: Optional[str] = None) -> int:
        with self._lock:
            position = self._positions.get(symbol.strip().upper())
            if position is None:
                return 0
            if strategy_id is None:
                return position.quantity
            return position.strategy_quantities.get(strategy_id, 0)

    def fills(self) -> tuple[Fill, ...]:
        with self._lock:
            return tuple(self._fills)

    def open_orders(self) -> tuple[ManagedOrder, ...]:
        with self._lock:
            return tuple(replace(x) for x in self._open_orders.values())
