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
class BrokerPosition:
    symbol: str
    quantity: int
    average_entry_price_ticks: int


@dataclass(frozen=True, slots=True)
class BrokerOrder:
    client_order_id: str
    venue_order_id: str
    symbol: str
    side: Side
    quantity: int
    filled_quantity: int
    status: str


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
        self._broker_orders: dict[str, BrokerOrder] = {}
        self._broker_revision_ns = 0
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

    def replace_from_broker(
        self,
        positions: tuple[BrokerPosition, ...],
        orders: tuple[BrokerOrder, ...],
        observed_ns: int,
    ) -> None:
        """Replace broker-owned totals; strategy ownership is never fabricated."""
        if observed_ns <= 0:
            raise ValueError("broker observation timestamp must be positive")
        with self._lock:
            if observed_ns < self._broker_revision_ns:
                raise ValueError("stale broker portfolio snapshot")
            internal_symbols = set(self._positions)
            broker_symbols = {position.symbol for position in positions}
            for position in positions:
                existing = self._positions.get(position.symbol)
                ownership = {} if existing is None else dict(existing.strategy_quantities)
                if ownership and sum(ownership.values()) != position.quantity:
                    raise RuntimeError(
                        f"strategy ownership diverged from broker for {position.symbol}"
                    )
                self._positions[position.symbol] = Position(
                    symbol=position.symbol,
                    quantity=position.quantity,
                    cost_ticks=position.quantity * position.average_entry_price_ticks,
                    strategy_quantities=ownership,
                )
            for symbol in internal_symbols - broker_symbols:
                if self._positions[symbol].quantity != 0:
                    raise RuntimeError(f"broker is missing internal position {symbol}")
                self._positions.pop(symbol)
            self._broker_orders = {order.client_order_id: order for order in orders}
            self._broker_revision_ns = observed_ns

    def assert_consistent(self) -> None:
        with self._lock:
            for client_id, managed in self._open_orders.items():
                broker = self._broker_orders.get(client_id)
                if broker is None:
                    raise RuntimeError(f"internal open order absent at broker: {client_id}")
                if managed.venue_order_id and managed.venue_order_id != broker.venue_order_id:
                    raise RuntimeError(f"venue order mismatch for {client_id}")
            unknown = set(self._broker_orders) - set(self._open_orders)
            if unknown:
                raise RuntimeError(f"unowned broker open orders: {sorted(unknown)}")

    def broker_orders(self) -> tuple[BrokerOrder, ...]:
        with self._lock:
            return tuple(self._broker_orders.values())

    def fills(self) -> tuple[Fill, ...]:
        with self._lock:
            return tuple(self._fills)

    def open_orders(self) -> tuple[ManagedOrder, ...]:
        with self._lock:
            return tuple(replace(x) for x in self._open_orders.values())

    def positions(self) -> tuple[Position, ...]:
        with self._lock:
            return tuple(
                Position(
                    item.symbol,
                    item.quantity,
                    item.cost_ticks,
                    dict(item.strategy_quantities),
                )
                for item in self._positions.values()
            )
