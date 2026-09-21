"""Strategy inventory state derived solely from authoritative execution outcomes."""

from __future__ import annotations

from dataclasses import dataclass

from order_manager import TERMINAL_STATUSES, ExecutionUpdate, ManagedOrder, OrderStatus, Side


@dataclass(slots=True)
class StrategyInventory:
    quantity: int = 0
    rejected_orders: int = 0


class StrategyStateBook:
    def __init__(self) -> None:
        self._state: dict[tuple[str, str], StrategyInventory] = {}
        self._pending: dict[tuple[str, str], str] = {}

    def on_order(self, order: ManagedOrder) -> None:
        key = (order.intent.strategy_id, order.intent.symbol)
        if order.status in TERMINAL_STATUSES or order.status is OrderStatus.ERROR:
            if self._pending.get(key) == order.client_order_id:
                self._pending.pop(key, None)
        else:
            existing = self._pending.get(key)
            if existing not in {None, order.client_order_id}:
                raise RuntimeError(f"multiple pending orders for strategy/symbol {key}")
            self._pending[key] = order.client_order_id

    def can_submit(self, strategy_id: str, symbol: str) -> bool:
        return (strategy_id, symbol) not in self._pending

    def reconcile(self, order: ManagedOrder, update: ExecutionUpdate) -> None:
        key = (order.intent.strategy_id, order.intent.symbol)
        state = self._state.setdefault(key, StrategyInventory())
        if update.last_fill_quantity:
            direction = 1 if order.intent.side is Side.BUY else -1
            state.quantity += direction * update.last_fill_quantity
        if update.status in {OrderStatus.REJECTED, OrderStatus.RISK_REJECTED}:
            state.rejected_orders += 1

    def quantity(self, strategy_id: str, symbol: str) -> int:
        return self._state.get((strategy_id, symbol), StrategyInventory()).quantity
