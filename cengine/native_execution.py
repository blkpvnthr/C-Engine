"""Translate native order-book executions into authoritative Python updates."""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from typing import Protocol

from order_manager import ExecutionUpdate, OrderStatus


class NativeExecutionLike(Protocol):
    maker_id: int
    taker_id: int
    quantity: int
    price_ticks: int


class NativeExecutionBridge:
    def __init__(self, reconcile: Callable[[ExecutionUpdate], Awaitable[None]]) -> None:
        self._reconcile = reconcile
        self._orders: dict[int, tuple[str, str, int, int]] = {}
        self._sequence = 0

    def register(
        self,
        native_order_id: int,
        client_order_id: str,
        venue_order_id: str,
        quantity: int,
    ) -> None:
        if native_order_id in self._orders:
            raise ValueError("duplicate native order id")
        self._orders[native_order_id] = (client_order_id, venue_order_id, quantity, 0)

    async def on_execution(self, execution: NativeExecutionLike) -> None:
        self._sequence += 1
        for order_id in (int(execution.maker_id), int(execution.taker_id)):
            tracked = self._orders.get(order_id)
            if tracked is None:
                continue
            client_id, venue_id, total, prior = tracked
            cumulative = prior + int(execution.quantity)
            if cumulative > total:
                raise RuntimeError("native execution exceeds registered order quantity")
            self._orders[order_id] = (client_id, venue_id, total, cumulative)
            await self._reconcile(
                ExecutionUpdate(
                    client_order_id=client_id,
                    venue_order_id=venue_id,
                    status=(
                        OrderStatus.FILLED if cumulative == total else OrderStatus.PARTIALLY_FILLED
                    ),
                    event_ns=time.time_ns(),
                    cumulative_filled_quantity=cumulative,
                    last_fill_quantity=int(execution.quantity),
                    last_fill_price_ticks=int(execution.price_ticks),
                    venue_sequence=self._sequence,
                )
            )
