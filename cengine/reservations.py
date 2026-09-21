"""Atomic risk reservations spanning pending orders and authoritative positions."""

from __future__ import annotations

from dataclasses import dataclass
from threading import RLock

from order_manager import Side


@dataclass(frozen=True, slots=True)
class RiskReservation:
    client_order_id: str
    symbol: str
    side: Side
    quantity: int
    price_ticks: int

    @property
    def signed_quantity(self) -> int:
        return self.quantity if self.side is Side.BUY else -self.quantity


class RiskReservationBook:
    def __init__(self) -> None:
        self._items: dict[str, RiskReservation] = {}
        self._lock = RLock()

    def reserve(self, reservation: RiskReservation) -> None:
        if reservation.quantity <= 0 or reservation.price_ticks <= 0:
            raise ValueError("reservation quantity and price must be positive")
        with self._lock:
            if reservation.client_order_id in self._items:
                raise ValueError("duplicate risk reservation")
            self._items[reservation.client_order_id] = reservation

    def release(self, client_order_id: str) -> None:
        with self._lock:
            self._items.pop(client_order_id, None)

    def projected_position(self, symbol: str, authoritative_position: int) -> int:
        with self._lock:
            return authoritative_position + sum(
                item.signed_quantity for item in self._items.values() if item.symbol == symbol
            )

    def gross_notional_ticks(self) -> int:
        with self._lock:
            return sum(item.quantity * item.price_ticks for item in self._items.values())

    def snapshot(self) -> tuple[RiskReservation, ...]:
        with self._lock:
            return tuple(self._items.values())
