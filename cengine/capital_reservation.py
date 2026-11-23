"""Atomic capital reservation across concurrent recommendations (Milestone 3, §16).

Prevents multiple portfolio recommendations from spending the same buying power.
Wraps the existing :class:`cengine.reservations.RiskReservationBook` (the native
risk path's reservation ledger) with pre-order capital accounting and an explicit
per-reservation lifecycle. Unknown broker outcomes do NOT release capital
prematurely; capital is released only on confirmed terminal, non-unknown states.

Idempotent by ``decision_id``: re-reserving the same key is a no-op (safe across
restarts / duplicate recommendations).
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from enum import Enum

from order_manager import Side

RESERVATION_VERSION = "capital-reservation-1.0.0"


class ReservationStatus(str, Enum):
    RESERVED = "reserved"          # capital reserved, order not yet submitted
    SUBMITTED = "submitted"        # order submitted to the venue
    PARTIAL = "partial"            # partially filled (capital still held for remainder)
    FILLED = "filled"              # fully filled (terminal; capital consumed/released)
    REJECTED = "rejected"          # terminal; release
    CANCELED = "canceled"          # terminal; release
    UNKNOWN = "unknown"            # outcome unknown; DO NOT release


_TERMINAL_RELEASE = {ReservationStatus.FILLED, ReservationStatus.REJECTED,
                     ReservationStatus.CANCELED}


@dataclass
class CapitalReservation:
    decision_id: str
    symbol: str
    side: Side
    quantity: int
    price_ticks: int
    status: ReservationStatus = ReservationStatus.RESERVED

    @property
    def notional_ticks(self) -> int:
        return self.quantity * self.price_ticks

    @property
    def is_buy(self) -> bool:
        return self.side is Side.BUY


class CapitalReservationLedger:
    """Thread-safe pre-order capital reservation across concurrent recommendations."""

    def __init__(self, *, risk_reservations=None) -> None:
        # Optional bridge to the native-path RiskReservationBook so downstream risk
        # sees the same reserved capital. Kept optional for isolated testing.
        self._risk = risk_reservations
        self._reservations: dict[str, CapitalReservation] = {}
        self._lock = threading.RLock()

    def reserved_buy_ticks(self) -> int:
        with self._lock:
            return sum(r.notional_ticks for r in self._reservations.values()
                       if r.is_buy and r.status not in _TERMINAL_RELEASE)

    def available_ticks(self, base_available_ticks: int) -> int:
        """Capital still available after outstanding BUY reservations."""
        return max(0, int(base_available_ticks) - self.reserved_buy_ticks())

    def try_reserve(self, *, decision_id: str, symbol: str, side: Side,
                    quantity: int, price_ticks: int,
                    base_available_ticks: int) -> tuple[bool, str]:
        """Atomically reserve capital for a BUY if it fits remaining availability.

        SELL reservations never consume buying power (they reduce exposure) and are
        always accepted for tracking. Idempotent by ``decision_id``.
        """
        if quantity <= 0 or price_ticks <= 0:
            return False, "invalid_quantity_or_price"
        with self._lock:
            existing = self._reservations.get(decision_id)
            if existing is not None:
                return True, "already_reserved"  # idempotent
            notional = quantity * price_ticks
            if side is Side.BUY:
                remaining = self.available_ticks(base_available_ticks)
                if notional > remaining:
                    return False, f"insufficient_capital:{notional}>{remaining}"
            reservation = CapitalReservation(decision_id, symbol, side, quantity, price_ticks)
            self._reservations[decision_id] = reservation
            if self._risk is not None:
                try:
                    from cengine.reservations import RiskReservation
                    self._risk.reserve(RiskReservation(
                        decision_id, symbol, side, quantity, price_ticks))
                except Exception:  # noqa: BLE001 — duplicate/again is non-fatal
                    pass
            return True, "reserved"

    def _set_status(self, decision_id: str, status: ReservationStatus) -> None:
        with self._lock:
            r = self._reservations.get(decision_id)
            if r is None:
                return
            r.status = status
            if status in _TERMINAL_RELEASE:
                self._release_locked(decision_id)

    def mark_submitted(self, decision_id: str) -> None:
        self._set_status(decision_id, ReservationStatus.SUBMITTED)

    def mark_partial(self, decision_id: str) -> None:
        self._set_status(decision_id, ReservationStatus.PARTIAL)

    def mark_filled(self, decision_id: str) -> None:
        self._set_status(decision_id, ReservationStatus.FILLED)

    def mark_rejected(self, decision_id: str) -> None:
        self._set_status(decision_id, ReservationStatus.REJECTED)

    def mark_canceled(self, decision_id: str) -> None:
        self._set_status(decision_id, ReservationStatus.CANCELED)

    def mark_unknown(self, decision_id: str) -> None:
        # Unknown outcome: keep the reservation (do NOT release capital).
        with self._lock:
            r = self._reservations.get(decision_id)
            if r is not None:
                r.status = ReservationStatus.UNKNOWN

    def _release_locked(self, decision_id: str) -> None:
        self._reservations.pop(decision_id, None)
        if self._risk is not None:
            try:
                self._risk.release(decision_id)
            except Exception:  # noqa: BLE001
                pass

    def snapshot(self) -> tuple:
        with self._lock:
            return tuple(CapitalReservation(r.decision_id, r.symbol, r.side, r.quantity,
                                          r.price_ticks, r.status)
                         for r in self._reservations.values())

    def outstanding_count(self) -> int:
        with self._lock:
            return sum(1 for r in self._reservations.values()
                       if r.status not in _TERMINAL_RELEASE)
