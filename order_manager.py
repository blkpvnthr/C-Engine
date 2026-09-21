#!/usr/bin/env python3
"""
order_manager.py

Python orchestration layer for order lifecycle management.

Authority boundary
-------------------
Strategy -> OrderManager -> native/C++ RiskEngine -> ExecutionVenue -> Broker/Simulator
                       |
                       +-> reconciliation / audit state

The OrderManager:
- does NOT approve its own risk;
- does NOT bypass the native RiskEngine;
- does NOT infer broker fills;
- does NOT fabricate market prices or timestamps;
- does NOT contain Alpaca credentials;
- does NOT treat submission acknowledgement as a fill;
- does NOT let Metal kernels submit/cancel/amend orders.

All market-dependent values, timestamps, quantities, prices, and policy
decisions must come from explicit callers, the native risk boundary, or the
execution venue.
"""

from __future__ import annotations

import asyncio
import inspect
from dataclasses import dataclass, replace
from enum import Enum
from typing import Awaitable, Callable, Optional, Protocol, TypeVar
from uuid import uuid4


class OrderManagerError(RuntimeError):
    pass


class DuplicateClientOrderId(OrderManagerError):
    pass


class UnknownOrder(OrderManagerError):
    pass


class InvalidOrderTransition(OrderManagerError):
    pass


class RiskRejected(OrderManagerError):
    pass


class ExecutionRejected(OrderManagerError):
    pass


class ReconciliationError(OrderManagerError):
    pass


class Side(str, Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class TimeInForce(str, Enum):
    DAY = "day"
    GTC = "gtc"
    IOC = "ioc"
    FOK = "fok"


class OrderStatus(str, Enum):
    CREATED = "created"
    RISK_PENDING = "risk_pending"
    RISK_REJECTED = "risk_rejected"
    APPROVED = "approved"
    SUBMITTING = "submitting"
    SUBMITTED = "submitted"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCEL_PENDING = "cancel_pending"
    CANCELED = "canceled"
    REJECTED = "rejected"
    EXPIRED = "expired"
    ERROR = "error"


TERMINAL_STATUSES = frozenset(
    {
        OrderStatus.RISK_REJECTED,
        OrderStatus.FILLED,
        OrderStatus.CANCELED,
        OrderStatus.REJECTED,
        OrderStatus.EXPIRED,
    }
)

_ALLOWED_TRANSITIONS: dict[OrderStatus, frozenset[OrderStatus]] = {
    OrderStatus.CREATED: frozenset(
        {
            OrderStatus.RISK_PENDING,
            OrderStatus.ERROR,
        }
    ),
    OrderStatus.RISK_PENDING: frozenset(
        {
            OrderStatus.RISK_REJECTED,
            OrderStatus.APPROVED,
            OrderStatus.ERROR,
        }
    ),
    OrderStatus.RISK_REJECTED: frozenset(),
    OrderStatus.APPROVED: frozenset(
        {
            OrderStatus.SUBMITTING,
            OrderStatus.ERROR,
        }
    ),
    OrderStatus.SUBMITTING: frozenset(
        {
            OrderStatus.SUBMITTED,
            OrderStatus.PARTIALLY_FILLED,
            OrderStatus.FILLED,
            OrderStatus.REJECTED,
            OrderStatus.ERROR,
        }
    ),
    OrderStatus.SUBMITTED: frozenset(
        {
            OrderStatus.PARTIALLY_FILLED,
            OrderStatus.FILLED,
            OrderStatus.CANCEL_PENDING,
            OrderStatus.CANCELED,
            OrderStatus.REJECTED,
            OrderStatus.EXPIRED,
            OrderStatus.ERROR,
        }
    ),
    OrderStatus.PARTIALLY_FILLED: frozenset(
        {
            OrderStatus.PARTIALLY_FILLED,
            OrderStatus.FILLED,
            OrderStatus.CANCEL_PENDING,
            OrderStatus.CANCELED,
            OrderStatus.REJECTED,
            OrderStatus.EXPIRED,
            OrderStatus.ERROR,
        }
    ),
    OrderStatus.FILLED: frozenset(),
    OrderStatus.CANCEL_PENDING: frozenset(
        {
            OrderStatus.PARTIALLY_FILLED,
            OrderStatus.FILLED,
            OrderStatus.CANCELED,
            OrderStatus.REJECTED,
            OrderStatus.ERROR,
        }
    ),
    OrderStatus.CANCELED: frozenset(),
    OrderStatus.REJECTED: frozenset(),
    OrderStatus.EXPIRED: frozenset(),
    OrderStatus.ERROR: frozenset(),
}


@dataclass(frozen=True, slots=True)
class OrderIntent:
    symbol: str
    side: Side
    quantity: int
    order_type: OrderType
    time_in_force: TimeInForce

    limit_price_ticks: Optional[int] = None
    stop_price_ticks: Optional[int] = None

    strategy_id: str = ""
    correlation_id: str = ""
    client_order_id: str = ""

    # Caller-provided observation time used by risk policy for freshness.
    created_ns: int = 0

    def validate(self) -> None:
        symbol = self.symbol.strip().upper()
        if not symbol:
            raise ValueError("symbol is required")
        if self.quantity <= 0:
            raise ValueError("quantity must be positive")
        if self.created_ns <= 0:
            raise ValueError("created_ns must be explicitly supplied")

        if self.order_type in {OrderType.LIMIT, OrderType.STOP_LIMIT}:
            if self.limit_price_ticks is None or self.limit_price_ticks <= 0:
                raise ValueError("positive limit_price_ticks required for limit orders")
        elif self.limit_price_ticks is not None:
            raise ValueError("limit_price_ticks is only valid for LIMIT/STOP_LIMIT")

        if self.order_type in {OrderType.STOP, OrderType.STOP_LIMIT}:
            if self.stop_price_ticks is None or self.stop_price_ticks <= 0:
                raise ValueError("positive stop_price_ticks required for stop orders")
        elif self.stop_price_ticks is not None:
            raise ValueError("stop_price_ticks is only valid for STOP/STOP_LIMIT")


@dataclass(frozen=True, slots=True)
class RiskDecision:
    approved: bool
    decision_id: str
    policy_version: str
    decided_ns: int
    reason: str = ""

    # Risk engine may reduce quantity but never increase the requested amount.
    approved_quantity: Optional[int] = None

    def validate_for(self, intent: OrderIntent) -> None:
        if not self.decision_id:
            raise ValueError("risk decision_id is required")
        if not self.policy_version:
            raise ValueError("risk policy_version is required")
        if self.decided_ns <= 0:
            raise ValueError("risk decided_ns must be positive")

        if self.approved:
            qty = intent.quantity if self.approved_quantity is None else self.approved_quantity
            if qty <= 0:
                raise ValueError("approved quantity must be positive")
            if qty > intent.quantity:
                raise ValueError("risk engine cannot increase requested quantity")


@dataclass(frozen=True, slots=True)
class ExecutionAck:
    client_order_id: str
    venue_order_id: str
    accepted_ns: int

    def validate(self) -> None:
        if not self.client_order_id:
            raise ValueError("client_order_id is required")
        if not self.venue_order_id:
            raise ValueError("venue_order_id is required")
        if self.accepted_ns <= 0:
            raise ValueError("accepted_ns must be positive")


@dataclass(frozen=True, slots=True)
class ExecutionUpdate:
    client_order_id: str
    venue_order_id: str
    status: OrderStatus
    event_ns: int

    cumulative_filled_quantity: int = 0
    last_fill_quantity: int = 0
    last_fill_price_ticks: Optional[int] = None
    reason: str = ""
    venue_sequence: Optional[int] = None

    def validate(self) -> None:
        if not self.client_order_id:
            raise ValueError("client_order_id is required")
        if not self.venue_order_id:
            raise ValueError("venue_order_id is required")
        if self.event_ns <= 0:
            raise ValueError("event_ns must be positive")
        if self.cumulative_filled_quantity < 0:
            raise ValueError("cumulative fill cannot be negative")
        if self.last_fill_quantity < 0:
            raise ValueError("last fill cannot be negative")
        if self.last_fill_quantity > 0:
            if self.last_fill_price_ticks is None:
                raise ValueError("fill price required when last_fill_quantity > 0")
            if self.last_fill_price_ticks <= 0:
                raise ValueError("fill price must be positive")
        if self.venue_sequence is not None and self.venue_sequence < 0:
            raise ValueError("venue_sequence cannot be negative")


@dataclass(slots=True)
class ManagedOrder:
    intent: OrderIntent
    client_order_id: str
    status: OrderStatus = OrderStatus.CREATED

    approved_quantity: int = 0
    cumulative_filled_quantity: int = 0
    average_fill_price_ticks: Optional[float] = None

    venue_order_id: Optional[str] = None
    risk_decision_id: Optional[str] = None
    risk_policy_version: Optional[str] = None

    last_event_ns: int = 0
    last_venue_sequence: Optional[int] = None
    rejection_reason: str = ""

    def remaining_quantity(self) -> int:
        target = self.approved_quantity or self.intent.quantity
        return max(0, target - self.cumulative_filled_quantity)


class NativeRiskEngine(Protocol):
    """Adapter implemented by the C++/pybind11 risk boundary."""

    def evaluate(
        self,
        intent: OrderIntent,
    ) -> RiskDecision | Awaitable[RiskDecision]: ...


class ExecutionVenue(Protocol):
    """Broker or CPU simulator adapter. No risk authority."""

    def submit(
        self,
        intent: OrderIntent,
    ) -> ExecutionAck | Awaitable[ExecutionAck]: ...

    def cancel(
        self,
        *,
        client_order_id: str,
        venue_order_id: str,
    ) -> None | Awaitable[None]: ...


AuditHandler = Callable[
    [str, ManagedOrder],
    None | Awaitable[None],
]

T = TypeVar("T")


async def _maybe_await(value: T | Awaitable[T]) -> T:
    if inspect.isawaitable(value):
        return await value
    return value


def _normalize_symbol(symbol: str) -> str:
    return symbol.strip().upper()


def _new_client_order_id() -> str:
    # UUID is an identifier, not market data or policy state. The caller may
    # provide its own deterministic/idempotency key instead.
    return uuid4().hex


class OrderManager:
    """Coordinates risk-approved order submission and broker reconciliation."""

    def __init__(
        self,
        *,
        risk_engine: NativeRiskEngine,
        venue: ExecutionVenue,
        on_audit: Optional[AuditHandler] = None,
    ) -> None:
        self._risk_engine = risk_engine
        self._venue = venue
        self._on_audit = on_audit

        self._orders: dict[str, ManagedOrder] = {}
        self._venue_to_client: dict[str, str] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._registry_lock = asyncio.Lock()

    async def submit(self, intent: OrderIntent) -> ManagedOrder:
        intent.validate()

        normalized = replace(
            intent,
            symbol=_normalize_symbol(intent.symbol),
            client_order_id=(
                intent.client_order_id.strip()
                if intent.client_order_id.strip()
                else _new_client_order_id()
            ),
        )

        async with self._registry_lock:
            if normalized.client_order_id in self._orders:
                raise DuplicateClientOrderId(normalized.client_order_id)

            order = ManagedOrder(
                intent=normalized,
                client_order_id=normalized.client_order_id,
                last_event_ns=normalized.created_ns,
            )
            self._orders[order.client_order_id] = order
            self._locks[order.client_order_id] = asyncio.Lock()

        await self._audit("created", order)

        lock = self._locks[order.client_order_id]
        async with lock:
            self._transition(order, OrderStatus.RISK_PENDING)
            await self._audit("risk_pending", order)

            try:
                decision = await _maybe_await(self._risk_engine.evaluate(normalized))
                decision.validate_for(normalized)
            except Exception:
                self._transition(order, OrderStatus.ERROR)
                await self._audit("risk_error", order)
                raise

            order.risk_decision_id = decision.decision_id
            order.risk_policy_version = decision.policy_version
            order.last_event_ns = max(order.last_event_ns, decision.decided_ns)

            if not decision.approved:
                order.rejection_reason = decision.reason
                self._transition(order, OrderStatus.RISK_REJECTED)
                await self._audit("risk_rejected", order)
                raise RiskRejected(decision.reason or "native risk engine rejected order")

            approved_quantity = (
                normalized.quantity
                if decision.approved_quantity is None
                else decision.approved_quantity
            )
            order.approved_quantity = approved_quantity

            approved_intent = replace(
                normalized,
                quantity=approved_quantity,
            )
            order.intent = approved_intent

            self._transition(order, OrderStatus.APPROVED)
            await self._audit("approved", order)

            self._transition(order, OrderStatus.SUBMITTING)
            await self._audit("submitting", order)

            try:
                ack = await _maybe_await(self._venue.submit(approved_intent))
                ack.validate()
            except Exception:
                self._transition(order, OrderStatus.ERROR)
                await self._audit("submission_error", order)
                raise

            if ack.client_order_id != order.client_order_id:
                self._transition(order, OrderStatus.ERROR)
                await self._audit("client_id_mismatch", order)
                raise ReconciliationError("execution venue returned a different client_order_id")

            if (
                ack.venue_order_id in self._venue_to_client
                and self._venue_to_client[ack.venue_order_id] != order.client_order_id
            ):
                self._transition(order, OrderStatus.ERROR)
                await self._audit("venue_id_collision", order)
                raise ReconciliationError(f"venue_order_id collision: {ack.venue_order_id}")

            order.venue_order_id = ack.venue_order_id
            order.last_event_ns = max(order.last_event_ns, ack.accepted_ns)
            self._venue_to_client[ack.venue_order_id] = order.client_order_id

            # An acknowledgement means accepted/submitted, never filled.
            self._transition(order, OrderStatus.SUBMITTED)
            await self._audit("submitted", order)

            return self.snapshot(order.client_order_id)

    async def cancel(self, client_order_id: str) -> ManagedOrder:
        order = self._require_order(client_order_id)
        lock = self._locks[client_order_id]

        async with lock:
            if order.status in TERMINAL_STATUSES:
                return self.snapshot(client_order_id)

            if not order.venue_order_id:
                raise InvalidOrderTransition("cannot cancel before venue acknowledgement")

            self._transition(order, OrderStatus.CANCEL_PENDING)
            await self._audit("cancel_pending", order)

            try:
                await _maybe_await(
                    self._venue.cancel(
                        client_order_id=order.client_order_id,
                        venue_order_id=order.venue_order_id,
                    )
                )
            except Exception:
                self._transition(order, OrderStatus.ERROR)
                await self._audit("cancel_error", order)
                raise

            # Remain CANCEL_PENDING until the venue explicitly confirms
            # CANCELED or FILLED through reconcile().
            return self.snapshot(client_order_id)

    async def reconcile(self, update: ExecutionUpdate) -> ManagedOrder:
        update.validate()

        order = self._require_order(update.client_order_id)
        lock = self._locks[update.client_order_id]

        async with lock:
            if order.venue_order_id is None:
                raise ReconciliationError("received execution update before venue acknowledgement")

            if update.venue_order_id != order.venue_order_id:
                raise ReconciliationError("execution update venue_order_id does not match order")

            if (
                update.venue_sequence is not None
                and order.last_venue_sequence is not None
                and update.venue_sequence <= order.last_venue_sequence
            ):
                # Idempotent/stale venue event: preserve authoritative current
                # state rather than applying it twice.
                return self.snapshot(order.client_order_id)

            if update.venue_sequence is None and update.event_ns < order.last_event_ns:
                raise ReconciliationError("out-of-order execution update without venue sequence")

            target_quantity = order.approved_quantity or order.intent.quantity

            if update.cumulative_filled_quantity > target_quantity:
                raise ReconciliationError("venue cumulative fill exceeds approved quantity")

            if update.cumulative_filled_quantity < order.cumulative_filled_quantity:
                raise ReconciliationError("venue cumulative fill moved backwards")

            fill_delta = update.cumulative_filled_quantity - order.cumulative_filled_quantity

            if update.last_fill_quantity > fill_delta:
                raise ReconciliationError("last_fill_quantity exceeds cumulative fill delta")

            if fill_delta > 0:
                if update.last_fill_price_ticks is None:
                    raise ReconciliationError("new fill requires last_fill_price_ticks")

                previous_qty = order.cumulative_filled_quantity
                previous_avg = order.average_fill_price_ticks or 0.0

                # If the venue reports a cumulative increase larger than the
                # explicit last-fill quantity, the missing fill prices cannot
                # be inferred. Reject the update instead of inventing them.
                if update.last_fill_quantity != fill_delta:
                    raise ReconciliationError(
                        "cannot calculate average fill: cumulative fill delta "
                        "does not equal explicitly priced last fill quantity"
                    )

                total_notional_ticks = (
                    previous_avg * previous_qty + update.last_fill_price_ticks * fill_delta
                )
                order.average_fill_price_ticks = (
                    total_notional_ticks / update.cumulative_filled_quantity
                )

            order.cumulative_filled_quantity = update.cumulative_filled_quantity
            order.last_event_ns = max(order.last_event_ns, update.event_ns)
            order.last_venue_sequence = update.venue_sequence

            expected_status = update.status

            if (
                order.cumulative_filled_quantity == target_quantity
                and expected_status != OrderStatus.FILLED
            ):
                raise ReconciliationError("fully filled quantity requires FILLED venue status")

            if (
                expected_status == OrderStatus.FILLED
                and order.cumulative_filled_quantity != target_quantity
            ):
                raise ReconciliationError("FILLED status requires approved quantity to be filled")

            if expected_status == OrderStatus.PARTIALLY_FILLED and not (
                0 < order.cumulative_filled_quantity < target_quantity
            ):
                raise ReconciliationError("PARTIALLY_FILLED requires a nonzero partial quantity")

            self._transition(order, expected_status)

            if expected_status == OrderStatus.REJECTED:
                order.rejection_reason = update.reason

            await self._audit("reconciled", order)
            return self.snapshot(order.client_order_id)

    def snapshot(self, client_order_id: str) -> ManagedOrder:
        order = self._require_order(client_order_id)
        return replace(order)

    def all_orders(self) -> tuple[ManagedOrder, ...]:
        return tuple(replace(order) for order in self._orders.values())

    def open_orders(self) -> tuple[ManagedOrder, ...]:
        return tuple(
            replace(order)
            for order in self._orders.values()
            if order.status not in TERMINAL_STATUSES
        )

    def by_venue_order_id(self, venue_order_id: str) -> ManagedOrder:
        try:
            client_order_id = self._venue_to_client[venue_order_id]
        except KeyError as exc:
            raise UnknownOrder(venue_order_id) from exc
        return self.snapshot(client_order_id)

    def _require_order(self, client_order_id: str) -> ManagedOrder:
        try:
            return self._orders[client_order_id]
        except KeyError as exc:
            raise UnknownOrder(client_order_id) from exc

    @staticmethod
    def _transition(
        order: ManagedOrder,
        new_status: OrderStatus,
    ) -> None:
        if new_status == order.status:
            if new_status == OrderStatus.PARTIALLY_FILLED:
                return
            raise InvalidOrderTransition(f"duplicate transition {order.status.value}")

        allowed = _ALLOWED_TRANSITIONS[order.status]
        if new_status not in allowed:
            raise InvalidOrderTransition(
                f"{order.status.value} -> {new_status.value} is not allowed"
            )

        order.status = new_status

    async def _audit(
        self,
        event: str,
        order: ManagedOrder,
    ) -> None:
        if self._on_audit is None:
            return

        # Pass a snapshot so audit consumers cannot mutate manager state.
        await _maybe_await(self._on_audit(event, replace(order)))


__all__ = [
    "DuplicateClientOrderId",
    "ExecutionAck",
    "ExecutionRejected",
    "ExecutionUpdate",
    "ExecutionVenue",
    "InvalidOrderTransition",
    "ManagedOrder",
    "NativeRiskEngine",
    "OrderIntent",
    "OrderManager",
    "OrderManagerError",
    "OrderStatus",
    "OrderType",
    "ReconciliationError",
    "RiskDecision",
    "RiskRejected",
    "Side",
    "TimeInForce",
    "UnknownOrder",
]
