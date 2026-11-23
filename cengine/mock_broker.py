"""Isolated, deterministic mock broker for mock-execution certification (§9).

``SimulatedBroker`` implements the :class:`order_manager.ExecutionVenue` protocol
with NO network and NO wall-clock dependency, so the full production decision path
(regime → families → arbitration → ranking → bridge → CandidateRouter → exposure
gate → native RiskEngine → OrderManager → broker → fill reconciliation) can be
exercised end-to-end under isolated test authorization.

It is a *venue*, not a risk authority and not a position book. It only:
  * acknowledges submissions with deterministic venue order ids, and
  * produces a configurable, deterministic fill schedule (full, partial-then-full,
    rejection, or no fill) as :class:`order_manager.ExecutionUpdate` objects that
    the caller feeds back through ``OrderManager.reconcile``.

Determinism: ids and event timestamps come from monotonic counters seeded from the
submitted intent, never from ``time``/``random``.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from order_manager import ExecutionAck, ExecutionUpdate, OrderIntent, OrderStatus


class FillPlan(str, Enum):
    FULL = "full"                       # single complete fill
    PARTIAL_THEN_FULL = "partial_then_full"   # one partial, then the remainder
    REJECT = "reject"                   # venue rejects after ack
    NOFILL = "nofill"                   # acknowledged, never fills (resting)


class SimulatedBroker:
    """Deterministic in-memory execution venue.

    Parameters
    ----------
    default_plan:
        Fill plan applied to symbols without an explicit override.
    prices_ticks:
        Optional per-symbol fill price (ticks). Market orders have no limit price,
        so a fill price must come from here (or ``default_fill_price_ticks``).
    plans:
        Optional per-symbol :class:`FillPlan` overrides.
    default_fill_price_ticks:
        Fallback fill price for symbols absent from ``prices_ticks``.
    """

    def __init__(self, *, default_plan: FillPlan = FillPlan.FULL,
                 prices_ticks: Optional[dict] = None,
                 plans: Optional[dict] = None,
                 default_fill_price_ticks: int = 1_000_000) -> None:
        self._default_plan = default_plan
        self._prices = dict(prices_ticks or {})
        self._plans = dict(plans or {})
        self._default_price = int(default_fill_price_ticks)
        self._seq = 0
        self._clock = 0
        # venue_order_id -> list[ExecutionUpdate] (FIFO schedule)
        self._schedule: dict[str, list] = {}
        self._acks: dict[str, ExecutionAck] = {}
        self.submitted = 0
        self.canceled = 0

    # -- configuration -------------------------------------------------------
    def set_plan(self, symbol: str, plan: FillPlan) -> None:
        self._plans[symbol.upper()] = plan

    def set_price(self, symbol: str, price_ticks: int) -> None:
        self._prices[symbol.upper()] = int(price_ticks)

    # -- ExecutionVenue protocol --------------------------------------------
    def submit(self, intent: OrderIntent) -> ExecutionAck:
        intent.validate()
        self._seq += 1
        venue_order_id = f"MOCK-{self._seq:08d}"
        self._clock = max(self._clock, intent.created_ns) + 1
        ack = ExecutionAck(client_order_id=intent.client_order_id,
                           venue_order_id=venue_order_id, accepted_ns=self._clock)
        self._acks[venue_order_id] = ack
        self._schedule[venue_order_id] = self._build_schedule(intent, venue_order_id)
        self.submitted += 1
        return ack

    def cancel(self, *, client_order_id: str, venue_order_id: str) -> None:
        # A canceled resting order simply stops producing fills in this simulator.
        self._schedule.pop(venue_order_id, None)
        self.canceled += 1

    # -- fill driving --------------------------------------------------------
    def next_update(self, venue_order_id: str) -> Optional[ExecutionUpdate]:
        """Pop and return the next scheduled update for an order, or None."""
        queue = self._schedule.get(venue_order_id)
        if not queue:
            return None
        return queue.pop(0)

    def drain_updates(self, venue_order_id: str) -> list:
        """Pop and return ALL remaining scheduled updates for an order, in order."""
        queue = self._schedule.get(venue_order_id)
        if not queue:
            return []
        self._schedule[venue_order_id] = []
        return queue

    def pending_order_ids(self) -> list:
        return [vid for vid, q in self._schedule.items() if q]

    # -- internals -----------------------------------------------------------
    def _price_for(self, symbol: str, intent: OrderIntent) -> int:
        if intent.limit_price_ticks:
            return int(intent.limit_price_ticks)
        return int(self._prices.get(symbol.upper(), self._default_price))

    def _tick(self) -> int:
        self._clock += 1
        return self._clock

    def _build_schedule(self, intent: OrderIntent, venue_order_id: str) -> list:
        plan = self._plans.get(intent.symbol.upper(), self._default_plan)
        qty = intent.quantity
        price = self._price_for(intent.symbol, intent)
        cid = intent.client_order_id
        updates: list = []
        seq = 0

        def upd(status, cum, last, px, reason=""):
            nonlocal seq
            seq += 1
            return ExecutionUpdate(
                client_order_id=cid, venue_order_id=venue_order_id, status=status,
                event_ns=self._tick(), cumulative_filled_quantity=cum,
                last_fill_quantity=last, last_fill_price_ticks=px, reason=reason,
                venue_sequence=seq)

        if plan is FillPlan.REJECT:
            updates.append(upd(OrderStatus.REJECTED, 0, 0, None, "mock_venue_rejected"))
        elif plan is FillPlan.NOFILL:
            pass  # acknowledged only; remains resting
        elif plan is FillPlan.PARTIAL_THEN_FULL and qty >= 2:
            half = qty // 2
            updates.append(upd(OrderStatus.PARTIALLY_FILLED, half, half, price))
            updates.append(upd(OrderStatus.FILLED, qty, qty - half, price))
        else:  # FULL (or PARTIAL requested on a 1-share order -> single fill)
            updates.append(upd(OrderStatus.FILLED, qty, qty, price))
        return updates


__all__ = ["FillPlan", "SimulatedBroker"]
