"""Strategy → execution bridge (directive §3, §4, §14-17).

Closes the gap between the ADVISORY five-family regime pipeline and the EXISTING
execution authority. It converts maturity-authorized, condition-passing
:class:`StrategyRecommendation` objects into :class:`strategies.TradeCandidate`
objects and routes them through the EXISTING ``CandidateRouter`` — which enforces
the single-pending guard, the exposure-authorization gate, buy/sell conflict
withholding, position sizing, the native C++ RiskEngine, and the OrderManager.

It creates NO parallel engine and bypasses NO financial control. Two gates must
both open before anything routes:

1. ``recommendation.conditions_passed`` — the strategy/entry/regime conditions
   themselves passed (independent of advisory eligibility), AND
2. ``authorization_permits(family_maturity, authorization)`` — the family's
   independently-verified maturity permits the run's authorization.

The default authorization is ``NONE`` → nothing routes (advisory only). Isolated
mock authorization (``MOCK``) routes only families explicitly at ``MOCK_EXECUTION``
or beyond, and NEVER grants paper/live eligibility. Mandatory protective exits are
out of scope here and are never touched.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

from cengine.decision_events import DecisionEvent, DecisionEventSink, DecisionEventType
from cengine.strategy_families import ExposureDirection, StrategyFamily, StrategyFamilyRegistry
from cengine.strategy_maturity import (
    ExecutionAuthorization,
    authorization_permits,
    describe_authorization,
)
from order_manager import ManagedOrder, OrderStatus
from strategies import SignalSide, StrategyName, TradeCandidate

# Family -> the proven strategy identity it maps to (for attribution only; the
# routing state identity is StrategyName.REGIME_INTELLIGENCE to keep the bridge's
# single-pending guard separate from the legacy per-strategy signals).
FAMILY_TO_MAPPED_STRATEGY = {
    StrategyFamily.MOMENTUM_CONTINUATION: StrategyName.DUAL_MA,
    StrategyFamily.VOLATILITY_BREAKOUT: StrategyName.TURTLE,
    StrategyFamily.TREND_PULLBACK: StrategyName.DI_BB,
    StrategyFamily.MEAN_REVERSION: StrategyName.APO_MEAN_REVERSION,
    StrategyFamily.MULTI_DAY_SWING: StrategyName.REGIME_INTELLIGENCE,
}


@dataclass
class RouteDisposition:
    decision_id: str
    symbol_id: str
    family: str
    outcome: str          # "submitted" | "skipped" | "conflicted" | "risk_rejected"
                          # | "exposure_blocked" | "venue_rejected" | "error"
    reason: str = ""


@dataclass
class BridgeOutcome:
    authorization: str
    submitted: int = 0
    skipped: int = 0
    dispositions: list = field(default_factory=list)

    def by_outcome(self, outcome: str) -> list:
        return [d for d in self.dispositions if d.outcome == outcome]


class StrategyExecutionBridge:
    """Routes authorized recommendations through the existing execution authority."""

    def __init__(self, *, router, registry: StrategyFamilyRegistry,
                 sink: DecisionEventSink,
                 price_ticks_fn: Callable[[str], int],
                 authorization: ExecutionAuthorization = ExecutionAuthorization.NONE,
                 max_route: int = 10,
                 correlation_index: Optional[dict] = None) -> None:
        self._router = router
        self._registry = registry
        self._sink = sink
        self._price_ticks = price_ticks_fn
        self._authorization = authorization
        self._max_route = int(max_route)
        # correlation_id -> decision_id, so the OrderManager audit handler can
        # attribute ack/fill/risk events back to the originating recommendation.
        self._corr_to_decision = correlation_index if correlation_index is not None else {}

    @property
    def authorization(self) -> ExecutionAuthorization:
        return self._authorization

    @property
    def correlation_index(self) -> dict:
        return self._corr_to_decision

    def _eligible(self, rec, now_ns: int) -> tuple[bool, str]:
        if self._authorization is ExecutionAuthorization.NONE:
            return False, "authorization_none"
        if not getattr(rec, "conditions_passed", False):
            return False, "conditions_not_passed"
        if rec.exposure_direction is not ExposureDirection.LONG:
            # The strict elective-entry contract authorizes LONG entries only
            # (including BUYING authorized inverse ETFs).
            return False, f"non_long_direction:{rec.exposure_direction.value}"
        if rec.expired(now_ns):
            return False, "expired"
        maturity = self._registry.maturity(rec.strategy_family)
        if not authorization_permits(maturity, self._authorization):
            return False, (f"maturity_{maturity.value}_not_permitted_under_"
                           f"{self._authorization.value}")
        return True, "authorized"

    def _to_candidate(self, rec, now_ns: int) -> TradeCandidate:
        price_ticks = int(self._price_ticks(rec.symbol_id))
        return TradeCandidate(
            strategy=StrategyName.REGIME_INTELLIGENCE, symbol=rec.symbol_id,
            side=SignalSide.LONG, timestamp_ns=now_ns,
            reference_price_ticks=price_ticks,
            reason=f"{rec.strategy_family.value}:{rec.strategy_version}:{rec.decision_id}")

    async def route_recommendations(self, recommendations: list, *,
                                    now_ns: int) -> BridgeOutcome:
        """Route authorized recommendations through the existing CandidateRouter.

        Recommendations are routed in descending rank order (economic ranking is
        consumed here, §15). Each is routed independently so one rejection never
        blocks the others. Returns a :class:`BridgeOutcome` with per-recommendation
        dispositions. Never raises into the caller.
        """
        outcome = BridgeOutcome(authorization=self._authorization.value)
        ordered = sorted(
            recommendations,
            key=lambda r: (r.rank_score if r.rank_score is not None else -1e9),
            reverse=True)
        routed = 0
        for rec in ordered:
            ok, why = self._eligible(rec, now_ns)
            if not ok:
                outcome.skipped += 1
                outcome.dispositions.append(RouteDisposition(
                    rec.decision_id, rec.symbol_id, rec.strategy_family.value,
                    "skipped", why))
                continue
            if routed >= self._max_route:
                outcome.skipped += 1
                outcome.dispositions.append(RouteDisposition(
                    rec.decision_id, rec.symbol_id, rec.strategy_family.value,
                    "skipped", "bridge_max_route_cap"))
                continue

            try:
                candidate = self._to_candidate(rec, now_ns)
            except Exception as exc:  # noqa: BLE001 — e.g. no authoritative price yet
                outcome.skipped += 1
                outcome.dispositions.append(RouteDisposition(
                    rec.decision_id, rec.symbol_id, rec.strategy_family.value,
                    "skipped", f"price_unavailable:{type(exc).__name__}"))
                continue
            correlation_id = (f"{candidate.symbol}:{candidate.timestamp_ns}:"
                              f"{candidate.strategy.value}")
            self._corr_to_decision[correlation_id] = rec.decision_id
            # ORDER_INTENT: recommendation -> candidate linkage (quantity is decided
            # downstream by the sizing policy + native risk and surfaces at ack/fill).
            self._sink.emit(DecisionEvent(
                event_type=DecisionEventType.ORDER_INTENT, timestamp_ns=now_ns,
                decision_id=rec.decision_id, symbol_id=rec.symbol_id,
                strategy_family=rec.strategy_family.value,
                strategy_version=rec.strategy_version,
                regime_version=rec.regime_version, feature_version=rec.feature_version,
                payload={"correlation_id": correlation_id,
                         "authorization": self._authorization.value,
                         "reference_price_ticks": candidate.reference_price_ticks,
                         "side": candidate.side.value,
                         "rank_score": rec.rank_score, "rank_basis": rec.rank_basis}))

            disp = await self._route_one(rec, candidate)
            outcome.dispositions.append(disp)
            if disp.outcome == "submitted":
                outcome.submitted += 1
                routed += 1
            else:
                outcome.skipped += 1
        return outcome

    async def _route_one(self, rec, candidate: TradeCandidate) -> RouteDisposition:
        from order_manager import ExecutionRejected, RiskRejected  # local: avoid cycle
        fam = rec.strategy_family.value
        try:
            submitted, conflicted = await self._router.route((candidate,))
        except RiskRejected as exc:
            return RouteDisposition(rec.decision_id, rec.symbol_id, fam,
                                    "risk_rejected", str(exc))
        except ExecutionRejected as exc:
            return RouteDisposition(rec.decision_id, rec.symbol_id, fam,
                                    "venue_rejected", str(exc))
        except Exception as exc:  # noqa: BLE001 — bridge must never crash the loop
            return RouteDisposition(rec.decision_id, rec.symbol_id, fam,
                                    "error", f"{type(exc).__name__}:{exc}")
        if conflicted:
            return RouteDisposition(rec.decision_id, rec.symbol_id, fam,
                                    "conflicted", "router_withheld_conflicting_batch")
        if submitted == 0:
            # Router accepted the call but routed nothing: either the exposure gate
            # blocked an unauthorized geared/inverse entry, or the single-pending
            # guard withheld it.
            return RouteDisposition(rec.decision_id, rec.symbol_id, fam,
                                    "exposure_blocked",
                                    "router_submitted_zero (exposure gate or pending guard)")
        return RouteDisposition(rec.decision_id, rec.symbol_id, fam, "submitted", "")

    def describe(self) -> str:
        return describe_authorization(self._authorization)


def make_decision_audit_handler(sink: DecisionEventSink,
                                correlation_index: dict) -> Callable:
    """Build an OrderManager ``on_audit`` handler that emits execution-lifecycle
    decision events, attributing them to the originating recommendation via
    ``correlation_index`` (correlation_id -> decision_id).

    Compose this with the existing audit handlers (StrategyStateBook.on_order,
    journal) rather than replacing them.
    """

    def handler(event: str, order: ManagedOrder) -> None:
        corr = order.intent.correlation_id
        decision_id = correlation_index.get(corr)
        symbol = order.intent.symbol
        if event == "approved":
            sink.record_risk_decision(
                decision_id=decision_id, symbol_id=symbol, correlation_id=corr,
                approved=True, reason="native_risk_approved",
                now_ns=order.last_event_ns)
        elif event == "risk_rejected":
            sink.record_risk_decision(
                decision_id=decision_id, symbol_id=symbol, correlation_id=corr,
                approved=False, reason=order.rejection_reason or "native_risk_rejected",
                now_ns=order.last_event_ns)
        elif event == "submitted":
            sink.record_broker_ack(order, order.last_event_ns, decision_id=decision_id)
        elif event == "reconciled":
            if order.status is OrderStatus.FILLED:
                _emit_fill(sink, order, decision_id, DecisionEventType.COMPLETE_FILL)
            elif order.status is OrderStatus.PARTIALLY_FILLED:
                _emit_fill(sink, order, decision_id, DecisionEventType.PARTIAL_FILL)

    return handler


def _emit_fill(sink: DecisionEventSink, order: ManagedOrder, decision_id,
               event_type: DecisionEventType) -> None:
    sink.emit(DecisionEvent(
        event_type=event_type, timestamp_ns=order.last_event_ns,
        decision_id=decision_id, symbol_id=order.intent.symbol,
        payload={"correlation_id": order.intent.correlation_id,
                 "venue_order_id": order.venue_order_id,
                 "cumulative_filled_quantity": order.cumulative_filled_quantity,
                 "average_fill_price_ticks": order.average_fill_price_ticks}))


__all__ = [
    "BridgeOutcome",
    "FAMILY_TO_MAPPED_STRATEGY",
    "RouteDisposition",
    "StrategyExecutionBridge",
    "make_decision_audit_handler",
]
