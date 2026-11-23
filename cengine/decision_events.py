"""Decision-evidence contract (milestone §17).

A consistent, NONBLOCKING decision-event schema that captures the full elective
decision lifecycle for subsequent local Parquet/DuckDB integration. This module
intentionally does NOT build the ML/RL platform — it only defines the event
contract and a nonblocking sink that preserves point-in-time timestamps, decision
IDs, feature/strategy/regime attribution, and can hand its buffer to a later
Parquet writer or reuse an existing audit journal.

Nonblocking guarantee: ``emit`` is O(1) (append to a bounded in-memory deque) and
NEVER raises — it must never perturb market ingestion, strategy evaluation, or the
mandatory protective-exit path.
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Optional

DECISION_EVENT_SCHEMA_VERSION = "decision-events-1.0.0"


class DecisionEventType(str, Enum):
    REGIME_TRANSITION = "regime_transition"
    STRATEGY_RECOMMENDATION = "strategy_recommendation"
    REJECTED_OPPORTUNITY = "rejected_opportunity"
    RANKING_COMPONENTS = "ranking_components"
    ARBITRATION_DECISION = "arbitration_decision"
    RISK_APPROVAL = "risk_approval"
    RISK_REJECTION = "risk_rejection"
    ORDER_INTENT = "order_intent"
    BROKER_ACK = "broker_ack"
    PARTIAL_FILL = "partial_fill"
    COMPLETE_FILL = "complete_fill"
    EXIT_DECISION = "exit_decision"
    REALIZED_OUTCOME = "realized_outcome"


@dataclass
class DecisionEvent:
    event_type: DecisionEventType
    timestamp_ns: int                       # point-in-time (observation/decision)
    decision_id: Optional[str] = None
    symbol_id: Optional[str] = None
    strategy_family: Optional[str] = None
    strategy_version: Optional[str] = None
    regime_version: Optional[str] = None
    feature_version: Optional[str] = None
    payload: dict = field(default_factory=dict)
    schema_version: str = DECISION_EVENT_SCHEMA_VERSION

    def to_row(self) -> dict:
        """Flat, serializable row suitable for Parquet/DuckDB ingestion later."""
        d = asdict(self)
        d["event_type"] = self.event_type.value
        return d


class DecisionEventSink:
    """Bounded, nonblocking decision-event buffer.

    ``emit`` appends in O(1) and never raises. If an audit ``journal`` with an
    ``append(kind, ts_ns, obj)`` method is supplied, decision-relevant events are
    ALSO mirrored to it (reusing existing audit persistence, §17) — but a journal
    failure is swallowed so it can never block the decision path. ``drain`` yields
    the buffered rows for a later Parquet/DuckDB writer.
    """

    def __init__(self, *, capacity: int = 100_000, journal: Any = None,
                 mirror_to_journal: bool = True) -> None:
        self._buf: deque = deque(maxlen=max(1, int(capacity)))
        self._journal = journal
        self._mirror = bool(mirror_to_journal) and journal is not None
        self.emitted = 0
        self.dropped = 0

    def emit(self, event: DecisionEvent) -> None:
        try:
            if len(self._buf) == self._buf.maxlen:
                self.dropped += 1       # oldest silently evicted by deque(maxlen=)
            self._buf.append(event)
            self.emitted += 1
            if self._mirror:
                try:
                    self._journal.append("decision_event", event.timestamp_ns,
                                         event.to_row())
                except Exception:  # noqa: BLE001 — journaling must never block decisions
                    pass
        except Exception:  # noqa: BLE001 — the sink must be totally nonblocking
            pass

    # -- convenience emitters (attribution-preserving) -----------------------
    def record_regime_transition(self, assessment, now_ns: int) -> None:
        dims = assessment.dimensions
        self.emit(DecisionEvent(
            event_type=DecisionEventType.REGIME_TRANSITION, timestamp_ns=now_ns,
            symbol_id=assessment.scope, regime_version=assessment.version,
            payload={"level": assessment.level.value,
                     "direction": dims.direction.value, "trend": dims.trend.value,
                     "volatility": dims.volatility.value,
                     "liquidity": dims.liquidity.value,
                     "confidence": dims.confidence,
                     "previous_direction": (assessment.previous_direction.value
                                            if assessment.previous_direction else None)}))

    def record_recommendation(self, rec, now_ns: int,
                              event_type: DecisionEventType =
                              DecisionEventType.STRATEGY_RECOMMENDATION) -> None:
        self.emit(DecisionEvent(
            event_type=event_type, timestamp_ns=now_ns,
            decision_id=rec.decision_id, symbol_id=rec.symbol_id,
            strategy_family=rec.strategy_family.value,
            strategy_version=rec.strategy_version, regime_version=rec.regime_version,
            feature_version=rec.feature_version,
            payload={"exposure_direction": rec.exposure_direction.value,
                     "eligibility": rec.eligibility.value,
                     "rejection_reason": rec.rejection_reason,
                     "expected_gross_return": rec.expected_gross_return,
                     "estimated_transaction_cost": rec.estimated_transaction_cost,
                     "expected_net_return": rec.expected_net_return,
                     "expected_downside": rec.expected_downside,
                     "confidence": rec.confidence,
                     "rank_score": rec.rank_score, "rank_basis": rec.rank_basis}))

    def record_arbitration(self, result: dict, now_ns: int) -> None:
        self.emit(DecisionEvent(
            event_type=DecisionEventType.ARBITRATION_DECISION, timestamp_ns=now_ns,
            payload={"selected": [r.decision_id for r in result.get("selected", [])],
                     "rejected": [(r.decision_id, why)
                                  for r, why in result.get("rejected", [])],
                     "research_count": len(result.get("research", []))}))

    # -- execution-lifecycle emitters (§8: traceable recommendation->outcome) --
    def record_order_intent(self, rec, intent, now_ns: int, *,
                            authorization: Optional[str] = None,
                            reference_price_ticks: Optional[int] = None) -> None:
        """Link a recommendation's ``decision_id`` to a concrete order intent.

        ``correlation_id`` on the intent is the stable key that connects this
        intent to its subsequent ack/fill/outcome events.
        """
        self.emit(DecisionEvent(
            event_type=DecisionEventType.ORDER_INTENT, timestamp_ns=now_ns,
            decision_id=getattr(rec, "decision_id", None),
            symbol_id=getattr(rec, "symbol_id", None),
            strategy_family=(rec.strategy_family.value
                             if getattr(rec, "strategy_family", None) else None),
            strategy_version=getattr(rec, "strategy_version", None),
            regime_version=getattr(rec, "regime_version", None),
            feature_version=getattr(rec, "feature_version", None),
            payload={"correlation_id": intent.correlation_id,
                     "strategy_id": intent.strategy_id,
                     "side": intent.side.value, "quantity": intent.quantity,
                     "order_type": intent.order_type.value,
                     "time_in_force": intent.time_in_force.value,
                     "authorization": authorization,
                     "reference_price_ticks": reference_price_ticks}))

    def record_risk_decision(self, *, decision_id: Optional[str], symbol_id: str,
                             correlation_id: str, approved: bool, reason: str,
                             now_ns: int) -> None:
        self.emit(DecisionEvent(
            event_type=(DecisionEventType.RISK_APPROVAL if approved
                        else DecisionEventType.RISK_REJECTION),
            timestamp_ns=now_ns, decision_id=decision_id, symbol_id=symbol_id,
            payload={"correlation_id": correlation_id, "approved": approved,
                     "reason": reason}))

    def record_broker_ack(self, order, now_ns: int, *,
                          decision_id: Optional[str] = None) -> None:
        self.emit(DecisionEvent(
            event_type=DecisionEventType.BROKER_ACK, timestamp_ns=now_ns,
            decision_id=decision_id, symbol_id=order.intent.symbol,
            payload={"correlation_id": order.intent.correlation_id,
                     "client_order_id": order.client_order_id,
                     "venue_order_id": order.venue_order_id,
                     "status": order.status.value}))

    def record_fill(self, order, update, now_ns: int, *,
                    decision_id: Optional[str] = None) -> None:
        """Emit PARTIAL_FILL or COMPLETE_FILL from a reconciled execution update."""
        from order_manager import OrderStatus  # local import avoids a cycle
        complete = update.status is OrderStatus.FILLED
        self.emit(DecisionEvent(
            event_type=(DecisionEventType.COMPLETE_FILL if complete
                        else DecisionEventType.PARTIAL_FILL),
            timestamp_ns=now_ns, decision_id=decision_id,
            symbol_id=order.intent.symbol,
            payload={"correlation_id": order.intent.correlation_id,
                     "venue_order_id": update.venue_order_id,
                     "last_fill_quantity": update.last_fill_quantity,
                     "cumulative_filled_quantity": update.cumulative_filled_quantity,
                     "last_fill_price_ticks": update.last_fill_price_ticks,
                     "average_fill_price_ticks": order.average_fill_price_ticks}))

    def record_exit_decision(self, *, symbol_id: str, strategy_id: str,
                             reason: str, now_ns: int,
                             unrealized_return: Optional[float] = None) -> None:
        self.emit(DecisionEvent(
            event_type=DecisionEventType.EXIT_DECISION, timestamp_ns=now_ns,
            symbol_id=symbol_id,
            payload={"strategy_id": strategy_id, "reason": reason,
                     "unrealized_return": unrealized_return}))

    def record_realized_outcome(self, *, decision_id: Optional[str], symbol_id: str,
                                correlation_id: str, now_ns: int,
                                realized_return: Optional[float] = None,
                                holding_ns: Optional[int] = None,
                                realized_pnl_ticks: Optional[int] = None) -> None:
        self.emit(DecisionEvent(
            event_type=DecisionEventType.REALIZED_OUTCOME, timestamp_ns=now_ns,
            decision_id=decision_id, symbol_id=symbol_id,
            payload={"correlation_id": correlation_id,
                     "realized_return": realized_return, "holding_ns": holding_ns,
                     "realized_pnl_ticks": realized_pnl_ticks}))

    # -- draining for later Parquet/DuckDB ingestion -------------------------
    def drain(self) -> list:
        """Return + clear the buffered events as flat rows (point-in-time preserved)."""
        rows = [e.to_row() for e in self._buf]
        self._buf.clear()
        return rows

    def peek(self) -> list:
        return list(self._buf)

    def rehydrate(self, rows: list) -> int:
        """Rebuild events from flat rows (e.g. recovered from the authoritative
        journal) into the buffer. Used for missing-event recovery (§8) when the
        bounded in-memory buffer has evicted older events. Returns the count added.
        """
        added = 0
        for ev in events_from_rows(rows):
            self._buf.append(ev)
            added += 1
        return added

    @property
    def size(self) -> int:
        return len(self._buf)


def events_from_rows(rows: list) -> list:
    """Reconstruct :class:`DecisionEvent` objects from flat ``to_row()`` dicts.

    Tolerant of unknown/extra keys so journal rows from a newer schema still load.
    Rows whose ``event_type`` is unknown are skipped (never fabricated).
    """
    out: list = []
    valid = {e.value for e in DecisionEventType}
    for row in rows or []:
        et = row.get("event_type")
        if et not in valid:
            continue
        out.append(DecisionEvent(
            event_type=DecisionEventType(et),
            timestamp_ns=int(row.get("timestamp_ns", 0)),
            decision_id=row.get("decision_id"), symbol_id=row.get("symbol_id"),
            strategy_family=row.get("strategy_family"),
            strategy_version=row.get("strategy_version"),
            regime_version=row.get("regime_version"),
            feature_version=row.get("feature_version"),
            payload=dict(row.get("payload") or {}),
            schema_version=row.get("schema_version", DECISION_EVENT_SCHEMA_VERSION)))
    return out
