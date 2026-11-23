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

    # -- draining for later Parquet/DuckDB ingestion -------------------------
    def drain(self) -> list:
        """Return + clear the buffered events as flat rows (point-in-time preserved)."""
        rows = [e.to_row() for e in self._buf]
        self._buf.clear()
        return rows

    def peek(self) -> list:
        return list(self._buf)

    @property
    def size(self) -> int:
        return len(self._buf)
