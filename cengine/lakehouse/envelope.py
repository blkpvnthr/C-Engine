"""Canonical versioned event envelope (Milestone 5, §9).

Every lakehouse record is wrapped in a common envelope that reuses C-Engine's
existing identifiers (decision_id, correlation_id, order ids, feature/config
versions) rather than minting incompatible replacements. It distinguishes MARKET
event time from the time information became AVAILABLE to C-Engine — mandatory for
point-in-time learning integrity (no look-ahead).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Optional

ENVELOPE_SCHEMA_VERSION = "event-envelope-1.0.0"


@dataclass(frozen=True)
class EventEnvelope:
    event_id: str
    event_type: str
    schema_version: str
    source_system: str                   # "cengine"
    source_feed: Optional[str]           # "sip" | "iex" | "internal"
    session_id: Optional[str]
    run_id: Optional[str]
    symbol_id: Optional[str]
    instrument_generation: Optional[int]
    event_timestamp_ns: int              # when the market event occurred
    receipt_timestamp_ns: Optional[int]  # when C-Engine received it
    processing_timestamp_ns: Optional[int]
    availability_timestamp_ns: int       # when it became usable for a decision (point-in-time)
    sequence_number: Optional[int]
    correlation_id: Optional[str]
    causation_id: Optional[str]
    strategy_decision_id: Optional[str]
    portfolio_decision_id: Optional[str]
    execution_plan_id: Optional[str]
    order_intent_id: Optional[str]
    broker_order_id: Optional[str]
    strategy_family: Optional[str]
    strategy_version: Optional[str]
    regime_version: Optional[str]
    feature_version: Optional[str]
    configuration_version: Optional[str]
    quality_flags: tuple = ()
    payload: dict = field(default_factory=dict)

    def to_row(self) -> dict:
        d = {
            "event_id": self.event_id, "event_type": self.event_type,
            "schema_version": self.schema_version, "source_system": self.source_system,
            "source_feed": self.source_feed, "session_id": self.session_id,
            "run_id": self.run_id, "symbol_id": self.symbol_id,
            "instrument_generation": self.instrument_generation,
            "event_timestamp_ns": self.event_timestamp_ns,
            "receipt_timestamp_ns": self.receipt_timestamp_ns,
            "processing_timestamp_ns": self.processing_timestamp_ns,
            "availability_timestamp_ns": self.availability_timestamp_ns,
            "sequence_number": self.sequence_number,
            "correlation_id": self.correlation_id, "causation_id": self.causation_id,
            "strategy_decision_id": self.strategy_decision_id,
            "portfolio_decision_id": self.portfolio_decision_id,
            "execution_plan_id": self.execution_plan_id,
            "order_intent_id": self.order_intent_id, "broker_order_id": self.broker_order_id,
            "strategy_family": self.strategy_family, "strategy_version": self.strategy_version,
            "regime_version": self.regime_version, "feature_version": self.feature_version,
            "configuration_version": self.configuration_version,
            "quality_flags": list(self.quality_flags),
            # payload stored as a JSON string (schema-flexible; the typed columns above
            # carry the stable, queryable identity + time fields).
            "payload_json": json.dumps(self.payload, default=str, sort_keys=True),
        }
        return d


def _stable_event_id(event_type: str, ts_ns: int, payload: dict,
                     decision_id: Optional[str]) -> str:
    """Deterministic id from identity fields -> idempotent dedup across restarts."""
    basis = f"{event_type}|{ts_ns}|{decision_id}|" + json.dumps(
        payload, default=str, sort_keys=True)
    return hashlib.sha256(basis.encode()).hexdigest()[:32]


def envelope_from_decision_row(row: dict, *, session_id: Optional[str] = None,
                               run_id: Optional[str] = None,
                               source_feed: Optional[str] = None,
                               now_ns: Optional[int] = None) -> EventEnvelope:
    """Wrap a DecisionEventSink ``to_row()`` dict in the canonical envelope, reusing
    its identifiers. The decision event's timestamp is both the event time and the
    availability time (decisions are computed from already-available observations).
    """
    payload = dict(row.get("payload") or {})
    ts = int(row.get("timestamp_ns", 0))
    event_type = str(row.get("event_type", "unknown"))
    decision_id = row.get("decision_id")
    corr = payload.get("correlation_id")
    return EventEnvelope(
        event_id=_stable_event_id(event_type, ts, payload, decision_id),
        event_type=event_type, schema_version=ENVELOPE_SCHEMA_VERSION,
        source_system="cengine", source_feed=source_feed, session_id=session_id,
        run_id=run_id, symbol_id=row.get("symbol_id"),
        instrument_generation=payload.get("instrument_generation"),
        event_timestamp_ns=ts, receipt_timestamp_ns=ts, processing_timestamp_ns=now_ns,
        availability_timestamp_ns=ts, sequence_number=payload.get("venue_sequence"),
        correlation_id=corr, causation_id=row.get("decision_id"),
        strategy_decision_id=(row.get("decision_id")
                              if "strategy" in event_type or "recommendation" in event_type
                              else payload.get("source_strategy_decision_id")),
        portfolio_decision_id=payload.get("portfolio_decision_id")
        or (row.get("decision_id") if "portfolio" in event_type else None),
        execution_plan_id=(row.get("decision_id") if "execution_plan" in event_type else None),
        order_intent_id=payload.get("client_order_id"),
        broker_order_id=payload.get("venue_order_id"),
        strategy_family=row.get("strategy_family"),
        strategy_version=row.get("strategy_version"),
        regime_version=row.get("regime_version"),
        feature_version=row.get("feature_version"),
        configuration_version=payload.get("configuration_version"),
        quality_flags=tuple(payload.get("quality_flags", ()) or ()),
        payload=payload)
