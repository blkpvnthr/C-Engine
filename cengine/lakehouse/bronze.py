"""Bronze layer: raw immutable source observations + engine/execution events (§8/§10).

The :class:`AnalyticalIngestionSink` drains the existing nonblocking
:class:`cengine.decision_events.DecisionEventSink` (and raw market events), wraps each
record in the canonical envelope, and writes it to a bronze Parquet dataset. It is
ANALYTICAL only: bounded, nonblocking, with explicit event-loss accounting; it NEVER
blocks the order path, native RiskEngine, or broker reconciliation. Corrections are
represented as explicit records — bronze is never silently rewritten.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

from cengine.lakehouse.config import LakehouseConfig
from cengine.lakehouse.envelope import envelope_from_decision_row
from cengine.lakehouse.parquet_writer import ParquetWriter, trade_date_from_ns

# DecisionEventType.value -> bronze dataset (§10). Unknown types -> engine_decisions.
_EVENT_DATASET = {
    "regime_transition": "engine_decisions",
    "strategy_recommendation": "engine_decisions",
    "rejected_opportunity": "engine_decisions",
    "ranking_components": "engine_decisions",
    "arbitration_decision": "engine_decisions",
    "risk_approval": "risk_decisions",
    "risk_rejection": "risk_decisions",
    "portfolio_state": "portfolio_decisions",
    "portfolio_optimization": "portfolio_decisions",
    "portfolio_recommendation": "portfolio_decisions",
    "portfolio_constraint_binding": "portfolio_decisions",
    "capital_reservation": "portfolio_decisions",
    "replacement_decision": "portfolio_decisions",
    "execution_plan": "execution_plans",
    "execution_plan_rejected": "execution_plans",
    "microstructure_snapshot": "execution_plans",
    "transaction_cost_analysis": "execution_plans",
    "order_intent": "order_events",
    "broker_ack": "order_events",
    "partial_fill": "fill_events",
    "complete_fill": "fill_events",
    "exit_decision": "position_events",
    "position_management": "position_events",
    "profit_protection": "position_events",
    "realized_outcome": "position_events",
}

BRONZE_DATASETS = sorted(set(_EVENT_DATASET.values()) | {
    "market_bars", "market_trades", "market_quotes", "market_corrections",
    "instrument_metadata", "universe_membership", "subscription_events",
    "market_data_quality_events", "reconciliation_events", "system_health_events"})


@dataclass
class IngestionStats:
    ingested: int = 0
    written: int = 0
    dropped: int = 0          # could not be written (disk/cap) — explicitly recorded
    quarantined: int = 0
    by_dataset: dict = field(default_factory=dict)
    losses: list = field(default_factory=list)


class AnalyticalIngestionSink:
    """Drains decision/market events to bronze Parquet. Nonblocking + loss-accounted."""

    def __init__(self, config: Optional[LakehouseConfig] = None, *,
                 session_id: Optional[str] = None, run_id: Optional[str] = None,
                 source_feed: Optional[str] = None, overflow_capacity: int = 500_000) -> None:
        self.config = config or LakehouseConfig.from_env()
        self.writer = ParquetWriter(self.config)
        self.writer.recover_incomplete()
        self.session_id = session_id
        self.run_id = run_id
        self.source_feed = source_feed
        self._pending: deque = deque(maxlen=overflow_capacity)  # spill buffer if a flush fails
        self.stats = IngestionStats()

    def ingest_decision_rows(self, rows: list, *, now_ns: Optional[int] = None) -> IngestionStats:
        """Wrap decision rows in envelopes, route to bronze datasets, write Parquet."""
        now_ns = now_ns if now_ns is not None else time.time_ns()
        by_dataset: dict[str, list] = {}
        for row in rows:
            self.stats.ingested += 1
            try:
                env = envelope_from_decision_row(
                    row, session_id=self.session_id, run_id=self.run_id,
                    source_feed=self.source_feed, now_ns=now_ns)
                dataset = _EVENT_DATASET.get(env.event_type, "engine_decisions")
                by_dataset.setdefault(dataset, []).append(env.to_row())
            except Exception as exc:  # noqa: BLE001 — bad record -> quarantine, never crash
                self.stats.quarantined += 1
                self.writer.quarantine("engine_decisions", [row],
                                       f"envelope_error:{type(exc).__name__}")
        self._write_groups(by_dataset)
        return self.stats

    def ingest_market_rows(self, dataset: str, rows: list) -> IngestionStats:
        """Ingest raw market observation rows (bars/trades/quotes/metadata) to bronze.
        Rows must carry ``event_timestamp_ns`` + ``source_feed`` for partitioning."""
        self.stats.ingested += len(rows)
        self._write_groups({dataset: list(rows)})
        return self.stats

    def _write_groups(self, by_dataset: dict) -> None:
        ok, reason = self.config.accepts_writes()
        if not ok:
            # Fail closed for analytics: record the loss explicitly; do NOT crash.
            total = sum(len(v) for v in by_dataset.values())
            self.stats.dropped += total
            self.stats.losses.append({"reason": reason, "rows": total, "ns": time.time_ns()})
            return
        for dataset, rows in by_dataset.items():
            results = self.writer.write_rows("bronze", dataset, rows)
            written = sum(r.rows for r in results if r.ok)
            self.stats.written += written
            self.stats.by_dataset[dataset] = self.stats.by_dataset.get(dataset, 0) + written
            for r in results:
                if not r.ok and r.reason not in ("empty",):
                    self.stats.dropped += len(rows)
                    self.stats.losses.append({"reason": r.reason, "dataset": dataset,
                                              "ns": time.time_ns()})

    def drain_decision_sink(self, decision_sink) -> IngestionStats:
        """Drain a DecisionEventSink's buffered rows into bronze (post-session / periodic)."""
        rows = decision_sink.drain()
        return self.ingest_decision_rows(rows)

    def record_correction(self, dataset: str, original_event_id: str, correction: dict) -> None:
        """Append an explicit correction record (never rewrite the original, §10)."""
        row = {"event_timestamp_ns": time.time_ns(), "source_feed": self.source_feed,
               "corrects_event_id": original_event_id, "correction": correction,
               "trade_date": trade_date_from_ns(time.time_ns())}
        self._write_groups({"market_corrections": [row]})
