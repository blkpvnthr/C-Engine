"""Local Parquet audit-journal sink — replaces the MongoDB journal sink.

The MongoDB sink blocked engine startup: ``AuditJournal.__init__`` replays every
record to each sink, and ``MongoDailyJournalStore.write`` issues a synchronous
``w=majority, journal=True`` ``update_one`` with no socket timeout, which hangs
indefinitely against a paused/degraded Atlas primary (the cluster accepts a ping
but blocks writes).

This sink is **local** (no network), **batched**, **idempotent across restarts**,
and **never raises** into the journal (it reuses the lakehouse
:class:`~cengine.lakehouse.parquet_writer.ParquetWriter`, whose writes return a
result object rather than raising). The append-only, checksummed JSONL file in
:class:`~cengine.journal.AuditJournal` remains the authoritative, durable record;
this Parquet dataset is the DuckDB-queryable replica.
"""

from __future__ import annotations

import json
from typing import Any

from cengine.lakehouse.config import LakehouseConfig
from cengine.lakehouse.parquet_writer import ParquetWriter, trade_date_from_ns

_AUDIT_LAYER = "audit"
_JOURNAL_DATASET = "journal"


class ParquetJournalStore:
    """Append verified audit-journal records to a partitioned local Parquet dataset.

    Implements the ``JournalSink`` protocol (``write`` / ``close``). Records are
    buffered and flushed in batches (never one tiny file per record) and
    partitioned by UTC ``trade_date``. Idempotent: a record whose ``sequence`` is
    ``<=`` the highest already-persisted sequence is skipped, so ``AuditJournal``'s
    startup replay of the whole JSONL file is a cheap no-op for records already
    stored in a previous run.
    """

    def __init__(self, config: LakehouseConfig, *, flush_threshold: int = 256) -> None:
        self._config = config
        self._writer = ParquetWriter(config)
        # Clean up any orphaned temp files from an interrupted previous write.
        self._writer.recover_incomplete()
        self._flush_threshold = max(1, int(flush_threshold))
        self._pending: list[dict[str, Any]] = []
        self._seen: set[int] = set()
        self._max_persisted = self._load_max_persisted_sequence()

    def write(self, record: dict[str, Any]) -> None:
        try:
            seq = int(record["sequence"])
        except (KeyError, TypeError, ValueError):
            # Without a usable sequence it cannot be deduped; the JSONL keeps it.
            return
        if seq <= self._max_persisted or seq in self._seen:
            return  # already persisted (prior run) or buffered this run — idempotent
        self._seen.add(seq)
        self._pending.append(self._row(record, seq))
        if len(self._pending) >= self._flush_threshold:
            self._flush()

    def close(self) -> None:
        self._flush()

    def _flush(self) -> None:
        if not self._pending:
            return
        rows, self._pending = self._pending, []
        # write_rows groups by partition and NEVER raises (returns WriteResult list).
        self._writer.write_rows(
            _AUDIT_LAYER,
            _JOURNAL_DATASET,
            rows,
            partition_fn=lambda r: {"trade_date": r["trade_date"]},
        )

    @staticmethod
    def _row(record: dict[str, Any], seq: int) -> dict[str, Any]:
        ts = int(record.get("timestamp_ns") or 0)
        return {
            "sequence": seq,
            "kind": str(record.get("kind", "")),
            "timestamp_ns": ts,
            "sha256": str(record.get("sha256", "")),
            # Arbitrary/heterogeneous payloads collapse to a stable JSON string column
            # (default=str so a non-serializable value degrades to its repr, never
            # raising and never clashing with another record's payload shape).
            "payload_json": json.dumps(record.get("payload"), default=str, sort_keys=True),
            "trade_date": trade_date_from_ns(ts),
        }

    def _load_max_persisted_sequence(self) -> int:
        """Highest sequence already on disk, so the startup replay skips old records."""
        try:
            import pyarrow.parquet as pq

            dpath = self._config.dataset_path(_AUDIT_LAYER, _JOURNAL_DATASET)
            if not dpath.exists():
                return 0
            mx = 0
            for part in dpath.rglob("*.parquet"):
                try:
                    col = pq.read_table(part, columns=["sequence"]).column("sequence").to_pylist()
                    if col:
                        mx = max(mx, max(int(x) for x in col))
                except Exception:  # noqa: BLE001 — a bad part file must not break startup
                    continue
            return mx
        except Exception:  # noqa: BLE001 — never block startup on replica bookkeeping
            return 0

    def __repr__(self) -> str:
        return (
            f"ParquetJournalStore(root={self._config.root}, "
            f"dataset={_AUDIT_LAYER}/{_JOURNAL_DATASET})"
        )
