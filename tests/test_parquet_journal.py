"""Parquet audit-journal sink — local replacement for the MongoDB journal sink.

The MongoDB sink blocked engine startup: AuditJournal.__init__ replays every
record to each sink, and MongoDailyJournalStore.write does a synchronous
w=majority+journal update_one with no socket timeout, which hangs forever against a
paused/degraded Atlas primary. The Parquet sink is local (no network), batched,
idempotent across restarts, and never raises into the journal (reuses the
lakehouse ParquetWriter). The append-only JSONL file remains the authoritative,
durable record; this sink is the queryable replica.
"""

from __future__ import annotations

import json

import pyarrow.parquet as pq

from cengine.lakehouse.config import LakehouseConfig
from cengine.parquet_journal import ParquetJournalStore

# 2026-11-23 ~14:00 UTC and a second distinct day, in epoch-ns.
_NS_DAY1 = 1_795_000_000_000_000_000
_NS_DAY2 = _NS_DAY1 + 2 * 86_400 * 1_000_000_000


def _rec(seq: int, ts: int = _NS_DAY1, kind: str = "engine_state", payload=None) -> dict:
    return {
        "sequence": seq,
        "kind": kind,
        "timestamp_ns": ts,
        "payload": payload if payload is not None else {"state": "reconciling"},
        "sha256": f"deadbeef{seq:08d}",
    }


def _all_sequences(root) -> list[int]:
    seqs: list[int] = []
    for f in (root / "audit" / "journal").rglob("*.parquet"):
        seqs.extend(pq.read_table(f, columns=["sequence"]).column("sequence").to_pylist())
    return seqs


def test_writes_records_to_local_parquet(tmp_path):
    store = ParquetJournalStore(LakehouseConfig(root=tmp_path))
    store.write(_rec(1, payload={"state": "reconciling"}))
    store.write(_rec(2, payload={"order_id": "abc", "qty": 3}))
    store.close()

    seqs = _all_sequences(tmp_path)
    assert sorted(seqs) == [1, 2]
    # Heterogeneous payloads survive via a stable JSON column (no schema clash).
    files = list((tmp_path / "audit" / "journal").rglob("*.parquet"))
    table = pq.read_table(files[0])
    assert "payload_json" in table.column_names
    payloads = [json.loads(p) for p in table.column("payload_json").to_pylist()]
    assert {"state": "reconciling"} in payloads


def test_idempotent_across_restart_no_duplicate_rows(tmp_path):
    cfg = LakehouseConfig(root=tmp_path)
    first = ParquetJournalStore(cfg)
    first.write(_rec(1))
    first.write(_rec(2))
    first.close()

    # Simulate engine restart: AuditJournal replays ALL records, then one new append.
    second = ParquetJournalStore(cfg)
    second.write(_rec(1))  # replay — already persisted
    second.write(_rec(2))  # replay — already persisted
    second.write(_rec(3))  # genuinely new
    second.close()

    seqs = sorted(_all_sequences(tmp_path))
    assert seqs == [1, 2, 3]  # no duplicates despite the replay


def test_within_run_dedup(tmp_path):
    store = ParquetJournalStore(LakehouseConfig(root=tmp_path))
    store.write(_rec(1))
    store.write(_rec(1))  # duplicate in the same run
    store.close()
    assert sorted(_all_sequences(tmp_path)) == [1]


def test_partitioned_by_trade_date(tmp_path):
    store = ParquetJournalStore(LakehouseConfig(root=tmp_path))
    store.write(_rec(1, ts=_NS_DAY1))
    store.write(_rec(2, ts=_NS_DAY2))
    store.close()
    part_dirs = {p.name for p in (tmp_path / "audit" / "journal").glob("trade_date=*")}
    assert len(part_dirs) == 2  # two distinct day partitions


def test_write_never_raises_on_bad_payload(tmp_path):
    store = ParquetJournalStore(LakehouseConfig(root=tmp_path))
    # A non-serializable payload must not raise into the journal/engine.
    store.write(_rec(1, payload={"obj": object()}))
    store.close()  # must not raise
