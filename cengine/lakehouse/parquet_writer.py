"""Atomic, partitioned Parquet writer (Milestone 5, §3/§4/§24).

Writes Arrow tables to ``dataset/trade_date=YYYY-MM-DD/feed=.../part-<hash>.parquet``
with zstd compression, configurable row groups, per-file checksums, append-only
dataset manifests, atomic publication (temp-file + fsync + rename), incomplete-write
recovery (orphan ``.tmp`` cleanup), and a quarantine path for invalid rows. Never
raises into the caller's hot path — write failures return a result object.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pyarrow as pa
import pyarrow.parquet as pq

from cengine.lakehouse.config import LakehouseConfig


@dataclass(frozen=True)
class WriteResult:
    ok: bool
    path: Optional[str]
    rows: int
    checksum: Optional[str]
    reason: str = ""


def trade_date_from_ns(ts_ns: int) -> str:
    try:
        return datetime.fromtimestamp(ts_ns / 1e9, tz=timezone.utc).strftime("%Y-%m-%d")
    except (ValueError, OverflowError, OSError):
        return "1970-01-01"


class ParquetWriter:
    def __init__(self, config: LakehouseConfig) -> None:
        self.config = config
        self.config.ensure_dirs()

    # -- recovery (§4/§33) ---------------------------------------------------
    def recover_incomplete(self) -> int:
        """Delete orphan ``.tmp`` files from interrupted writes. Returns count removed."""
        removed = 0
        for tmp in self.config.root.rglob("*.parquet.tmp"):
            try:
                tmp.unlink()
                removed += 1
            except OSError:
                pass
        return removed

    def _codec(self) -> str:
        # zstd preferred; snappy fallback if the build lacks zstd support.
        return self.config.compression if self.config.compression in {
            "zstd", "snappy", "gzip", "lz4", "none"} else "snappy"

    def write(self, layer: str, dataset: str, table: pa.Table, *,
              partition: Optional[dict] = None) -> WriteResult:
        """Atomically write one Arrow table as a single part file under a partition.

        ``partition`` is an ordered dict of partition_col -> value (e.g.
        {"trade_date": "2026-11-20", "feed": "sip"}).
        """
        if table.num_rows == 0:
            return WriteResult(True, None, 0, None, "empty")
        ok, reason = self.config.accepts_writes()
        if not ok:
            return WriteResult(False, None, 0, None, reason)
        part_dir = self.config.dataset_path(layer, dataset)
        for col, val in (partition or {}).items():
            part_dir = part_dir / f"{col}={val}"
        try:
            part_dir.mkdir(parents=True, exist_ok=True)
            # content hash over the serialized table -> idempotent, dedup-friendly name.
            sink = pa.BufferOutputStream()
            codec = self._codec()
            compression = None if codec == "none" else codec
            pq.write_table(table, sink, compression=compression,
                           row_group_size=self.config.row_group_size)
            buf = sink.getvalue()
            data = buf.to_pybytes()
            checksum = hashlib.sha256(data).hexdigest()
            final = part_dir / f"part-{checksum[:16]}.parquet"
            if final.exists():
                return WriteResult(True, str(final), table.num_rows, checksum, "idempotent")
            # atomic publish: temp file in the same dir, fsync, rename.
            fd, tmp_name = tempfile.mkstemp(suffix=".parquet.tmp", dir=str(part_dir))
            try:
                with os.fdopen(fd, "wb") as fh:
                    fh.write(data)
                    fh.flush()
                    os.fsync(fh.fileno())
                os.replace(tmp_name, final)
            except Exception:
                try:
                    os.unlink(tmp_name)
                except OSError:
                    pass
                raise
            self._append_manifest(layer, dataset, final, table.num_rows, checksum, partition)
            return WriteResult(True, str(final), table.num_rows, checksum, "written")
        except Exception as exc:  # noqa: BLE001 — analytical write failure is never fatal
            return WriteResult(False, None, 0, None, f"{type(exc).__name__}:{exc}")

    def write_rows(self, layer: str, dataset: str, rows: list, *,
                   partition_fn=None) -> list:
        """Group rows by partition (via ``partition_fn(row) -> dict``) and write each
        group as a part file. Default partition = {trade_date, feed} from the envelope."""
        if not rows:
            return []
        partition_fn = partition_fn or _default_partition
        groups: dict = {}
        for row in rows:
            key = tuple(sorted(partition_fn(row).items()))
            groups.setdefault(key, []).append(row)
        results = []
        for key, group in groups.items():
            partition = dict(key)
            table = pa.Table.from_pylist(group)
            results.append(self.write(layer, dataset, table, partition=partition))
        return results

    def _append_manifest(self, layer: str, dataset: str, path: Path, rows: int,
                         checksum: str, partition: Optional[dict]) -> None:
        manifest = self.config.manifests / f"{layer}.{dataset}.jsonl"
        entry = {"path": str(path), "rows": rows, "checksum": checksum,
                 "partition": partition or {}, "written_ns": time.time_ns(),
                 "layer": layer, "dataset": dataset}
        try:
            with open(manifest, "a") as fh:
                fh.write(json.dumps(entry) + "\n")
        except OSError:
            pass

    def quarantine(self, dataset: str, rows: list, reason: str) -> WriteResult:
        """Persist invalid records with an explicit reason instead of discarding (§23)."""
        if not rows:
            return WriteResult(True, None, 0, None, "empty")
        tagged = [{**r, "_quarantine_reason": reason,
                   "_quarantine_ns": time.time_ns()} for r in rows]
        table = pa.Table.from_pylist([{"record_json": json.dumps(r, default=str)}
                                      for r in tagged])
        return self.write("quarantine", dataset, table,
                          partition={"reason": reason[:40].replace("/", "_")})


def _default_partition(row: dict) -> dict:
    ts = int(row.get("event_timestamp_ns") or row.get("timestamp_ns") or 0)
    feed = row.get("source_feed") or "internal"
    return {"trade_date": trade_date_from_ns(ts), "feed": feed}
