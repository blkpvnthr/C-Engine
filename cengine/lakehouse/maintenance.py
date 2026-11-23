"""Compaction + retention maintenance (Milestone 5, §24/§4).

Compaction merges many small part files within a single ``trade_date=.../feed=...``
partition into one larger file (fewer files -> faster scans, better row groups),
deduplicating by ``event_id`` so re-ingested/overlapping parts never double-count.
It is safe-by-construction: the compacted part is written atomically FIRST, then the
originals are removed only after the replacement is durable.

Retention optionally prunes BRONZE partitions older than ``retention_days``. Bronze
holds analytical raw events, NOT the authoritative financial journal (that lives in
cengine/journal.py and is never touched here). Retention defaults to keep-forever and
logs every pruned partition — it never silently deletes financial evidence.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import duckdb
import pyarrow.parquet as pq

from cengine.lakehouse.config import LakehouseConfig
from cengine.lakehouse.parquet_writer import ParquetWriter

MAINTENANCE_VERSION = "lakehouse-maintenance-1.0.0"


@dataclass
class MaintenanceStats:
    compacted_partitions: int = 0
    files_before: int = 0
    files_after: int = 0
    rows_in: int = 0
    rows_out: int = 0
    pruned_partitions: int = 0
    pruned_files: int = 0
    errors: list = field(default_factory=list)


class LakehouseMaintenance:
    def __init__(self, config: Optional[LakehouseConfig] = None) -> None:
        self.config = config or LakehouseConfig.from_env()
        self.writer = ParquetWriter(self.config)

    # -- compaction (§24) ----------------------------------------------------
    def compact_layer(self, layer: str, *, min_files: Optional[int] = None,
                      dedup_key: str = "event_id") -> MaintenanceStats:
        """Compact every partition in ``layer`` that has >= ``min_files`` part files."""
        stats = MaintenanceStats()
        min_files = min_files if min_files is not None else self.config.compaction_min_files
        layer_root = self.config.root / layer
        if not layer_root.exists():
            return stats
        # a partition is any leaf dir containing *.parquet part files
        partitions: dict[Path, list[Path]] = {}
        for part in layer_root.rglob("part-*.parquet"):
            partitions.setdefault(part.parent, []).append(part)
        for part_dir, files in partitions.items():
            if len(files) < min_files:
                continue
            try:
                self._compact_partition(layer, part_dir, files, dedup_key, stats)
            except Exception as exc:  # noqa: BLE001 — one bad partition never aborts the sweep
                stats.errors.append({"partition": str(part_dir),
                                     "error": f"{type(exc).__name__}:{exc}"})
        return stats

    def _compact_partition(self, layer: str, part_dir: Path, files: list,
                           dedup_key: str, stats: MaintenanceStats) -> None:
        dataset = self._dataset_of(layer, part_dir)
        partition = self._partition_of(layer, dataset, part_dir)
        glob = str(part_dir / "part-*.parquet").replace("'", "''")
        con = duckdb.connect()
        try:
            cols = con.sql(f"SELECT * FROM read_parquet('{glob}', union_by_name=true) "
                           f"LIMIT 0").to_arrow_table().schema.names
            if dedup_key in cols:
                sql = (f"SELECT * FROM read_parquet('{glob}', union_by_name=true) "
                       f"QUALIFY row_number() OVER (PARTITION BY {dedup_key} "
                       f"ORDER BY {dedup_key}) = 1")
            else:
                sql = f"SELECT * FROM read_parquet('{glob}', union_by_name=true)"
            table = con.sql(sql).to_arrow_table()
        finally:
            con.close()
        rows_in = sum(pq.read_metadata(f).num_rows for f in files)
        # write the compacted part (atomic, content-hashed, manifest-appended)
        result = self.writer.write(layer, dataset, table, partition=partition)
        if not result.ok:
            stats.errors.append({"partition": str(part_dir), "error": result.reason})
            return
        # remove the originals only now that the replacement is durable
        removed = 0
        new_part = Path(result.path) if result.path else None
        for f in files:
            if new_part is not None and f.resolve() == new_part.resolve():
                continue  # never delete the file we just wrote
            try:
                f.unlink()
                removed += 1
            except OSError:
                pass
        stats.compacted_partitions += 1
        stats.files_before += len(files)
        stats.files_after += 1
        stats.rows_in += rows_in
        stats.rows_out += table.num_rows
        self.writer._append_manifest(layer, dataset, new_part or part_dir,  # noqa: SLF001
                                     table.num_rows, result.checksum or "",
                                     {**(partition or {}), "_compaction": "true",
                                      "_files_merged": removed + 1})

    def _dataset_of(self, layer: str, part_dir: Path) -> str:
        rel = part_dir.relative_to(self.config.root / layer)
        return rel.parts[0] if rel.parts else ""

    def _partition_of(self, layer: str, dataset: str, part_dir: Path) -> dict:
        rel = part_dir.relative_to(self.config.root / layer / dataset)
        partition = {}
        for seg in rel.parts:
            if "=" in seg:
                k, v = seg.split("=", 1)
                partition[k] = v
        return partition

    # -- retention (§4) ------------------------------------------------------
    def prune_bronze(self, *, retention_days: Optional[int] = None,
                     now_ns: Optional[int] = None) -> MaintenanceStats:
        """Prune BRONZE ``trade_date=`` partitions older than the retention window.

        Keep-forever when retention_days <= 0. Logs every pruned partition. Never
        touches silver/gold or the authoritative journal.
        """
        stats = MaintenanceStats()
        days = retention_days if retention_days is not None else self.config.retention_days
        if not days or days <= 0:
            return stats
        now_ns = now_ns if now_ns is not None else time.time_ns()
        cutoff = (datetime.fromtimestamp(now_ns / 1e9, tz=timezone.utc)
                  - timedelta(days=days)).strftime("%Y-%m-%d")
        bronze = self.config.bronze
        if not bronze.exists():
            return stats
        for td_dir in bronze.rglob("trade_date=*"):
            if not td_dir.is_dir():
                continue
            td = td_dir.name.split("=", 1)[1]
            if td < cutoff:
                files = list(td_dir.rglob("*.parquet"))
                for f in files:
                    try:
                        f.unlink()
                        stats.pruned_files += 1
                    except OSError:
                        pass
                stats.pruned_partitions += 1
                self._log_prune(td_dir, td, cutoff, len(files))
        return stats

    def _log_prune(self, td_dir: Path, trade_date: str, cutoff: str, files: int) -> None:
        log = self.config.manifests / "retention_prune.jsonl"
        import json
        try:
            with open(log, "a") as fh:
                fh.write(json.dumps({"partition": str(td_dir), "trade_date": trade_date,
                                     "cutoff": cutoff, "files_pruned": files,
                                     "pruned_ns": time.time_ns(),
                                     "layer": "bronze"}) + "\n")
        except OSError:
            pass

    def run(self) -> dict:
        """Run compaction across bronze/silver/gold + bronze retention. Idempotent."""
        out = {"version": MAINTENANCE_VERSION}
        for layer in ("bronze", "silver", "gold"):
            s = self.compact_layer(layer)
            out[f"compact_{layer}"] = {
                "partitions": s.compacted_partitions, "files_before": s.files_before,
                "files_after": s.files_after, "rows_in": s.rows_in,
                "rows_out": s.rows_out, "errors": s.errors}
        p = self.prune_bronze()
        out["retention"] = {"pruned_partitions": p.pruned_partitions,
                            "pruned_files": p.pruned_files}
        return out
