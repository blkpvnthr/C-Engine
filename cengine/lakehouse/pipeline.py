"""Lakehouse batch pipeline orchestration (Milestone 5, §27).

Chains the post-session analytical stages into one idempotent, resumable run:

    drain decision events -> bronze -> silver -> gold -> data quality ->
    attribution -> maintenance (compaction + retention) -> storage report ->
    DuckDB view registration

Every stage is incremental (silver/gold use high-water checkpoints; Parquet parts are
content-hashed so re-ingestion is idempotent) and failure-isolated — one stage raising
never aborts the rest, and never touches the order path or the authoritative journal.
A run record is appended to ``_checkpoints/pipeline_runs.jsonl`` for auditability.
"""

from __future__ import annotations

import json
import time
from typing import Optional

from cengine.lakehouse.attribution import DailyAttribution
from cengine.lakehouse.config import LakehouseConfig
from cengine.lakehouse.gold import GoldBuilder
from cengine.lakehouse.maintenance import LakehouseMaintenance
from cengine.lakehouse.parquet_writer import ParquetWriter
from cengine.lakehouse.quality import DataQualityValidator
from cengine.lakehouse.silver import SilverTransformer
from cengine.lakehouse.storage_monitor import StorageMonitor
from cengine.lakehouse.warehouse import DuckDBWarehouse

PIPELINE_VERSION = "lakehouse-pipeline-1.0.0"


class LakehousePipeline:
    """Orchestrates the batch analytical stages. Zero required args — all state lives
    in the configured lakehouse root so re-runs are safe and resumable."""

    def __init__(self, config: Optional[LakehouseConfig] = None) -> None:
        self.config = config or LakehouseConfig.from_env()
        self.config.ensure_dirs()
        self.writer = ParquetWriter(self.config)

    def _stage(self, out: dict, name: str, fn) -> None:
        try:
            out[name] = fn()
        except Exception as exc:  # noqa: BLE001 — stage isolation; never abort the pipeline
            out[name] = {"error": f"{type(exc).__name__}:{exc}"}

    def run(self, *, decision_sink=None, compact: bool = True,
            now_ns: Optional[int] = None) -> dict:
        """Run the full batch pipeline. If ``decision_sink`` is given, its buffered
        rows are drained to bronze first (post-session); otherwise bronze is assumed
        already populated by the live analytical sink."""
        now_ns = now_ns if now_ns is not None else time.time_ns()
        out: dict = {"pipeline": PIPELINE_VERSION, "started_ns": now_ns}

        if decision_sink is not None:
            from cengine.lakehouse.bronze import AnalyticalIngestionSink
            sink = AnalyticalIngestionSink(self.config)
            self._stage(out, "bronze_drain",
                        lambda: _stats_dict(sink.drain_decision_sink(decision_sink)))

        self._stage(out, "silver", lambda: SilverTransformer(self.config).run_all())
        self._stage(out, "gold", lambda: GoldBuilder(self.config).run_all())
        self._stage(out, "quality", self._quality)
        self._stage(out, "attribution", lambda: _compact_attrib(
            DailyAttribution(self.config).attribute(now_ns=now_ns)))
        if compact:
            self._stage(out, "maintenance", lambda: LakehouseMaintenance(self.config).run())
        self._stage(out, "storage", lambda: _storage_dict(
            StorageMonitor(self.config).report()))
        self._stage(out, "warehouse_views",
                    lambda: {"views": DuckDBWarehouse(self.config).register_views()})

        out["completed_ns"] = time.time_ns()
        out["duration_s"] = round((out["completed_ns"] - now_ns) / 1e9, 3)
        self._record_run(out)
        return out

    def _quality(self) -> dict:
        profile = DataQualityValidator(self.config).profile()
        # persist the quality profile so gold/data_quality_results is queryable (§22/§23)
        row = {"report": profile["report"], "version": profile["version"],
               "total_checks": profile["total_checks"], "passed": profile["passed"],
               "failed": profile["failed"],
               "profile_json": json.dumps(profile, default=str),
               "event_timestamp_ns": profile["generated_ns"], "source_feed": "internal"}
        self.writer.write_rows("gold", "data_quality_results", [row])
        return {"total_checks": profile["total_checks"], "passed": profile["passed"],
                "failed": profile["failed"]}

    def _record_run(self, out: dict) -> None:
        path = self.config.checkpoints / "pipeline_runs.jsonl"
        summary = {k: v for k, v in out.items()
                   if k in ("pipeline", "started_ns", "completed_ns", "duration_s")}
        summary["stages"] = {k: ("error" if isinstance(v, dict) and "error" in v else "ok")
                             for k, v in out.items() if isinstance(v, dict)}
        try:
            with open(path, "a") as fh:
                fh.write(json.dumps(summary, default=str) + "\n")
        except OSError:
            pass


def _stats_dict(stats) -> dict:
    return {"ingested": stats.ingested, "written": stats.written,
            "dropped": stats.dropped, "quarantined": stats.quarantined,
            "by_dataset": stats.by_dataset}


def _storage_dict(rep) -> dict:
    return {"health": rep.health, "free_disk_gib": rep.free_disk_gib,
            "used_lakehouse_gib": rep.used_lakehouse_gib,
            "accepts_writes": rep.accepts_writes,
            "compaction_candidates": len(rep.compaction_candidates),
            "recommendations": rep.recommendations}


def _compact_attrib(report: dict) -> dict:
    return {k: v for k, v in report.items()
            if k in ("closed_positions", "winners", "win_rate", "avg_net_return",
                     "sum_net_return", "error")}
