"""Local storage monitoring (Milestone 5, §25).

Reports lakehouse disk footprint, free-space headroom, per-layer / per-dataset sizes
and file counts, compaction pressure (partitions with many small parts), and an
overall health verdict with recommended actions. Pure read-only observation — it
never mutates data and never blocks the trading path.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

from cengine.lakehouse.config import LakehouseConfig

STORAGE_MONITOR_VERSION = "storage-monitor-1.0.0"

HEALTH_OK = "OK"
HEALTH_WARN = "WARN"
HEALTH_CRITICAL = "CRITICAL"


@dataclass
class DatasetUsage:
    layer: str
    dataset: str
    files: int = 0
    bytes: int = 0
    partitions: int = 0
    max_files_in_partition: int = 0


@dataclass
class StorageReport:
    version: str = STORAGE_MONITOR_VERSION
    generated_ns: int = 0
    root: str = ""
    free_disk_gib: float = 0.0
    used_lakehouse_gib: float = 0.0
    min_free_disk_gib: float = 0.0
    max_storage_gib: float = 0.0
    accepts_writes: bool = True
    accepts_reason: str = "ok"
    health: str = HEALTH_OK
    datasets: list = field(default_factory=list)
    compaction_candidates: list = field(default_factory=list)
    recommendations: list = field(default_factory=list)


class StorageMonitor:
    def __init__(self, config: Optional[LakehouseConfig] = None) -> None:
        self.config = config or LakehouseConfig.from_env()

    def report(self) -> StorageReport:
        cfg = self.config
        ok, reason = cfg.accepts_writes()
        rep = StorageReport(
            generated_ns=time.time_ns(), root=str(cfg.root),
            free_disk_gib=round(cfg.free_disk_gib(), 3),
            used_lakehouse_gib=round(cfg.used_gib(), 4),
            min_free_disk_gib=cfg.min_free_disk_gib,
            max_storage_gib=cfg.max_storage_gib,
            accepts_writes=ok, accepts_reason=reason)
        rep.datasets = [self._usage_dict(u) for u in self._scan_datasets()]
        rep.compaction_candidates = [
            {"layer": d["layer"], "dataset": d["dataset"],
             "max_files_in_partition": d["max_files_in_partition"]}
            for d in rep.datasets
            if d["max_files_in_partition"] >= cfg.compaction_min_files]
        rep.health, rep.recommendations = self._verdict(rep)
        return rep

    def _scan_datasets(self) -> list:
        usages = []
        for layer in ("bronze", "silver", "gold", "quarantine"):
            layer_root = self.config.root / layer
            if not layer_root.exists():
                continue
            for ds_dir in sorted(p for p in layer_root.iterdir() if p.is_dir()):
                u = DatasetUsage(layer=layer, dataset=ds_dir.name)
                per_partition: dict = {}
                for part in ds_dir.rglob("part-*.parquet"):
                    u.files += 1
                    try:
                        u.bytes += part.stat().st_size
                    except OSError:
                        pass
                    per_partition[part.parent] = per_partition.get(part.parent, 0) + 1
                u.partitions = len(per_partition)
                u.max_files_in_partition = max(per_partition.values(), default=0)
                if u.files:
                    usages.append(u)
        return usages

    @staticmethod
    def _usage_dict(u: DatasetUsage) -> dict:
        return {"layer": u.layer, "dataset": u.dataset, "files": u.files,
                "mib": round(u.bytes / (1024 ** 2), 3), "partitions": u.partitions,
                "max_files_in_partition": u.max_files_in_partition}

    def _verdict(self, rep: StorageReport) -> tuple:
        recs = []
        health = HEALTH_OK
        if not rep.accepts_writes:
            health = HEALTH_CRITICAL
            recs.append(f"analytical writes are HALTED ({rep.accepts_reason}); "
                        "free disk or raise the cap — order path is unaffected")
        elif rep.free_disk_gib < rep.min_free_disk_gib * 2:
            health = HEALTH_WARN
            recs.append(f"free disk {rep.free_disk_gib:.1f}GiB approaching floor "
                        f"{rep.min_free_disk_gib}GiB; consider retention/compaction "
                        "or an external SSD via CENGINE_LAKEHOUSE_ROOT")
        if (rep.max_storage_gib > 0
                and rep.used_lakehouse_gib >= rep.max_storage_gib * 0.8):
            health = HEALTH_WARN if health == HEALTH_OK else health
            recs.append(f"lakehouse at {rep.used_lakehouse_gib:.1f}GiB of "
                        f"{rep.max_storage_gib}GiB cap; enable retention/compaction")
        if rep.compaction_candidates:
            names = ", ".join(f"{c['layer']}/{c['dataset']}"
                              for c in rep.compaction_candidates[:5])
            recs.append(f"compaction recommended for: {names}")
        return health, recs
