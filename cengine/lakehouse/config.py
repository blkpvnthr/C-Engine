"""Local lakehouse storage configuration (Milestone 5, §4).

Configurable local paths (default ``data/lakehouse/``), overridable via environment
for an external local SSD WITHOUT a code change. Carries compression, retention,
storage limits, row-group/batch sizing and compaction thresholds. No cloud.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path

LAKEHOUSE_SCHEMA_VERSION = "lakehouse-1.0.0"
DEFAULT_PRICE_SCALE = 10_000  # native financial scale; preserved end-to-end (§11)


@dataclass(frozen=True)
class LakehouseConfig:
    root: Path
    compression: str = "zstd"           # Parquet codec; falls back to snappy if absent
    compression_level: int = 3
    row_group_size: int = 128_000
    write_batch_size: int = 10_000
    compaction_min_files: int = 16       # compact a partition once it has >= N parts
    max_storage_gib: float = 0.0         # 0 = unlimited; else warn/stop near the cap
    min_free_disk_gib: float = 5.0       # refuse new writes below this free-space floor
    retention_days: int = 0              # 0 = keep forever; else prune bronze beyond N days
    duckdb_filename: str = "cengine.duckdb"

    @staticmethod
    def from_env() -> "LakehouseConfig":
        root = Path(os.environ.get("CENGINE_LAKEHOUSE_ROOT",
                                   str(Path.cwd() / "data" / "lakehouse"))).expanduser()
        return LakehouseConfig(
            root=root,
            compression=os.environ.get("CENGINE_LAKEHOUSE_COMPRESSION", "zstd"),
            compression_level=int(os.environ.get("CENGINE_LAKEHOUSE_ZSTD_LEVEL", "3")),
            row_group_size=int(os.environ.get("CENGINE_LAKEHOUSE_ROW_GROUP", "128000")),
            write_batch_size=int(os.environ.get("CENGINE_LAKEHOUSE_BATCH", "10000")),
            max_storage_gib=float(os.environ.get("CENGINE_LAKEHOUSE_MAX_GIB", "0")),
            min_free_disk_gib=float(os.environ.get("CENGINE_LAKEHOUSE_MIN_FREE_GIB", "5")),
            retention_days=int(os.environ.get("CENGINE_LAKEHOUSE_RETENTION_DAYS", "0")),
        )

    # -- layer roots ---------------------------------------------------------
    @property
    def bronze(self) -> Path:
        return self.root / "bronze"

    @property
    def silver(self) -> Path:
        return self.root / "silver"

    @property
    def gold(self) -> Path:
        return self.root / "gold"

    @property
    def quarantine(self) -> Path:
        return self.root / "quarantine"

    @property
    def manifests(self) -> Path:
        return self.root / "_manifests"

    @property
    def checkpoints(self) -> Path:
        return self.root / "_checkpoints"

    @property
    def duckdb_path(self) -> Path:
        return self.root / self.duckdb_filename

    def dataset_path(self, layer: str, dataset: str) -> Path:
        return self.root / layer / dataset

    def ensure_dirs(self) -> None:
        for p in (self.bronze, self.silver, self.gold, self.quarantine,
                  self.manifests, self.checkpoints):
            p.mkdir(parents=True, exist_ok=True)

    # -- disk / capacity (§4/§25/§30) ----------------------------------------
    def free_disk_gib(self) -> float:
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            usage = shutil.disk_usage(self.root)
            return usage.free / (1024 ** 3)
        except Exception:  # noqa: BLE001 — never raise into the ingestion path
            return float("inf")

    def used_gib(self) -> float:
        total = 0
        if self.root.exists():
            for p in self.root.rglob("*.parquet"):
                try:
                    total += p.stat().st_size
                except OSError:
                    pass
        return total / (1024 ** 3)

    def accepts_writes(self) -> tuple[bool, str]:
        """Return (ok, reason). Fails CLOSED on low disk / over-cap so analytical
        writes stop gracefully — this NEVER affects the order path."""
        free = self.free_disk_gib()
        if free < self.min_free_disk_gib:
            return False, f"low_disk:{free:.2f}GiB<{self.min_free_disk_gib}GiB"
        if self.max_storage_gib > 0 and self.used_gib() >= self.max_storage_gib:
            return False, f"storage_cap_reached:{self.max_storage_gib}GiB"
        return True, "ok"
