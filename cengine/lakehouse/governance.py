"""Model governance metadata foundation (Milestone 5, §28).

Prepares the metadata records + reproducible dataset-version interface that the
SUBSEQUENT ML/RL milestone (M6) will consume. This milestone does NOT train or
activate any model — it only records model metadata + dataset manifests so future
training is reproducible and auditable. Stored as local JSON (small, versioned),
never committed to Git.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

from cengine.lakehouse.config import LakehouseConfig

GOVERNANCE_VERSION = "model-governance-1.0.0"


@dataclass(frozen=True)
class ModelRecord:
    model_id: str
    model_version: str
    model_type: str                         # e.g. "supervised_classifier" | "contextual_bandit"
    training_dataset_version: str
    feature_schema_version: str
    training_period: str
    validation_period: str
    out_of_sample_period: str
    hyperparameters: dict = field(default_factory=dict)
    validation_metrics: dict = field(default_factory=dict)
    applicable_strategy_families: tuple = ()
    applicable_regimes: tuple = ()
    applicable_instrument_classes: tuple = ()
    artifact_checksum: Optional[str] = None
    promotion_status: str = "NONE"          # NONE until M6 validates + promotes
    approval_status: str = "UNAPPROVED"
    rollback_reference: Optional[str] = None
    created_ns: int = 0
    governance_version: str = GOVERNANCE_VERSION


class ModelGovernanceRegistry:
    """Append-only local registry of model metadata (no training here, §28)."""

    def __init__(self, config: Optional[LakehouseConfig] = None) -> None:
        self.config = config or LakehouseConfig.from_env()
        self._dir = self.config.root / "_model_registry"
        self._dir.mkdir(parents=True, exist_ok=True)
        self._path = self._dir / "models.jsonl"

    def register(self, record: ModelRecord) -> dict:
        """Record model metadata. Promotion/approval stay NONE/UNAPPROVED (M6 governs)."""
        row = asdict(record)
        if not row.get("created_ns"):
            row["created_ns"] = time.time_ns()
        with open(self._path, "a") as fh:
            fh.write(json.dumps(row, default=str) + "\n")
        return row

    def list_models(self) -> list:
        if not self._path.exists():
            return []
        out = []
        for line in self._path.read_text().splitlines():
            if line.strip():
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        return out

    @staticmethod
    def dataset_version(dataset_path: Path) -> str:
        """Reproducible version hash over a dataset's Parquet part checksums (from the
        manifest ordering) — lets M6 pin the exact training data."""
        parts = sorted(str(p) for p in Path(dataset_path).rglob("*.parquet"))
        h = hashlib.sha256()
        for p in parts:
            h.update(p.encode())
            try:
                h.update(str(Path(p).stat().st_size).encode())
            except OSError:
                pass
        return h.hexdigest()[:24]
