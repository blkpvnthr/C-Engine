"""Operational learning-model registry (Milestone 6, §22/§23 + addendum §18).

Extends the M5 lakehouse governance metadata into a full operational registry: rich,
GPU-aware ModelCards with reproducibility provenance + maturity, atomic checksummed
artifact publication, authorized maturity transitions, champion/challenger selection, and
rollback to the last approved model or the deterministic baseline.

Storage is an append-only JSONL ledger (one line per state revision — cards are never
mutated in place) at ``<root>/_learning_models/registry.jsonl`` plus checksummed artifact
blobs alongside. Nothing here submits orders or grants financial authority; promotion to
live requires explicit approval (see maturity.check_transition).
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Optional

from cengine.lakehouse.config import LakehouseConfig
from cengine.lakehouse.governance import ModelGovernanceRegistry, ModelRecord
from cengine.learning.registry.artifacts import ArtifactStore
from cengine.learning.registry.maturity import (
    ModelMaturity,
    check_transition,
)

REGISTRY_VERSION = "learning-model-registry-1.0.0"


@dataclass
class ModelCard:
    # identity + reproducibility (§22)
    model_id: str
    model_version: str
    model_type: str                         # e.g. 'forward_return_regressor'
    model_architecture: str = ""            # e.g. 'mlx_mlp[13-64-32-1]' | 'sklearn_hgbr'
    framework: str = ""                     # 'mlx' | 'sklearn' | 'numpy'
    framework_version: str = ""
    training_device: str = "cpu"            # 'gpu' (Metal) | 'cpu'
    inference_device: str = "cpu"
    numerical_precision: str = "float32"
    dataset_version: str = ""
    dataset_checksum: str = ""
    feature_schema_version: str = ""
    label_definition_version: str = ""
    training_period: str = ""
    validation_period: str = ""
    out_of_sample_period: str = ""
    hyperparameters: dict = field(default_factory=dict)
    random_seed: Optional[int] = None
    software_versions: dict = field(default_factory=dict)
    # evaluation
    validation_metrics: dict = field(default_factory=dict)
    calibration_metrics: dict = field(default_factory=dict)
    economic_metrics: dict = field(default_factory=dict)
    hardware_benchmark_reference: str = ""
    # applicability
    applicable_regimes: tuple = ()
    applicable_instrument_classes: tuple = ()
    applicable_strategy_families: tuple = ()
    execution_environment: str = "research"
    # artifacts (name -> ArtifactRef dict)
    artifacts: dict = field(default_factory=dict)
    artifact_checksum: str = ""
    # governance / lifecycle
    maturity: str = ModelMaturity.UNTRAINED.value
    promotion_status: str = "NONE"
    approval_status: str = "UNAPPROVED"
    rollback_reference: Optional[str] = None
    revision: int = 0
    created_ns: int = 0
    transition_reason: str = ""
    card_version: str = REGISTRY_VERSION

    def to_dict(self) -> dict:
        return asdict(self)

    def to_model_record(self) -> ModelRecord:
        """Project onto the M5 governance ModelRecord for continuity."""
        return ModelRecord(
            model_id=self.model_id, model_version=self.model_version,
            model_type=self.model_type,
            training_dataset_version=self.dataset_version,
            feature_schema_version=self.feature_schema_version,
            training_period=self.training_period, validation_period=self.validation_period,
            out_of_sample_period=self.out_of_sample_period,
            hyperparameters=self.hyperparameters,
            validation_metrics=self.validation_metrics,
            applicable_strategy_families=self.applicable_strategy_families,
            applicable_regimes=self.applicable_regimes,
            applicable_instrument_classes=self.applicable_instrument_classes,
            artifact_checksum=self.artifact_checksum or None,
            promotion_status=self.promotion_status, approval_status=self.approval_status,
            rollback_reference=self.rollback_reference, created_ns=self.created_ns)


class LearningModelRegistry:
    def __init__(self, config: Optional[LakehouseConfig] = None) -> None:
        self.config = config or LakehouseConfig.from_env()
        self.artifacts = ArtifactStore(self.config)
        self._dir = self.config.root / "_learning_models"
        self._dir.mkdir(parents=True, exist_ok=True)
        self._path = self._dir / "registry.jsonl"
        self._governance = ModelGovernanceRegistry(self.config)

    # -- persistence (append-only, atomic-ish) --------------------------------
    def _append(self, card: ModelCard) -> dict:
        row = card.to_dict()
        line = json.dumps(row, default=str) + "\n"
        # append then fsync for durability (JSONL append is the ledger of record)
        with open(self._path, "a") as fh:
            fh.write(line)
            fh.flush()
            os.fsync(fh.fileno())
        # mirror to M5 governance metadata for continuity
        try:
            self._governance.register(card.to_model_record())
        except Exception:  # noqa: BLE001 — governance mirror is best-effort
            pass
        return row

    def _all_rows(self) -> list:
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

    # -- registration ---------------------------------------------------------
    def register(self, card: ModelCard) -> ModelCard:
        """Register (or re-revision) a model card. created_ns stamped if unset."""
        if not card.created_ns:
            card.created_ns = time.time_ns()
        # combined artifact checksum over all attached artifact checksums
        if card.artifacts and not card.artifact_checksum:
            import hashlib
            h = hashlib.sha256()
            for name in sorted(card.artifacts):
                chk = (card.artifacts[name] or {}).get("checksum", "")
                h.update(f"{name}:{chk}".encode())
            card.artifact_checksum = h.hexdigest()[:24]
        self._append(card)
        return card

    # -- queries --------------------------------------------------------------
    def latest(self, model_id: str) -> Optional[dict]:
        rows = [r for r in self._all_rows() if r.get("model_id") == model_id]
        return rows[-1] if rows else None

    def list_latest(self) -> list:
        by_id: dict = {}
        for r in self._all_rows():
            by_id[r.get("model_id")] = r  # last wins
        return list(by_id.values())

    def by_status(self, maturity: ModelMaturity) -> list:
        return [r for r in self.list_latest() if r.get("maturity") == maturity.value]

    def champion(self, model_type: Optional[str] = None) -> Optional[dict]:
        """Latest ACTIVE model (falls back to SHADOW) for a type, else None."""
        def _match(r):
            return model_type is None or r.get("model_type") == model_type
        actives = [r for r in self.by_status(ModelMaturity.ACTIVE) if _match(r)]
        if actives:
            return max(actives, key=lambda r: r.get("created_ns", 0))
        shadows = [r for r in self.by_status(ModelMaturity.SHADOW) if _match(r)]
        return max(shadows, key=lambda r: r.get("created_ns", 0)) if shadows else None

    def challenger(self, model_type: str) -> Optional[dict]:
        """Latest VALIDATED/SHADOW model of a type that is not the champion."""
        champ = self.champion(model_type)
        champ_key = (champ or {}).get("model_id"), (champ or {}).get("model_version")
        cands = [r for r in self.list_latest()
                 if r.get("model_type") == model_type
                 and r.get("maturity") in (ModelMaturity.VALIDATED.value,
                                           ModelMaturity.SHADOW.value)
                 and ((r.get("model_id"), r.get("model_version")) != champ_key)]
        return max(cands, key=lambda r: r.get("created_ns", 0)) if cands else None

    # -- lifecycle transitions ------------------------------------------------
    def transition(self, model_id: str, dst: ModelMaturity, *, approved: bool = False,
                   reason: str = "") -> ModelCard:
        cur = self.latest(model_id)
        if cur is None:
            raise KeyError(f"unknown model_id {model_id!r}")
        src = ModelMaturity(cur.get("maturity", ModelMaturity.UNTRAINED.value))
        check_transition(src, dst, approved=approved)
        card = _card_from_row(cur)
        card.maturity = dst.value
        card.revision = int(cur.get("revision", 0)) + 1
        card.created_ns = time.time_ns()
        card.transition_reason = reason
        if dst in (ModelMaturity.LIVE_APPROVED, ModelMaturity.ACTIVE) and approved:
            card.approval_status = "APPROVED"
            card.promotion_status = "PROMOTED"
        self._append(card)
        return card

    def approve(self, model_id: str, *, reason: str = "operator_approval") -> ModelCard:
        cur = self.latest(model_id)
        if cur is None:
            raise KeyError(f"unknown model_id {model_id!r}")
        card = _card_from_row(cur)
        card.approval_status = "APPROVED"
        card.revision = int(cur.get("revision", 0)) + 1
        card.created_ns = time.time_ns()
        card.transition_reason = reason
        self._append(card)
        return card

    def rollback(self, model_id: str, *, to_reference: str = "baseline",
                 reason: str = "rollback") -> ModelCard:
        """Demote a model to DEGRADED and record the rollback target (last approved
        model or the deterministic baseline). Always legal."""
        cur = self.latest(model_id)
        if cur is None:
            raise KeyError(f"unknown model_id {model_id!r}")
        card = _card_from_row(cur)
        card.maturity = ModelMaturity.DEGRADED.value
        card.rollback_reference = to_reference
        card.revision = int(cur.get("revision", 0)) + 1
        card.created_ns = time.time_ns()
        card.transition_reason = reason
        self._append(card)
        return card


def _card_from_row(row: dict) -> ModelCard:
    known = {f for f in ModelCard.__dataclass_fields__}  # noqa: C416
    return ModelCard(**{k: v for k, v in row.items() if k in known})
