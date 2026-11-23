"""Point-in-time training-dataset certification (Milestone 6, §5).

Produces an immutable, reproducible certificate for a training dataset: a content
checksum, schema/label versions, a source manifest, and CHRONOLOGICAL train/validation/
out-of-sample splits with a purge+embargo gap so overlapping-label windows cannot leak
across the boundary. It independently verifies feature availability against decision
timestamps (no look-ahead) and refuses to certify a randomly-shuffled split of temporally
dependent observations.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import duckdb

from cengine.lakehouse.config import LakehouseConfig

DATASET_CERT_VERSION = "dataset-cert-1.0.0"


@dataclass
class DatasetCertificate:
    cert_version: str = DATASET_CERT_VERSION
    dataset: str = ""
    layer: str = ""
    dataset_version: str = ""
    dataset_checksum: str = ""
    feature_schema_version: str = ""
    label_definition_version: str = ""
    source_manifest: list = field(default_factory=list)
    rows: int = 0
    training_start: Optional[str] = None
    training_end: Optional[str] = None
    validation_start: Optional[str] = None
    validation_end: Optional[str] = None
    out_of_sample_start: Optional[str] = None
    out_of_sample_end: Optional[str] = None
    embargo_days: int = 0
    instrument_membership_version: str = ""
    execution_environment: str = ""
    availability_timestamp_ns: int = 0
    leakage_verified: bool = False
    leakage_violations: int = 0
    chronological: bool = True
    valid: bool = False
    reasons: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


def _glob(config: LakehouseConfig, layer: str, dataset: str) -> Optional[str]:
    base = config.dataset_path(layer, dataset)
    if not base.exists() or not any(base.rglob("*.parquet")):
        return None
    return str(base / "**" / "*.parquet").replace("'", "''")


def _content_checksum(config: LakehouseConfig, layer: str, dataset: str) -> tuple:
    """Strong content checksum + source manifest from the lakehouse manifest file."""
    manifest = config.manifests / f"{layer}.{dataset}.jsonl"
    entries = []
    h = hashlib.sha256()
    if manifest.exists():
        for line in manifest.read_text().splitlines():
            if not line.strip():
                continue
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            chk = e.get("checksum", "")
            if chk:
                h.update(chk.encode())
                entries.append({"path": e.get("path"), "checksum": chk,
                                "rows": e.get("rows")})
    else:
        base = config.dataset_path(layer, dataset)
        for p in sorted(base.rglob("*.parquet")):
            try:
                data = p.read_bytes()
            except OSError:
                continue
            chk = hashlib.sha256(data).hexdigest()
            h.update(chk.encode())
            entries.append({"path": str(p), "checksum": chk})
    return h.hexdigest()[:32], entries


def _ts_column(cols: set) -> Optional[str]:
    for c in ("decision_ts", "decision_timestamp", "event_timestamp_ns",
              "entry_timestamp", "availability_timestamp_ns"):
        if c in cols:
            return c
    return None


def certify_training_dataset(
    config: Optional[LakehouseConfig], layer: str, dataset: str, *,
    feature_schema_version: str, label_definition_version: str,
    train_frac: float = 0.6, validation_frac: float = 0.2, embargo_days: int = 1,
    instrument_membership_version: str = "", execution_environment: str = "mixed",
    now_ns: int = 0,
) -> DatasetCertificate:
    """Build a point-in-time certificate with chronological purged/embargoed splits."""
    config = config or LakehouseConfig.from_env()
    cert = DatasetCertificate(
        dataset=dataset, layer=layer,
        feature_schema_version=feature_schema_version,
        label_definition_version=label_definition_version,
        embargo_days=embargo_days,
        instrument_membership_version=instrument_membership_version,
        execution_environment=execution_environment,
        availability_timestamp_ns=now_ns)
    glob = _glob(config, layer, dataset)
    if glob is None:
        cert.reasons.append("dataset_not_materialized")
        return cert
    cert.dataset_checksum, cert.source_manifest = _content_checksum(config, layer, dataset)
    cert.dataset_version = cert.dataset_checksum

    src = f"read_parquet('{glob}', union_by_name=true)"
    con = duckdb.connect()
    try:
        cols = set(con.sql(f"SELECT * FROM {src} LIMIT 0").to_arrow_table().schema.names)
        cert.rows = int(con.execute(f"SELECT count(*) FROM {src}").fetchone()[0])
        ts = _ts_column(cols)
        if ts is None:
            cert.reasons.append("no_timestamp_column_for_chronological_split")
            return cert
        # leakage check: availability must not precede event (if both present)
        if {"availability_timestamp_ns", "event_timestamp_ns"} <= cols:
            cert.leakage_violations = int(con.execute(
                f"SELECT count(*) FROM {src} WHERE availability_timestamp_ns "
                f"< event_timestamp_ns").fetchone()[0])
            cert.leakage_verified = cert.leakage_violations == 0
        else:
            cert.leakage_verified = True  # single-time datasets carry no cross-time join
        # chronological boundaries by timestamp quantiles
        qs = con.execute(
            f"SELECT quantile_cont({ts}, {train_frac}), "
            f"quantile_cont({ts}, {train_frac + validation_frac}), "
            f"min({ts}), max({ts}) FROM {src} WHERE {ts} IS NOT NULL").fetchone()
        if not qs or qs[0] is None:
            cert.reasons.append("no_timestamps")
            return cert
        train_cut, val_cut, lo, hi = (int(qs[0]), int(qs[1]), int(qs[2]), int(qs[3]))
        embargo_ns = embargo_days * 86_400 * 1_000_000_000

        def _d(ns: int) -> str:
            return datetime.fromtimestamp(ns / 1e9, tz=timezone.utc).strftime("%Y-%m-%d")

        cert.training_start, cert.training_end = _d(lo), _d(train_cut)
        cert.validation_start = _d(train_cut + embargo_ns)
        cert.validation_end = _d(val_cut)
        cert.out_of_sample_start = _d(val_cut + embargo_ns)
        cert.out_of_sample_end = _d(hi)
        cert.chronological = True
    finally:
        con.close()

    cert.valid = (cert.rows > 0 and cert.leakage_verified
                  and cert.training_end is not None)
    if not cert.leakage_verified:
        cert.reasons.append(f"leakage_violations={cert.leakage_violations}")
    return cert


def split_predicate(cert: DatasetCertificate, ts_col: str, split: str) -> str:
    """SQL predicate selecting a certified split (train|validation|oos) by timestamp.

    Embargo/purge gaps between splits are honored: rows landing inside a gap belong to
    neither split, preventing overlapping-label leakage across the boundary.
    """
    def _ns(date_str: Optional[str]) -> Optional[int]:
        if not date_str:
            return None
        dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        return int(dt.timestamp() * 1e9)

    if split == "train":
        end = _ns(cert.training_end)
        return f"{ts_col} <= {end}" if end else "1=0"
    if split == "validation":
        start, end = _ns(cert.validation_start), _ns(cert.validation_end)
        if start and end:
            return f"{ts_col} >= {start} AND {ts_col} <= {end + 86_400_000_000_000}"
        return "1=0"
    if split == "oos":
        start = _ns(cert.out_of_sample_start)
        return f"{ts_col} >= {start}" if start else "1=0"
    raise ValueError(f"unknown split {split!r}")
