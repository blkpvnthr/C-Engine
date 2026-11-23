"""Point-in-time feature store (Milestone 5, §18).

Backed by silver Parquet + DuckDB ASOF joins. Every training observation uses ONLY
information available at its decision timestamp (``availability_timestamp_ns <=
decision_ts``) — preventing future leakage, look-ahead bias, and label contamination.
Dataset construction is reproducible (deterministic as-of semantics on immutable
Parquet). Feature freshness/completeness are reported, not fabricated.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import duckdb
import pyarrow as pa

from cengine.lakehouse.config import LakehouseConfig

FEATURE_STORE_VERSION = "feature-store-1.0.0"


@dataclass(frozen=True)
class FeatureSet:
    """A silver feature dataset + the columns to pull + its availability timestamp col."""
    name: str                                   # silver dataset name
    columns: tuple
    availability_col: str = "availability_timestamp_ns"
    prefix: str = ""                            # column prefix to avoid collisions


class PointInTimeFeatureStore:
    def __init__(self, config: Optional[LakehouseConfig] = None) -> None:
        self.config = config or LakehouseConfig.from_env()

    def _glob(self, dataset: str) -> Optional[str]:
        base = self.config.dataset_path("silver", dataset)
        if not base.exists() or not any(base.rglob("*.parquet")):
            return None
        return str(base / "**" / "*.parquet").replace("'", "''")

    def as_of(self, symbol: str, as_of_ns: int, feature_set: FeatureSet) -> Optional[dict]:
        """Latest row for ``symbol`` with availability <= as_of_ns (no leakage)."""
        glob = self._glob(feature_set.name)
        if glob is None:
            return None
        cols = ", ".join(feature_set.columns)
        con = duckdb.connect()
        try:
            sql = (f"SELECT {cols}, {feature_set.availability_col} FROM read_parquet('{glob}', "
                   f"union_by_name=true) WHERE symbol_id = ? AND "
                   f"{feature_set.availability_col} <= ? "
                   f"ORDER BY {feature_set.availability_col} DESC LIMIT 1")
            res = con.execute(sql, [symbol, int(as_of_ns)]).fetchone()
            if res is None:
                return None
            names = list(feature_set.columns) + [feature_set.availability_col]
            return dict(zip(names, res, strict=False))
        finally:
            con.close()

    def training_snapshot(self, decisions: pa.Table, feature_sets: list,
                          *, decision_ts_col: str = "decision_ts",
                          symbol_col: str = "symbol_id") -> pa.Table:
        """ASOF-join a table of decisions (symbol + decision_ts) to each feature set,
        pulling only features available at or before each decision's timestamp.

        This is the reproducible, leakage-free training-dataset constructor (§18).
        """
        con = duckdb.connect()
        try:
            con.register("decisions", decisions)
            select_cols = [f"d.{c}" for c in decisions.column_names]
            joins = []
            for i, fs in enumerate(feature_sets):
                glob = self._glob(fs.name)
                if glob is None:
                    continue
                alias = f"f{i}"
                con.execute(f"CREATE OR REPLACE TEMP VIEW {alias}_src AS SELECT * FROM "
                            f"read_parquet('{glob}', union_by_name=true)")
                for col in fs.columns:
                    select_cols.append(f"{alias}.{col} AS {fs.prefix}{col}")
                # ASOF: most recent feature row with availability <= decision_ts
                joins.append(
                    f"ASOF LEFT JOIN {alias}_src {alias} "
                    f"ON d.{symbol_col} = {alias}.symbol_id "
                    f"AND d.{decision_ts_col} >= {alias}.{fs.availability_col}")
            sql = (f"SELECT {', '.join(select_cols)} FROM decisions d " + " ".join(joins))
            return con.sql(sql).to_arrow_table()
        finally:
            con.close()

    def freshness(self, feature_set: FeatureSet, *, now_ns: Optional[int] = None) -> dict:
        """Per-symbol feature freshness (age of the latest available observation)."""
        now_ns = now_ns if now_ns is not None else time.time_ns()
        glob = self._glob(feature_set.name)
        if glob is None:
            return {"dataset": feature_set.name, "available": False}
        con = duckdb.connect()
        try:
            row = con.execute(
                f"SELECT count(distinct symbol_id), max({feature_set.availability_col}), "
                f"count(*) FROM read_parquet('{glob}', union_by_name=true)").fetchone()
            latest = row[1] or 0
            return {"dataset": feature_set.name, "available": True, "symbols": row[0],
                    "rows": row[2], "latest_ns": latest,
                    "staleness_s": max(0.0, (now_ns - latest) / 1e9) if latest else None}
        finally:
            con.close()
