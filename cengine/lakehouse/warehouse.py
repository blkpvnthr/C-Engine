"""DuckDB analytical warehouse (Milestone 5, §22).

A persistent local DuckDB catalog exposing documented Parquet-backed views over the
bronze/silver/gold lakehouse — no duplicate bulk storage of the lakehouse inside
DuckDB. Reads use a short-lived connection; materialization uses a single coordinated
writer. Views tolerate not-yet-materialized datasets (missing globs are skipped).
"""

from __future__ import annotations

from typing import Optional

import duckdb

from cengine.lakehouse.config import LakehouseConfig

# view_name -> (layer, dataset) ; Parquet-backed read_parquet views (§22).
_VIEWS = {
    # bronze
    "engine_decisions": ("bronze", "engine_decisions"),
    "risk_decisions": ("bronze", "risk_decisions"),
    "portfolio_decision_events": ("bronze", "portfolio_decisions"),
    "execution_plan_events": ("bronze", "execution_plans"),
    "order_events": ("bronze", "order_events"),
    "fill_events": ("bronze", "fill_events"),
    "position_events": ("bronze", "position_events"),
    "market_bars": ("bronze", "market_bars"),
    "market_trades": ("bronze", "market_trades"),
    "market_quotes": ("bronze", "market_quotes"),
    "instrument_metadata": ("bronze", "instrument_metadata"),
    "universe_membership": ("bronze", "universe_membership"),
    # silver
    "market_observations": ("silver", "market_observations"),
    "technical_features": ("silver", "technical_features"),
    "microstructure_features": ("silver", "microstructure_features"),
    "market_regimes": ("silver", "market_regimes"),
    "strategy_recommendations": ("silver", "strategy_recommendations"),
    "strategy_rejections": ("silver", "strategy_rejections"),
    "portfolio_decisions": ("silver", "portfolio_decisions"),
    "execution_plans": ("silver", "execution_plans"),
    "orders": ("silver", "orders"),
    "fills": ("silver", "fills"),
    "transaction_costs": ("silver", "transaction_costs"),
    # gold
    "position_outcomes": ("gold", "position_outcomes"),
    "supervised_training_datasets": ("gold", "supervised"),
    "offline_rl_transitions": ("gold", "offline_rl"),
    "daily_performance": ("gold", "daily_performance"),
    "strategy_performance": ("gold", "strategy_performance"),
    "regime_performance": ("gold", "regime_performance"),
    "execution_policy_performance": ("gold", "execution_policy_performance"),
    "portfolio_performance": ("gold", "portfolio_performance"),
    "data_quality_results": ("gold", "data_quality_results"),
    # adaptive learning platform (Milestone 6)
    "learning_events": ("bronze", "learning_events"),
    "model_events": ("bronze", "model_events"),
    "predictions": ("silver", "predictions"),
    "shadow_recommendations": ("silver", "shadow_recommendations"),
    "model_lifecycle": ("silver", "model_lifecycle"),
}


class DuckDBWarehouse:
    def __init__(self, config: Optional[LakehouseConfig] = None) -> None:
        self.config = config or LakehouseConfig.from_env()
        self.config.ensure_dirs()

    def _glob(self, layer: str, dataset: str) -> str:
        return str(self.config.dataset_path(layer, dataset) / "**" / "*.parquet")

    def _has_parts(self, layer: str, dataset: str) -> bool:
        base = self.config.dataset_path(layer, dataset)
        return base.exists() and any(base.rglob("*.parquet"))

    def connect(self, *, read_only: bool = False) -> duckdb.DuckDBPyConnection:
        return duckdb.connect(str(self.config.duckdb_path), read_only=read_only)

    def register_views(self) -> list[str]:
        """(Re)create views over materialized datasets. Returns the view names created."""
        created = []
        con = self.connect()
        try:
            for view, (layer, dataset) in _VIEWS.items():
                if not self._has_parts(layer, dataset):
                    continue
                glob = self._glob(layer, dataset).replace("'", "''")
                con.execute(f"CREATE OR REPLACE VIEW {view} AS "
                            f"SELECT * FROM read_parquet('{glob}', union_by_name=true, "
                            f"filename=true)")
                created.append(view)
        finally:
            con.close()
        return created

    def query(self, sql: str) -> list:
        con = self.connect(read_only=False)
        try:
            self._register_into(con)
            return con.execute(sql).fetchall()
        finally:
            con.close()

    def query_df(self, sql: str):
        con = self.connect(read_only=False)
        try:
            self._register_into(con)
            return con.sql(sql).to_arrow_table()
        finally:
            con.close()

    def _register_into(self, con) -> None:
        for view, (layer, dataset) in _VIEWS.items():
            if self._has_parts(layer, dataset):
                glob = self._glob(layer, dataset).replace("'", "''")
                con.execute(f"CREATE OR REPLACE VIEW {view} AS "
                            f"SELECT * FROM read_parquet('{glob}', union_by_name=true)")

    def table_counts(self) -> dict:
        counts = {}
        con = self.connect()
        try:
            for view, (layer, dataset) in _VIEWS.items():
                if not self._has_parts(layer, dataset):
                    continue
                glob = self._glob(layer, dataset).replace("'", "''")
                try:
                    n = con.execute(f"SELECT count(*) FROM read_parquet('{glob}', "
                                    f"union_by_name=true)").fetchone()[0]
                    counts[view] = int(n)
                except Exception:  # noqa: BLE001
                    counts[view] = -1
        finally:
            con.close()
        return counts
