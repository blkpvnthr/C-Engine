"""Silver layer: validated, normalized, typed datasets from bronze (§11/§12/§13-16).

DuckDB SQL reads bronze Parquet (the canonical envelope + JSON payload) and projects
typed, queryable columns into per-concern silver datasets, preserving
DEFAULT_PRICE_SCALE=10000 for monetary fields. Deduplicated by ``event_id``;
incremental via a per-dataset checkpoint (``availability_timestamp_ns`` high-water).
Observed values are kept distinct from inferred L1 microstructure features; no L2
depth / aggressor / queue is ever fabricated.
"""

from __future__ import annotations

import json
from typing import Optional

import duckdb

from cengine.lakehouse.config import LakehouseConfig
from cengine.lakehouse.parquet_writer import ParquetWriter

# silver dataset -> (bronze dataset, [event_types], {column: json path or envelope col})
# A value starting with "$." is extracted from payload_json; otherwise it is an
# envelope column copied through.
_TRANSFORMS = {
    "strategy_recommendations": ("engine_decisions", ["strategy_recommendation"], {
        "decision_id": "causation_id", "symbol_id": "symbol_id",
        "strategy_family": "strategy_family", "strategy_version": "strategy_version",
        "regime_version": "regime_version", "feature_version": "feature_version",
        "exposure_direction": "$.exposure_direction", "eligibility": "$.eligibility",
        "rejection_reason": "$.rejection_reason", "confidence": "$.confidence",
        "expected_gross_return": "$.expected_gross_return",
        "estimated_transaction_cost": "$.estimated_transaction_cost",
        "expected_net_return": "$.expected_net_return", "rank_score": "$.rank_score",
        "rank_basis": "$.rank_basis"}),
    "strategy_rejections": ("engine_decisions", ["rejected_opportunity"], {
        "decision_id": "causation_id", "symbol_id": "symbol_id",
        "strategy_family": "strategy_family", "rejection_reason": "$.rejection_reason"}),
    "market_regimes": ("engine_decisions", ["regime_transition"], {
        "symbol_id": "symbol_id", "regime_version": "regime_version",
        "level": "$.level", "direction": "$.direction", "trend": "$.trend",
        "volatility": "$.volatility", "liquidity": "$.liquidity",
        "confidence": "$.confidence", "previous_direction": "$.previous_direction"}),
    "portfolio_decisions": ("portfolio_decisions", ["portfolio_recommendation"], {
        "decision_id": "causation_id", "symbol_id": "symbol_id",
        "portfolio_decision_id": "portfolio_decision_id",
        "source_strategy_decision_id": "$.source_strategy_decision_id",
        "target_weight": "$.target_weight", "current_weight": "$.current_weight",
        "requested_quantity": "$.requested_quantity",
        "capital_required_ticks": "$.capital_required_ticks",
        "expected_risk_contribution": "$.expected_risk_contribution",
        "allocation_policy": "$.allocation_policy", "return_basis": "$.return_basis",
        "constraint_status": "$.constraint_status", "eligibility": "$.eligibility"}),
    "execution_plans": ("execution_plans", ["execution_plan", "execution_plan_rejected"], {
        "execution_plan_id": "execution_plan_id", "symbol_id": "symbol_id",
        "portfolio_decision_id": "portfolio_decision_id",
        "side": "$.side", "execution_policy": "$.execution_policy", "urgency": "$.urgency",
        "limit_price_ticks": "$.limit_price_ticks",
        "decision_price_ticks": "$.decision_price_ticks",
        "observed_bid_ticks": "$.observed_bid_ticks", "observed_ask_ticks": "$.observed_ask_ticks",
        "spread_bps": "$.spread_bps",
        "estimated_transaction_cost_bps": "$.estimated_transaction_cost_bps",
        "estimated_slippage_bps": "$.estimated_slippage_bps",
        "rejection_reason": "$.rejection_reason"}),
    "microstructure_features": ("execution_plans", ["microstructure_snapshot"], {
        "symbol_id": "symbol_id", "feature_version": "feature_version",
        "spread_bps": "$.spread_bps", "quote_imbalance": "$.quote_imbalance",
        "order_flow_imbalance": "$.order_flow_imbalance",
        "microprice_displacement_bps": "$.microprice_displacement_bps",
        "kyle_lambda": "$.kyle_lambda", "amihud_illiquidity": "$.amihud_illiquidity",
        "trade_arrival_intensity": "$.trade_arrival_intensity",
        "adverse_selection_bps": "$.adverse_selection_bps", "quality": "$.quality",
        "is_stale": "$.is_stale", "crossed_or_locked": "$.crossed_or_locked",
        "confidence": "$.confidence"}),
    "transaction_costs": ("execution_plans", ["transaction_cost_analysis"], {
        "decision_id": "causation_id", "symbol_id": "symbol_id", "side": "$.side",
        "implementation_shortfall_bps": "$.implementation_shortfall_bps",
        "realized_slippage_bps": "$.realized_slippage_bps",
        "expected_slippage_bps": "$.expected_slippage_bps",
        "adverse_price_movement_bps": "$.adverse_price_movement_bps",
        "filled_quantity": "$.filled_quantity", "unfilled_quantity": "$.unfilled_quantity",
        "average_fill_price_ticks": "$.average_fill_price_ticks"}),
    "orders": ("order_events", ["order_intent", "broker_ack"], {
        "decision_id": "causation_id", "symbol_id": "symbol_id", "event_type": "event_type",
        "correlation_id": "$.correlation_id", "client_order_id": "$.client_order_id",
        "venue_order_id": "$.venue_order_id", "side": "$.side", "quantity": "$.quantity",
        "order_type": "$.order_type", "status": "$.status"}),
    "fills": ("fill_events", ["partial_fill", "complete_fill"], {
        "decision_id": "causation_id", "symbol_id": "symbol_id", "event_type": "event_type",
        "correlation_id": "$.correlation_id", "venue_order_id": "$.venue_order_id",
        "cumulative_filled_quantity": "$.cumulative_filled_quantity",
        "last_fill_quantity": "$.last_fill_quantity",
        "last_fill_price_ticks": "$.last_fill_price_ticks",
        "average_fill_price_ticks": "$.average_fill_price_ticks"}),
    # adaptive learning evidence (Milestone 6, §32)
    "predictions": ("learning_events", ["prediction"], {
        "symbol_id": "symbol_id", "task": "$.task", "prediction": "$.prediction",
        "prediction_uncertainty": "$.prediction_uncertainty",
        "applicability": "$.applicability", "data_quality": "$.data_quality",
        "model_id": "$.model_id", "model_version": "$.model_version",
        "model_maturity": "$.model_maturity", "device": "$.device",
        "abstention_reason": "$.abstention_reason"}),
    "shadow_recommendations": ("learning_events", ["shadow_recommendation"], {
        "symbol_id": "symbol_id", "model_id": "$.model_id",
        "model_version": "$.model_version", "champion_score": "$.champion_score",
        "challenger_score": "$.challenger_score", "selected": "$.selected",
        "rank_basis": "$.rank_basis"}),
    "model_lifecycle": ("model_events", ["training_result", "model_promotion",
                                         "model_rollback", "drift_detection"], {
        "event_type": "event_type", "model_id": "$.model_id",
        "model_version": "$.model_version", "to_maturity": "$.to_maturity",
        "from_maturity": "$.from_maturity", "approved": "$.approved",
        "to_reference": "$.to_reference", "reason": "$.reason"}),
    # dynamic universe management (§9): per-symbol selection decisions
    "universe_selection": ("universe_selection",
                           ["universe_promotion", "universe_demotion",
                            "universe_selection"], {
        "event_type": "event_type", "symbol_id": "symbol_id",
        "generation": "$.generation", "is_baseline": "$.is_baseline",
        "opportunity_score": "$.opportunity_score", "model_version": "$.model_version",
        "selection_reason": "$.selection_reason", "removal_reason": "$.removal_reason"}),
}


class SilverTransformer:
    def __init__(self, config: Optional[LakehouseConfig] = None) -> None:
        self.config = config or LakehouseConfig.from_env()
        self.writer = ParquetWriter(self.config)
        self._ckpt = self.config.checkpoints / "silver_checkpoints.json"

    def _load_checkpoints(self) -> dict:
        if self._ckpt.exists():
            try:
                return json.loads(self._ckpt.read_text())
            except (OSError, json.JSONDecodeError):
                return {}
        return {}

    def _save_checkpoints(self, ckpts: dict) -> None:
        tmp = self._ckpt.with_suffix(".tmp")
        tmp.write_text(json.dumps(ckpts, indent=2))
        tmp.replace(self._ckpt)

    def transform(self, silver_dataset: str) -> dict:
        bronze_dataset, event_types, cols = _TRANSFORMS[silver_dataset]
        base = self.config.dataset_path("bronze", bronze_dataset)
        if not base.exists() or not any(base.rglob("*.parquet")):
            return {"dataset": silver_dataset, "rows": 0, "reason": "no_bronze"}
        ckpts = self._load_checkpoints()
        since = int(ckpts.get(silver_dataset, 0))
        glob = str(base / "**" / "*.parquet").replace("'", "''")
        types = ",".join(f"'{t}'" for t in event_types)
        selects = ["event_id", "availability_timestamp_ns", "event_timestamp_ns", "source_feed"]
        for col, src in cols.items():
            if src.startswith("$."):
                selects.append(f"json_extract_string(payload_json, '{src}') AS {col}")
            elif col == src:
                selects.append(col)
            else:
                selects.append(f"{src} AS {col}")
        sql = (f"SELECT {', '.join(selects)} FROM read_parquet('{glob}', union_by_name=true) "
               f"WHERE event_type IN ({types}) AND availability_timestamp_ns > {since} "
               f"QUALIFY row_number() OVER (PARTITION BY event_id "
               f"ORDER BY availability_timestamp_ns) = 1")
        con = duckdb.connect()
        try:
            table = con.sql(sql).to_arrow_table()
        finally:
            con.close()
        if table.num_rows == 0:
            return {"dataset": silver_dataset, "rows": 0, "reason": "up_to_date"}
        rows = table.to_pylist()
        results = self.writer.write_rows("silver", silver_dataset, rows)
        written = sum(r.rows for r in results if r.ok)
        max_ts = max(int(r["availability_timestamp_ns"]) for r in rows)
        ckpts[silver_dataset] = max_ts
        self._save_checkpoints(ckpts)
        return {"dataset": silver_dataset, "rows": written, "checkpoint_ns": max_ts}

    def run_all(self) -> dict:
        return {name: self.transform(name) for name in _TRANSFORMS}
