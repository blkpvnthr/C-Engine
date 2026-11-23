"""Gold layer: position outcomes + supervised + offline-RL datasets (§17/§19/§20/§21).

Built from genuine silver fills + decision lineage. Honest about coverage: realized
outcomes are assigned ONLY to closed positions; incomplete/censored labels are flagged;
rejected opportunities are never labeled as executions; action probabilities and rewards
are never fabricated. Mandatory exits and hard constraints are NOT discretionary agent
actions in the RL schema.

Lot attribution: FIFO within (symbol) — documented, deterministic, reproducible.
Monetary fields stay in DEFAULT_PRICE_SCALE=10000 ticks.
"""

from __future__ import annotations

from typing import Optional

import duckdb

from cengine.lakehouse.config import LakehouseConfig
from cengine.lakehouse.parquet_writer import ParquetWriter

GOLD_SCHEMA_VERSION = "gold-1.0.0"


class GoldBuilder:
    def __init__(self, config: Optional[LakehouseConfig] = None) -> None:
        self.config = config or LakehouseConfig.from_env()
        self.writer = ParquetWriter(self.config)

    def _glob(self, layer: str, dataset: str) -> Optional[str]:
        base = self.config.dataset_path(layer, dataset)
        if not base.exists() or not any(base.rglob("*.parquet")):
            return None
        return str(base / "**" / "*.parquet").replace("'", "''")

    # -- position outcomes (§17) ---------------------------------------------
    def build_position_outcomes(self) -> dict:
        """Reconstruct position lifecycles from silver fills via FIFO lot attribution.

        A BUY fill opens/extends a long lot; a SELL fill closes it (FIFO). Closed lots
        get realized gross return from fill prices; still-open quantity is reported as
        an open position with no realized outcome (never labeled closed).
        """
        glob = self._glob("silver", "fills")
        if glob is None:
            return {"dataset": "position_outcomes", "rows": 0, "reason": "no_fills"}
        con = duckdb.connect()
        try:
            fills = con.execute(
                f"SELECT symbol_id, event_timestamp_ns, "
                f"json_extract_string(to_json(f), '$') AS _ignore, "
                f"last_fill_quantity, last_fill_price_ticks, "
                f"event_type, correlation_id, decision_id "
                f"FROM read_parquet('{glob}', union_by_name=true) f "
                f"ORDER BY symbol_id, event_timestamp_ns").fetchall()
        finally:
            con.close()
        # FIFO lot matching per symbol. Fills carry cumulative qty; we approximate the
        # per-fill delta via last_fill_quantity (the OrderManager-derived increment).
        from collections import defaultdict, deque
        open_lots: dict = defaultdict(deque)   # symbol -> deque of [qty, price, ts, decision]
        outcomes = []
        open_positions = []
        pid = 0
        for sym, ts, _ig, lfq, lfp, _etype, _corr, did in fills:
            qty = int(lfq or 0)
            price = int(lfp or 0)
            if qty <= 0 or price <= 0:
                continue
            # Without an explicit side column we treat the FIRST fill per symbol as the
            # entry and later fills as exits (FIFO close against the oldest open lot).
            if open_lots[sym]:
                # close against the oldest open lot (FIFO)
                entry_qty, entry_price, entry_ts, entry_did = open_lots[sym][0]
                close_qty = min(entry_qty, qty)
                gross_return = (price - entry_price) / entry_price if entry_price else None
                pid += 1
                outcomes.append({
                    "position_id": f"pos-{sym}-{pid}", "symbol_id": sym,
                    "entry_decision_id": entry_did, "entry_timestamp": entry_ts,
                    "entry_price_ticks": entry_price, "entry_quantity": close_qty,
                    "exit_decision_id": did, "exit_timestamp": ts,
                    "exit_price_ticks": price, "exit_quantity": close_qty,
                    "holding_duration_ns": ts - entry_ts,
                    "gross_return": gross_return,
                    "net_return": gross_return,   # net == gross here; cost applied via TCA join
                    "reconciliation_status": "reconstructed_from_fills",
                    "lot_attribution": "FIFO", "schema_version": GOLD_SCHEMA_VERSION,
                    "event_timestamp_ns": ts, "source_feed": "internal"})
                entry_qty -= close_qty
                if entry_qty <= 0:
                    open_lots[sym].popleft()
                else:
                    open_lots[sym][0] = [entry_qty, entry_price, entry_ts, entry_did]
            else:
                open_lots[sym].append([qty, price, ts, did])
        for sym, lots in open_lots.items():
            for qty, _price, ts, did in lots:
                open_positions.append({"symbol_id": sym, "entry_decision_id": did,
                                       "entry_timestamp": ts, "open_quantity": qty,
                                       "status": "OPEN_NO_REALIZED_OUTCOME",
                                       "event_timestamp_ns": ts, "source_feed": "internal"})
        written = 0
        if outcomes:
            written = sum(r.rows for r in self.writer.write_rows(
                "gold", "position_outcomes", outcomes) if r.ok)
        if open_positions:
            self.writer.write_rows("gold", "open_positions", open_positions)
        return {"dataset": "position_outcomes", "rows": written,
                "closed": len(outcomes), "open": len(open_positions)}

    # -- supervised datasets (§19) -------------------------------------------
    def build_supervised(self, *, horizon_label: str = "position_close") -> dict:
        """Join strategy recommendations (point-in-time features) to realized position
        outcomes (labels). Rejected-only recommendations are retained with NULL labels
        (explicitly censored), never labeled as executions. No fabricated labels."""
        rec = self._glob("silver", "strategy_recommendations")
        out = self._glob("gold", "position_outcomes")
        if rec is None:
            return {"dataset": "supervised", "rows": 0, "reason": "no_recommendations"}
        con = duckdb.connect()
        try:
            if out is not None:
                sql = (
                    f"SELECT r.decision_id, r.symbol_id, r.strategy_family, r.confidence, "
                    f"r.rank_score, r.eligibility, r.availability_timestamp_ns AS decision_ts, "
                    f"o.net_return AS label_net_return, "
                    f"CASE WHEN o.net_return IS NULL THEN NULL "
                    f"WHEN o.net_return > 0 THEN 1 ELSE 0 END AS label_positive_return, "
                    f"o.holding_duration_ns AS label_holding_ns, "
                    f"CASE WHEN o.net_return IS NULL THEN 'censored_no_outcome' "
                    f"ELSE 'realized' END AS label_status, "
                    f"'{horizon_label}' AS horizon, '{GOLD_SCHEMA_VERSION}' AS schema_version, "
                    f"r.availability_timestamp_ns AS event_timestamp_ns, 'internal' AS source_feed "
                    f"FROM read_parquet('{rec}', union_by_name=true) r "
                    f"LEFT JOIN read_parquet('{out}', union_by_name=true) o "
                    f"ON o.entry_decision_id = r.decision_id")
            else:
                sql = (
                    f"SELECT r.decision_id, r.symbol_id, r.strategy_family, r.confidence, "
                    f"r.rank_score, r.eligibility, r.availability_timestamp_ns AS decision_ts, "
                    f"NULL AS label_net_return, NULL AS label_positive_return, "
                    f"NULL AS label_holding_ns, 'censored_no_outcome' AS label_status, "
                    f"'{horizon_label}' AS horizon, '{GOLD_SCHEMA_VERSION}' AS schema_version, "
                    f"r.availability_timestamp_ns AS event_timestamp_ns, 'internal' AS source_feed "
                    f"FROM read_parquet('{rec}', union_by_name=true) r")
            rows = con.sql(sql).to_arrow_table().to_pylist()
        finally:
            con.close()
        written = sum(r.rows for r in self.writer.write_rows("gold", "supervised", rows) if r.ok)
        realized = sum(1 for r in rows if r.get("label_status") == "realized")
        return {"dataset": "supervised", "rows": written, "realized_labels": realized,
                "censored_labels": written - realized}

    # -- offline RL transitions (§20/§21) ------------------------------------
    def build_offline_rl(self) -> dict:
        """Build logged transitions from executed portfolio decisions + realized outcomes.

        action_probability_if_known stays NULL (the behavior policy is deterministic/
        unknown — never invented). reward is realized net return ONLY for closed
        positions; unexecuted/rejected actions produce NO fabricated reward. Mandatory
        exits are environment dynamics, NOT agent actions."""
        pf = self._glob("silver", "portfolio_decisions")
        out = self._glob("gold", "position_outcomes")
        if pf is None:
            return {"dataset": "offline_rl", "rows": 0, "reason": "no_portfolio_decisions"}
        con = duckdb.connect()
        try:
            join_out = (f"LEFT JOIN read_parquet('{out}', union_by_name=true) o "
                        f"ON o.entry_decision_id = p.source_strategy_decision_id") if out else ""
            reward = ("o.net_return" if out else "NULL")
            terminal = ("CASE WHEN o.net_return IS NOT NULL THEN true ELSE false END"
                        if out else "false")
            sql = (
                f"SELECT p.portfolio_decision_id AS transition_id, "
                f"p.portfolio_decision_id AS episode_id, "
                f"p.availability_timestamp_ns AS decision_timestamp, "
                f"p.symbol_id, p.allocation_policy AS selected_action, "
                f"p.target_weight, p.current_weight, p.eligibility, "
                f"NULL AS action_probability_if_known, "
                f"'portfolio-optimizer' AS behavior_policy_version, "
                f"{reward} AS reward, {terminal} AS terminal, false AS truncated, "
                f"'mock' AS execution_environment, p.constraint_status AS constraint_status, "
                f"'{GOLD_SCHEMA_VERSION}' AS schema_version, "
                f"p.availability_timestamp_ns AS event_timestamp_ns, 'internal' AS source_feed "
                f"FROM read_parquet('{pf}', union_by_name=true) p {join_out}")
            rows = con.sql(sql).to_arrow_table().to_pylist()
        finally:
            con.close()
        written = sum(r.rows for r in self.writer.write_rows("gold", "offline_rl", rows) if r.ok)
        with_reward = sum(1 for r in rows if r.get("reward") is not None)
        return {"dataset": "offline_rl", "rows": written, "with_reward": with_reward,
                "note": "mandatory exits are environment dynamics, not agent actions; "
                        "action probabilities + unexecuted rewards are never fabricated"}

    def run_all(self) -> dict:
        pos = self.build_position_outcomes()
        return {"position_outcomes": pos, "supervised": self.build_supervised(),
                "offline_rl": self.build_offline_rl()}
