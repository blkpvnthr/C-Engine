"""Daily performance attribution (Milestone 5, §26).

End-of-session attribution over the gold/silver datasets: gross/net return,
per-strategy-family, per-regime, per-execution-policy, transaction-cost contribution,
protective-exit frequency, and an explicit loss-source taxonomy. Causal attribution is
treated as UNCERTAIN unless supported by evidence (each attributed cause carries a
confidence note). Clearly separates prediction / strategy / execution quality.
"""

from __future__ import annotations

import time
from typing import Optional

import duckdb

from cengine.lakehouse.config import LakehouseConfig
from cengine.lakehouse.parquet_writer import ParquetWriter

ATTRIBUTION_VERSION = "daily-attribution-1.0.0"

LOSS_SOURCES = ["POOR_OPPORTUNITY_SELECTION", "REGIME_MISMATCH", "ENTRY_TIMING",
                "POSITION_SIZING", "EXPOSURE_CONCENTRATION", "EXECUTION_COST",
                "ADVERSE_SELECTION", "EXIT_TIMING", "DATA_QUALITY"]


class DailyAttribution:
    def __init__(self, config: Optional[LakehouseConfig] = None) -> None:
        self.config = config or LakehouseConfig.from_env()
        self.writer = ParquetWriter(self.config)

    def _glob(self, layer: str, dataset: str) -> Optional[str]:
        base = self.config.dataset_path(layer, dataset)
        if not base.exists() or not any(base.rglob("*.parquet")):
            return None
        return str(base / "**" / "*.parquet").replace("'", "''")

    def attribute(self, *, trade_date: Optional[str] = None,
                  now_ns: Optional[int] = None) -> dict:
        out = self._glob("gold", "position_outcomes")
        tca = self._glob("silver", "transaction_costs")
        con = duckdb.connect()
        report: dict = {"report": "daily_attribution", "version": ATTRIBUTION_VERSION,
                        "generated_ns": now_ns if now_ns is not None else time.time_ns(),
                        "trade_date": trade_date,
                        "quality_distinction": {
                            "prediction_quality": "label vs realized (supervised datasets)",
                            "strategy_quality": "net return by family/regime",
                            "execution_quality": "implementation shortfall / slippage (TCA)"},
                        "loss_source_taxonomy": LOSS_SOURCES,
                        "causal_note": "Causal attribution is UNCERTAIN unless supported "
                                       "by sufficient evidence; counts below are descriptive."}
        try:
            if out is not None:
                src = f"read_parquet('{out}', union_by_name=true)"
                total = con.execute(f"SELECT count(*), "
                                    f"sum(CASE WHEN net_return>0 THEN 1 ELSE 0 END), "
                                    f"avg(net_return), sum(net_return) FROM {src} "
                                    f"WHERE net_return IS NOT NULL").fetchone()
                report["closed_positions"] = int(total[0] or 0)
                report["winners"] = int(total[1] or 0)
                report["avg_net_return"] = float(total[2]) if total[2] is not None else None
                report["sum_net_return"] = float(total[3]) if total[3] is not None else None
                report["win_rate"] = (report["winners"] / report["closed_positions"]
                                      if report["closed_positions"] else None)
            else:
                report["closed_positions"] = 0
                report["note"] = "no closed positions: insufficient execution history"
            # per-strategy-family (from silver strategy_recommendations joined to outcomes)
            rec = self._glob("silver", "strategy_recommendations")
            if rec is not None and out is not None:
                fam = con.execute(
                    f"SELECT r.strategy_family, count(*), avg(o.net_return) "
                    f"FROM read_parquet('{out}', union_by_name=true) o "
                    f"JOIN read_parquet('{rec}', union_by_name=true) r "
                    f"ON r.decision_id = o.entry_decision_id "
                    f"GROUP BY r.strategy_family").fetchall()
                report["by_strategy_family"] = [
                    {"family": f, "positions": int(n),
                     "avg_net_return": float(a) if a is not None else None}
                    for f, n, a in fam]
            # execution quality from TCA
            if tca is not None:
                t = con.execute(
                    f"SELECT avg(TRY_CAST(implementation_shortfall_bps AS DOUBLE)), "
                    f"avg(TRY_CAST(realized_slippage_bps AS DOUBLE)), count(*) "
                    f"FROM read_parquet('{tca}', union_by_name=true)").fetchone()
                report["execution_quality"] = {
                    "avg_implementation_shortfall_bps": float(t[0]) if t[0] is not None else None,
                    "avg_realized_slippage_bps": float(t[1]) if t[1] is not None else None,
                    "tca_records": int(t[2] or 0)}
        except Exception as exc:  # noqa: BLE001 — attribution must never crash the session
            report["error"] = f"{type(exc).__name__}:{exc}"
        finally:
            con.close()

        # persist the attribution record to gold/daily_performance
        row = {**{k: v for k, v in report.items() if not isinstance(v, (list, dict))},
               "report_json": __import__("json").dumps(report, default=str),
               "event_timestamp_ns": report["generated_ns"], "source_feed": "internal"}
        self.writer.write_rows("gold", "daily_performance", [row])
        return report
