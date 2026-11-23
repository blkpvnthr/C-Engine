"""Autonomous performance attribution engine (Milestone 6B, §8).

Identifies economically-significant weaknesses in C-Engine from genuine trading records in
the lakehouse (gold position_outcomes / daily_performance / fills), slicing by the §8.1
dimensions and detecting the §8.2 weakness families. Every finding is a descriptive
Diagnostic that recommends a controlled experiment — causation is never inferred from
correlation. When genuine outcome data is absent (the current reality — no live session has
produced closed positions), it honestly emits an INSUFFICIENT_DATA diagnostic rather than
fabricating weaknesses.
"""

from __future__ import annotations

import time
from typing import Optional

import duckdb

from cengine.lakehouse.config import LakehouseConfig
from cengine.optimization.contracts import Diagnostic, Severity, stable_id

ATTRIBUTION_ENGINE_VERSION = "performance-attribution-1.0.0"


def _glob(config: LakehouseConfig, layer: str, dataset: str) -> Optional[str]:
    base = config.dataset_path(layer, dataset)
    if not base.exists() or not any(base.rglob("*.parquet")):
        return None
    return str(base / "**" / "*.parquet").replace("'", "''")


class PerformanceAttributionEngine:
    def __init__(self, config: Optional[LakehouseConfig] = None) -> None:
        self.config = config or LakehouseConfig.from_env()

    def _outcomes_src(self) -> Optional[str]:
        g = _glob(self.config, "gold", "position_outcomes")
        return f"read_parquet('{g}', union_by_name=true)" if g else None

    def attribute(self, *, now_ns: Optional[int] = None) -> dict:
        """Run attribution; return a machine-readable report with diagnostics."""
        now_ns = now_ns if now_ns is not None else time.time_ns()
        report = {"report": "performance_attribution",
                  "version": ATTRIBUTION_ENGINE_VERSION, "generated_ns": now_ns,
                  "diagnostics": [], "dimensions_analyzed": [], "data_available": False,
                  "causal_note": "descriptive attribution; each finding recommends a "
                                 "controlled experiment — no causation is asserted"}
        src = self._outcomes_src()
        if src is None:
            report["diagnostics"].append(self._insufficient(now_ns).to_dict())
            return report
        con = duckdb.connect()
        try:
            total = con.execute(f"SELECT count(*), avg(net_return), "
                                f"sum(CASE WHEN net_return>0 THEN 1 ELSE 0 END) "
                                f"FROM {src} WHERE net_return IS NOT NULL").fetchone()
            n = int(total[0] or 0)
            report["closed_positions"] = n
            if n < 30:
                report["diagnostics"].append(self._insufficient(now_ns, n).to_dict())
                return report
            report["data_available"] = True
            avg_net = float(total[1]) if total[1] is not None else 0.0
            win_rate = (int(total[2] or 0) / n) if n else 0.0
            report["avg_net_return"] = avg_net
            report["win_rate"] = win_rate
            diags = []
            diags += self._by_dimension(con, src, "symbol_id", "symbol", now_ns)
            diags += self._by_dimension(con, src, "entry_decision_id", "entry_signal", now_ns)
            diags += self._exit_quality(con, src, now_ns)
            report["dimensions_analyzed"] = ["symbol", "entry_signal", "exit_behavior"]
            report["diagnostics"] = [d.to_dict() for d in diags]
        except Exception as exc:  # noqa: BLE001 — attribution must never crash the cycle
            report["error"] = f"{type(exc).__name__}:{exc}"
        finally:
            con.close()
        return report

    def _insufficient(self, now_ns: int, n: int = 0) -> Diagnostic:
        return Diagnostic(
            diagnostic_id=stable_id("diag-insufficient", now_ns),
            detected_ns=now_ns, component="trading_outcomes",
            period="all", sample_size=n, severity=Severity.INFO.value,
            suspected_cause="no_genuine_closed_position_outcomes_yet",
            recommended_experiment_family="data_collection",
            required_eligibility="genuine_execution_outcomes",
            status="insufficient_data",
            causal_note="attribution requires genuine closed-position outcomes; none/"
                        "too few exist (no live trading session has run). Not a weakness "
                        "in the engine — a data-coverage gap.")

    def _by_dimension(self, con, src, col: str, label: str, now_ns: int) -> list:
        """Flag dimension buckets with materially negative mean net return (descriptive)."""
        out = []
        try:
            rows = con.execute(
                f"SELECT {col}, count(*) c, avg(net_return) a FROM {src} "
                f"WHERE net_return IS NOT NULL GROUP BY {col} HAVING c >= 20 "
                f"ORDER BY a ASC LIMIT 3").fetchall()
        except Exception:  # noqa: BLE001
            return out
        for key, c, a in rows:
            if a is not None and a < 0:
                out.append(Diagnostic(
                    diagnostic_id=stable_id("diag", label, key, now_ns),
                    detected_ns=now_ns, component=f"{label}:{key}",
                    period="all", sample_size=int(c),
                    est_economic_impact=float(a) * int(c),
                    severity=Severity.MEDIUM.value if a < -0.002 else Severity.LOW.value,
                    suspected_cause=f"negative mean net return in {label}={key}",
                    alternatives=["regime effect", "small-sample noise", "cost drag"],
                    recommended_experiment_family=("entry_optimization"
                                                   if label == "entry_signal"
                                                   else "strategy_optimization"),
                    required_eligibility="genuine_execution_outcomes",
                    priority=abs(float(a)) * int(c)))
        return out

    def _exit_quality(self, con, src, now_ns: int) -> list:
        """Flag frequent near-threshold exits (candidate premature-liquidation signal)."""
        out = []
        try:
            row = con.execute(
                f"SELECT count(*), sum(CASE WHEN net_return BETWEEN -0.007 AND -0.004 "
                f"THEN 1 ELSE 0 END) FROM {src} WHERE net_return IS NOT NULL").fetchone()
        except Exception:  # noqa: BLE001
            return out
        n, near = int(row[0] or 0), int(row[1] or 0)
        if n >= 30 and near / n > 0.15:
            out.append(Diagnostic(
                diagnostic_id=stable_id("diag-exit", now_ns),
                detected_ns=now_ns, component="protective_exit", period="all",
                sample_size=n, est_economic_impact=None,
                severity=Severity.HIGH.value,
                suspected_cause=f"{near}/{n} exits cluster near the -0.5% protective "
                                f"threshold — possible premature liquidation",
                alternatives=["genuine stop-losses", "regime-driven losses"],
                recommended_experiment_family="protective_exit_matrix",
                required_eligibility="counterfactual_exit_observations",
                priority=float(near)))
        return out
