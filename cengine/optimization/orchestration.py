"""Autonomous optimization orchestration (Milestone 6B, §22).

Wires the genuine experiment runners (protective-exit + supervised) into the MetaOptimizer and
runs a performance-driven cycle, persisting a report. This is the operational face of the
closed loop: measure -> diagnose -> hypothesize -> experiment (OOS on genuine data) -> evaluate
-> accept(shadow)/reject -> memory -> repeat. Off the trading-critical path; no live activation.
"""

from __future__ import annotations

import json
import time
from typing import Optional

from cengine.lakehouse.config import LakehouseConfig
from cengine.optimization.config import OptimizationConfig, load_config
from cengine.optimization.controller import MetaOptimizer
from cengine.optimization.economic import EconomicScoreConfig
from cengine.optimization.experiments import ExperimentBudget

ORCHESTRATION_VERSION = "autonomous-orchestration-1.0.0"


class AutonomousOptimizationCycle:
    def __init__(self, config: Optional[LakehouseConfig] = None, *,
                 opt_config: Optional[OptimizationConfig] = None,
                 symbols: Optional[list] = None) -> None:
        self.config = config or LakehouseConfig.from_env()
        self.opt = opt_config or load_config()
        self.symbols = symbols or ["AAPL"]

    def _runners(self, symbol: str) -> dict:
        runners = {}
        try:
            from cengine.optimization.exits.runner import make_exit_runner
            exit_runner, _ = make_exit_runner(symbol, config=self.config)
            runners["protective_exit"] = exit_runner
        except Exception:  # noqa: BLE001
            pass
        try:
            from cengine.optimization.supervised_runner import make_supervised_runner
            runners["supervised_model"] = make_supervised_runner(config=self.config)
        except Exception:  # noqa: BLE001
            pass
        return runners

    def run(self, *, now_ns: Optional[int] = None, symbol: Optional[str] = None) -> dict:
        now_ns = now_ns if now_ns is not None else time.time_ns()
        sym = symbol or self.symbols[0]
        out = {"report": "autonomous_optimization_cycle", "version": ORCHESTRATION_VERSION,
               "started_ns": now_ns, "symbol": sym, "config": self.opt.to_dict()}
        if not self.opt.enabled:
            out["skipped"] = "optimization disabled"
            return out
        mo = MetaOptimizer(
            self.config, runners=self._runners(sym),
            score_cfg=EconomicScoreConfig(
                drawdown_penalty=self.opt.reward_coefficients.get("drawdown_penalty", 1.0)),
            budget=ExperimentBudget(max_configs_per_hypothesis=self.opt.max_configs_per_hypothesis,
                                    method=self.opt.search_method))
        cycle = mo.run(now_ns=now_ns)
        out["cycle"] = cycle
        out["completed_ns"] = time.time_ns()
        out["duration_s"] = round((out["completed_ns"] - now_ns) / 1e9, 2)
        out["boundary"] = ("shadow-only; no live activation / risk expansion / production-exit "
                           "relaxation; Milestone 7 certifies")
        self._record(out)
        return out

    def _record(self, out: dict) -> None:
        path = self.config.root / "_optimization" / "cycle_runs.jsonl"
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            summary = {"version": out["version"], "symbol": out["symbol"],
                       "started_ns": out["started_ns"], "duration_s": out.get("duration_s"),
                       "outcomes": out.get("cycle", {}).get("outcomes")}
            with open(path, "a") as fh:
                fh.write(json.dumps(summary, default=str) + "\n")
        except OSError:
            pass


def run_autonomous_cycle(config: Optional[LakehouseConfig] = None, *,
                         symbol: str = "AAPL", now_ns: Optional[int] = None) -> dict:
    return AutonomousOptimizationCycle(config, symbols=[symbol]).run(now_ns=now_ns, symbol=symbol)
