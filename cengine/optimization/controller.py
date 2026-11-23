"""Autonomous ML/RL meta-optimization controller (Milestone 6B, §9 — core deliverable).

The persistent closed loop:

  attribution -> diagnostics -> data-sufficiency gate -> hypotheses -> experiment planning
  (memory-informed) -> eligible-experiment execution (pluggable runners) -> independent
  economic evaluation vs the baseline hierarchy -> accept(shadow)/reject/defer -> memory
  update -> better subsequent selection.

It optimizes the LEARNING PROCESS, not just model weights, and it consults prior experiment
outcomes so it does not repeat the same work. It NEVER activates a live policy, expands risk,
relaxes the production protective exit, or modifies its own executable code — accepted
candidates become shadow-eligible only (Milestone 7 certifies live).

Experiment execution is delegated to pluggable `runners` keyed by search-space name; a
hypothesis whose runner is absent or whose data is ineligible is DEFERRED (infrastructure
present, conclusion withheld) rather than faked.
"""

from __future__ import annotations

import time
from typing import Callable, Optional

from cengine.lakehouse.config import LakehouseConfig
from cengine.optimization.attribution import PerformanceAttributionEngine
from cengine.optimization.contracts import Diagnostic, ExperimentStatus
from cengine.optimization.data_sufficiency import DataSufficiencyController
from cengine.optimization.economic import EconomicScoreConfig, compare_to_baselines
from cengine.optimization.experiments import ExperimentBudget, ExperimentPlanner
from cengine.optimization.hypotheses import HypothesisGenerator
from cengine.optimization.memory import OptimizationMemory

META_OPTIMIZER_VERSION = "meta-optimizer-1.0.0"

# search-space -> the learning task whose eligibility gates it (None = always allowed to
# attempt, e.g. protective_exit which runs on genuine bars via counterfactual/replay).
_SPACE_TASK = {
    "supervised_model": "forward_return_prediction",
    "rl_reward": "offline_rl_execution",
    "entry_policy": "strategy_net_return",
    "strategy_policy": "strategy_net_return",
    "protective_exit": None,
}


class MetaOptimizer:
    """A ``runner(space, config, now_ns) -> dict`` returns at least:
    {'evaluation': CandidateEvaluation-like dict, 'baselines': {name: CandidateEval dict},
     'data_label': 'real'|'synthetic', 'runtime_s': float, 'device': str}. Returning None
    means the runner declined (treated as deferred)."""

    def __init__(self, config: Optional[LakehouseConfig] = None, *,
                 runners: Optional[dict] = None,
                 score_cfg: Optional[EconomicScoreConfig] = None,
                 budget: Optional[ExperimentBudget] = None) -> None:
        self.config = config or LakehouseConfig.from_env()
        self.attribution = PerformanceAttributionEngine(self.config)
        self.sufficiency = DataSufficiencyController(self.config)
        self.hypotheses = HypothesisGenerator()
        self.memory = OptimizationMemory(self.config)
        self.planner = ExperimentPlanner(self.memory)
        self.runners: dict = runners or {}
        self.score_cfg = score_cfg or EconomicScoreConfig()
        self.budget = budget or ExperimentBudget()

    def register_runner(self, space: str, runner: Callable) -> None:
        self.runners[space] = runner

    def run(self, *, now_ns: Optional[int] = None,
            max_hypotheses: int = 6) -> dict:
        now_ns = now_ns if now_ns is not None else time.time_ns()
        report = {"report": "meta_optimization_cycle", "version": META_OPTIMIZER_VERSION,
                  "started_ns": now_ns}

        # 1. attribution -> diagnostics
        attr = self.attribution.attribute(now_ns=now_ns)
        diags = [Diagnostic(**{k: v for k, v in d.items()
                               if k in Diagnostic.__dataclass_fields__})
                 for d in attr.get("diagnostics", [])]
        report["diagnostics"] = len(diags)

        # 2. data sufficiency (eligibility + collection tasks)
        suff = self.sufficiency.assess(now_ns=now_ns)
        report["eligibility"] = suff["eligibility"]
        report["collection_tasks"] = len(suff["collection_tasks"])
        eligible_tasks = set(suff["experiment_eligible_tasks"])

        # 3. hypotheses: diagnostic-driven + standing research (protective-exit matrix always
        # runs on genuine bars; supervised search fires when forward-return is eligible)
        hyps = self.hypotheses.generate(diags)
        standing = self.hypotheses.standing(eligible_tasks=eligible_tasks)
        seen = {h.hypothesis_id for h in hyps}
        hyps += [h for h in standing if h.hypothesis_id not in seen]
        hyps = hyps[:max_hypotheses]
        report["hypotheses"] = [h.hypothesis_id for h in hyps]

        results = {"accepted": [], "rejected": [], "deferred": [], "errors": []}
        for h in hyps:
            self._process_hypothesis(h, eligible_tasks, results, now_ns)

        report["outcomes"] = {k: len(v) for k, v in results.items()}
        report["details"] = results
        report["memory_effectiveness"] = self.memory.effectiveness()
        report["completed_ns"] = time.time_ns()
        report["boundary"] = ("candidates are shadow-eligible only; no live activation, no "
                              "risk expansion, no production-exit relaxation (Milestone 7 "
                              "certifies)")
        return report

    def _process_hypothesis(self, h, eligible_tasks, results, now_ns) -> None:
        space = h.search_space
        if space in ("none", ""):
            return  # data-collection hypothesis; handled by the sufficiency controller
        runner = self.runners.get(space)
        task = _SPACE_TASK.get(space)
        # eligibility gate: a task-bound space needs eligible data; protective_exit is gated
        # by its own runner on genuine bars.
        if task is not None and task not in eligible_tasks:
            results["deferred"].append({"hypothesis": h.hypothesis_id, "space": space,
                                        "reason": f"task_not_eligible:{task}"})
            self.memory.record(hypothesis_id=h.hypothesis_id, search_space=space,
                               config={}, score=None, economic_metrics={},
                               decision=ExperimentStatus.DEFERRED.value,
                               rejection_reason="insufficient_data", now_ns=now_ns)
            return
        if runner is None:
            results["deferred"].append({"hypothesis": h.hypothesis_id, "space": space,
                                        "reason": "no_runner_registered"})
            return
        configs = self.planner.plan(h, budget=self.budget)
        if not configs:
            results["deferred"].append({"hypothesis": h.hypothesis_id, "space": space,
                                        "reason": "no_candidate_configs (dedup/budget)"})
            return
        best = None
        for config in configs:
            try:
                out = runner(space, config, now_ns)
            except Exception as exc:  # noqa: BLE001 — one runner failure never aborts the loop
                results["errors"].append({"hypothesis": h.hypothesis_id, "config": config,
                                          "error": f"{type(exc).__name__}:{exc}"})
                continue
            if out is None:
                results["deferred"].append({"hypothesis": h.hypothesis_id, "space": space,
                                            "reason": "runner_declined"})
                continue
            verdict = self._evaluate(out)
            decision = verdict["recommendation"]
            self.memory.record(
                hypothesis_id=h.hypothesis_id, search_space=space, config=config,
                score=out.get("evaluation", {}).get("score"),
                economic_metrics=out.get("evaluation", {}).get("metrics", {}),
                decision=(ExperimentStatus.ELIGIBLE_FOR_SHADOW.value
                          if decision == "eligible_for_shadow"
                          else ExperimentStatus.REJECTED.value),
                rejection_reason=str(verdict.get("reason", "")),
                data_label=out.get("data_label", "synthetic"),
                cost_s=float(out.get("runtime_s", 0.0)), now_ns=now_ns)
            record = {"hypothesis": h.hypothesis_id, "space": space, "config": config,
                      "verdict": verdict, "data_label": out.get("data_label")}
            if decision == "eligible_for_shadow":
                if best is None or (out.get("evaluation", {}).get("score") or -1e9) > \
                        (best.get("verdict", {}).get("candidate_score") or -1e9):
                    best = record
            else:
                results["rejected"].append(record)
        if best is not None:
            results["accepted"].append(best)

    def _evaluate(self, out: dict) -> dict:
        ev = out.get("evaluation")
        baselines = out.get("baselines", {})
        if ev is None:
            return {"recommendation": "reject", "reason": "no_evaluation"}
        cand = _AsEval(ev)
        base_evals = {name: _AsEval(b) for name, b in baselines.items()}
        return compare_to_baselines(cand, base_evals)


class _AsEval:
    """Adapt a plain evaluation dict to the attribute interface compare_to_baselines uses."""

    def __init__(self, d: dict) -> None:
        self.score = d.get("score")
        self.gates_passed = d.get("gates_passed", True)
        self.gate_failures = d.get("gate_failures", [])
        self.data_label = d.get("data_label", "synthetic")
