"""Experiment planner + manifest builder (Milestone 6B, §9.3/§23.1).

Turns a hypothesis + its approved search space + prior memory into a bounded, deduplicated,
budgeted list of candidate configurations to evaluate, and records a reproducible manifest
for each. Experiments that would exceed the budget, duplicate prior work, or target a
repeatedly-failing region are dropped before any compute is spent.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

from cengine.optimization.contracts import (
    ExperimentManifest,
    config_fingerprint,
    stable_id,
)
from cengine.optimization.memory import OptimizationMemory
from cengine.optimization.search import (
    default_search_spaces,
    grid_search,
    random_search,
    tpe_suggest,
)

EXPERIMENT_PLANNER_VERSION = "experiment-planner-1.0.0"


@dataclass
class ExperimentBudget:
    max_configs_per_hypothesis: int = 8
    method: str = "tpe"                 # 'random' | 'grid' | 'tpe'
    seed: int = 7


class ExperimentPlanner:
    def __init__(self, memory: Optional[OptimizationMemory] = None) -> None:
        self.memory = memory or OptimizationMemory()
        self.spaces = default_search_spaces()

    def plan(self, hypothesis, *, budget: Optional[ExperimentBudget] = None) -> list:
        """Return a deduped, budgeted list of candidate configs for a hypothesis."""
        budget = budget or ExperimentBudget()
        space = self.spaces.get(hypothesis.search_space)
        if space is None:
            return []
        n = budget.max_configs_per_hypothesis
        if budget.method == "grid" and (space.grid_size() or 10 ** 9) <= 64:
            candidates = grid_search(space, max_points=n * 2)
        elif budget.method == "tpe":
            history = self.memory.history_for(hypothesis.search_space)
            candidates = tpe_suggest(space, history, n * 2, seed=budget.seed)
        else:
            candidates = random_search(space, n * 2, seed=budget.seed)
        # validate + dedupe vs memory + drop repeatedly-failing fingerprints
        failing = set(self.memory.repeated_failures(hypothesis.search_space))
        valid = []
        for c in candidates:
            ok, _ = space.validate(c)
            if not ok:
                continue
            if config_fingerprint({"space": hypothesis.search_space, **c}) in failing:
                continue
            valid.append(c)
        valid = self.memory.dedupe(hypothesis.search_space, valid)
        return valid[:n]

    def manifest(self, *, hypothesis, config: dict, result: dict, decision: str,
                 git_commit: str = "", now_ns: Optional[int] = None) -> ExperimentManifest:
        now_ns = now_ns if now_ns is not None else time.time_ns()
        eval_result = result.get("evaluation", {})
        return ExperimentManifest(
            experiment_id=stable_id("exp", hypothesis.hypothesis_id,
                                    config_fingerprint(config), now_ns),
            parent_experiment=None, diagnostic_id=hypothesis.diagnostic_id,
            hypothesis_id=hypothesis.hypothesis_id,
            search_space_version=EXPERIMENT_PLANNER_VERSION,
            model_family=str(config.get("model_family", hypothesis.search_space)),
            model_architecture=str(config.get("policy", config.get("hidden_width", ""))),
            hyperparameters=config, feature_definition=result.get("feature_definition", ""),
            dataset_version=result.get("dataset_version", ""),
            training_window=result.get("training_window", ""),
            evaluation_window=result.get("evaluation_window", ""),
            reward_definition=result.get("reward_definition", ""),
            policy_configuration=result.get("policy_configuration", config),
            random_seed=int(result.get("seed", 7)),
            execution_device=result.get("device", "cpu"),
            runtime_s=float(result.get("runtime_s", 0.0)),
            resource=result.get("resource", {}), git_commit=git_commit,
            artifact_hashes=result.get("artifact_hashes", {}),
            economic_metrics=eval_result.get("metrics", {}),
            risk_metrics=result.get("risk_metrics", {}),
            statistical_uncertainty={"net_return_ci": eval_result.get("net_return_ci")},
            evaluation_result=eval_result, decision=decision,
            rejection_reason=result.get("rejection_reason", ""),
            data_label=result.get("data_label", "synthetic"), created_ns=now_ns)
