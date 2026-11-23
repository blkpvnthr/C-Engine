"""Autonomous data-sufficiency controller (Milestone 6B, §7).

Builds on the Milestone 6 dataset-eligibility semantics (reused, not reinvented) and adds:
(1) richer eligibility states, (2) autonomous generation of prioritized data-collection
tasks when evidence is missing, and (3) a gate the meta-optimizer consults so it never
launches experiments that cannot reach a reliable conclusion. It never fabricates
observations, orders, fills, rewards or behavior-policy probabilities — a deficiency yields
a collection task, not synthetic data.
"""

from __future__ import annotations

import time
from typing import Optional

from cengine.lakehouse.config import LakehouseConfig
from cengine.learning.datasets.eligibility import (
    EligibilityState,
    assess_all,
    default_task_specs,
)
from cengine.optimization.contracts import CollectionTask, stable_id

DATA_SUFFICIENCY_VERSION = "data-sufficiency-controller-1.0.0"

# economic-importance priors per learning task (used to prioritize collection, §7.3).
_IMPORTANCE = {
    "forward_return_prediction": 0.4,
    "strategy_net_return": 1.0,
    "probability_positive_return": 0.8,
    "regime_transition": 0.5,
    "execution_quality": 0.7,
    "policy_bandit": 0.9,
    "offline_rl_execution": 0.9,
    "universe_selection": 0.85,
}

# where each task's missing evidence could be recovered from, if at all.
_RECOVERABLE = {
    "forward_return_prediction": ("historical_sip_bars", True),
    "regime_transition": ("historical_sip_bars", True),
    "strategy_net_return": ("live_or_mock_execution_outcomes", False),
    "probability_positive_return": ("live_or_mock_execution_outcomes", False),
    "execution_quality": ("genuine_order_fill_events", False),
    "policy_bandit": ("logged_policy_actions_with_rewards", False),
    "offline_rl_execution": ("logged_sequential_transitions", False),
    "universe_selection": ("logged_universe_selections_with_outcomes", False),
}

_TRAINABLE = {EligibilityState.TRAINING_ELIGIBLE.value,
              EligibilityState.VALIDATION_ELIGIBLE.value}


class DataSufficiencyController:
    def __init__(self, config: Optional[LakehouseConfig] = None) -> None:
        self.config = config or LakehouseConfig.from_env()

    def assess(self, *, now_ns: Optional[int] = None) -> dict:
        """Assess every learning task and emit collection tasks for deficiencies."""
        now_ns = now_ns if now_ns is not None else time.time_ns()
        base = assess_all(self.config, specs=default_task_specs())
        tasks = base.get("tasks", [])
        collection: list = []
        eligible_for_experiments: list = []
        for t in tasks:
            name = t["task"]
            state = t["eligibility"]
            if state in _TRAINABLE:
                eligible_for_experiments.append(name)
                continue
            if state == EligibilityState.DATA_QUALITY_FAILED.value:
                continue  # a quality failure is handled by the quarantine pipeline, not collection
            source, recoverable = _RECOVERABLE.get(name, ("unknown", False))
            collection.append(CollectionTask(
                task_id=stable_id("collect", name, now_ns),
                learning_task=name,
                missing_evidence=self._missing_evidence(name, state, t.get("reasons", [])),
                recoverable_historically=recoverable,
                priority=_IMPORTANCE.get(name, 0.5) * (1.2 if recoverable else 1.0),
                estimated_economic_importance=_IMPORTANCE.get(name),
                suggested_source=source).to_dict())
        collection.sort(key=lambda c: c["priority"], reverse=True)
        return {"report": "data_sufficiency", "version": DATA_SUFFICIENCY_VERSION,
                "generated_ns": now_ns,
                "eligibility": {t["task"]: t["eligibility"] for t in tasks},
                "experiment_eligible_tasks": eligible_for_experiments,
                "collection_tasks": collection,
                "policy": "no fabricated observations; deficiencies become collection tasks"}

    @staticmethod
    def _missing_evidence(task: str, state: str, reasons: list) -> str:
        base = {
            EligibilityState.INSUFFICIENT_DATA.value: "insufficient eligible observations",
            EligibilityState.RESEARCH_ELIGIBLE.value: "below training-eligible threshold",
        }.get(state, state)
        return f"{base}: {', '.join(str(r) for r in reasons[:3])}" if reasons else base

    def is_experiment_eligible(self, task: str, *, now_ns: Optional[int] = None) -> bool:
        """Gate the meta-optimizer consults before launching an experiment for a task."""
        return task in self.assess(now_ns=now_ns)["experiment_eligible_tasks"]
