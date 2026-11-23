"""Constrained offline-RL research framework (Milestone 6, §6/§7).

RECOMMENDATION-ONLY research infrastructure. Loads versioned logged transitions from the
gold ``offline_rl_transitions`` dataset and provides conservative, data-sized offline
learners — Behavior Cloning (BC), Fitted Q Evaluation (FQE), Conservative Q-Learning (CQL)
and Implicit Q-Learning (IQL) — on the Apple Metal GPU via MLX, each with a documented,
observable numpy CPU fallback.

Non-negotiable invariants enforced throughout:

* No orders, no broker, no bypass of financial authority; nothing here touches the mandatory
  -0.5% protective exit, which is modeled as ENVIRONMENT DYNAMICS and excluded from every
  learnable action space (never an agent action).
* Behavior-policy probabilities are unknown and never fabricated, so IPS / doubly-robust
  estimators that need them are not provided; the shipped methods do not require them.
* Insufficient or synthetic-as-real data cannot produce a promoted model: the
  :class:`TrainingGuard` refuses training on insufficient data and marks any result on
  explicitly-synthetic data as research-only / non-promotable, while the pipeline keeps
  working (returning an untrained result) rather than crashing.
"""

from cengine.learning.offline_rl.bc import (
    BC_VERSION,
    BCConfig,
    BCResult,
    BehaviorCloner,
)
from cengine.learning.offline_rl.cql_iql import (
    CQL_IQL_VERSION,
    ConservativeQLearner,
    CQLConfig,
    CQLResult,
    ImplicitQLearner,
    IQLConfig,
    IQLResult,
)
from cengine.learning.offline_rl.fqe import (
    FQE_VERSION,
    FittedQEvaluator,
    FQEConfig,
    FQEResult,
)
from cengine.learning.offline_rl.transitions import (
    MANDATORY_EXIT_ACTIONS,
    TRANSITIONS_VERSION,
    ActionSpace,
    DataQualityReport,
    GuardDecision,
    InsufficientDataError,
    SyntheticDataError,
    TrainingGuard,
    TransitionDataset,
    TransitionLoader,
    build_dataset_from_records,
    make_synthetic_dataset,
)

__all__ = [
    "BC_VERSION",
    "CQL_IQL_VERSION",
    "FQE_VERSION",
    "MANDATORY_EXIT_ACTIONS",
    "TRANSITIONS_VERSION",
    "ActionSpace",
    "BCConfig",
    "BCResult",
    "BehaviorCloner",
    "CQLConfig",
    "CQLResult",
    "ConservativeQLearner",
    "DataQualityReport",
    "FQEConfig",
    "FQEResult",
    "FittedQEvaluator",
    "GuardDecision",
    "IQLConfig",
    "IQLResult",
    "ImplicitQLearner",
    "InsufficientDataError",
    "SyntheticDataError",
    "TrainingGuard",
    "TransitionDataset",
    "TransitionLoader",
    "build_dataset_from_records",
    "make_synthetic_dataset",
]
