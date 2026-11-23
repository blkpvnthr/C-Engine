"""C-Engine autonomous performance-driven ML/RL optimization platform (Milestone 6B).

A governed, closed-loop system that continuously: measures actual trading-system
performance, diagnoses economically-significant weaknesses, generates + prioritizes
improvement hypotheses, plans + runs bounded experiments on local CPU/Metal-GPU, evaluates
candidates on independent transaction-cost-adjusted economics, learns from prior experiments
(persistent optimization memory), and promotes qualified candidates into SHADOW only.

It never activates a live policy, expands risk, relaxes the production -0.5% protective exit,
enables uncontrolled exploration, or modifies its own executable code. Milestone 7 owns
independent economic certification and controlled live deployment. Builds on (does not
rebuild) the Milestone 5 lakehouse and the Milestone 6 learning layer.
"""

from cengine.optimization.attribution import PerformanceAttributionEngine
from cengine.optimization.config import OptimizationConfig, load_config
from cengine.optimization.contracts import (
    OPTIMIZATION_CONTRACTS_VERSION,
    CollectionTask,
    Diagnostic,
    ExperimentManifest,
    ExperimentStatus,
    Hypothesis,
    Severity,
)
from cengine.optimization.controller import META_OPTIMIZER_VERSION, MetaOptimizer
from cengine.optimization.data_sufficiency import DataSufficiencyController
from cengine.optimization.economic import (
    CandidateEvaluation,
    EconomicScoreConfig,
    compare_to_baselines,
    evaluate_candidate,
)
from cengine.optimization.experiments import ExperimentBudget, ExperimentPlanner
from cengine.optimization.hypotheses import HypothesisGenerator
from cengine.optimization.memory import OptimizationMemory
from cengine.optimization.orchestration import (
    AutonomousOptimizationCycle,
    run_autonomous_cycle,
)
from cengine.optimization.search import SearchSpace, default_search_spaces
from cengine.optimization.validation import oos_windows, purged_walk_forward

OPTIMIZATION_PLATFORM_VERSION = "optimization-platform-1.0.0"

__all__ = [
    "META_OPTIMIZER_VERSION",
    "OPTIMIZATION_CONTRACTS_VERSION",
    "OPTIMIZATION_PLATFORM_VERSION",
    "AutonomousOptimizationCycle",
    "CandidateEvaluation",
    "OptimizationConfig",
    "load_config",
    "oos_windows",
    "purged_walk_forward",
    "run_autonomous_cycle",
    "CollectionTask",
    "DataSufficiencyController",
    "Diagnostic",
    "EconomicScoreConfig",
    "ExperimentBudget",
    "ExperimentManifest",
    "ExperimentPlanner",
    "ExperimentStatus",
    "HypothesisGenerator",
    "Hypothesis",
    "MetaOptimizer",
    "OptimizationMemory",
    "PerformanceAttributionEngine",
    "SearchSpace",
    "Severity",
    "compare_to_baselines",
    "default_search_spaces",
    "evaluate_candidate",
]
