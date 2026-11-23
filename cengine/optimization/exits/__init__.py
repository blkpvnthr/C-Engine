"""Protective-exit research: counterfactual data, adaptive candidates, state machine, matrix
(Milestone 6B, §14-17). All candidates are SHADOW/research-only; the production -0.5% exit is
preserved untouched until Milestone 7 certification."""

from cengine.optimization.exits.adaptive import (
    ADAPTIVE_EXIT_VERSION,
    PRODUCTION_EXIT_PCT,
    AdaptiveExitConfig,
    AdaptiveExitPolicy,
    ExitAction,
    ExitDecision,
    asset_class_stop_range,
    risk_normalized_notional,
    volatility_adjusted_stop,
)
from cengine.optimization.exits.counterfactual import (
    COUNTERFACTUAL_VERSION,
    CounterfactualReport,
)
from cengine.optimization.exits.counterfactual import (
    analyze as analyze_counterfactual_exits,
)
from cengine.optimization.exits.experiment_matrix import (
    EXIT_MATRIX_VERSION,
    ExitMatrixResult,
    run_exit_matrix,
)
from cengine.optimization.exits.runner import make_exit_runner
from cengine.optimization.exits.state_machine import (
    EXIT_STATE_MACHINE_VERSION,
    ExitState,
    ProtectiveExitStateMachine,
)

__all__ = [
    "ADAPTIVE_EXIT_VERSION",
    "COUNTERFACTUAL_VERSION",
    "EXIT_MATRIX_VERSION",
    "EXIT_STATE_MACHINE_VERSION",
    "CounterfactualReport",
    "ExitMatrixResult",
    "analyze_counterfactual_exits",
    "make_exit_runner",
    "run_exit_matrix",
    "PRODUCTION_EXIT_PCT",
    "AdaptiveExitConfig",
    "AdaptiveExitPolicy",
    "ExitAction",
    "ExitDecision",
    "ExitState",
    "ProtectiveExitStateMachine",
    "asset_class_stop_range",
    "risk_normalized_notional",
    "volatility_adjusted_stop",
]
