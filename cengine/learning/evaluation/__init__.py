"""Independent evaluation + offline policy evaluation (Milestone 6, §19/§21).

Economic metrics over per-decision NET returns with dependence-aware (block/stationary
bootstrap) uncertainty and honest effective-sample-size reporting, plus offline policy
evaluation (Direct Method + Fitted-Q hooks; IPS/doubly-robust gated off when behavior
probabilities are unknown) with an action-support/extrapolation audit and candidate-vs-
baseline comparison on identical decisions. Pure numpy/scipy (CPU). Every output carries a
'synthetic'/'real' provenance label: synthetic evidence can never claim profitability.
"""

from cengine.learning.evaluation.economic import (
    DATA_LABEL_HISTORICAL_REPLAY,
    DATA_LABEL_REAL,
    DATA_LABEL_SYNTHETIC,
    ECONOMIC_METRICS_VERSION,
    BootstrapCI,
    EconomicEvaluation,
    EconomicMetrics,
    SampleSizeReport,
    bootstrap_metric_ci,
    economic_metrics,
    evaluate_returns,
    moving_block_bootstrap_indices,
    sample_size_report,
    stationary_bootstrap_indices,
    validate_data_label,
)
from cengine.learning.evaluation.ope import (
    OPE_VERSION,
    ActionSupportAudit,
    DirectMethodEvaluator,
    FittedQEvaluator,
    LinearRewardModel,
    LoggedDecisions,
    OPEResult,
    PerActionMeanRewardModel,
    PolicyComparison,
    RewardModel,
    action_support_audit,
    compare_policies,
    doubly_robust_value,
    ips_value,
    policy_action_matrix,
)

__all__ = [
    "DATA_LABEL_HISTORICAL_REPLAY",
    "DATA_LABEL_REAL",
    "DATA_LABEL_SYNTHETIC",
    "ECONOMIC_METRICS_VERSION",
    "OPE_VERSION",
    "ActionSupportAudit",
    "BootstrapCI",
    "DirectMethodEvaluator",
    "EconomicEvaluation",
    "EconomicMetrics",
    "FittedQEvaluator",
    "LinearRewardModel",
    "LoggedDecisions",
    "OPEResult",
    "PerActionMeanRewardModel",
    "PolicyComparison",
    "RewardModel",
    "SampleSizeReport",
    "action_support_audit",
    "bootstrap_metric_ci",
    "compare_policies",
    "doubly_robust_value",
    "economic_metrics",
    "evaluate_returns",
    "ips_value",
    "moving_block_bootstrap_indices",
    "policy_action_matrix",
    "sample_size_report",
    "stationary_bootstrap_indices",
    "validate_data_label",
]
