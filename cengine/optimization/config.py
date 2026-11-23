"""Governed autonomous-optimization configuration (Milestone 6B, §24).

A validated settings object with the safety invariants baked in: signed P/L thresholds are
decimal fractions distinct from positive stop distances; production activation is OFF;
autonomous risk expansion is OFF; the mandatory emergency threshold cannot be loosened; and the
production -0.5% exit is preserved. Reads overrides from CENGINE_OPT_* env vars; invalid or
unsafe settings are rejected.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

OPTIMIZATION_CONFIG_VERSION = "optimization-config-1.0.0"


class ConfigError(ValueError):
    """Raised when a configuration would violate a hard safety invariant."""


@dataclass(frozen=True)
class OptimizationConfig:
    enabled: bool = True
    mode: str = "governed_autonomous_research"
    local_only: bool = True
    # compute
    preferred_device: str = "metal_gpu"
    allow_silent_cpu_fallback: bool = False
    training_during_market_hours: bool = False
    # governed toggles (hard-off)
    production_activation_enabled: bool = False
    autonomous_live_activation: bool = False
    autonomous_risk_expansion: bool = False
    hard_stop_optimization: bool = False
    live_exploration: bool = False
    milestone_7_certification_required: bool = True
    # protective exit (candidate vs production)
    production_exit_pct: float = -0.005          # signed fraction; PRESERVED
    candidate_warning_pct: float = -0.005
    candidate_initial_stop_pct: float = -0.010
    candidate_max_stop_pct: float = -0.025
    candidate_emergency_pct: float = -0.030       # may be TIGHTENED, never loosened
    shadow_recommendations: bool = True
    # experiment budgets
    max_configs_per_hypothesis: int = 8
    search_method: str = "tpe"
    reward_coefficients: dict = field(default_factory=lambda: {
        "turnover_penalty": 0.1, "drawdown_penalty": 1.0, "tail_risk_penalty": 0.5})
    version: str = OPTIMIZATION_CONFIG_VERSION

    def validate(self) -> "OptimizationConfig":
        if self.production_activation_enabled or self.autonomous_live_activation:
            raise ConfigError("live activation must remain disabled in Milestone 6B")
        if self.autonomous_risk_expansion:
            raise ConfigError("autonomous risk expansion is prohibited")
        if self.hard_stop_optimization:
            raise ConfigError("hard-stop / mandatory-protection optimization is prohibited")
        if self.live_exploration:
            raise ConfigError("live exploration is prohibited")
        if self.production_exit_pct != -0.005:
            raise ConfigError("the production -0.5% exit must be preserved unchanged")
        # emergency may be tightened (more negative is looser) -> never looser than -3%
        if self.candidate_emergency_pct < -0.030 - 1e-9:
            raise ConfigError("emergency threshold may not be loosened beyond -3%")
        if self.candidate_initial_stop_pct < self.candidate_max_stop_pct - 1e-9:
            raise ConfigError("initial stop must not be wider than the max adaptive stop")
        return self

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


def _env_bool(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    return default if v is None else v not in ("", "0", "false", "False")


def load_config() -> OptimizationConfig:
    """Load the governed config from defaults + CENGINE_OPT_* overrides, validated."""
    cfg = OptimizationConfig(
        enabled=_env_bool("CENGINE_OPT_ENABLED", True),
        training_during_market_hours=_env_bool("CENGINE_OPT_TRAIN_IN_SESSION", False),
        preferred_device=os.environ.get("CENGINE_OPT_DEVICE", "metal_gpu"),
        max_configs_per_hypothesis=int(os.environ.get("CENGINE_OPT_MAX_CONFIGS", "8")),
        search_method=os.environ.get("CENGINE_OPT_SEARCH", "tpe"))
    return cfg.validate()
