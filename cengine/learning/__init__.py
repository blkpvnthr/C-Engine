"""C-Engine adaptive ML / constrained offline-RL learning layer (Milestone 6).

A GPU-first (Apple Metal via MLX), locally operated learning platform that learns from
historical market data, the engine's own decisions, portfolio allocations, execution
policies and realized outcomes — to IMPROVE RANKING and ELIGIBLE-POLICY SELECTION only.

Non-negotiable invariants (enforced structurally, not by convention):
  * Models are recommendation systems. They NEVER submit orders, open a broker
    connection, run a parallel order manager, or bypass the native financial authority
    (RiskEngine / constraints / reservations / exposure authorization).
  * The strict elective-entry contract (DI+>DI- AND price>VWAP|SMA20|BB-mid AND
    developing-1m>prev-1m-close) is preserved; learning cannot override it.
  * The mandatory -0.5% protective exit stays fully independent of MLX/Metal/GPU, model
    training/inference, the registry, and the lakehouse. No reward, forecast or policy
    may suppress or defer it.
  * No fabricated labels, rewards, or behavior-policy probabilities. Insufficient
    evidence yields an UNTRAINED / RESEARCH-only model, never a promoted one.
  * GPU-first with a DOCUMENTED, OBSERVABLE, MEASURABLE CPU fallback — never a silent
    claim of GPU acceleration when work ran on the CPU.

The lakehouse (cengine/lakehouse) is the authoritative analytical data interface; the
authoritative financial journal stays separate and untouched.
"""

from cengine.learning.hardware import (
    HARDWARE_REPORT_VERSION,
    DevicePolicy,
    HardwareCapabilities,
    inspect_hardware,
    mlx_available,
)

LEARNING_PLATFORM_VERSION = "learning-platform-1.0.0"
FEATURE_SCHEMA_VERSION = "feature-schema-1.0.0"
LABEL_DEFINITION_VERSION = "label-def-1.0.0"

__all__ = [
    "FEATURE_SCHEMA_VERSION",
    "HARDWARE_REPORT_VERSION",
    "LABEL_DEFINITION_VERSION",
    "LEARNING_PLATFORM_VERSION",
    "DevicePolicy",
    "HardwareCapabilities",
    "inspect_hardware",
    "mlx_available",
]
