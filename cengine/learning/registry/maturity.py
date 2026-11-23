"""Model maturity lifecycle (Milestone 6, §23).

Model maturity is an INDEPENDENT dimension from strategy maturity and from financial
execution authorization. The cardinal rules enforced here:

  * No model auto-promotes itself to live execution.
  * Independent validation must precede any promotion past SHADOW.
  * Explicit human/operator approval is required for LIVE_APPROVED / ACTIVE.
  * Rollback to the last approved model (or the deterministic baseline) is always legal.

This module only governs the STATE MACHINE + authorization; it holds no model weights and
submits no orders.
"""

from __future__ import annotations

from enum import Enum

MATURITY_VERSION = "model-maturity-1.0.0"


class ModelMaturity(str, Enum):
    UNTRAINED = "untrained"
    RESEARCH = "research"
    TRAINED = "trained"
    VALIDATED = "validated"
    SHADOW = "shadow"
    PAPER_ELIGIBLE = "paper_eligible"
    PAPER_CERTIFIED = "paper_certified"
    LIVE_ELIGIBLE = "live_eligible"
    LIVE_APPROVED = "live_approved"
    ACTIVE = "active"
    DEGRADED = "degraded"
    BLOCKED = "blocked"
    RETIRED = "retired"


class PromotionError(RuntimeError):
    """Raised when an illegal or unauthorized maturity transition is attempted."""


# Legal forward/diagnostic transitions. BLOCKED/RETIRED/DEGRADED are reachable from most
# states (safety sinks). RETIRED is terminal.
_LEGAL: dict = {
    ModelMaturity.UNTRAINED: {ModelMaturity.RESEARCH, ModelMaturity.TRAINED},
    ModelMaturity.RESEARCH: {ModelMaturity.TRAINED},
    ModelMaturity.TRAINED: {ModelMaturity.VALIDATED},
    ModelMaturity.VALIDATED: {ModelMaturity.SHADOW},
    ModelMaturity.SHADOW: {ModelMaturity.PAPER_ELIGIBLE},
    ModelMaturity.PAPER_ELIGIBLE: {ModelMaturity.PAPER_CERTIFIED},
    ModelMaturity.PAPER_CERTIFIED: {ModelMaturity.LIVE_ELIGIBLE},
    ModelMaturity.LIVE_ELIGIBLE: {ModelMaturity.LIVE_APPROVED},
    ModelMaturity.LIVE_APPROVED: {ModelMaturity.ACTIVE},
    ModelMaturity.ACTIVE: set(),
    ModelMaturity.DEGRADED: {ModelMaturity.ACTIVE, ModelMaturity.SHADOW},
    ModelMaturity.BLOCKED: {ModelMaturity.RESEARCH},
    ModelMaturity.RETIRED: set(),
}

# Safety sinks reachable from any non-terminal state.
_SINKS = {ModelMaturity.DEGRADED, ModelMaturity.BLOCKED, ModelMaturity.RETIRED}

# Transitions that REQUIRE explicit operator approval (never automatic).
_APPROVAL_REQUIRED = {
    (ModelMaturity.LIVE_ELIGIBLE, ModelMaturity.LIVE_APPROVED),
    (ModelMaturity.LIVE_APPROVED, ModelMaturity.ACTIVE),
}

# States that constitute "live" financial authority (gated hard).
_LIVE_STATES = {ModelMaturity.LIVE_APPROVED, ModelMaturity.ACTIVE}


def is_live(state: ModelMaturity) -> bool:
    return state in _LIVE_STATES


def requires_explicit_approval(src: ModelMaturity, dst: ModelMaturity) -> bool:
    return (src, dst) in _APPROVAL_REQUIRED or dst in _LIVE_STATES


def can_transition(src: ModelMaturity, dst: ModelMaturity) -> bool:
    if src == ModelMaturity.RETIRED:
        return False
    if dst in _SINKS and src not in _SINKS | {ModelMaturity.RETIRED}:
        return True
    return dst in _LEGAL.get(src, set())


def check_transition(src: ModelMaturity, dst: ModelMaturity, *, approved: bool) -> None:
    """Raise PromotionError if the transition is illegal or unauthorized."""
    if not can_transition(src, dst):
        raise PromotionError(f"illegal maturity transition {src.value} -> {dst.value}")
    if requires_explicit_approval(src, dst) and not approved:
        raise PromotionError(
            f"transition {src.value} -> {dst.value} requires explicit approval "
            f"(live financial authority is never granted automatically)")
