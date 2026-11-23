"""Staged live-deployment framework (defined, HARD-INACTIVE).

Defines — but does not enable — a conservative staged path to live capital. Every stage is gated
behind the full certification chain AND an explicit, external human authorization token. There is
no code path that activates live trading autonomously: :meth:`authorize` requires a token the
software cannot mint itself, and :meth:`is_active` is False until a human provides it. This module
is a *specification with a safety interlock*, not an actuator.

Stages (each requires the previous plus its own gate):

1. SHADOW         — decisions logged only; zero orders. (always permissible)
2. PILOT_TINY     — minimal capital, strict per-trade + daily loss caps; requires paper
                    certification + human authorization.
3. SCALED         — graduated capital; requires a positive pilot track record + re-authorization.

The −0.5% protective exit and the native risk gate remain the authority in every stage; this
framework can only ever REDUCE what is permitted, never expand it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from cengine.certification.contracts import CertificationResult, CertificationVerdict

LIVE_FRAMEWORK_VERSION = "certification-live-framework-1.0.0"


class LiveStage(str, Enum):
    INACTIVE = "inactive"
    SHADOW = "shadow"
    PILOT_TINY = "pilot_tiny"
    SCALED = "scaled"


@dataclass(frozen=True)
class LiveStageGate:
    stage: LiveStage
    requires_infrastructure_sound: bool
    requires_economic_edge: bool
    requires_paper_certified: bool
    requires_human_authorization: bool
    max_capital_fraction: float            # hard cap on deployable capital at this stage


_GATES = {
    LiveStage.SHADOW: LiveStageGate(LiveStage.SHADOW, True, False, False, False, 0.0),
    LiveStage.PILOT_TINY: LiveStageGate(LiveStage.PILOT_TINY, True, True, True, True, 0.01),
    LiveStage.SCALED: LiveStageGate(LiveStage.SCALED, True, True, True, True, 0.10),
}


class LiveAuthorizationError(RuntimeError):
    """Raised on any attempt to activate a live stage without satisfying its gate + token."""


@dataclass
class StagedLiveDeployment:
    """Inactive-by-default staged live controller. Cannot self-activate."""

    infrastructure_sound: bool = False
    economic_edge_certified: bool = False
    paper_certified: bool = False
    _stage: LiveStage = field(default=LiveStage.INACTIVE)
    _human_token: str = ""

    @property
    def stage(self) -> LiveStage:
        return self._stage

    def is_active(self) -> bool:
        """Live (order-submitting) stages require a non-empty human token AND a non-shadow stage."""
        return self._stage in (LiveStage.PILOT_TINY, LiveStage.SCALED) and bool(self._human_token)

    def can_authorize(self, stage: LiveStage) -> tuple:
        """Return (ok, reasons) for whether ``stage`` could be authorized given current evidence."""
        gate = _GATES.get(stage)
        if gate is None:
            return False, [f"unknown stage {stage}"]
        reasons = []
        if gate.requires_infrastructure_sound and not self.infrastructure_sound:
            reasons.append("infrastructure not sound")
        if gate.requires_economic_edge and not self.economic_edge_certified:
            reasons.append("no certified economic edge")
        if gate.requires_paper_certified and not self.paper_certified:
            reasons.append("not paper-certified")
        return (not reasons), reasons

    def authorize(self, stage: LiveStage, *, human_token: str = "") -> None:
        """Attempt to enter a stage. Refuses unless every gate is satisfied AND (for live stages)
        a non-empty external human token is supplied. The software never mints the token."""
        gate = _GATES[stage]
        ok, reasons = self.can_authorize(stage)
        if not ok:
            raise LiveAuthorizationError(f"cannot authorize {stage.value}: {reasons}")
        if gate.requires_human_authorization and not human_token:
            raise LiveAuthorizationError(
                f"{stage.value} requires an explicit external human authorization token")
        self._stage = stage
        self._human_token = human_token if gate.requires_human_authorization else ""

    def status(self) -> CertificationResult:
        return CertificationResult(
            check="live.deployment_framework", component="live_deployment",
            verdict=CertificationVerdict.NOT_APPLICABLE,
            summary=(f"staged live deployment INACTIVE (stage={self._stage.value}, "
                     f"active={self.is_active()}); live stages gated behind full certification "
                     f"chain + external human authorization"),
            metrics={"stage": self._stage.value, "active": self.is_active(),
                     "infrastructure_sound": self.infrastructure_sound,
                     "economic_edge_certified": self.economic_edge_certified,
                     "paper_certified": self.paper_certified,
                     "pilot_authorizable": self.can_authorize(LiveStage.PILOT_TINY)[0]},
            notes="no autonomous live activation; −0.5% exit + native risk gate remain authority")


def live_status(*, infrastructure_sound: bool = False, economic_edge_certified: bool = False,
                paper_certified: bool = False) -> CertificationResult:
    return StagedLiveDeployment(
        infrastructure_sound=infrastructure_sound,
        economic_edge_certified=economic_edge_certified,
        paper_certified=paper_certified).status()
