"""Deterministic protective-exit state machine (Milestone 6B, §15).

An explicit, testable candidate state machine composing the AdaptiveExitPolicy. It enforces:
deterministic precedence (a triggered mandatory/emergency boundary always wins, from ANY
state), idempotent transitions, duplicate-event protection, stale-data and invalid-prediction
handling (-> fallback to the incumbent exit), and reconciliation before replacement. It is a
RECOMMENDATION state machine — it emits intended actions, never orders; native C++ retains
authority. The production baseline is untouched until certification.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from cengine.optimization.exits.adaptive import (
    AdaptiveExitConfig,
    AdaptiveExitPolicy,
    ExitAction,
)

EXIT_STATE_MACHINE_VERSION = "exit-state-machine-1.0.0"


class ExitState(str, Enum):
    ACTIVE_POSITION = "active_position"
    WARNING = "warning"
    EVALUATE = "evaluate_position"
    HOLD = "hold"
    REDUCE = "reduce"
    RECONCILE_REDUCTION = "reconcile_reduction"
    EXIT = "exit"
    REPLACE = "replace"
    MANDATORY_EXIT = "mandatory_exit"
    FALLBACK = "fallback"
    RECONCILE_EXIT = "reconcile_exit"
    REPLACEMENT_ELIGIBILITY = "replacement_eligibility"
    REPLACEMENT_CANDIDATE = "replacement_candidate"
    CLOSED = "closed"


_TERMINAL = {ExitState.CLOSED}


@dataclass
class ExitEvent:
    event_id: str
    total_pl_frac: float
    active_stop_pct: float
    now_ns: int
    intelligence: dict = field(default_factory=dict)
    replacement_available: bool = False
    reconciliation_ok: bool = True
    stale: bool = False


@dataclass
class Transition:
    from_state: str
    to_state: str
    action: Optional[str]
    reason: str


class ProtectiveExitStateMachine:
    def __init__(self, config: Optional[AdaptiveExitConfig] = None) -> None:
        self.cfg = config or AdaptiveExitConfig()
        self.policy = AdaptiveExitPolicy(self.cfg)
        self.state = ExitState.ACTIVE_POSITION
        self.history: list = []
        self._seen_events: set = set()

    def _emit(self, to: ExitState, action, reason: str) -> Transition:
        t = Transition(self.state.value, to.value, action, reason)
        self.state = to
        self.history.append(t)
        return t

    def step(self, event: ExitEvent) -> Transition:
        """Advance the machine by one observation. Idempotent on duplicate event ids and
        deterministic: a triggered boundary always routes to MANDATORY_EXIT."""
        if self.state in _TERMINAL:
            return Transition(self.state.value, self.state.value, None, "terminal")
        if event.event_id in self._seen_events:
            return Transition(self.state.value, self.state.value, None, "duplicate_event")
        self._seen_events.add(event.event_id)

        # highest precedence: emergency / adaptive-stop boundary from ANY non-terminal state
        if event.total_pl_frac <= self.cfg.emergency_pct:
            return self._emit(ExitState.MANDATORY_EXIT, ExitAction.EXIT_POSITION.value,
                              "emergency_trigger")
        if event.total_pl_frac <= event.active_stop_pct:
            return self._emit(ExitState.MANDATORY_EXIT, ExitAction.EXIT_POSITION.value,
                              "adaptive_stop_hit")

        # stale data or (at/under warning with) invalid intelligence -> fallback to incumbent
        at_warning = event.total_pl_frac <= self.cfg.warning_pct
        if self.state in (ExitState.RECONCILE_EXIT, ExitState.REPLACEMENT_ELIGIBILITY,
                          ExitState.REPLACEMENT_CANDIDATE):
            return self._reconcile(event)
        if event.stale and at_warning:
            self._emit(ExitState.FALLBACK, None, "stale_data")
            return self._emit(ExitState.MANDATORY_EXIT, ExitAction.EXIT_POSITION.value,
                              "incumbent_threshold")

        if not at_warning:
            return self._emit(ExitState.ACTIVE_POSITION, ExitAction.HOLD.value,
                              "above_warning")

        # at/under warning -> WARNING -> EVALUATE via the adaptive policy
        self._emit(ExitState.WARNING, None, "warning_threshold")
        self._emit(ExitState.EVALUATE, None, "reassess")
        decision = self.policy.decide(
            total_pl_frac=event.total_pl_frac, active_stop_pct=event.active_stop_pct,
            intelligence=event.intelligence,
            replacement_available=event.replacement_available)
        if decision.fallback:
            self._emit(ExitState.FALLBACK, None, "invalid_intelligence")
            return self._emit(ExitState.MANDATORY_EXIT, ExitAction.EXIT_POSITION.value,
                              "incumbent_threshold")
        if decision.action == ExitAction.HOLD.value:
            return self._emit(ExitState.ACTIVE_POSITION, ExitAction.HOLD.value,
                              decision.reason)
        if decision.action == ExitAction.REDUCE_POSITION.value:
            self._emit(ExitState.REDUCE, ExitAction.REDUCE_POSITION.value, decision.reason)
            return self._emit(ExitState.RECONCILE_REDUCTION, None, "await_reduction_fill")
        if decision.action == ExitAction.EXIT_AND_REPLACE.value:
            self._emit(ExitState.REPLACE, ExitAction.EXIT_AND_REPLACE.value, decision.reason)
            return self._emit(ExitState.EXIT, ExitAction.EXIT_POSITION.value,
                              "exit_before_replace")
        return self._emit(ExitState.EXIT, ExitAction.EXIT_POSITION.value, decision.reason)

    def _reconcile(self, event: ExitEvent) -> Transition:
        if self.state == ExitState.RECONCILE_EXIT:
            if not event.reconciliation_ok:
                return Transition(self.state.value, self.state.value, None,
                                  "await_reconciliation")
            return self._emit(ExitState.REPLACEMENT_ELIGIBILITY, None, "reconciled")
        if self.state == ExitState.REPLACEMENT_ELIGIBILITY:
            if event.replacement_available and event.reconciliation_ok:
                return self._emit(ExitState.REPLACEMENT_CANDIDATE, "enter_candidate",
                                  "independent_approval")
            return self._emit(ExitState.CLOSED, None, "not_eligible")
        return self._emit(ExitState.CLOSED, None, "replacement_done")

    def post_exit(self, event: ExitEvent) -> Transition:
        """Drive the reconcile->replacement->closed tail after an EXIT/MANDATORY_EXIT."""
        if self.state in (ExitState.EXIT, ExitState.MANDATORY_EXIT):
            self._emit(ExitState.RECONCILE_EXIT, None, "await_exit_fill")
        return self.step(event)

    def terminated(self) -> bool:
        return self.state in _TERMINAL
