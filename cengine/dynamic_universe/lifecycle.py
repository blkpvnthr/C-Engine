from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from typing import Iterable

from .config import UniverseRotationConfig


class LifecycleState(str, Enum):
    DISCOVERED = "discovered"
    ELIGIBLE = "eligible"
    RANKED = "ranked"
    ACTIVE = "active"
    ROTATED_OUT = "rotated_out"
    REASSESSMENT = "reassessment"
    EXPIRED = "expired"


@dataclass(frozen=True)
class CandidateState:
    symbol: str
    state: LifecycleState
    is_baseline: bool
    last_eval_s: float
    entered_active_s: float | None = None
    rotated_out_s: float | None = None
    last_score: float = 0.0
    reason: str = ""


class LifecycleTracker:
    """Per-symbol candidate lifecycle with residence/revisit/hysteresis rules.

    Pure: all time is passed in as ``now_s`` so behavior is reproducible.
    Baseline membership is permanent — baseline symbols can be ROTATED_OUT or
    REASSESSMENT but never EXPIRED.

    ``last_eval_s`` semantics: the last time a symbol was *actually evaluated*
    (i.e. placed in the active scanning window where it receives live data).
    Registration via :meth:`observe` with ``stamp_eval=False`` records
    membership/score/state without advancing ``last_eval_s`` — so a rotated-out
    baseline symbol grows progressively "stale" until the revisit guarantee
    re-schedules it. Only active placement (:meth:`mark_active`) refreshes it.
    """

    def __init__(self, config: UniverseRotationConfig) -> None:
        self._cfg = config
        self._states: dict[str, CandidateState] = {}

    def get(self, symbol: str) -> CandidateState | None:
        return self._states.get(symbol)

    def all(self) -> dict[str, CandidateState]:
        return dict(self._states)

    def restore(self, states: "Iterable[CandidateState]") -> None:
        """Replace the tracked states (used by persistent restart recovery)."""
        self._states = {s.symbol: s for s in states}

    def observe(
        self,
        symbol: str,
        *,
        is_baseline: bool,
        eligible: bool,
        ranked: bool,
        score: float,
        now_s: float,
        stamp_eval: bool = True,
    ) -> CandidateState:
        cur = self._states.get(symbol)
        if ranked:
            new_state = LifecycleState.RANKED
        elif eligible:
            new_state = LifecycleState.ELIGIBLE
        else:
            new_state = LifecycleState.DISCOVERED
        # Do not regress an ACTIVE symbol back to RANKED via observe(); active
        # transitions are explicit (mark_active/mark_rotated_out).
        if cur is not None and cur.state is LifecycleState.ACTIVE:
            new_state = LifecycleState.ACTIVE
        # last_eval_s advances only on a genuine evaluation (stamp_eval), or when
        # the symbol is first seen (no prior state to preserve).
        if stamp_eval or cur is None:
            last_eval = now_s
        else:
            last_eval = cur.last_eval_s
        state = CandidateState(
            symbol=symbol,
            state=new_state,
            is_baseline=is_baseline,
            last_eval_s=last_eval,
            entered_active_s=cur.entered_active_s if cur else None,
            rotated_out_s=cur.rotated_out_s if cur else None,
            last_score=score,
            reason=cur.reason if cur else "",
        )
        self._states[symbol] = state
        return state

    def mark_active(self, symbol: str, now_s: float, reason: str) -> None:
        cur = self._states.get(symbol)
        base = cur.is_baseline if cur else False
        already_active = cur is not None and cur.state is LifecycleState.ACTIVE
        # Preserve the original entry time across consecutive active cycles so
        # minimum-residence measures time-since-first-entry, not since last cycle.
        entered = (
            cur.entered_active_s
            if (already_active and cur is not None and cur.entered_active_s is not None)
            else now_s
        )
        self._states[symbol] = replace(
            cur or CandidateState(symbol, LifecycleState.ACTIVE, base, now_s),
            state=LifecycleState.ACTIVE,
            entered_active_s=entered,
            reason=reason,
            last_eval_s=now_s,
        )

    def mark_rotated_out(self, symbol: str, now_s: float, reason: str) -> None:
        cur = self._states.get(symbol)
        base = cur.is_baseline if cur else False
        self._states[symbol] = replace(
            cur or CandidateState(symbol, LifecycleState.ROTATED_OUT, base, now_s),
            state=LifecycleState.ROTATED_OUT,
            rotated_out_s=now_s,
            reason=reason,
        )

    def can_demote(self, symbol: str, now_s: float, *, force: bool = False) -> bool:
        if force:
            return True
        cur = self._states.get(symbol)
        if cur is None or cur.entered_active_s is None:
            return True
        return (now_s - cur.entered_active_s) >= self._cfg.candidate_min_residence_seconds

    def needs_revisit(self, symbol: str, now_s: float) -> bool:
        cur = self._states.get(symbol)
        if cur is None or not cur.is_baseline:
            return False
        return (now_s - cur.last_eval_s) >= self._cfg.baseline_max_revisit_seconds

    def reassess(self, symbol: str, now_s: float) -> None:
        cur = self._states.get(symbol)
        if cur is None:
            return
        self._states[symbol] = replace(
            cur, state=LifecycleState.REASSESSMENT, last_eval_s=now_s
        )

    def expire(self, symbol: str, now_s: float, reason: str) -> None:
        cur = self._states.get(symbol)
        if cur is not None and cur.is_baseline:
            raise ValueError(f"baseline symbol {symbol!r} may never EXPIRE")
        base = cur.is_baseline if cur else False
        self._states[symbol] = replace(
            cur or CandidateState(symbol, LifecycleState.EXPIRED, base, now_s),
            state=LifecycleState.EXPIRED,
            reason=reason,
            last_eval_s=now_s,
        )
