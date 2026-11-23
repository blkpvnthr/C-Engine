"""Continuous-runtime operating-state machine (Milestone 7 addendum §4).

Derives the engine's operating state from the exchange calendar (time-driven) with operational
overrides (DEGRADED/RECOVERING/SHUTTING_DOWN set by the supervisor). Continuous operation does
NOT imply continuous execution: only REGULAR_SESSION (and, when explicitly authorized, PRE_MARKET/
AFTER_HOURS) permit elective orders; research/analytics run in the off-session states. Protective
monitoring + reconciliation run in ALL states.
"""

from __future__ import annotations

import datetime as _dt
from enum import Enum
from zoneinfo import ZoneInfo

from cengine.runtime.calendar import SessionPhase, is_trading_day, session_phase

STATE_MACHINE_VERSION = "operating-state-machine-1.0.0"
_ET = ZoneInfo("America/New_York")
NS = 1_000_000_000


class OperatingState(str, Enum):
    INITIALIZING = "initializing"
    OVERNIGHT_RESEARCH = "overnight_research"
    PRE_MARKET = "pre_market"
    REGULAR_SESSION = "regular_session"
    AFTER_HOURS = "after_hours"
    POST_SESSION_PROCESSING = "post_session_processing"
    WEEKEND_RESEARCH = "weekend_research"
    HOLIDAY_RESEARCH = "holiday_research"
    DEGRADED = "degraded"
    RECOVERING = "recovering"
    SHUTTING_DOWN = "shutting_down"


# states in which elective equity orders may be submitted (if separately authorized).
_EXECUTION_STATES = {OperatingState.REGULAR_SESSION}
# states where research/training/backfill are appropriate (ample off-session GPU).
_RESEARCH_STATES = {OperatingState.OVERNIGHT_RESEARCH, OperatingState.WEEKEND_RESEARCH,
                    OperatingState.HOLIDAY_RESEARCH, OperatingState.POST_SESSION_PROCESSING}


def derive_state(ts_ns: int) -> OperatingState:
    """Time-driven operating state (before operational overrides)."""
    now = _dt.datetime.fromtimestamp(ts_ns / 1e9, tz=_ET)
    d = now.date()
    if not is_trading_day(d):
        return (OperatingState.WEEKEND_RESEARCH if d.weekday() >= 5
                else OperatingState.HOLIDAY_RESEARCH)
    phase = session_phase(ts_ns)
    if phase == SessionPhase.PRE_MARKET:
        return OperatingState.PRE_MARKET
    if phase == SessionPhase.REGULAR:
        return OperatingState.REGULAR_SESSION
    if phase == SessionPhase.AFTER_HOURS:
        return OperatingState.AFTER_HOURS
    # CLOSED on a trading day: post-session processing window just after after-hours, else overnight
    mins = now.hour * 60 + now.minute
    if 20 * 60 <= mins < 21 * 60:
        return OperatingState.POST_SESSION_PROCESSING
    return OperatingState.OVERNIGHT_RESEARCH


def execution_permitted(state: OperatingState) -> bool:
    """Whether elective equity orders are permitted in this state (authorization still required)."""
    return state in _EXECUTION_STATES


def research_appropriate(state: OperatingState) -> bool:
    return state in _RESEARCH_STATES


class OperatingStateMachine:
    """Tracks the operating state with operational overrides. Protective monitoring +
    reconciliation are ALWAYS active regardless of state."""

    def __init__(self) -> None:
        self.state = OperatingState.INITIALIZING
        self._override: OperatingState | None = None
        self.transitions: list = []

    def set_degraded(self, reason: str, now_ns: int) -> None:
        self._override = OperatingState.DEGRADED
        self._record(OperatingState.DEGRADED, now_ns, reason)

    def set_recovering(self, now_ns: int) -> None:
        self._override = OperatingState.RECOVERING
        self._record(OperatingState.RECOVERING, now_ns, "recovering")

    def clear_override(self) -> None:
        self._override = None

    def begin_shutdown(self, now_ns: int) -> None:
        self._override = OperatingState.SHUTTING_DOWN
        self._record(OperatingState.SHUTTING_DOWN, now_ns, "shutdown")

    def tick(self, now_ns: int) -> OperatingState:
        new = self._override or derive_state(now_ns)
        if new != self.state:
            self._record(new, now_ns, "time_transition")
            self.state = new
        return self.state

    def _record(self, to: OperatingState, now_ns: int, reason: str) -> None:
        self.transitions.append({"to": to.value, "ns": now_ns, "reason": reason})
        if len(self.transitions) > 1000:
            self.transitions = self.transitions[-500:]

    def execution_permitted(self) -> bool:
        return (self._override is None) and execution_permitted(self.state)

    def protection_active(self) -> bool:
        # protective monitoring + reconciliation are active in EVERY state (even degraded/shutdown)
        return True
