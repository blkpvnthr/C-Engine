"""Continuous 24/7 autonomous runtime (Milestone 7 addendum).

Exchange-calendar-aware operating-state machine + runtime supervisor that keeps C-Engine
running indefinitely across sessions, overnight, weekends and holidays — reusing the existing
M1–M6B workers. Continuous operation never implies continuous execution: elective orders are
confined to authorized sessions, while protective monitoring + reconciliation run always and
research/learning use off-session GPU capacity. No new financial authority.
"""

from cengine.runtime.calendar import (
    CALENDAR_VERSION,
    CalendarSummary,
    SessionPhase,
    is_regular_session,
    is_trading_day,
    next_transition,
    session_phase,
)
from cengine.runtime.state_machine import (
    STATE_MACHINE_VERSION,
    OperatingState,
    OperatingStateMachine,
    derive_state,
    execution_permitted,
    research_appropriate,
)
from cengine.runtime.supervisor import (
    SUPERVISOR_VERSION,
    Priority,
    RuntimeSupervisor,
    SingleInstanceError,
    SingleInstanceLock,
    read_checkpoint,
    write_checkpoint,
)

__all__ = [
    "CALENDAR_VERSION",
    "STATE_MACHINE_VERSION",
    "SUPERVISOR_VERSION",
    "CalendarSummary",
    "OperatingState",
    "OperatingStateMachine",
    "Priority",
    "RuntimeSupervisor",
    "SessionPhase",
    "SingleInstanceError",
    "SingleInstanceLock",
    "derive_state",
    "execution_permitted",
    "is_regular_session",
    "is_trading_day",
    "next_transition",
    "read_checkpoint",
    "research_appropriate",
    "session_phase",
    "write_checkpoint",
]
