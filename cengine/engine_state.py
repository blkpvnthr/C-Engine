"""Explicit engine lifecycle and operator kill switch."""

from __future__ import annotations

from enum import Enum
from threading import RLock


class EngineState(str, Enum):
    CREATED = "created"
    RECONCILING = "reconciling"
    READY = "ready"
    RUNNING = "running"
    DEGRADED = "degraded"
    KILLED = "killed"
    STOPPING = "stopping"
    STOPPED = "stopped"


_ALLOWED = {
    EngineState.CREATED: {EngineState.RECONCILING, EngineState.KILLED},
    EngineState.RECONCILING: {EngineState.READY, EngineState.DEGRADED, EngineState.KILLED},
    EngineState.READY: {EngineState.RUNNING, EngineState.KILLED, EngineState.STOPPING},
    EngineState.RUNNING: {EngineState.DEGRADED, EngineState.KILLED, EngineState.STOPPING},
    EngineState.DEGRADED: {EngineState.RECONCILING, EngineState.KILLED, EngineState.STOPPING},
    EngineState.KILLED: {EngineState.RECONCILING, EngineState.STOPPING},
    EngineState.STOPPING: {EngineState.STOPPED},
    EngineState.STOPPED: set(),
}


class EngineStateMachine:
    def __init__(self) -> None:
        self._state = EngineState.CREATED
        self._reason = ""
        self._lock = RLock()

    @property
    def state(self) -> EngineState:
        with self._lock:
            return self._state

    @property
    def reason(self) -> str:
        with self._lock:
            return self._reason

    def transition(self, target: EngineState, reason: str = "") -> None:
        with self._lock:
            if target not in _ALLOWED[self._state]:
                raise RuntimeError(
                    f"invalid engine transition {self._state.value} -> {target.value}"
                )
            self._state, self._reason = target, reason

    def kill(self, reason: str) -> None:
        if not reason.strip():
            raise ValueError("kill-switch reason is required")
        with self._lock:
            if self._state is not EngineState.STOPPED:
                self._state, self._reason = EngineState.KILLED, reason

    def require_ordering_enabled(self) -> None:
        if self.state is not EngineState.RUNNING:
            raise RuntimeError(f"order submission blocked while engine is {self.state.value}")
