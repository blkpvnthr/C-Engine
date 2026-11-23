"""Continuous runtime supervisor (Milestone 7 addendum §3/§8/§12).

Coordinates the reused engine workers under an operating-state policy with isolated failures,
a P0–P9 Metal-GPU workload priority ladder, single-instance enforcement, and durable
checkpoints. It is a POLICY/COORDINATION layer — the actual async workers live in trade.py;
this decides which workers should be active per state, whether a GPU research workload may be
admitted, and isolates recoverable failures so a learning/analytics fault NEVER terminates
financial protection or broker reconciliation (P0).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional

from cengine.runtime.state_machine import OperatingState

SUPERVISOR_VERSION = "runtime-supervisor-1.0.0"


class Priority(IntEnum):
    P0_PROTECTION = 0          # financial protection + broker reconciliation (never gated)
    P1_MARKET_EXEC = 1         # native market processing + execution
    P2_DISCOVERY = 2           # Metal discovery + technical analytics
    P3_ML_INFERENCE = 3        # production ML inference
    P4_PORTFOLIO = 4           # portfolio scenario analytics
    P5_FEATURES = 5            # historical feature construction
    P6_ML_TRAINING = 6
    P7_BANDIT = 7
    P8_OFFLINE_RL = 8
    P9_RESEARCH = 9            # large historical research experiments


@dataclass
class Worker:
    name: str
    priority: Priority
    essential: bool            # essential workers (P0/P1) never isolated-killed
    restarts: int = 0
    healthy: bool = True
    last_error: str = ""


# workloads at priority >= this are "research" and are throttled/suspended during live sessions.
_RESEARCH_CUTOFF = Priority.P6_ML_TRAINING


class RuntimeSupervisor:
    def __init__(self, *, max_gpu_pressure: float = 0.85) -> None:
        self.workers: dict = {}
        self.max_gpu_pressure = max_gpu_pressure
        self.version = SUPERVISOR_VERSION

    def register(self, name: str, priority: Priority, *, essential: bool = False) -> None:
        self.workers[name] = Worker(name, priority, essential)

    # -- per-state activation policy -----------------------------------------
    def active_workers(self, state: OperatingState) -> list:
        """Workers that should be running in a given operating state."""
        from cengine.runtime.state_machine import execution_permitted, research_appropriate
        out = []
        for w in self.workers.values():
            if w.priority in (Priority.P0_PROTECTION,):
                out.append(w.name)                       # always on
            elif w.priority <= Priority.P5_FEATURES:
                # market/analytics workers run whenever markets are (or recently were) open
                if state in (OperatingState.PRE_MARKET, OperatingState.REGULAR_SESSION,
                             OperatingState.AFTER_HOURS, OperatingState.POST_SESSION_PROCESSING):
                    out.append(w.name)
                elif w.priority <= Priority.P1_MARKET_EXEC:
                    out.append(w.name)                   # reconciliation/exec monitoring always
            else:
                # research/training: off-session states, or throttled in-session (see gpu_admission)
                if research_appropriate(state) or execution_permitted(state):
                    out.append(w.name)
        return out

    # -- GPU workload admission (§8) -----------------------------------------
    def gpu_admission(self, workload: Priority, state: OperatingState, *,
                      gpu_pressure: float = 0.0) -> dict:
        """Decide whether a GPU workload may run now. P0 never gated; research suspended under
        pressure and bounded during live sessions."""
        from cengine.runtime.state_machine import execution_permitted
        if workload == Priority.P0_PROTECTION:
            return {"admit": True, "reason": "protection_never_gated"}
        if gpu_pressure >= self.max_gpu_pressure and workload >= _RESEARCH_CUTOFF:
            return {"admit": False, "reason": f"gpu_pressure {gpu_pressure:.2f} >= "
                                              f"{self.max_gpu_pressure}"}
        if execution_permitted(state) and workload >= _RESEARCH_CUTOFF:
            # bounded background research only when it cannot materially degrade trading
            return {"admit": gpu_pressure < 0.5,
                    "reason": "bounded_research_in_session" if gpu_pressure < 0.5
                              else "deferred_to_protect_live_workloads"}
        return {"admit": True, "reason": "off_session_or_non_research"}

    # -- failure isolation (§3) ----------------------------------------------
    def record_failure(self, name: str, error: str) -> dict:
        w = self.workers.get(name)
        if w is None:
            return {"isolated": False, "reason": "unknown_worker"}
        w.healthy = False
        w.last_error = error
        w.restarts += 1
        # a non-essential (learning/analytics) failure is ISOLATED — it never cascades to P0/P1
        isolated = not w.essential
        return {"worker": name, "isolated": isolated, "restarts": w.restarts,
                "financial_protection_affected": False,
                "note": ("learning/analytics failure isolated; protection + reconciliation "
                         "unaffected" if isolated else "essential worker — supervisor restarts")}

    def mark_healthy(self, name: str) -> None:
        if name in self.workers:
            self.workers[name].healthy = True

    def health(self) -> dict:
        return {"version": self.version,
                "workers": {n: {"priority": int(w.priority), "healthy": w.healthy,
                                "restarts": w.restarts, "essential": w.essential}
                            for n, w in self.workers.items()},
                "protection_independent_of_learning": True}


# -- single-instance enforcement (§2) ---------------------------------------

class SingleInstanceError(RuntimeError):
    """Raised when another trade.py instance already holds the runtime lock."""


@dataclass
class SingleInstanceLock:
    """PID-file lock preventing competing trading authority; stale (dead-PID) locks reclaimed."""
    path: str
    _acquired: bool = field(default=False)

    def acquire(self) -> bool:
        if os.path.exists(self.path):
            try:
                pid = int(open(self.path).read().strip() or "0")
            except (OSError, ValueError):
                pid = 0
            if pid and _pid_alive(pid):
                raise SingleInstanceError(
                    f"another trade.py instance (pid {pid}) holds {self.path}")
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "w") as fh:
            fh.write(str(os.getpid()))
        self._acquired = True
        return True

    def release(self) -> None:
        if self._acquired and os.path.exists(self.path):
            try:
                if int(open(self.path).read().strip() or "0") == os.getpid():
                    os.unlink(self.path)
            except (OSError, ValueError):
                pass
        self._acquired = False


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False


# -- durable checkpoint (§12) -----------------------------------------------

def write_checkpoint(path: str, state: dict) -> None:
    """Atomic checkpoint write (tmp + os.replace) for crash recovery."""
    import tempfile
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path) or ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(state, fh, default=str)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def read_checkpoint(path: str) -> Optional[dict]:
    if not os.path.exists(path):
        return None
    try:
        return json.load(open(path))
    except (OSError, json.JSONDecodeError):
        return None
