"""Continuous runtime service (addendum §1–§3 / §6 / §10 / §12).

The persistent 24/7 supervisor that keeps C-Engine running across sessions, overnight, weekends
and holidays. It is a COORDINATION layer: it ticks the exchange-calendar-driven operating-state
machine, decides which reused workers/loops should be active per state, drives the off-session
continuous services under the P0–P9 GPU priority ladder, verifies the safety invariants every
tick, isolates failures (a research/learning fault can never stop protection), checkpoints its
state atomically, and enforces single-instance trading authority.

Continuous operation is NOT continuous execution: elective orders are confined to REGULAR_SESSION
(and only via an explicitly-injected, authorized trading-engine runner — absent by default, so
the service is safe to run with no broker). Protective monitoring + governance run in every state.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Optional

from cengine.runtime.continuous import (
    ContinuousAcquisitionService,
    ContinuousGovernanceService,
    ContinuousOptimizationService,
)
from cengine.runtime.state_machine import (
    OperatingState,
    OperatingStateMachine,
    execution_permitted,
    research_appropriate,
)
from cengine.runtime.supervisor import (
    Priority,
    RuntimeSupervisor,
    SingleInstanceLock,
    read_checkpoint,
    write_checkpoint,
)

SERVICE_VERSION = "continuous-runtime-service-1.0.0"
logger = logging.getLogger("cengine.runtime.service")

_MIN = 60 * 1_000_000_000
_HOUR = 60 * _MIN


@dataclass
class ContinuousRuntimeConfig:
    symbols: list = field(default_factory=lambda: ["AAPL", "MSFT", "NVDA", "SPY", "QQQ"])
    tick_interval_s: float = 30.0
    optimization_interval_ns: int = 2 * _HOUR     # off-session, bounded
    acquisition_interval_ns: int = 6 * _HOUR
    governance_full_interval_ns: int = 12 * _HOUR
    checkpoint_interval_ns: int = 15 * _MIN
    lock_path: str = "data/runtime/trade.lock"
    checkpoint_path: str = "data/runtime/checkpoint.json"
    enable_fetch: bool = False                     # acquisition stays dry-run unless enabled


@dataclass
class TickPlan:
    state: OperatingState
    active_workers: list
    run_optimization: bool
    run_acquisition: bool
    run_governance_full: bool
    execution_window: bool
    gpu_pressure: float
    reasons: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"state": self.state.value, "active_workers": self.active_workers,
                "run_optimization": self.run_optimization,
                "run_acquisition": self.run_acquisition,
                "run_governance_full": self.run_governance_full,
                "execution_window": self.execution_window,
                "gpu_pressure": self.gpu_pressure, "reasons": self.reasons}


class ContinuousRuntimeService:
    """Persistent supervisory runtime. Pure planning (:meth:`plan_tick`) is separated from the
    async loop (:meth:`run`) so the scheduling logic is deterministically testable."""

    def __init__(self, config: Optional[ContinuousRuntimeConfig] = None, *,
                 trading_engine_runner: Optional[Callable[[asyncio.Event], Awaitable]] = None,
                 opt_config=None) -> None:
        self.config = config or ContinuousRuntimeConfig()
        self.machine = OperatingStateMachine()
        self.supervisor = RuntimeSupervisor()
        self.opt = ContinuousOptimizationService(symbols=self.config.symbols)
        self.acq = ContinuousAcquisitionService(symbols=self.config.symbols,
                                                enabled_fetch=self.config.enable_fetch)
        self.gov = ContinuousGovernanceService()
        self._trading_runner = trading_engine_runner
        self._opt_config = opt_config
        self._last = {"optimization": 0, "acquisition": 0, "governance_full": 0, "checkpoint": 0}
        self._degraded_reason = ""
        self._register_workers()

    def _register_workers(self) -> None:
        s = self.supervisor
        s.register("protection", Priority.P0_PROTECTION, essential=True)
        s.register("execution", Priority.P1_MARKET_EXEC, essential=True)
        s.register("discovery", Priority.P2_DISCOVERY, essential=False)
        s.register("acquisition", Priority.P5_FEATURES, essential=False)
        s.register("optimization", Priority.P9_RESEARCH, essential=False)

    # -- pure planning -------------------------------------------------------
    def plan_tick(self, now_ns: int, *, gpu_pressure: float = 0.0) -> TickPlan:
        state = self.machine.tick(now_ns)
        active = self.supervisor.active_workers(state)
        reasons: dict = {}

        def _due(key: str, interval: int) -> bool:
            return (now_ns - self._last[key]) >= interval

        research_ok = research_appropriate(state)
        # optimization: off-session only, admitted by the GPU ladder, on cadence
        opt_admit = self.supervisor.gpu_admission(Priority.P9_RESEARCH, state,
                                                   gpu_pressure=gpu_pressure)
        run_opt = (research_ok and opt_admit["admit"]
                   and _due("optimization", self.config.optimization_interval_ns))
        reasons["optimization"] = opt_admit["reason"] if not run_opt else "scheduled"

        acq_admit = self.supervisor.gpu_admission(Priority.P5_FEATURES, state,
                                                  gpu_pressure=gpu_pressure)
        run_acq = (research_ok and acq_admit["admit"]
                   and _due("acquisition", self.config.acquisition_interval_ns))
        reasons["acquisition"] = acq_admit["reason"] if not run_acq else "scheduled"

        run_gov_full = _due("governance_full", self.config.governance_full_interval_ns)

        return TickPlan(state=state, active_workers=active, run_optimization=run_opt,
                        run_acquisition=run_acq, run_governance_full=run_gov_full,
                        execution_window=execution_permitted(state), gpu_pressure=gpu_pressure,
                        reasons=reasons)

    # -- governance / recovery ----------------------------------------------
    def governance_tick(self, now_ns: int) -> dict:
        """Cheap safety-invariant check every tick; DEGRADED on regression, recover when healthy."""
        inv = self.gov.check_invariants()
        if not inv["ok"]:
            self._degraded_reason = f"safety invariant regression: {inv['failures']}"
            self.machine.set_degraded(self._degraded_reason, now_ns)
            self.machine.tick(now_ns)                 # materialize the override onto .state
            logger.error("[runtime] %s", self._degraded_reason)
        elif self._degraded_reason:
            # invariants healthy again -> clear the override and return to the time-driven state
            self.machine.clear_override()
            self.machine.tick(now_ns)
            self._degraded_reason = ""
        return inv

    def _checkpoint(self, now_ns: int, extra: Optional[dict] = None) -> None:
        state = {"version": SERVICE_VERSION, "ns": now_ns,
                 "state": self.machine.state.value, "last": dict(self._last),
                 "degraded_reason": self._degraded_reason, "health": self.supervisor.health()}
        if extra:
            state.update(extra)
        try:
            write_checkpoint(self.config.checkpoint_path, state)
        except OSError as exc:
            logger.warning("[runtime] checkpoint write failed: %s", exc)

    # -- dispatch (sync; called via executor in the loop) --------------------
    def dispatch(self, plan: TickPlan, now_ns: int) -> dict:
        out: dict = {}
        if plan.run_optimization:
            try:
                out["optimization"] = self.opt.run_once(now_ns=now_ns, config=self._opt_config)
                self._last["optimization"] = now_ns
                self.supervisor.mark_healthy("optimization")
            except Exception as exc:  # noqa: BLE001
                out["optimization"] = self.supervisor.record_failure("optimization", str(exc))
        if plan.run_acquisition:
            try:
                out["acquisition"] = self.acq.run_once(config=self._opt_config)
                self._last["acquisition"] = now_ns
                self.supervisor.mark_healthy("acquisition")
            except Exception as exc:  # noqa: BLE001
                out["acquisition"] = self.supervisor.record_failure("acquisition", str(exc))
        if plan.run_governance_full:
            try:
                out["governance_full"] = self.gov.full_recert(self.config.symbols)
                self._last["governance_full"] = now_ns
            except Exception as exc:  # noqa: BLE001
                out["governance_full"] = {"ok": False, "error": str(exc)}
        return out

    # -- async loop ----------------------------------------------------------
    async def run(self, stop: Optional[asyncio.Event] = None, *, max_ticks: int = 0) -> dict:
        """Run the persistent loop until ``stop`` is set (or ``max_ticks`` for bounded runs)."""
        stop = stop or asyncio.Event()
        lock = SingleInstanceLock(self.config.lock_path)
        lock.acquire()                       # raises SingleInstanceError if another instance holds
        prior = read_checkpoint(self.config.checkpoint_path)
        if prior and isinstance(prior.get("last"), dict):
            self._last.update({k: int(v) for k, v in prior["last"].items() if k in self._last})
        logger.info("[runtime] continuous service start (recovered=%s)", bool(prior))
        ticks = 0
        loop = asyncio.get_event_loop()
        trading_task: Optional[asyncio.Task] = None
        trading_stop = asyncio.Event()
        try:
            while not stop.is_set():
                now_ns = time.time_ns()
                self.governance_tick(now_ns)
                plan = self.plan_tick(now_ns)
                # execution window -> hand off to the (optional, authorized) trading engine
                if plan.execution_window and self._trading_runner and trading_task is None:
                    trading_stop = asyncio.Event()
                    trading_task = loop.create_task(self._trading_runner(trading_stop))
                elif not plan.execution_window and trading_task is not None:
                    trading_stop.set()
                    trading_task = None
                # off-session heavy work runs in an executor so the loop never blocks
                if plan.run_optimization or plan.run_acquisition or plan.run_governance_full:
                    await loop.run_in_executor(None, self.dispatch, plan, now_ns)
                if (now_ns - self._last["checkpoint"]) >= self.config.checkpoint_interval_ns:
                    self._checkpoint(now_ns)
                    self._last["checkpoint"] = now_ns
                ticks += 1
                if max_ticks and ticks >= max_ticks:
                    break
                try:
                    await asyncio.wait_for(stop.wait(), timeout=self.config.tick_interval_s)
                except asyncio.TimeoutError:
                    pass
        finally:
            trading_stop.set()
            self._checkpoint(time.time_ns(), extra={"shutdown": True})
            lock.release()
            logger.info("[runtime] continuous service stopped after %d ticks", ticks)
        return {"ticks": ticks, "final_state": self.machine.state.value}

    def status(self) -> dict:
        now_ns = time.time_ns()
        plan = self.plan_tick(now_ns)
        return {"version": SERVICE_VERSION, "now_ns": now_ns, "plan": plan.to_dict(),
                "health": self.supervisor.health(), "degraded_reason": self._degraded_reason,
                "acquisition_plan": self.acq.plan()}
