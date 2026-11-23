"""Tests for the continuous runtime service (addendum §1–§3/§6/§10/§12).

Planning is tested deterministically by driving the state machine with fixed timestamps for
known operating states; the async loop is tested bounded with stubbed off-session services so it
stays fast and hermetic (no real optimization/acquisition passes).
"""

from __future__ import annotations

import asyncio
import datetime as _dt
from zoneinfo import ZoneInfo

import pytest

from cengine.runtime.service import ContinuousRuntimeConfig, ContinuousRuntimeService
from cengine.runtime.state_machine import OperatingState

_ET = ZoneInfo("America/New_York")


def _ns(y, mo, d, h, mi) -> int:
    return int(_dt.datetime(y, mo, d, h, mi, tzinfo=_ET).timestamp() * 1e9)


# Tuesday 2026-11-24 is a trading day; Saturday 2026-11-28 is a weekend.
_REGULAR = _ns(2026, 11, 24, 10, 0)
_OVERNIGHT = _ns(2026, 11, 24, 2, 0)
_WEEKEND = _ns(2026, 11, 28, 10, 0)


def _svc(tmp_path, **cfg_kw) -> ContinuousRuntimeService:
    cfg = ContinuousRuntimeConfig(
        lock_path=str(tmp_path / "trade.lock"),
        checkpoint_path=str(tmp_path / "state" / "checkpoint.json"),
        **cfg_kw)
    return ContinuousRuntimeService(cfg)


# --------------------------------------------------------------------------- planning


def test_regular_session_is_execution_window_no_research(tmp_path):
    svc = _svc(tmp_path)
    plan = svc.plan_tick(_REGULAR)
    assert plan.state == OperatingState.REGULAR_SESSION
    assert plan.execution_window is True
    # research is NOT appropriate in-session -> no off-session optimization/acquisition
    assert plan.run_optimization is False
    assert plan.run_acquisition is False


def test_overnight_runs_research_when_due_and_low_pressure(tmp_path):
    svc = _svc(tmp_path)
    plan = svc.plan_tick(_OVERNIGHT, gpu_pressure=0.1)
    assert plan.state == OperatingState.OVERNIGHT_RESEARCH
    assert plan.execution_window is False
    assert plan.run_optimization is True
    assert plan.run_acquisition is True


def test_high_gpu_pressure_defers_research(tmp_path):
    svc = _svc(tmp_path)
    plan = svc.plan_tick(_OVERNIGHT, gpu_pressure=0.95)
    # research is suspended under GPU pressure even off-session
    assert plan.run_optimization is False
    assert "gpu_pressure" in plan.reasons["optimization"]


def test_cadence_prevents_rerun(tmp_path):
    svc = _svc(tmp_path)
    p1 = svc.plan_tick(_OVERNIGHT, gpu_pressure=0.1)
    assert p1.run_optimization is True
    svc._last["optimization"] = _OVERNIGHT          # mark just-run
    p2 = svc.plan_tick(_OVERNIGHT + 60 * 1_000_000_000, gpu_pressure=0.1)
    assert p2.run_optimization is False             # within the 2h interval


def test_weekend_is_research_state(tmp_path):
    svc = _svc(tmp_path)
    plan = svc.plan_tick(_WEEKEND, gpu_pressure=0.1)
    assert plan.state == OperatingState.WEEKEND_RESEARCH
    assert plan.execution_window is False
    assert plan.run_optimization is True


# --------------------------------------------------------------------------- governance / recovery


def test_governance_degrades_and_recovers(tmp_path, monkeypatch):
    svc = _svc(tmp_path)
    # force an invariant regression
    monkeypatch.setattr(svc.gov, "check_invariants",
                        lambda: {"service": "governance", "ok": False, "failures": ["risk_gate"],
                                 "checks": {}})
    svc.governance_tick(_OVERNIGHT)
    assert svc.machine.state == OperatingState.DEGRADED
    assert "risk_gate" in svc._degraded_reason
    # invariants healthy again -> recover
    monkeypatch.setattr(svc.gov, "check_invariants",
                        lambda: {"service": "governance", "ok": True, "failures": [], "checks": {}})
    svc.governance_tick(_OVERNIGHT + 1)
    assert svc.machine.state != OperatingState.DEGRADED
    assert svc._degraded_reason == ""


def test_real_invariant_check_passes(tmp_path):
    svc = _svc(tmp_path)
    inv = svc.gov.check_invariants()   # runs the REAL fast safety checks
    assert inv["ok"] is True, inv["failures"]


# --------------------------------------------------------------------------- dispatch + loop


def test_dispatch_runs_due_services_and_updates_cadence(tmp_path):
    svc = _svc(tmp_path)

    class _FakeOpt:
        def run_once(self, **k):
            return {"service": "optimization", "ok": True}

    class _FakeAcq:
        def run_once(self, **k):
            return {"service": "acquisition", "mode": "dry_run"}

    class _FakeGov:
        def full_recert(self, symbols=None):
            return {"service": "governance_full", "readiness": "research_validated"}

    svc.opt, svc.acq, svc.gov = _FakeOpt(), _FakeAcq(), _FakeGov()
    plan = svc.plan_tick(_OVERNIGHT, gpu_pressure=0.1)
    plan.run_governance_full = True
    out = svc.dispatch(plan, _OVERNIGHT)
    assert out["optimization"]["ok"] is True
    assert out["acquisition"]["mode"] == "dry_run"
    assert out["governance_full"]["readiness"] == "research_validated"
    assert svc._last["optimization"] == _OVERNIGHT


def test_bounded_run_acquires_lock_checkpoints_and_stops(tmp_path, monkeypatch):
    svc = _svc(tmp_path, tick_interval_s=0.01)
    # stub the heavy services so the loop is instant
    monkeypatch.setattr(svc.opt, "run_once", lambda **k: {"ok": True})
    monkeypatch.setattr(svc.acq, "run_once", lambda **k: {"mode": "dry_run"})
    monkeypatch.setattr(svc.gov, "full_recert", lambda symbols=None: {"ok": True})

    async def _go():
        return await svc.run(max_ticks=1)

    result = asyncio.run(_go())
    assert result["ticks"] == 1
    assert "final_state" in result
    # checkpoint was written and the lock released (reusable)
    import os
    assert os.path.exists(svc.config.checkpoint_path)
    assert not os.path.exists(svc.config.lock_path)   # released on shutdown


def test_single_instance_blocks_second_runner(tmp_path):
    from cengine.runtime.supervisor import SingleInstanceError, SingleInstanceLock
    svc = _svc(tmp_path)
    held = SingleInstanceLock(svc.config.lock_path)
    held.acquire()
    try:
        async def _go():
            return await svc.run(max_ticks=1)
        with pytest.raises(SingleInstanceError):
            asyncio.run(_go())
    finally:
        held.release()
