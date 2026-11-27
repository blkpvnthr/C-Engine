"""B1 regression: a closed market session must NOT crash the engine.

Original defect: _process_bar called session_gate.require_open() at the top; a
closed-session bar raised, escaped _consume_market_events (no except) and run()'s
FIRST_COMPLETED re-raise, tearing down the whole engine — including the independent
protective exit — outside RTH. Market-data processing now runs unconditionally and
only elective order routing is session-gated, as a nonfatal decision; the consumer
isolates per-event failures.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from cengine.engine_state import EngineState, EngineStateMachine
from cengine.safety import (
    MarketClosedError,
    MarketSessionGate,
    SessionConfig,
)
from market_data.alpaca_stream import BarEvent
from trade import EngineHealth, MarketEngine

_GATE = MarketSessionGate(SessionConfig("America/New_York", 570, 960, (0, 1, 2, 3, 4)))
# Tue 10:00 ET (open) and Sunday (closed).
_OPEN_NS = int(datetime(2026, 11, 24, 15, 0, tzinfo=timezone.utc).timestamp() * 1e9)
_CLOSED_NS = int(datetime(2026, 11, 22, 15, 0, tzinfo=timezone.utc).timestamp() * 1e9)


def test_session_gate_nonfatal_and_fatal_variants():
    assert _GATE.is_open(_OPEN_NS) is True
    assert _GATE.is_open(_CLOSED_NS) is False
    assert _GATE.session_reason(_OPEN_NS) == ""
    assert "weekday" in _GATE.session_reason(_CLOSED_NS)
    _GATE.require_open(_OPEN_NS)  # no raise
    with pytest.raises(MarketClosedError):
        _GATE.require_open(_CLOSED_NS)


class _FakeMarketGate:
    def require_fresh(self, symbol, now_ns):  # always fresh in the test
        pass


class _FakeStrategies:
    def __init__(self, candidates):
        self._c = candidates
        self.called = False

    def on_bar(self, bar):
        self.called = True
        return SimpleNamespace(symbol=bar.symbol, timestamp_ns=bar.timestamp_ns,
                               candidates=self._c)


class _FakeRouter:
    def __init__(self):
        self.calls = 0

    async def route(self, candidates):
        self.calls += 1
        return (len(candidates), False)


def _engine(candidates):
    e = MarketEngine.__new__(MarketEngine)
    e.market_gate = _FakeMarketGate()
    e.state = EngineStateMachine()
    e.state.transition(EngineState.RECONCILING)
    e.state.transition(EngineState.READY)
    e.state.transition(EngineState.RUNNING)
    e.session_gate = _GATE
    e.strategies = _FakeStrategies(candidates)
    e.router = _FakeRouter()
    e.health = EngineHealth()
    e._bar_observer = None
    e._on_decision = None
    return e


def _bar(ns):
    return BarEvent(symbol="SPY", timestamp_ns=ns, open_ticks=100, high_ticks=101,
                    low_ticks=99, close_ticks=100, volume=10, trade_count=1,
                    vwap_ticks=100, sequence=1, received_ns=ns)


def test_closed_session_processes_bar_but_skips_routing_without_crashing():
    e = _engine(("candidate",))
    asyncio.run(e._process_bar(_bar(_CLOSED_NS)))  # MUST NOT raise
    assert e.strategies.called is True      # data / features / discovery still ran
    assert e.router.calls == 0              # elective routing skipped
    assert e.health.session_blocked == 1


def test_open_session_routes_candidates():
    e = _engine(("candidate",))
    asyncio.run(e._process_bar(_bar(_OPEN_NS)))
    assert e.router.calls == 1
    assert e.health.session_blocked == 0


def test_consumer_isolates_a_raising_event():
    e = _engine(("candidate",))
    e._stop = asyncio.Event()
    e.strategy_events = SimpleNamespace(queue=asyncio.Queue())

    async def _boom(event):
        raise RuntimeError("boom")

    e._process_bar = _boom

    async def driver():
        task = asyncio.create_task(e._consume_market_events())
        e.strategy_events.queue.put_nowait(_bar(_OPEN_NS))
        await e.strategy_events.queue.join()   # first event fully handled
        e._stop.set()
        e.strategy_events.queue.put_nowait(_bar(_OPEN_NS))  # unblock get() so loop exits
        await asyncio.wait_for(task, timeout=2)

    asyncio.run(driver())
    assert e.health.market_data_errors >= 1   # raised event isolated, not propagated
