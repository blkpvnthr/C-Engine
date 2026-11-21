"""Tests for UniverseSubscriptionSync (dynamic resubscription coordinator)."""

from __future__ import annotations

import asyncio

from universe_sync import UniverseSubscriptionSync


class FakeSnapshot:
    def __init__(self, symbols):
        self.symbols = tuple(symbols)


class FakeManager:
    """refresh_from_file returns the next scripted snapshot (or None = unchanged)."""
    def __init__(self, initial, refreshes=()):
        self._current = FakeSnapshot(initial)
        self._refreshes = list(refreshes)

    def refresh_from_file(self, **_):
        if not self._refreshes:
            return None
        nxt = self._refreshes.pop(0)
        if nxt is None:
            return None
        self._current = FakeSnapshot(nxt)
        return self._current

    def current(self):
        return self._current


class FakeStream:
    def __init__(self, symbols):
        self._symbols = tuple(symbols)
        self.calls = []

    @property
    def symbols(self):
        return self._symbols

    async def update_symbols(self, *, add=(), remove=()):
        cur = list(self._symbols)
        cur_set = set(cur)
        add_eff = tuple(s for s in add if s not in cur_set)
        remove_set = set(remove)
        new = tuple(s for s in cur if s not in remove_set) + add_eff
        self._symbols = tuple(dict.fromkeys(new))
        removed = tuple(s for s in cur if s in remove_set)
        self.calls.append({"add": tuple(add_eff), "remove": removed})
        return (add_eff, removed)


class _Pos:
    def __init__(self, symbol, quantity):
        self.symbol = symbol
        self.quantity = quantity


class FakePositions:
    def __init__(self, positions, *, raises=False):
        self._p = positions
        self._raises = raises

    def positions(self):
        if self._raises:
            raise RuntimeError("boom")
        return list(self._p)


class _Order:
    def __init__(self, symbol):
        self.intent = type("I", (), {"symbol": symbol})()


def _sync(**kw):
    return UniverseSubscriptionSync(interval_s=0.0, **kw)


def _run(coro):
    return asyncio.run(coro)


def test_new_canonical_symbol_subscribed():
    stream = FakeStream(["QQQ", "SQQQ"])
    mgr = FakeManager(["QQQ", "SQQQ"], refreshes=[["QQQ", "SQQQ", "NVDA"]])
    s = _sync(stream=stream, universe_manager=mgr)
    added, removed = _run(s.sync_once())
    assert "NVDA" in added
    assert "NVDA" in stream.symbols


def test_dropped_canonical_symbol_unsubscribed():
    stream = FakeStream(["QQQ", "SQQQ"])
    mgr = FakeManager(["QQQ", "SQQQ"], refreshes=[["QQQ"]])
    s = _sync(stream=stream, universe_manager=mgr)
    _run(s.sync_once())
    assert "SQQQ" not in stream.symbols
    assert "QQQ" in stream.symbols


def test_held_position_never_unsubscribed():
    # SQQQ dropped from canonical but held as a position -> must stay subscribed.
    stream = FakeStream(["QQQ", "SQQQ"])
    mgr = FakeManager(["QQQ", "SQQQ"], refreshes=[["QQQ"]])
    s = _sync(stream=stream, universe_manager=mgr,
              positions=FakePositions([_Pos("SQQQ", 10)]))
    _run(s.sync_once())
    assert "SQQQ" in stream.symbols


def test_pending_order_symbol_never_unsubscribed():
    stream = FakeStream(["QQQ", "SQQQ"])
    mgr = FakeManager(["QQQ", "SQQQ"], refreshes=[["QQQ"]])
    s = _sync(stream=stream, universe_manager=mgr,
              open_orders_fn=lambda: [_Order("SQQQ")])
    _run(s.sync_once())
    assert "SQQQ" in stream.symbols


def test_extra_reference_symbols_never_unsubscribed():
    stream = FakeStream(["QQQ", "SPY"])
    mgr = FakeManager(["QQQ"], refreshes=[["QQQ"]])
    s = _sync(stream=stream, universe_manager=mgr, extra_symbols=["SPY"])
    _run(s.sync_once())
    assert "SPY" in stream.symbols


def test_zero_quantity_position_not_forced():
    # A flat (qty 0) position is not a reason to keep a dropped symbol.
    stream = FakeStream(["QQQ", "SQQQ"])
    mgr = FakeManager(["QQQ", "SQQQ"], refreshes=[["QQQ"]])
    s = _sync(stream=stream, universe_manager=mgr,
              positions=FakePositions([_Pos("SQQQ", 0)]))
    _run(s.sync_once())
    assert "SQQQ" not in stream.symbols


def test_no_change_no_update_call():
    stream = FakeStream(["QQQ", "SQQQ"])
    mgr = FakeManager(["QQQ", "SQQQ"], refreshes=[None])  # unchanged
    s = _sync(stream=stream, universe_manager=mgr)
    added, removed = _run(s.sync_once())
    assert added == () and removed == ()
    assert stream.calls == []


def test_unchanged_file_still_reconciles_new_position():
    # refresh returns None (file unchanged) but a new position appeared -> subscribe it.
    stream = FakeStream(["QQQ"])
    mgr = FakeManager(["QQQ"], refreshes=[None])
    s = _sync(stream=stream, universe_manager=mgr,
              positions=FakePositions([_Pos("TSLA", 5)]))
    _run(s.sync_once())
    assert "TSLA" in stream.symbols


def test_position_read_failure_is_failsafe():
    stream = FakeStream(["QQQ"])
    mgr = FakeManager(["QQQ"], refreshes=[["QQQ", "NVDA"]])
    s = _sync(stream=stream, universe_manager=mgr,
              positions=FakePositions([], raises=True))
    # Must not raise; canonical change still applied.
    _run(s.sync_once())
    assert "NVDA" in stream.symbols


# -------------------------------------------------- discovery candidate folding
def test_discovery_candidates_are_subscribed():
    stream = FakeStream(["QQQ"])
    mgr = FakeManager(["QQQ"], refreshes=[["QQQ"]])
    s = _sync(stream=stream, universe_manager=mgr,
              discovery_symbols_fn=lambda: ["NVDA", "AMD"])
    _run(s.sync_once())
    assert {"NVDA", "AMD"} <= set(stream.symbols)


def test_discovery_bounded_by_max_symbols_never_trims_required():
    # required = QQQ (canonical) + SQQQ (held) = 2; budget 3 -> only 1 candidate.
    stream = FakeStream(["QQQ", "SQQQ"])
    mgr = FakeManager(["QQQ"], refreshes=[["QQQ"]])
    s = _sync(stream=stream, universe_manager=mgr,
              positions=FakePositions([_Pos("SQQQ", 10)]),
              discovery_symbols_fn=lambda: ["AAA", "BBB", "CCC"],
              max_symbols=3)
    _run(s.sync_once())
    syms = set(stream.symbols)
    assert {"QQQ", "SQQQ"} <= syms           # required never trimmed
    assert len(syms) == 3                     # budget respected
    assert len(syms & {"AAA", "BBB", "CCC"}) == 1


def test_required_exceeding_budget_keeps_all_required_drops_discovery():
    stream = FakeStream(["QQQ", "SQQQ", "SPY"])
    mgr = FakeManager(["QQQ", "SQQQ", "SPY"], refreshes=[["QQQ", "SQQQ", "SPY"]])
    s = _sync(stream=stream, universe_manager=mgr,
              discovery_symbols_fn=lambda: ["AAA", "BBB"],
              max_symbols=2)  # smaller than required (3)
    _run(s.sync_once())
    syms = set(stream.symbols)
    assert {"QQQ", "SQQQ", "SPY"} <= syms     # all required kept
    assert not (syms & {"AAA", "BBB"})        # no discovery added over budget


def test_discovery_read_failure_is_failsafe():
    stream = FakeStream(["QQQ"])
    mgr = FakeManager(["QQQ"], refreshes=[["QQQ", "NVDA"]])

    def boom():
        raise RuntimeError("scanner down")

    s = _sync(stream=stream, universe_manager=mgr, discovery_symbols_fn=boom)
    _run(s.sync_once())  # must not raise
    assert "NVDA" in stream.symbols
