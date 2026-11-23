"""P0.5 regression: seed portfolio returns from already-warmed native bars.

Root cause of `optimizer=warming_up cov=n/a reserved_n=0 routed=0`: the portfolio
pipeline only runs for symbols that already have >= min_obs (default 20) entries in
self._portfolio_returns, and that map is fed ONLY from live tick-to-tick deltas
(one per symbol per engine tick) and starts empty every run. So covariance stays
unavailable for ~20 live ticks per symbol after every restart, even though the
native engine has already warmed real historical bars.

Fix: GpuDiscoveryProvider.export_returns derives close-to-close returns from the
already-warmed native bar window (real closes ONLY -- a non-positive close breaks
the chain, never fabricated), keyed by right-aligned synthetic bar timestamps so
cross-symbol covariance alignment stays valid. _portfolio_tick seeds from it once.
"""

from __future__ import annotations

import pytest

from cengine.gpu_discovery import GpuDiscoveryProvider, _closes_to_returns


def test_closes_to_returns_computes_close_to_close_keyed_right_aligned():
    series = _closes_to_returns([100.0, 110.0, 121.0], ref_ns=1000, interval_ns=10)
    # Two returns; most-recent keyed at ref_ns, the prior one one interval earlier.
    assert series == pytest.approx({990: 0.10, 1000: 0.10})


def test_closes_to_returns_never_fabricates_across_nonpositive_closes():
    # A zero/negative close breaks the chain -- no return spans the gap.
    assert _closes_to_returns([100.0, 0.0, 120.0], ref_ns=1000, interval_ns=10) == {}
    partial = _closes_to_returns([100.0, -5.0, 120.0, 132.0], ref_ns=1000, interval_ns=10)
    assert partial == pytest.approx({1000: 0.10})  # only 120->132


def test_closes_to_returns_empty_for_single_bar():
    assert _closes_to_returns([100.0], ref_ns=1000, interval_ns=10) == {}


class _FakeBarWindow:
    def __init__(self, closes, counts, max_window):
        self.closes = closes
        self.counts = counts
        self.max_window = max_window


class _FakeEngine:
    def __init__(self, symbol_ids, bar_window):
        self._symbol_ids = symbol_ids
        self._bw = bar_window
        self.export_calls = []

    def symbol_id(self, symbol):
        return self._symbol_ids.get(symbol)

    def export_bar_window(self, ids, timeframe, window):
        self.export_calls.append((tuple(ids), timeframe, window))
        return self._bw


def _provider_with(engine, window=4):
    provider = GpuDiscoveryProvider.__new__(GpuDiscoveryProvider)
    provider.engine = engine
    provider._window = window
    return provider


def test_export_returns_seeds_real_returns_and_skips_unknown_and_short():
    # row0 AAA: 3 bars [100,110,121]; row1 BBB: 2 bars [50,55]; padding beyond count.
    bw = _FakeBarWindow(
        closes=[100.0, 110.0, 121.0, 0.0, 50.0, 55.0, 0.0, 0.0],
        counts=[3, 2],
        max_window=4,
    )
    engine = _FakeEngine({"AAA": 0, "BBB": 1}, bw)  # CCC unknown
    provider = _provider_with(engine)

    out = provider.export_returns(["AAA", "BBB", "CCC"], ref_ns=1000, interval_ns=10)

    assert set(out) == {"AAA", "BBB"}  # CCC (unknown) skipped
    assert out["AAA"] == pytest.approx({990: 0.10, 1000: 0.10})
    assert out["BBB"] == pytest.approx({1000: 0.10})
    # Only the two valid symbol ids were requested from the native engine.
    assert engine.export_calls and engine.export_calls[0][0] == (0, 1)


def test_export_returns_skips_symbols_with_fewer_than_two_bars():
    bw = _FakeBarWindow(closes=[100.0, 0.0], counts=[1], max_window=2)
    engine = _FakeEngine({"AAA": 0}, bw)
    provider = _provider_with(engine, window=2)

    out = provider.export_returns(["AAA"], ref_ns=1000, interval_ns=10)

    assert out == {}  # single bar -> no return, not seeded


def test_export_returns_empty_when_no_known_symbols():
    engine = _FakeEngine({}, _FakeBarWindow([], [], 0))
    provider = _provider_with(engine)
    assert provider.export_returns(["ZZZ"], ref_ns=1000, interval_ns=10) == {}
