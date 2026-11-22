"""Regression tests for the default-on order-producing strategies (§16).

The execution audit (reports/execution_strategy_audit.json) found Turtle, DualMA,
and APO are enabled by default under `python3 trade.py` but had NO dedicated unit
tests. These pin their signal-generation invariants: valid recommendation,
insufficient-history (no premature signal), and invalid-observation rejection.
These tests establish correctness/regression safety ONLY — not profitability.
"""

from __future__ import annotations

import pytest

from strategies import (
    APOConfig,
    APOMeanReversionStrategy,
    Bar,
    DualMAConfig,
    DualMAStrategy,
    SignalSide,
    StrategyName,
    TurtleConfig,
    TurtleStrategy,
)


def _bar(ts, o, h, lo, c, symbol="AAPL", v=1000):
    return Bar(symbol=symbol, timestamp_ns=ts, open_ticks=o, high_ticks=h,
               low_ticks=lo, close_ticks=c, volume=v)


# ------------------------------------------------------------------ Turtle
def test_turtle_breakout_emits_long():
    s = TurtleStrategy(TurtleConfig(entry_lookback=3, exit_lookback=2))
    # 3 bars in a tight range, then a close above the prior-highs channel.
    for ts in (1, 2, 3):
        assert s.on_bar(_bar(ts, 100, 101, 99, 100)) == ()  # fills history
    out = s.on_bar(_bar(4, 101, 110, 101, 109))  # close 109 > entry_high 101
    assert len(out) == 1 and out[0].side == SignalSide.LONG
    assert out[0].strategy == StrategyName.TURTLE


def test_turtle_no_signal_before_lookback():
    s = TurtleStrategy(TurtleConfig(entry_lookback=5, exit_lookback=2))
    for ts in range(1, 5):  # only 4 bars (< entry_lookback 5)
        assert s.on_bar(_bar(ts, 100, 120, 90, 118)) == ()  # breakout-looking, ignored


def test_turtle_rejects_invalid_bar():
    s = TurtleStrategy(TurtleConfig(entry_lookback=3, exit_lookback=2))
    with pytest.raises(ValueError):
        s.on_bar(_bar(1, 100, 90, 95, 92))  # high < low


# ------------------------------------------------------------------ DualMA
def test_dual_ma_cross_up_emits_long():
    s = DualMAStrategy(DualMAConfig(fast_period=2, slow_period=3))
    # establish a prior (negative/zero) difference, then cross fast above slow.
    for ts, c in enumerate([100, 99, 98], start=1):
        s.on_bar(_bar(ts, c, c + 1, c - 1, c))
    out = ()
    for ts, c in enumerate([105, 115], start=4):  # sharp rise -> fast > slow
        out = s.on_bar(_bar(ts, c, c + 1, c - 1, c)) or out
    assert any(x.side == SignalSide.LONG for x in out)
    assert all(x.strategy == StrategyName.DUAL_MA for x in out)


def test_dual_ma_no_signal_before_slow_period():
    s = DualMAStrategy(DualMAConfig(fast_period=2, slow_period=4))
    for ts, c in enumerate([100, 110, 120], start=1):  # only 3 closes (< slow 4)
        assert s.on_bar(_bar(ts, c, c + 1, c - 1, c)) == ()


def test_dual_ma_rejects_invalid_bar():
    s = DualMAStrategy(DualMAConfig(fast_period=2, slow_period=3))
    with pytest.raises(ValueError):
        s.on_bar(_bar(1, 100, 100, 100, 0))  # non-positive close


# ------------------------------------------------------------------ APO
def test_apo_warmup_guard_no_premature_signal():
    s = APOMeanReversionStrategy(
        APOConfig(fast_period=2, slow_period=4, entry_threshold_ticks=5.0,
                  exit_threshold_ticks=2.0))
    # During EMA warm-up (fewer than slow_period bars) nothing is emitted.
    for ts, c in enumerate([100, 80, 60], start=1):  # steep drop, still warming
        assert s.on_bar(_bar(ts, c, c + 1, c - 1, c)) == ()


def test_apo_contrarian_long_on_decline():
    s = APOMeanReversionStrategy(
        APOConfig(fast_period=2, slow_period=4, entry_threshold_ticks=5.0,
                  exit_threshold_ticks=2.0))
    out = ()
    for ts, c in enumerate([100, 95, 90, 80, 60, 40], start=1):  # sustained decline
        res = s.on_bar(_bar(ts, c, c + 1, c - 1, c))
        out = res or out
    # fast EMA falls below slow EMA by > entry threshold -> contrarian LONG
    assert any(x.side == SignalSide.LONG for x in out)
    assert all(x.strategy == StrategyName.APO_MEAN_REVERSION for x in out)


def test_apo_rejects_invalid_bar():
    s = APOMeanReversionStrategy(
        APOConfig(fast_period=2, slow_period=4, entry_threshold_ticks=5.0,
                  exit_threshold_ticks=2.0))
    with pytest.raises(ValueError):
        s.on_bar(_bar(0, 100, 101, 99, 100))  # timestamp_ns must be positive
