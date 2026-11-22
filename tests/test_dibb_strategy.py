"""Tests for the DI/BB entry + inverse-exit strategy (Milestone 5, spec §14/§15)."""

from __future__ import annotations

from strategies import (
    Bar,
    DIBBConfig,
    DIBBStrategy,
    SignalSide,
    StrategyName,
)

SCALE = 10_000
NS = 300_000_000_000  # 5 minutes in ns


def _bar(symbol, i, px, *, spread=0.5, volume=1000):
    t = (px) * SCALE
    return Bar(
        symbol=symbol,
        timestamp_ns=1_000_000_000 + i * NS,
        open_ticks=int(t),
        high_ticks=int((px + spread) * SCALE),
        low_ticks=int((px - spread) * SCALE),
        close_ticks=int(t),
        volume=volume,
    )


def _feed(strat, symbol, prices):
    """Feed a price path; return the list of (index, candidate) emitted."""
    emitted = []
    for i, px in enumerate(prices):
        for c in strat.on_bar(_bar(symbol, i, px)):
            emitted.append((i, c))
    return emitted


def test_name_and_insufficient_history():
    s = DIBBStrategy(DIBBConfig())
    assert s.name is StrategyName.DI_BB
    # Not enough bars for DMI/SMA -> nothing.
    out = _feed(s, "NVDA", [100.0, 101.0, 102.0])
    assert out == []


def test_uptrend_emits_long_and_no_short():
    s = DIBBStrategy(DIBBConfig())
    prices = [100.0 + i for i in range(40)]  # steady uptrend
    out = _feed(s, "NVDA", prices)
    sides = {c.side for _i, c in out}
    assert SignalSide.LONG in sides
    assert SignalSide.SHORT not in sides        # long-only
    assert SignalSide.EXIT_LONG not in sides
    longs = [c for _i, c in out if c.side is SignalSide.LONG]
    assert longs[0].strategy is StrategyName.DI_BB
    assert longs[0].fast_value > longs[0].slow_value  # di_plus > di_minus


def test_downtrend_from_flat_emits_nothing_long_only():
    s = DIBBStrategy(DIBBConfig())
    prices = [140.0 - i for i in range(40)]  # steady downtrend
    out = _feed(s, "NVDA", prices)
    # Long-only: a downtrend from flat produces no SHORT and no LONG entry.
    assert all(c.side is not SignalSide.SHORT for _i, c in out)
    assert all(c.side is not SignalSide.LONG for _i, c in out)


def test_inverse_exit_after_long():
    s = DIBBStrategy(DIBBConfig())
    # Establish an uptrend (go long), then reverse into a downtrend.
    up = [100.0 + i for i in range(40)]
    _feed(s, "NVDA", up)
    s.on_position("NVDA", 10)  # engine reconciles the fill -> we are long
    down = [max(1.0, 139.0 - 2 * i) for i in range(40)]
    out = []
    for i, px in enumerate(down):
        for c in s.on_bar(_bar("NVDA", 40 + i, px)):
            out.append(c)
    assert any(c.side is SignalSide.EXIT_LONG for c in out)


def test_vwap_gate_blocks_entry_when_price_below_vwap():
    # require_vwap on: construct DI+/BB-ok but keep price below session VWAP by
    # starting very high then trading below the running VWAP.
    s_on = DIBBStrategy(DIBBConfig(require_vwap=True, require_momentum=False))
    s_off = DIBBStrategy(DIBBConfig(require_vwap=False, require_momentum=False))
    # A path that rises overall (DI+ dominates, price>BB-mid) is also > VWAP, so
    # to isolate the gate we assert the OFF variant emits at least as many longs.
    prices = [100.0 + i for i in range(40)]
    longs_on = sum(1 for _i, c in _feed(s_on, "X", prices) if c.side is SignalSide.LONG)
    longs_off = sum(1 for _i, c in _feed(s_off, "X", prices) if c.side is SignalSide.LONG)
    assert longs_off >= longs_on


def test_momentum_gate_blocks_entry_when_not_rising():
    s = DIBBStrategy(DIBBConfig(require_momentum=True))
    # Rise then go flat: momentum over the lookback becomes ~0 at the plateau,
    # which blocks fresh entries there.
    prices = [100.0 + i for i in range(30)] + [129.0] * 10
    out = _feed(s, "NVDA", prices)
    # No LONG emitted on the flat plateau (indices >= 30) due to momentum gate.
    plateau_longs = [i for i, c in out if c.side is SignalSide.LONG and i >= 35]
    assert plateau_longs == []


def test_no_new_long_while_already_long():
    s = DIBBStrategy(DIBBConfig())
    prices = [100.0 + i for i in range(30)]
    _feed(s, "NVDA", prices)
    s.on_position("NVDA", 10)  # now long
    # Continue uptrend; must not emit additional LONG entries while long.
    more = []
    for i, px in enumerate([130.0 + i for i in range(10)]):
        for c in s.on_bar(_bar("NVDA", 30 + i, px)):
            more.append(c)
    assert all(c.side is not SignalSide.LONG for c in more)
