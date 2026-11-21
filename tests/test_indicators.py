"""Tests for the shared DI/BB/VWAP indicators (Milestone 5)."""

from __future__ import annotations

import pytest

from indicators import SessionVWAP, momentum, sma, wilder_dmi


# ------------------------------------------------------------------------ SMA
def test_sma_exact():
    assert sma([1, 2, 3, 4, 5], 5) == pytest.approx(3.0)
    assert sma([2, 4, 6], 2) == pytest.approx(5.0)  # last two: (4+6)/2


def test_sma_insufficient_returns_none():
    assert sma([1, 2], 5) is None
    assert sma([], 1) is None


# ------------------------------------------------------------------- momentum
def test_momentum_sign():
    assert momentum([1, 2, 3, 4, 5, 6], 5) > 0   # rising
    assert momentum([6, 5, 4, 3, 2, 1], 5) < 0   # falling
    assert momentum([3, 3, 3], 2) == pytest.approx(0.0)


def test_momentum_insufficient_none():
    assert momentum([1, 2], 5) is None


# ----------------------------------------------------------------- Wilder DMI
def _series(trend, n=40, start=100.0, step=1.0, spread=0.5):
    highs, lows, closes = [], [], []
    px = start
    for _ in range(n):
        px += step * trend
        highs.append(px + spread)
        lows.append(px - spread)
        closes.append(px)
    return highs, lows, closes


def test_dmi_uptrend_di_plus_dominates():
    h, low, c = _series(trend=+1)
    r = wilder_dmi(h, low, c, period=14)
    assert r is not None
    assert r.di_plus > r.di_minus
    assert 0.0 <= r.adx <= 100.0


def test_dmi_downtrend_di_minus_dominates():
    h, low, c = _series(trend=-1)
    r = wilder_dmi(h, low, c, period=14)
    assert r is not None
    assert r.di_minus > r.di_plus


def test_dmi_insufficient_returns_none():
    assert wilder_dmi([1, 2], [1, 2], [1, 2], period=14) is None


def test_dmi_flat_series_low_directional():
    # A perfectly flat series has no directional movement.
    h = [100.5] * 40
    low = [99.5] * 40
    c = [100.0] * 40
    r = wilder_dmi(h, low, c, period=14)
    assert r is not None
    assert r.di_plus == pytest.approx(0.0, abs=1e-9)
    assert r.di_minus == pytest.approx(0.0, abs=1e-9)


# ----------------------------------------------------------------- SessionVWAP
def test_session_vwap_exact_within_session():
    v = SessionVWAP()
    # two bars same day; typical price = (h+l+c)/3
    day = 1_600_000_000_000_000_000  # some ns in a day
    # bar1 typ=10, vol=100 ; bar2 typ=20, vol=300 -> vwap=(10*100+20*300)/400=17.5
    assert v.update("X", day, typical_price=10.0, volume=100) == pytest.approx(10.0)
    assert v.update("X", day + 1_000, typical_price=20.0, volume=300) == pytest.approx(17.5)


def test_session_vwap_resets_next_day():
    v = SessionVWAP()
    ns_per_day = 86_400 * 1_000_000_000
    d0 = 5 * ns_per_day + 10
    d1 = 6 * ns_per_day + 10
    v.update("X", d0, typical_price=10.0, volume=100)
    # new day -> accumulator resets, so vwap == the new bar's typical price
    assert v.update("X", d1, typical_price=50.0, volume=10) == pytest.approx(50.0)


def test_session_vwap_zero_volume_uses_typical():
    v = SessionVWAP()
    assert v.update("X", 1_000, typical_price=42.0, volume=0) == pytest.approx(42.0)


def test_session_vwap_per_symbol_isolated():
    v = SessionVWAP()
    day = 1_000_000
    v.update("A", day, typical_price=10.0, volume=100)
    assert v.update("B", day, typical_price=99.0, volume=100) == pytest.approx(99.0)
