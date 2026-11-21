"""Shared technical indicators for engine strategies (Milestone 5).

Pure, dependency-light implementations of the indicators the DI/BB strategy and
replacement gates need, computed from completed bars to avoid look-ahead bias:

  * :func:`sma` — simple moving average (Bollinger middle band);
  * :func:`momentum` — close-to-close rate of change over a lookback;
  * :func:`wilder_dmi` — Wilder's Directional Movement Index (DI+/DI-/ADX);
  * :class:`SessionVWAP` — per-symbol session VWAP accumulator.

These mirror the semantics of the QQ-Trader source (``dmi_adx`` / VWAP in
``trade.py``). Prices may be passed in any consistent unit (ticks or dollars);
ratios (DI/ADX) are unit-free and VWAP is returned in the input unit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

NS_PER_DAY = 86_400 * 1_000_000_000


def sma(values: Sequence[float], period: int) -> Optional[float]:
    """Mean of the last ``period`` values, or None if there are too few."""
    if period <= 0 or len(values) < period:
        return None
    window = values[-period:]
    return sum(window) / period


def momentum(values: Sequence[float], lookback: int) -> Optional[float]:
    """Close-to-close change over ``lookback`` bars (``v[-1] - v[-1-lookback]``)."""
    if lookback <= 0 or len(values) < lookback + 1:
        return None
    return float(values[-1] - values[-1 - lookback])


@dataclass(frozen=True, slots=True)
class DMI:
    di_plus: float
    di_minus: float
    adx: float


def _wilder_smooth(values: Sequence[float], period: int) -> list[float]:
    """Wilder's running sum smoothing: seed = sum of first ``period`` values,
    then ``s = s - s/period + v``. Returns the smoothed series."""
    seed = sum(values[:period])
    out = [seed]
    s = seed
    for v in values[period:]:
        s = s - s / period + v
        out.append(s)
    return out


def wilder_dmi(
    highs: Sequence[float],
    lows: Sequence[float],
    closes: Sequence[float],
    period: int = 14,
) -> Optional[DMI]:
    """Wilder's DMI (DI+/DI-/ADX). Returns None if fewer than ``period+1`` bars.

    Uses the standard definitions: directional movement (+DM/-DM), true range,
    Wilder smoothing, ``DI = 100 * smoothed(DM) / smoothed(TR)``, and ADX as the
    Wilder-smoothed DX.
    """
    n = len(closes)
    if period < 1 or n < period + 1 or len(highs) != n or len(lows) != n:
        return None

    trs: list[float] = []
    plus_dm: list[float] = []
    minus_dm: list[float] = []
    for i in range(1, n):
        up = highs[i] - highs[i - 1]
        down = lows[i - 1] - lows[i]
        plus_dm.append(up if (up > down and up > 0) else 0.0)
        minus_dm.append(down if (down > up and down > 0) else 0.0)
        trs.append(
            max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i - 1]),
                abs(lows[i] - closes[i - 1]),
            )
        )

    if len(trs) < period:
        return None

    str_ = _wilder_smooth(trs, period)
    spdm = _wilder_smooth(plus_dm, period)
    smdm = _wilder_smooth(minus_dm, period)

    atr = str_[-1]
    di_plus = (100.0 * spdm[-1] / atr) if atr else 0.0
    di_minus = (100.0 * smdm[-1] / atr) if atr else 0.0

    # ADX = Wilder-smoothed DX over the DX series.
    dxs: list[float] = []
    for a, p, m in zip(str_, spdm, smdm, strict=False):
        if a == 0:
            dxs.append(0.0)
            continue
        pp = 100.0 * p / a
        mm = 100.0 * m / a
        denom = pp + mm
        dxs.append((100.0 * abs(pp - mm) / denom) if denom else 0.0)

    if len(dxs) >= period:
        adx = sum(dxs[:period]) / period
        for d in dxs[period:]:
            adx = (adx * (period - 1) + d) / period
    else:
        adx = dxs[-1] if dxs else 0.0

    return DMI(di_plus=di_plus, di_minus=di_minus, adx=adx)


class SessionVWAP:
    """Per-symbol session VWAP accumulator.

    A "session" is a UTC calendar day (``timestamp_ns // NS_PER_DAY``); when a
    new session is seen for a symbol the accumulator resets. This is a documented
    approximation of the market's regular-session VWAP (it does not model exact
    exchange session hours). ``update`` returns the running VWAP after folding in
    the bar; a zero-volume bar contributes its typical price as a fallback.
    """

    def __init__(self) -> None:
        self._state: dict[str, tuple[int, float, float]] = {}  # sym -> (session, sum_pv, sum_v)

    def update(
        self, symbol: str, timestamp_ns: int, *, typical_price: float, volume: float
    ) -> float:
        sym = symbol.strip().upper()
        session = timestamp_ns // NS_PER_DAY
        prev = self._state.get(sym)
        if prev is None or prev[0] != session:
            sum_pv = 0.0
            sum_v = 0.0
        else:
            _, sum_pv, sum_v = prev
        vol = max(0.0, float(volume))
        sum_pv += typical_price * vol
        sum_v += vol
        self._state[sym] = (session, sum_pv, sum_v)
        if sum_v <= 0.0:
            return float(typical_price)
        return sum_pv / sum_v

    def value(self, symbol: str) -> Optional[float]:
        prev = self._state.get(symbol.strip().upper())
        if prev is None:
            return None
        _, sum_pv, sum_v = prev
        return (sum_pv / sum_v) if sum_v > 0 else None
