"""US equity (NYSE) exchange calendar (Milestone 7 addendum §4).

Holiday + early-close + DST-aware session logic used by BOTH the continuous runtime state
machine and the certification session-coverage checks. Pure stdlib (``zoneinfo``) — no new
dependency. Regular session 09:30–16:00 ET; early-close days close 13:00 ET; pre-market
04:00–09:30; after-hours 16:00–20:00. Holidays include observed shifts and Good Friday
(computed). This is a best-effort calendar (documented); it is NOT a substitute for a broker's
authoritative trading-day feed and never authorizes orders by itself.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass
from enum import Enum
from zoneinfo import ZoneInfo

CALENDAR_VERSION = "nyse-calendar-1.0.0"
_ET = ZoneInfo("America/New_York")
NS = 1_000_000_000


class SessionPhase(str, Enum):
    CLOSED = "closed"             # overnight / weekend / holiday (markets shut)
    PRE_MARKET = "pre_market"     # 04:00–09:30 ET
    REGULAR = "regular_session"   # 09:30–16:00 ET (13:00 on early-close days)
    AFTER_HOURS = "after_hours"   # close–20:00 ET


def _easter(year: int) -> _dt.date:
    """Anonymous Gregorian algorithm (Good Friday = Easter - 2 days)."""
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    ll = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * ll) // 451
    month = (h + ll - 7 * m + 114) // 31
    day = ((h + ll - 7 * m + 114) % 31) + 1
    return _dt.date(year, month, day)


def _observed(d: _dt.date) -> _dt.date:
    """Shift a fixed holiday to the observed weekday (Sat->Fri, Sun->Mon)."""
    if d.weekday() == 5:
        return d - _dt.timedelta(days=1)
    if d.weekday() == 6:
        return d + _dt.timedelta(days=1)
    return d


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> _dt.date:
    d = _dt.date(year, month, 1)
    offset = (weekday - d.weekday()) % 7
    return d + _dt.timedelta(days=offset + 7 * (n - 1))


def _last_weekday(year: int, month: int, weekday: int) -> _dt.date:
    d = (_dt.date(year, month + 1, 1) if month < 12 else _dt.date(year + 1, 1, 1)) \
        - _dt.timedelta(days=1)
    return d - _dt.timedelta(days=(d.weekday() - weekday) % 7)


def holidays(year: int) -> set:
    """Full-closure NYSE holidays for a year (observed)."""
    h = {
        _observed(_dt.date(year, 1, 1)),                       # New Year's
        _nth_weekday(year, 1, 0, 3),                           # MLK (3rd Mon Jan)
        _nth_weekday(year, 2, 0, 3),                           # Presidents' (3rd Mon Feb)
        _easter(year) - _dt.timedelta(days=2),                 # Good Friday
        _last_weekday(year, 5, 0),                             # Memorial (last Mon May)
        _nth_weekday(year, 9, 0, 1),                           # Labor (1st Mon Sep)
        _nth_weekday(year, 11, 3, 4),                          # Thanksgiving (4th Thu Nov)
        _observed(_dt.date(year, 12, 25)),                     # Christmas
        _observed(_dt.date(year, 7, 4)),                       # Independence Day
    }
    if year >= 2021:
        h.add(_observed(_dt.date(year, 6, 19)))                # Juneteenth (from 2021)
    return h


def early_closes(year: int) -> set:
    """Half-days (close 13:00 ET): day after Thanksgiving, Christmas Eve (weekday), July 3."""
    out = set()
    tg = _nth_weekday(year, 11, 3, 4)
    out.add(tg + _dt.timedelta(days=1))                        # Black Friday
    xmas_eve = _dt.date(year, 12, 24)
    if xmas_eve.weekday() < 5:
        out.add(xmas_eve)
    jul3 = _dt.date(year, 7, 3)
    if jul3.weekday() < 5:
        out.add(jul3)
    return out


def is_trading_day(d: _dt.date) -> bool:
    return d.weekday() < 5 and d not in holidays(d.year)


def _et(ts_ns: int) -> _dt.datetime:
    return _dt.datetime.fromtimestamp(ts_ns / 1e9, tz=_ET)


def session_phase(ts_ns: int) -> SessionPhase:
    """Market phase for a UTC-ns timestamp, in ET with DST + early closes handled."""
    now = _et(ts_ns)
    d = now.date()
    if not is_trading_day(d):
        return SessionPhase.CLOSED
    close_h, close_m = (13, 0) if d in early_closes(d.year) else (16, 0)
    mins = now.hour * 60 + now.minute
    if 4 * 60 <= mins < 9 * 60 + 30:
        return SessionPhase.PRE_MARKET
    if 9 * 60 + 30 <= mins < close_h * 60 + close_m:
        return SessionPhase.REGULAR
    if close_h * 60 + close_m <= mins < 20 * 60:
        return SessionPhase.AFTER_HOURS
    return SessionPhase.CLOSED


def is_regular_session(ts_ns: int) -> bool:
    return session_phase(ts_ns) == SessionPhase.REGULAR


def next_transition(ts_ns: int, *, max_scan_days: int = 10) -> dict:
    """Next phase boundary from ts_ns (minute-resolution scan, bounded)."""
    cur = session_phase(ts_ns)
    step = 60 * NS
    t = ts_ns
    for _ in range(max_scan_days * 24 * 60):
        t += step
        if session_phase(t) != cur:
            return {"from": cur.value, "to": session_phase(t).value,
                    "at_ns": t, "at_et": _et(t).isoformat()}
    return {"from": cur.value, "to": "unknown", "at_ns": None}


@dataclass
class CalendarSummary:
    version: str = CALENDAR_VERSION
    tz: str = "America/New_York"

    def describe(self, ts_ns: int) -> dict:
        now = _et(ts_ns)
        return {"version": self.version, "tz": self.tz,
                "et": now.isoformat(), "date": now.date().isoformat(),
                "trading_day": is_trading_day(now.date()),
                "early_close": now.date() in early_closes(now.year),
                "phase": session_phase(ts_ns).value,
                "next_transition": next_transition(ts_ns)}
