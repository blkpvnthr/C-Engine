"""Regression: the market-session gate must honor exchange holidays and early
closes, not just a static weekday 09:30-16:00 window.

Root cause (2026-11-27): on the post-Thanksgiving half-day the exchange closed
at 13:00 ET, but MarketSessionGate's static window said the session was open
until 16:00, so the engine routed a market order at 15:41 ET that Alpaca
accepted but could never fill. The gate needs a holiday/early-close calendar.
"""

from __future__ import annotations

from datetime import datetime, timezone

from cengine.safety import MarketSessionGate, SessionConfig

_WEEKDAYS = (0, 1, 2, 3, 4)


def _ns(y: int, mo: int, d: int, h: int, mi: int) -> int:
    """Wall-clock UTC -> epoch ns (callers pass the UTC hour directly)."""
    return int(datetime(y, mo, d, h, mi, tzinfo=timezone.utc).timestamp() * 1e9)


def test_full_holiday_closes_session_during_rth_window():
    # Thanksgiving 2026-11-26 (Thursday) 15:00 ET == 20:00 UTC. Inside the
    # static 09:30-16:00 window, but the exchange is fully closed.
    plain = MarketSessionGate(SessionConfig("America/New_York", 570, 960, _WEEKDAYS))
    assert plain.is_open(_ns(2026, 11, 26, 20, 0)) is True  # today's buggy behavior

    gated = MarketSessionGate(
        SessionConfig(
            "America/New_York", 570, 960, _WEEKDAYS,
            holidays=frozenset({"2026-11-26"}),
        )
    )
    assert gated.is_open(_ns(2026, 11, 26, 20, 0)) is False
    assert "holiday" in gated.session_reason(_ns(2026, 11, 26, 20, 0))


def test_early_close_blocks_routing_after_the_early_close():
    # 2026-11-27 half-day: exchange closes 13:00 ET (minute 780).
    gated = MarketSessionGate(
        SessionConfig(
            "America/New_York", 570, 960, _WEEKDAYS,
            early_closes=(("2026-11-27", 780),),
        )
    )
    # 15:41 ET == 20:41 UTC: the exact time the stuck SPY order was routed.
    assert gated.is_open(_ns(2026, 11, 27, 20, 41)) is False
    # 12:00 ET == 17:00 UTC: before the early close, still open.
    assert gated.is_open(_ns(2026, 11, 27, 17, 0)) is True


def test_calendar_fields_default_to_the_prior_static_behavior():
    plain = MarketSessionGate(SessionConfig("America/New_York", 570, 960, _WEEKDAYS))
    # Regular Tuesday 10:00 ET == 15:00 UTC stays open; Sunday stays closed.
    assert plain.is_open(_ns(2026, 11, 24, 15, 0)) is True
    assert plain.is_open(_ns(2026, 11, 22, 15, 0)) is False


def test_overlays_from_runtime_calendar_cover_thanksgiving_week_2026():
    from cengine.runtime.calendar import session_gate_overlays

    holidays, early = session_gate_overlays([2026])
    assert "2026-11-26" in holidays        # Thanksgiving — full close
    assert "2026-12-25" in holidays        # Christmas — full close
    assert ("2026-11-27", 780) in early    # Black Friday half-day closes 13:00 ET


def test_gate_built_from_runtime_calendar_is_closed_on_the_half_day_afternoon():
    """End-to-end: the overlay helper + gate reproduce today's fix — a market
    order at 15:41 ET on the 2026-11-27 half-day is correctly gated closed."""
    from cengine.runtime.calendar import session_gate_overlays

    holidays, early = session_gate_overlays([2026])
    gate = MarketSessionGate(
        SessionConfig("America/New_York", 570, 960, _WEEKDAYS,
                      holidays=holidays, early_closes=early)
    )
    assert gate.is_open(_ns(2026, 11, 27, 20, 41)) is False  # 15:41 ET, post 13:00 close
    assert gate.is_open(_ns(2026, 11, 27, 17, 0)) is True    # 12:00 ET, still open
    assert gate.is_open(_ns(2026, 11, 26, 20, 0)) is False   # Thanksgiving, full close
