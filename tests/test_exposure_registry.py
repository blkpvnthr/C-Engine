"""Instrument exposure registry (§5-8): classification, accounting, safety gate."""

from __future__ import annotations

from cengine.exposure_registry import (
    ExposureDirection,
    ExposureRegistry,
    InstrumentType,
    detect_opposing,
    detect_overlap,
    effective_exposure,
    seed_exposure_table,
)


def test_classification_of_known_instruments():
    t = seed_exposure_table()
    assert t["SPY"].instrument_type == InstrumentType.BROAD_MARKET_ETF
    assert t["SPY"].target_daily_leverage == 1.0 and not t["SPY"].leveraged
    assert t["TQQQ"].instrument_type == InstrumentType.LEVERAGED_LONG_ETF
    assert t["TQQQ"].target_daily_leverage == 3.0 and t["TQQQ"].leveraged
    assert t["SQQQ"].instrument_type == InstrumentType.LEVERAGED_INVERSE_ETF
    assert t["SQQQ"].target_daily_leverage == -3.0 and t["SQQQ"].inverse
    assert t["SH"].instrument_type == InstrumentType.INVERSE_ETF
    assert t["SH"].target_daily_leverage == -1.0
    assert t["XLK"].instrument_type == InstrumentType.SECTOR_ETF


def test_exposure_direction_and_grouping():
    t = seed_exposure_table()
    assert t["TQQQ"].exposure_direction == ExposureDirection.LONG
    assert t["SQQQ"].exposure_direction == ExposureDirection.INVERSE
    # QQQ, TQQQ, SQQQ, QID, PSQ, QLD all map to the NASDAQ100 exposure group
    assert {t[s].underlying_exposure_group for s in ("QQQ", "TQQQ", "SQQQ", "QID")} == {"NASDAQ100"}


def test_safety_leveraged_inverse_never_auto_authorized():
    r = ExposureRegistry()
    # 1x long broad/sector ETFs auto-authorize; geared/inverse do NOT (§7).
    assert r.get("SPY").execution_authorized is True
    assert r.get("XLF").execution_authorized is True
    for sym in ("TQQQ", "SQQQ", "SH", "SOXL", "SOXS", "UPRO", "SPXU"):
        assert r.get(sym).execution_authorized is False, sym
    # explicit, user-authorized enablement is the only way to flip it
    assert r.authorize("TQQQ", True) is True
    assert r.get("TQQQ").execution_authorized is True


def test_effective_exposure_signed_for_inverse():
    t = seed_exposure_table()
    # $10k of SQQQ (-3x) => -$30k effective underlying exposure
    assert effective_exposure(10_000, t["SQQQ"].target_daily_leverage) == -30_000
    assert effective_exposure(10_000, t["TQQQ"].target_daily_leverage) == 30_000
    assert effective_exposure(10_000, t["SPY"].target_daily_leverage) == 10_000


def test_detect_opposing_positions():
    t = seed_exposure_table()
    pos = {s: t[s] for s in ("TQQQ", "SQQQ", "SPY")}  # TQQQ vs SQQQ oppose on NASDAQ100
    opp = detect_opposing(pos)
    assert len(opp) == 1
    a, b, group = opp[0]
    assert {a, b} == {"TQQQ", "SQQQ"} and group == "NASDAQ100"
    # same-direction holdings are not "opposing"
    assert detect_opposing({s: t[s] for s in ("TQQQ", "QLD")}) == []


def test_detect_overlap_by_group():
    t = seed_exposure_table()
    pos = {s: t[s] for s in ("QQQ", "TQQQ", "SPY")}
    ov = detect_overlap(pos)
    assert ov.get("NASDAQ100") == ["QQQ", "TQQQ"]
    assert "SP500" not in ov  # only SPY in SP500 -> no overlap


def test_reconcile_with_alpaca_marks_tradability():
    r = ExposureRegistry()
    assets = {"SPY": {"tradable": True, "status": "active"},
              "SQQQ": {"tradable": True, "status": "active"}}
    stats = r.reconcile(assets)
    assert stats["present_in_alpaca"] == 2
    assert r.get("SPY").tradable is True
    # a seeded symbol absent from Alpaca is kept, flagged non-tradable (not dropped)
    assert r.get("TQQQ").tradable is False
    assert r.get("TQQQ").asset_status == "not_in_alpaca_assets"


def test_counts_shape():
    r = ExposureRegistry()
    c = r.counts()
    assert c["leveraged_long"] >= 10 and c["inverse"] >= 4 and c["leveraged_inverse"] >= 10
    assert c["exposure_groups"] >= 8
    # only unlevered longs auto-authorize by default
    assert c["execution_authorized_default"] == c["broad_market"] + c["sector_industry"]
