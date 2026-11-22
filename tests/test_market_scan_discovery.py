"""Tests for market_scan.discovery — web discovery of US-listed securities.

Fetchers are injectable so tests never hit the network. Discovery normalizes/
validates/classifies via universe.build_universe + classify_symbol, always
includes the leveraged/inverse/sector ETF seed (so those stay discoverable),
and is fail-safe: a fetch error degrades gracefully, never raises.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from market_scan.discovery import DEFAULT_ETF_SEED, discover_us_listed


def test_combines_sources_validates_and_classifies():
    res = discover_us_listed(
        sp500_fetcher=lambda: ["AAPL", "MSFT", "NVD"],   # NVD is a known-invalid typo
        nasdaq_fetcher=lambda: ["AMD", "AAPL"],           # dup across sources
        extra=["TQQQ"],
    )
    assert "AAPL" in res.symbols and "AMD" in res.symbols and "MSFT" in res.symbols
    assert "NVD" not in res.symbols            # invalid dropped by build_universe
    assert res.symbols.count("AAPL") == 1      # de-duped
    assert "TQQQ" in res.etfs                  # leveraged ETF classified as ETF
    assert "AAPL" in res.equities
    assert res.error is None


def test_etf_seed_always_present_for_directional_eligibility():
    res = discover_us_listed(sp500_fetcher=lambda: [], nasdaq_fetcher=lambda: [])
    for etf in ("SQQQ", "SOXS", "TQQQ", "SPY"):  # inverse + leveraged + index
        assert etf in res.symbols
    assert "SQQQ" in res.etfs and "SOXS" in res.etfs


def test_fetch_error_is_fail_safe_not_raising():
    def boom():
        raise RuntimeError("network down")
    res = discover_us_listed(sp500_fetcher=boom, nasdaq_fetcher=lambda: ["AMD"])
    assert res.error is not None
    assert "AMD" in res.symbols            # the source that worked still contributes
    assert "SPY" in res.symbols            # seed still present


def test_total_fetch_failure_returns_seed_only_with_error():
    def boom():
        raise RuntimeError("down")
    res = discover_us_listed(sp500_fetcher=boom, nasdaq_fetcher=boom)
    assert res.error is not None
    assert res.symbols == sorted(set(res.symbols))  # defined, non-crashing
    assert "SPY" in res.symbols            # seed guarantees non-empty


def test_max_symbols_caps_after_validation():
    big = [f"SYM{i}" for i in range(5)] + ["AAPL", "AMD"]
    res = discover_us_listed(sp500_fetcher=lambda: big, nasdaq_fetcher=lambda: [],
                             max_symbols=4)
    assert len(res.symbols) == 4


def test_default_etf_seed_has_all_directional_classes():
    # sanity: the seed spans index, sector, leveraged-bull, inverse
    assert {"SPY", "QQQ", "DIA", "IWM"} <= set(DEFAULT_ETF_SEED)
    assert {"TQQQ", "SOXL"} <= set(DEFAULT_ETF_SEED)      # leveraged bull
    assert {"SQQQ", "SOXS", "SH"} <= set(DEFAULT_ETF_SEED)  # inverse
