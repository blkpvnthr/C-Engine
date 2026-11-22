"""Tests for market_scan.config.MarketScanConfig — defaults + env parsing.

Defaults must match the approved Section-4 parameter table. Env parsing is the
single place scanner knobs are read; MARKET_SCAN_ENABLED follows the project's
default-ON `!= "0"` idiom.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from market_scan.config import MarketScanConfig


def test_defaults_match_spec_table():
    c = MarketScanConfig()
    assert c.enabled is True
    assert c.replace_enabled is True
    assert c.data_feed == "iex"
    assert c.rank_refresh_seconds == 3.0
    assert c.discovery_interval_seconds == 300.0
    assert c.target_universe_size == 30
    assert c.min_price == 5.0
    assert c.min_avg_daily_volume == 500_000
    assert c.min_avg_daily_dollar_volume == 25_000_000.0
    assert c.max_spread_pct == 0.0010
    assert c.max_quote_age_seconds == 10.0
    assert c.order_flow_windows_s == (10, 30, 60, 300)
    assert c.auto_reconnect is True
    assert c.loss_removal_threshold == -0.005
    assert c.universe_max_age_seconds == 15.0
    assert tuple(c.market_refs) == ("SPY", "QQQ", "DIA", "IWM")
    assert c.max_subscriptions >= 1


def test_enabled_default_on(monkeypatch):
    monkeypatch.delenv("MARKET_SCAN_ENABLED", raising=False)
    assert MarketScanConfig.from_env().enabled is True


def test_enabled_only_disabled_by_zero(monkeypatch):
    for val, expect in [("0", False), ("1", True), ("true", True), (" 1 ", True), ("", True)]:
        monkeypatch.setenv("MARKET_SCAN_ENABLED", val)
        assert MarketScanConfig.from_env().enabled is expect, val


def test_replace_enabled_default_on_and_toggles(monkeypatch):
    monkeypatch.delenv("MARKET_SCAN_REPLACE_ENABLED", raising=False)
    assert MarketScanConfig.from_env().replace_enabled is True
    monkeypatch.setenv("MARKET_SCAN_REPLACE_ENABLED", "0")
    assert MarketScanConfig.from_env().replace_enabled is False


def test_numeric_overrides_from_env(monkeypatch):
    monkeypatch.setenv("MARKET_SCAN_MAX_SUBSCRIPTIONS", "25")
    monkeypatch.setenv("MARKET_SCAN_TARGET_SIZE", "12")
    monkeypatch.setenv("MARKET_SCAN_RANK_SECONDS", "2.5")
    monkeypatch.setenv("MARKET_SCAN_DISCOVERY_SECONDS", "120")
    monkeypatch.setenv("MARKET_SCAN_UNIVERSE_MAX_AGE", "20")
    monkeypatch.setenv("MARKET_SCAN_MIN_PRICE", "7")
    c = MarketScanConfig.from_env()
    assert c.max_subscriptions == 25
    assert c.target_universe_size == 12
    assert c.rank_refresh_seconds == 2.5
    assert c.discovery_interval_seconds == 120.0
    assert c.universe_max_age_seconds == 20.0
    assert c.min_price == 7.0


def test_bad_numeric_env_falls_back_to_default(monkeypatch):
    monkeypatch.setenv("MARKET_SCAN_TARGET_SIZE", "not-a-number")
    c = MarketScanConfig.from_env()
    assert c.target_universe_size == 30  # default preserved, no crash


def test_loss_threshold_clamped_strictly_negative(monkeypatch):
    # A positive/zero typo must coerce to the locked default, not invert the gate.
    monkeypatch.setenv("MARKET_SCAN_LOSS_THRESHOLD", "0.5")
    assert MarketScanConfig.from_env().loss_removal_threshold == -0.005
    monkeypatch.setenv("MARKET_SCAN_LOSS_THRESHOLD", "0")
    assert MarketScanConfig.from_env().loss_removal_threshold == -0.005
    monkeypatch.setenv("MARKET_SCAN_LOSS_THRESHOLD", "-0.01")
    assert MarketScanConfig.from_env().loss_removal_threshold == -0.01
    monkeypatch.setenv("MARKET_SCAN_LOSS_THRESHOLD", "-5")  # floored at -100%
    assert MarketScanConfig.from_env().loss_removal_threshold == -1.0


def test_data_feed_sip_honored_when_configured(monkeypatch):
    # Algo Trader Plus: SIP is now HONORED when explicitly configured (the engine
    # verifies entitlement separately via tools/sip_verify.py). It is no longer
    # coerced to IEX. Basic IEX remains the explicit compatibility profile.
    monkeypatch.setenv("MARKET_SCAN_DATA_FEED", "sip")
    assert MarketScanConfig.from_env().data_feed == "sip"


def test_data_feed_unknown_falls_back_to_iex(monkeypatch):
    # An unrecognized feed is never a silent SIP upgrade -> falls back to IEX.
    monkeypatch.setenv("MARKET_SCAN_DATA_FEED", "bogus")
    assert MarketScanConfig.from_env().data_feed == "iex"


def test_data_feed_defaults_to_iex(monkeypatch):
    monkeypatch.delenv("MARKET_SCAN_DATA_FEED", raising=False)
    monkeypatch.delenv("CENGINE_ALPACA_FEED", raising=False)
    assert MarketScanConfig.from_env().data_feed == "iex"


def test_sip_lifts_subscription_cap(monkeypatch):
    # The 30-symbol cap is a Basic-IEX constraint; SIP lifts it to a larger but
    # still hardware-bounded default (directive §5/§8).
    monkeypatch.delenv("MARKET_SCAN_MAX_SUBSCRIPTIONS", raising=False)
    monkeypatch.setenv("MARKET_SCAN_DATA_FEED", "sip")
    assert MarketScanConfig.from_env().max_subscriptions > 30
    monkeypatch.setenv("MARKET_SCAN_DATA_FEED", "iex")
    assert MarketScanConfig.from_env().max_subscriptions == 30


def test_scanner_feed_follows_engine_feed(monkeypatch):
    # A single CENGINE_ALPACA_FEED switch migrates the scanner too.
    monkeypatch.delenv("MARKET_SCAN_DATA_FEED", raising=False)
    monkeypatch.setenv("CENGINE_ALPACA_FEED", "sip")
    assert MarketScanConfig.from_env().data_feed == "sip"
