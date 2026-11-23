"""P0.2 regression: feed-aware market-wide SIP coverage (not capped at 30).

Correction to the initial hypothesis: ``python3 trade.py`` DOES load ``.env``
(via ``load_runtime_env`` -> ``_load_env_file`` in trade.py, called from both
``main`` and the composition root), so ``CENGINE_ALPACA_FEED=sip`` already
reaches the process -- the runtime evidence ``connecting feed=sip symbols=30``
confirms the feed is SIP. The genuine root cause of the 30-symbol cap is purely
hardcoded subscription limits:

  1. ``MarketScanConfig.from_env`` made ``max_subscriptions`` feed-aware (SIP 250
     / IEX 30) but left ``target_universe_size`` hardcoded at 30, capping the
     PUBLISHED discovery universe at 30 even on SIP.
  2. ``trade.py`` built ``UniverseSubscriptionSync`` with a hardcoded ``"30"``
     literal fallback instead of the feed-aware ``max_subscriptions``.

This tests the config-level fix (1); fix (2) reuses the same ``from_env`` budget.
"""

from __future__ import annotations

from market_scan.config import MarketScanConfig

_FEED_ENV = (
    "MARKET_SCAN_TARGET_SIZE",
    "MARKET_SCAN_DATA_FEED",
    "MARKET_SCAN_MAX_SUBSCRIPTIONS",
)


def test_target_universe_size_is_feed_aware_for_sip(monkeypatch):
    for var in _FEED_ENV:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("CENGINE_ALPACA_FEED", "sip")

    cfg = MarketScanConfig.from_env()

    # Previously hardcoded 30 regardless of feed -> capped SIP discovery at 30.
    assert cfg.target_universe_size == 250
    assert cfg.max_subscriptions == 250  # already feed-aware; must stay consistent


def test_target_universe_size_stays_conservative_for_iex(monkeypatch):
    for var in _FEED_ENV:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("CENGINE_ALPACA_FEED", "iex")

    cfg = MarketScanConfig.from_env()

    assert cfg.target_universe_size == 30
    assert cfg.max_subscriptions == 30


def test_explicit_target_size_overrides_feed_default(monkeypatch):
    monkeypatch.setenv("CENGINE_ALPACA_FEED", "sip")
    monkeypatch.setenv("MARKET_SCAN_TARGET_SIZE", "12")

    cfg = MarketScanConfig.from_env()

    assert cfg.target_universe_size == 12


def test_delayed_sip_is_also_feed_aware(monkeypatch):
    for var in _FEED_ENV:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("CENGINE_ALPACA_FEED", "delayed_sip")

    cfg = MarketScanConfig.from_env()

    assert cfg.target_universe_size == 250
