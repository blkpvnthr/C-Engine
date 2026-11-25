"""Configuration for the real-time market scanner (IEX or SIP).

All scanner knobs live here with the approved Section-4 defaults.
:meth:`MarketScanConfig.from_env` is the single place ``MARKET_SCAN_*``
environment variables are parsed; it never raises on malformed input (a bad
value falls back to the default) so a typo can never take the trading engine
down. Feed selection is entitlement-aware: ``sip``/``delayed_sip`` are honored
for Algo Trader Plus accounts (verified separately by the engine), ``iex`` is
the explicit Basic compatibility profile, and an unrecognized value falls back
to IEX — never a silent SIP upgrade or downgrade.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


def _env_flag(name: str, default: bool) -> bool:
    """Project default-ON/OFF idiom: present-and-not-"0" is True."""
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip() != "0"


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw.strip())
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(float(raw.strip()))
    except (TypeError, ValueError):
        return default


@dataclass
class MarketScanConfig:
    """Scanner configuration (Section-4 defaults)."""

    # --- lifecycle / mode ---
    enabled: bool = True                      # MARKET_SCAN_ENABLED (default ON)
    replace_enabled: bool = True              # MARKET_SCAN_REPLACE_ENABLED
    data_feed: str = "iex"                    # iex | sip | delayed_sip

    # --- cadence ---
    rank_refresh_seconds: float = 3.0         # in-memory ranking refresh
    discovery_interval_seconds: float = 300.0 # broader discovery sweep

    # --- universe sizing / freshness ---
    target_universe_size: int = 30            # up to N published
    universe_max_age_seconds: float = 15.0    # hot-reload freshness ceiling
    candidate_score_ttl_s: float = 45.0       # expire a candidate score if its data
                                              # has not been refreshed within this window
    stream_live_staleness_s: float = 20.0     # no inbound message within this -> not live
    position_refresh_s: float = 5.0           # cadence for the shared position/pending
                                              # snapshot (NOT polled every rank tick)

    # --- eligibility thresholds ---
    min_price: float = 5.0
    min_avg_daily_volume: int = 500_000
    min_avg_daily_dollar_volume: float = 25_000_000.0
    max_spread_pct: float = 0.0010            # 0.10%
    max_quote_age_seconds: float = 10.0

    # --- microstructure / subscriptions ---
    order_flow_windows_s: tuple[int, ...] = (10, 30, 60, 300)
    auto_reconnect: bool = True
    max_subscriptions: int = 30               # feed-aware via from_env: IEX 30, SIP 250

    # --- loss-removal / replacement ---
    loss_removal_threshold: float = -0.005    # decimal fraction (-0.5%)
    # Replacement gating: require momentum + DI+>DI- + >BB-mid + >VWAP + liquidity
    # + regime for every replacement. Seller exhaustion is OPTIONAL (default OFF)
    # so stopped-out capital is redeployed more readily into the best-ranked
    # trend/structure-confirmed candidate; set 1 to also require it.
    replacement_require_seller_exhaustion: bool = False

    # --- always-monitored market references ---
    market_refs: tuple[str, ...] = ("SPY", "QQQ", "DIA", "IWM")

    def __post_init__(self) -> None:
        # Entitlement-aware feed selection (directive §5). SIP / delayed_sip are
        # honored for Algo Trader Plus accounts; IEX remains the explicit Basic
        # compatibility profile. An unrecognized value falls back to IEX (never a
        # silent SIP upgrade). The engine's startup verifies SIP entitlement
        # separately (tools/sip_verify.py); this class does not silently downgrade.
        feed = str(self.data_feed).strip().lower() or "iex"
        if feed not in ("iex", "sip", "delayed_sip"):
            feed = "iex"
        self.data_feed = feed
        self.max_subscriptions = max(1, int(self.max_subscriptions))
        self.target_universe_size = max(1, int(self.target_universe_size))
        # The loss-removal threshold must be strictly negative (a decimal
        # fraction, e.g. -0.005). A zero/positive value (operator typo like
        # "0.5") would invert the gate into a full-book liquidation, so coerce
        # it back to the locked default and floor it at -100%.
        thr = float(self.loss_removal_threshold)
        if not (thr < 0.0):
            thr = -0.005
        self.loss_removal_threshold = max(-1.0, thr)

    @classmethod
    def from_env(cls) -> "MarketScanConfig":
        """Build config from ``MARKET_SCAN_*`` env vars (fail-soft).

        The scanner feed defaults to the engine-wide ``CENGINE_ALPACA_FEED`` so a
        single switch migrates the whole system to SIP. The subscription cap is
        feed-aware: Basic IEX keeps the conservative 30; SIP (unlimited provider
        symbols) uses a larger, still hardware-bounded default — "unlimited" at
        the provider is NOT unlimited compute (directive §5/§8).
        """
        feed = os.getenv("MARKET_SCAN_DATA_FEED",
                         os.getenv("CENGINE_ALPACA_FEED", "iex"))
        is_sip = str(feed).strip().lower() in ("sip", "delayed_sip")
        default_subs = 250 if is_sip else 30
        return cls(
            enabled=_env_flag("MARKET_SCAN_ENABLED", True),
            replace_enabled=_env_flag("MARKET_SCAN_REPLACE_ENABLED", True),
            data_feed=feed,
            rank_refresh_seconds=_env_float("MARKET_SCAN_RANK_SECONDS", 3.0),
            discovery_interval_seconds=_env_float(
                "MARKET_SCAN_DISCOVERY_SECONDS", 300.0
            ),
            # Feed-aware like max_subscriptions: Basic IEX keeps the conservative
            # 30; SIP publishes the full hardware-bounded discovery breadth so
            # market-wide SIP coverage is not silently capped at 30 (P0.2 / §10).
            target_universe_size=_env_int("MARKET_SCAN_TARGET_SIZE", default_subs),
            universe_max_age_seconds=_env_float(
                "MARKET_SCAN_UNIVERSE_MAX_AGE", 15.0
            ),
            min_price=_env_float("MARKET_SCAN_MIN_PRICE", 5.0),
            min_avg_daily_volume=_env_int("MARKET_SCAN_MIN_ADV", 500_000),
            min_avg_daily_dollar_volume=_env_float(
                "MARKET_SCAN_MIN_ADDV", 25_000_000.0
            ),
            max_spread_pct=_env_float("MARKET_SCAN_MAX_SPREAD_PCT", 0.0010),
            max_quote_age_seconds=_env_float("MARKET_SCAN_MAX_QUOTE_AGE", 10.0),
            auto_reconnect=_env_flag("MARKET_SCAN_AUTO_RECONNECT", True),
            max_subscriptions=_env_int("MARKET_SCAN_MAX_SUBSCRIPTIONS", default_subs),
            loss_removal_threshold=_env_float(
                "MARKET_SCAN_LOSS_THRESHOLD", -0.005
            ),
            replacement_require_seller_exhaustion=_env_flag(
                "MARKET_SCAN_REQUIRE_SELLER_EXHAUSTION", False
            ),
        )
