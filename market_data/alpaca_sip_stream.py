"""Backward-compatible shim for the renamed market-data stream module.

The live Alpaca market-data consumer moved to :mod:`market_data.alpaca_stream`
and its class was renamed to the provider-neutral ``AlpacaMarketDataStream``
when IEX became the default feed. This module re-exports the public surface so
existing imports of ``market_data.alpaca_sip_stream`` (and ``AlpacaSIPStream``)
continue to work unchanged. Prefer importing from ``market_data.alpaca_stream``.
"""

from __future__ import annotations

from .alpaca_stream import (  # noqa: F401
    DEFAULT_AUTH_TIMEOUT,
    DEFAULT_BACKOFF_INITIAL,
    DEFAULT_BACKOFF_MAX,
    DEFAULT_CONNECT_TIMEOUT,
    DEFAULT_HEARTBEAT_TIMEOUT,
    DEFAULT_PRICE_SCALE,
    DEFAULT_UNIVERSE,
    IEX_URL,
    INVERSE_EQUITY,
    PRIMARY_TARGET,
    SIP_URL,
    TEST_URL,
    VOLATILITY_FACTORS,
    AlpacaAuthenticationError,
    AlpacaConnectionLimitError,
    AlpacaEntitlementError,
    AlpacaMarketDataStream,
    AlpacaSIPStream,
    AlpacaStreamError,
    AlpacaSubscriptionError,
    BarEvent,
    ConnectionState,
    IndexEvent,
    IndexFactorBridge,
    MarketDataFanout,
    MarketDataQueue,
    QuoteEvent,
    StreamHealth,
    Subscription,
    TradeEvent,
    VolatilityFactorState,
)

__all__ = [
    "AlpacaAuthenticationError",
    "AlpacaConnectionLimitError",
    "AlpacaEntitlementError",
    "AlpacaMarketDataStream",
    "AlpacaSIPStream",
    "AlpacaStreamError",
    "AlpacaSubscriptionError",
    "BarEvent",
    "ConnectionState",
    "IEX_URL",
    "IndexEvent",
    "IndexFactorBridge",
    "MarketDataFanout",
    "MarketDataQueue",
    "QuoteEvent",
    "SIP_URL",
    "StreamHealth",
    "Subscription",
    "TEST_URL",
    "TradeEvent",
    "VolatilityFactorState",
]
