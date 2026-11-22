"""Tests for the renamed, provider-neutral Alpaca market-data stream.

These use mocked WebSocket frames and never touch the network or place orders.
"""

import asyncio
import json

import pytest

import market_data.alpaca_stream as mod
from market_data.alpaca_sip_stream import AlpacaSIPStream
from market_data.alpaca_stream import (
    IEX_URL,
    SIP_URL,
    AlpacaAuthenticationError,
    AlpacaConnectionLimitError,
    AlpacaEntitlementError,
    AlpacaMarketDataStream,
    AlpacaSubscriptionError,
    BarEvent,
    QuoteEvent,
    TradeEvent,
)


def make_stream(url=IEX_URL, **kwargs):
    return AlpacaMarketDataStream(
        ["QQQ", "SQQQ"],
        api_key="test-key",
        api_secret="test-secret",
        url=url,
        backoff_initial=0.001,
        backoff_max=0.002,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Fake transport
# ---------------------------------------------------------------------------


class FakeSocket:
    """Scripted WebSocket: recv() yields queued frames; send() records output."""

    def __init__(self, inbound, *, on_exhausted=None):
        self._inbound = list(inbound)
        self.sent: list[str] = []
        self._on_exhausted = on_exhausted

    async def recv(self):
        if not self._inbound:
            if self._on_exhausted is not None:
                self._on_exhausted()
            raise RuntimeError("fake socket drained")  # transient, not a CC
        item = self._inbound.pop(0)
        return item if isinstance(item, (str, bytes)) else json.dumps(item)

    async def send(self, data):
        self.sent.append(data)

    async def close(self, code=1000, reason=""):
        return None


class FakeConnect:
    """Callable replacement for websockets.connect returning an async CM."""

    def __init__(self, sockets, *, on_empty=None):
        self._sockets = list(sockets)
        self._on_empty = on_empty
        self.calls = 0
        self.last_socket: FakeSocket | None = None

    def __call__(self, url, **kwargs):
        self.calls += 1
        factory = self

        class _CM:
            async def __aenter__(self_inner):
                if not factory._sockets:
                    if factory._on_empty is not None:
                        factory._on_empty()
                    raise RuntimeError("no more fake connections")
                factory.last_socket = factory._sockets.pop(0)
                return factory.last_socket

            async def __aexit__(self_inner, *exc):
                return False

        return _CM()


def connected_frames(symbols=("QQQ", "SQQQ")):
    syms = list(symbols)
    return [
        [{"T": "success", "msg": "connected"}],
        [{"T": "success", "msg": "authenticated"}],
        [{"T": "subscription", "trades": syms, "quotes": syms, "bars": syms}],
    ]


QUOTE_MSG = {
    "T": "q", "S": "QQQ", "t": "2026-01-02T15:00:00.123456789Z",
    "bp": 400.10, "bs": 3, "bx": "V", "ap": 400.12, "as": 5, "ax": "V",
    "c": ["R"], "z": "C",
}
TRADE_MSG = {
    "T": "t", "S": "QQQ", "t": "2026-01-02T15:00:01Z",
    "i": 99, "x": "V", "p": 400.11, "s": 10, "c": ["@"], "z": "C",
}
BAR_MSG = {
    "T": "b", "S": "QQQ", "t": "2026-01-02T15:01:00Z",
    "o": 400.0, "h": 401.0, "l": 399.5, "c": 400.5, "v": 12345, "n": 42, "vw": 400.3,
}


# ---------------------------------------------------------------------------
# Feed derivation & health
# ---------------------------------------------------------------------------


def test_feed_and_endpoint_derived_from_url():
    assert make_stream().feed == "iex"
    assert make_stream().endpoint == IEX_URL
    assert make_stream(url=SIP_URL).feed == "sip"


def test_backward_compatible_alias_is_same_class():
    assert AlpacaSIPStream is AlpacaMarketDataStream


def test_health_reports_feed_and_idle_distinct_from_dead():
    stream = make_stream()
    health = stream.health()
    assert health.feed == "iex"
    assert health.endpoint == IEX_URL
    assert health.active_symbol_count == 2
    assert health.subscription_acknowledged is False
    # No event yet -> age is None (a quiet feed is not a timestamp of 0).
    assert health.data_age_ns(10**18) is None


def test_endpoint_contains_no_credentials():
    stream = make_stream()
    assert "test-key" not in stream.endpoint
    assert "test-secret" not in stream.endpoint


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------


def test_parse_quote_trade_bar_normalization():
    stream = make_stream()
    quote = stream._parse_quote(QUOTE_MSG)
    assert isinstance(quote, QuoteEvent)
    assert quote.symbol == "QQQ"
    assert quote.bid_price_ticks == 4_001_000  # 400.10 * 10_000
    assert quote.ask_price_ticks == 4_001_200
    assert quote.bid_size == 3 and quote.ask_size == 5
    # nanosecond precision preserved
    assert quote.timestamp_ns % 1_000_000_000 == 123_456_789

    trade = stream._parse_trade(TRADE_MSG)
    assert isinstance(trade, TradeEvent)
    assert trade.price_ticks == 4_001_100 and trade.size == 10

    bar = stream._parse_bar(BAR_MSG)
    assert isinstance(bar, BarEvent)
    assert bar.open_ticks == 4_000_000 and bar.close_ticks == 4_005_000
    assert bar.volume == 12345 and bar.vwap_ticks == 4_003_000


def test_zero_trade_size_is_malformed_and_counted():
    stream = make_stream()
    bad = dict(TRADE_MSG, s=0)
    asyncio.run(stream._dispatch_message(bad))
    assert stream.health().malformed_messages == 1
    assert stream.health().trades_received == 0


# ---------------------------------------------------------------------------
# Subscription acknowledgement
# ---------------------------------------------------------------------------


def test_partial_subscription_rejected():
    stream = make_stream()
    partial = {"trades": ["QQQ", "SQQQ"], "quotes": ["QQQ"], "bars": ["QQQ", "SQQQ"]}
    with pytest.raises(AlpacaSubscriptionError, match="did not subscribe quotes"):
        stream._verify_subscription(partial)


def test_full_subscription_accepted():
    stream = make_stream()
    both = ["QQQ", "SQQQ"]
    stream._verify_subscription({"trades": both, "quotes": both, "bars": both})


# ---------------------------------------------------------------------------
# Server-error categorization & counters
# ---------------------------------------------------------------------------


def test_entitlement_error_categorized_and_counted():
    stream = make_stream()
    with pytest.raises(AlpacaEntitlementError):
        stream._raise_server_error({"T": "error", "code": 402, "msg": "insufficient subscription"})
    assert stream.health().entitlement_errors == 1


def test_connection_limit_error_categorized():
    stream = make_stream()
    with pytest.raises(AlpacaConnectionLimitError):
        stream._raise_server_error({"T": "error", "code": 406, "msg": "connection limit"})
    assert stream.health().feed_errors == 1


def test_auth_error_categorized():
    stream = make_stream()
    with pytest.raises(AlpacaAuthenticationError):
        stream._raise_server_error({"T": "error", "code": 401, "msg": "auth failed"})


# ---------------------------------------------------------------------------
# Full connect -> auth -> subscribe -> consume cycle (mocked)
# ---------------------------------------------------------------------------


async def _run_with_connect(stream, fake_connect, monkeypatch, timeout=2.0):
    monkeypatch.setattr(mod.websockets, "connect", fake_connect)
    await asyncio.wait_for(stream.run_forever(), timeout=timeout)


async def test_successful_cycle_authenticates_subscribes_and_consumes(monkeypatch):
    stream = make_stream()
    received: list[QuoteEvent] = []

    def on_quote(event):
        received.append(event)
        # Stop as soon as we have proven end-to-end delivery.
        stream._stop_event.set()

    stream._on_quote = on_quote
    socket = FakeSocket(connected_frames() + [[QUOTE_MSG]])
    fake = FakeConnect([socket])
    await _run_with_connect(stream, fake, monkeypatch)

    assert len(received) == 1 and received[0].symbol == "QQQ"
    assert stream.subscription_acknowledged is True
    assert stream.health().quotes_received == 1
    assert stream.health().feed == "iex"
    assert fake.calls == 1
    assert stream.health().reconnects == 0
    # auth + subscribe messages were sent (never logging credentials).
    sent = [json.loads(s) for s in socket.sent]
    assert any(m.get("action") == "auth" for m in sent)
    assert any(m.get("action") == "subscribe" for m in sent)


async def test_authentication_failure_is_not_retried(monkeypatch):
    stream = make_stream()
    socket = FakeSocket(
        [
            [{"T": "success", "msg": "connected"}],
            [{"T": "error", "code": 401, "msg": "auth failed"}],
        ]
    )
    fake = FakeConnect([socket])
    with pytest.raises(AlpacaAuthenticationError):
        await _run_with_connect(stream, fake, monkeypatch)
    # Credential failures must not reconnect in a tight loop.
    assert fake.calls == 1
    assert stream.health().reconnects == 0


async def test_transient_disconnect_reconnects(monkeypatch):
    stream = make_stream()
    # Two transient failures, then connect runs dry and stops the loop.
    failing1 = FakeSocket([])  # recv raises transient immediately
    failing2 = FakeSocket([])
    fake = FakeConnect([failing1, failing2], on_empty=stream._stop_event.set)
    await _run_with_connect(stream, fake, monkeypatch)
    # attempt1 (first_connection) does not count; attempt2 does; attempt3 is
    # the stop. Exactly one reconnect is recorded.
    assert fake.calls == 3
    assert stream.health().reconnects == 1
