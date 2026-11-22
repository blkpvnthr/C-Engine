"""Tests for market_scan.stream_client.StreamClient.

The real Alpaca StockDataStream is never constructed in tests — a fake stream is
injected via stream_factory. Tests pin: handler parsing -> ingest callbacks,
dynamic subscribe/unsubscribe diffing, single-stream ownership, and
reconnect/backoff logic. No socket is ever opened.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from market_scan.stream_client import StreamClient


class _FakeStream:
    def __init__(self, *a, **k):
        self.trade_handler = None
        self.quote_handler = None
        self.subscribed_trades: set[str] = set()
        self.subscribed_quotes: set[str] = set()
        self.unsub_trades: list[str] = []
        self.run_calls = 0
        self.fail_times = 0

    def subscribe_trades(self, handler, *syms):
        self.trade_handler = handler
        self.subscribed_trades.update(syms)

    def subscribe_quotes(self, handler, *syms):
        self.quote_handler = handler
        self.subscribed_quotes.update(syms)

    def unsubscribe_trades(self, *syms):
        self.unsub_trades.extend(syms)
        self.subscribed_trades.difference_update(syms)

    def unsubscribe_quotes(self, *syms):
        self.subscribed_quotes.difference_update(syms)

    async def _run_forever(self):
        self.run_calls += 1
        if self.run_calls <= self.fail_times:
            raise ConnectionError("socket dropped")
        return


def _client(**kw):
    trades, quotes = [], []
    fake = _FakeStream()
    c = StreamClient(
        api_key="k", secret_key="s", feed="iex",
        on_trade=lambda *a: trades.append(a),
        on_quote=lambda *a: quotes.append(a),
        stream_factory=lambda *a, **k: fake,
        **kw,
    )
    return c, fake, trades, quotes


def test_apply_subscriptions_adds_and_removes():
    c, fake, _t, _q = _client()
    c.connect()
    asyncio.run(c.apply_subscriptions(to_add=["NVDA", "AMD"], to_remove=[]))
    assert fake.subscribed_trades == {"NVDA", "AMD"}
    assert fake.subscribed_quotes == {"NVDA", "AMD"}
    asyncio.run(c.apply_subscriptions(to_add=["TSLA"], to_remove=["AMD"]))
    assert "AMD" in fake.unsub_trades
    assert "TSLA" in fake.subscribed_trades
    assert c.subscribed == {"NVDA", "TSLA"}


def test_trade_handler_parses_and_ingests():
    c, fake, trades, _q = _client()
    c.connect()
    asyncio.run(c.apply_subscriptions(to_add=["NVDA"], to_remove=[]))

    async def go():
        await fake.trade_handler(SimpleNamespace(
            symbol="NVDA", price=100.5, size=10,
            timestamp=SimpleNamespace(timestamp=lambda: 1_000_000.0)))
    asyncio.run(go())
    assert trades and trades[0][0] == "NVDA"
    assert trades[0][2] == 100.5 and trades[0][3] == 10.0  # (sym, ts, price, size)


def test_quote_handler_parses_and_ingests():
    c, fake, _t, quotes = _client()
    c.connect()
    asyncio.run(c.apply_subscriptions(to_add=["AMD"], to_remove=[]))

    async def go():
        await fake.quote_handler(SimpleNamespace(
            symbol="AMD", bid_price=9.99, ask_price=10.01,
            bid_size=100, ask_size=200,
            timestamp=SimpleNamespace(timestamp=lambda: 1_000_000.0)))
    asyncio.run(go())
    assert quotes and quotes[0][0] == "AMD"
    assert quotes[0][2] == 9.99 and quotes[0][3] == 10.01  # (sym, ts, bid, ask, bid_sz, ask_sz)


def test_run_reconnects_with_backoff():
    sleeps = []
    c, fake, _t, _q = _client(reconnect=True, backoff_base=0.1, backoff_max=1.0,
                              sleep_fn=lambda d: sleeps.append(d))
    fake.fail_times = 2  # drop twice, then connect cleanly

    async def go():
        await c.run(max_reconnects=3)
    asyncio.run(go())
    assert fake.run_calls == 3          # 2 failures + 1 success
    assert len(sleeps) == 2             # backoff slept before each reconnect
    assert sleeps[0] <= sleeps[1]       # non-decreasing backoff


def test_run_without_reconnect_raises():
    c, fake, _t, _q = _client(reconnect=False)
    fake.fail_times = 1
    import pytest

    async def go():
        await c.run()
    with pytest.raises(ConnectionError):
        asyncio.run(go())


def test_only_one_stream_constructed():
    constructed = []

    def factory(*a, **k):
        s = _FakeStream()
        constructed.append(s)
        return s
    c = StreamClient(api_key="k", secret_key="s", on_trade=lambda *a: None,
                     on_quote=lambda *a: None, stream_factory=factory)
    c.connect()
    c.connect()  # idempotent
    assert len(constructed) == 1
