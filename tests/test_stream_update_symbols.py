"""Tests for live dynamic (re)subscription: AlpacaMarketDataStream.update_symbols."""

from __future__ import annotations

import asyncio
import json

import pytest

from market_data.alpaca_stream import (
    IEX_URL,
    AlpacaMarketDataStream,
    ConnectionState,
)


def make_stream(symbols=("QQQ", "SQQQ")):
    return AlpacaMarketDataStream(
        list(symbols),
        api_key="test-key",
        api_secret="test-secret",
        url=IEX_URL,
    )


class FakeSocket:
    def __init__(self):
        self.sent: list[dict] = []

    async def send(self, data):
        self.sent.append(json.loads(data))


def _run(coro):
    return asyncio.run(coro)


def _connected(stream):
    stream._socket = FakeSocket()
    stream._state = ConnectionState.SUBSCRIBED
    return stream._socket


def test_add_and_remove_updates_subscription_and_sends_frames():
    s = make_stream(("QQQ", "SQQQ"))
    sock = _connected(s)
    added, removed = _run(s.update_symbols(add=["NVDA"], remove=["SQQQ"]))
    assert added == ("NVDA",)
    assert removed == ("SQQQ",)
    assert set(s.symbols) == {"QQQ", "NVDA"}
    # Two frames: a subscribe for NVDA and an unsubscribe for SQQQ, all channels.
    actions = {f["action"]: f for f in sock.sent}
    assert actions["subscribe"]["trades"] == ["NVDA"]
    assert actions["subscribe"]["quotes"] == ["NVDA"]
    assert actions["subscribe"]["bars"] == ["NVDA"]
    assert actions["unsubscribe"]["quotes"] == ["SQQQ"]


def test_noop_when_nothing_changes():
    s = make_stream(("QQQ", "SQQQ"))
    sock = _connected(s)
    # QQQ already subscribed; TSLA not present -> no effective change.
    added, removed = _run(s.update_symbols(add=["QQQ"], remove=["TSLA"]))
    assert added == ()
    assert removed == ()
    assert sock.sent == []
    assert set(s.symbols) == {"QQQ", "SQQQ"}


def test_update_while_disconnected_still_updates_subscription():
    s = make_stream(("QQQ",))
    # no socket / not subscribed
    added, removed = _run(s.update_symbols(add=["NVDA"]))
    assert added == ("NVDA",)
    assert set(s.symbols) == {"QQQ", "NVDA"}  # applied for the next (re)connect


def test_refuse_to_remove_last_symbol():
    s = make_stream(("QQQ",))
    _connected(s)
    with pytest.raises(ValueError):
        _run(s.update_symbols(remove=["QQQ"]))
    assert set(s.symbols) == {"QQQ"}  # unchanged


def test_symbols_are_normalised():
    s = make_stream(("QQQ",))
    _connected(s)
    added, _ = _run(s.update_symbols(add=["  nvda "]))
    assert added == ("NVDA",)
    assert "NVDA" in s.symbols


def test_order_preserved_add_appends():
    s = make_stream(("QQQ", "SQQQ"))
    _connected(s)
    _run(s.update_symbols(add=["AAA", "BBB"]))
    assert s.symbols == ("QQQ", "SQQQ", "AAA", "BBB")
