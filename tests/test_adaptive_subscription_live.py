"""Adaptive subscription manager wired to the live SIP stream (directive item 2).

Verifies per-channel stream subscriptions (distinct bars/trades/quotes symbol sets
on ONE connection) and the AdaptiveSubscriptionController driving them, including
the Tier1->Tier2->Tier3 promotion progression. Uses a fake socket; no network.
"""

from __future__ import annotations

import json

import pytest

from cengine.adaptive_subscription_controller import AdaptiveSubscriptionController
from market_data.alpaca_stream import AlpacaMarketDataStream, ConnectionState
from market_scan.adaptive_subscriptions import Channel, ChannelCaps, SubState


class _FakeSocket:
    def __init__(self):
        self.sent = []

    async def send(self, data):
        self.sent.append(json.loads(data))


def _live_stream():
    s = AlpacaMarketDataStream(["AAPL"], api_key="k", api_secret="s")
    s._socket = _FakeSocket()
    s._state = ConnectionState.SUBSCRIBED
    return s


# --------------------------------------------------- stream per-channel frames
@pytest.mark.asyncio
async def test_set_channel_subscriptions_distinct_lists():
    s = _live_stream()
    res = await s.set_channel_subscriptions(
        bars=["AAA", "BBB", "CCC"], trades=["AAA", "BBB"], quotes=["AAA"])
    assert res["bars"][0] == ("AAA", "BBB", "CCC")
    assert res["trades"][0] == ("AAA", "BBB")
    assert res["quotes"][0] == ("AAA",)
    # subscribe frames carry DISTINCT per-channel lists
    frames = s._socket.sent
    bars_frame = next(f for f in frames if "bars" in f and f["action"] == "subscribe")
    assert bars_frame["bars"] == ["AAA", "BBB", "CCC"]
    quotes_frame = next(f for f in frames if "quotes" in f and f["action"] == "subscribe")
    assert quotes_frame["quotes"] == ["AAA"]


@pytest.mark.asyncio
async def test_set_channel_subscriptions_delta_and_unsubscribe():
    s = _live_stream()
    await s.set_channel_subscriptions(bars=["AAA", "BBB"], trades=["AAA"])
    s._socket.sent.clear()
    # drop BBB bars, add CCC bars + promote AAA... to quotes
    res = await s.set_channel_subscriptions(bars=["AAA", "CCC"], trades=["AAA"],
                                            quotes=["AAA"])
    assert res["bars"] == (("CCC",), ("BBB",))      # +CCC, -BBB
    assert res["trades"] == ((), ())                 # unchanged
    assert res["quotes"] == (("AAA",), ())           # +AAA
    actions = [(f["action"], tuple(k for k in f if k != "action")) for f in s._socket.sent]
    assert ("unsubscribe", ("bars",)) in actions
    assert ("subscribe", ("quotes",)) in actions


@pytest.mark.asyncio
async def test_reconnect_resubscribes_per_channel():
    s = _live_stream()
    await s.set_channel_subscriptions(bars=["AAA", "BBB"], trades=["AAA"], quotes=[])
    assert s._channel_subs_active()  # reconnect will use the per-channel frame
    fresh = _FakeSocket()
    await s._subscribe_channels(fresh)  # the frame _subscribe sends on reconnect
    frame = fresh.sent[0]
    assert frame["action"] == "subscribe"
    assert frame["bars"] == ["AAA", "BBB"] and frame["trades"] == ["AAA"]
    assert "quotes" not in frame  # empty channel omitted


def test_channel_subs_gate_selects_path():
    # Empty channel state -> legacy union path; populated -> per-channel path.
    s = AlpacaMarketDataStream(["AAPL"], api_key="k", api_secret="s")
    assert not s._channel_subs_active()
    s._channel_subs["bars"].add("AAA")
    assert s._channel_subs_active()


# --------------------------------------------------- controller + promotion
@pytest.mark.asyncio
async def test_controller_promotion_progression():
    s = _live_stream()
    ctl = AdaptiveSubscriptionController(s, caps=ChannelCaps(bars=100, trades=50, quotes=20))
    # Tier 1: WIN is bar-only (preliminary discovery, no trade dependency).
    r1 = await ctl.apply(tier1_bars=["WIN", "OTH"], now=0.0)
    assert "WIN" in s._channel_subs["bars"] and "WIN" not in s._channel_subs["trades"]
    assert r1["applied"] and not r1["degraded"]
    # Tier 2: promote WIN -> gains trades (no re-subscribe of bars needed).
    await ctl.apply(tier1_bars=["WIN", "OTH"], tier2_trades=["WIN"], now=60.0)
    assert "WIN" in s._channel_subs["trades"]
    # Tier 3: actionable -> gains quotes.
    await ctl.apply(tier1_bars=["WIN", "OTH"], tier2_trades=["WIN"],
                    tier3_quotes=["WIN"], now=120.0)
    assert "WIN" in s._channel_subs["quotes"]


@pytest.mark.asyncio
async def test_controller_protected_degraded_flag():
    s = _live_stream()
    ctl = AdaptiveSubscriptionController(s, caps=ChannelCaps(bars=100, trades=50, quotes=2))
    out = await ctl.apply(protected=["P1", "P2", "P3"], now=0.0)  # 3 protected > quotes cap 2
    assert out["degraded"] is True
    assert len(s._channel_subs["quotes"]) == 2
    # bars cap is ample -> protected fully covered on bars
    assert {"P1", "P2", "P3"} <= s._channel_subs["bars"]


@pytest.mark.asyncio
async def test_controller_single_connection_only():
    # The controller drives the ONE provided stream; it never constructs another.
    s = _live_stream()
    ctl = AdaptiveSubscriptionController(s)
    await ctl.apply(tier1_bars=["AAA"], now=0.0)
    assert ctl._stream is s


# ------------------------------------------- live trade/quote promotion lifecycle
def _state(ctl, channel, sym):
    sub = ctl.manager._subs.get((channel, sym))
    return sub.state if sub is not None else None


@pytest.mark.asyncio
async def test_observation_promotes_warming_to_fresh():
    """Directive item 6: a requested (symbol, channel) advances WARMING_UP -> FRESH
    only when a real tick is observed on that channel."""
    s = _live_stream()
    ctl = AdaptiveSubscriptionController(s)
    # Request AAA on bars+trades; apply() acks the adds -> WARMING_UP.
    await ctl.apply(tier1_bars=["AAA"], tier2_trades=["AAA"], now=0.0)
    assert _state(ctl, Channel.BARS, "AAA") == SubState.WARMING_UP
    assert _state(ctl, Channel.TRADES, "AAA") == SubState.WARMING_UP
    # A live trade tick promotes ONLY the trades channel to FRESH.
    ctl.on_observation(Channel.TRADES, "AAA", now=1.0)
    assert _state(ctl, Channel.TRADES, "AAA") == SubState.FRESH
    assert _state(ctl, Channel.BARS, "AAA") == SubState.WARMING_UP  # bars untouched
    # A bar tick then promotes bars too.
    ctl.on_observation(Channel.BARS, "AAA", now=1.5)
    assert _state(ctl, Channel.BARS, "AAA") == SubState.FRESH


@pytest.mark.asyncio
async def test_observation_of_unsubscribed_symbol_is_ignored():
    """Ticks for symbols never requested are a cheap no-op (no phantom subs)."""
    s = _live_stream()
    ctl = AdaptiveSubscriptionController(s)
    await ctl.apply(tier1_bars=["AAA"], now=0.0)
    ctl.on_observation(Channel.TRADES, "ZZZ", now=1.0)  # never requested
    assert _state(ctl, Channel.TRADES, "ZZZ") is None


@pytest.mark.asyncio
async def test_stale_demotes_fresh_after_window():
    """The lifecycle is symmetric: silence past the stale window demotes FRESH ->
    STALE; a still-ticking channel stays FRESH."""
    s = _live_stream()
    ctl = AdaptiveSubscriptionController(s)
    ctl.manager.stale_after_s = 20.0
    await ctl.apply(tier1_bars=["AAA"], tier2_trades=["AAA"], now=0.0)
    ctl.on_observation(Channel.BARS, "AAA", now=1.0)
    ctl.on_observation(Channel.TRADES, "AAA", now=100.0)  # trades keep ticking
    # 110s: bars last seen at 1 (>20s ago) -> stale; trades at 100 -> still fresh.
    staled = ctl.mark_stale(now=110.0)
    assert staled == 1
    assert _state(ctl, Channel.BARS, "AAA") == SubState.STALE
    assert _state(ctl, Channel.TRADES, "AAA") == SubState.FRESH
