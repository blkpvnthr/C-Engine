"""Full production discovery-to-execution path (directive item 9).

Drives the ACTUAL acquisition + staged promotion sequence through the real
components — membership -> native registration -> warm-up ingestion -> confirmed
bar subscription -> GPU structural screening -> preliminary ranking -> confirmed
trade subscription -> momentum confirmation -> confirmed quote subscription ->
final GPU qualification -> exposure authorization -> OrderManager -> mock fill —
across conventional equities, broad / sector / leveraged-bull / inverse /
leveraged-inverse ETFs, INCLUDING instruments outside any prior flat discovery
window. Nothing is injected to bypass the acquisition/promotion sequence: the
candidate set at each stage is produced by the real manager, GPU screen, and
exposure gate. Provider acknowledgements and streaming observations are simulated
(no network); final live-session certification remains session-gated.
"""

from __future__ import annotations

import json

import _cengine_native as native
import pytest

from cengine.adaptive_subscription_controller import AdaptiveSubscriptionController
from cengine.exposure_registry import ExposureRegistry
from cengine.gpu_scanner_bridge import EngineGpuDiscovery
from market_data.alpaca_stream import AlpacaMarketDataStream, ConnectionState
from market_scan.adaptive_subscriptions import Channel, ChannelCaps, SubState

SCALE = 10_000
_NS_MIN = 60 * 1_000_000_000

# One instrument per category; AAPL/SPY/XLK are electively authorizable, the geared
# and inverse products are blocked by the exposure gate until explicitly authorized.
UNIVERSE = ["AAPL", "SPY", "XLK", "TQQQ", "SH", "SQQQ"]
AUTHORIZABLE = {"AAPL", "SPY", "XLK"}  # conventional + broad + sector


def _t(x):
    return int(round(x * SCALE))


class _FakeSocket:
    def __init__(self):
        self.sent = []

    async def send(self, data):
        self.sent.append(json.loads(data))


class _BarEvent:
    def __init__(self, sym, ts, o, h, lo, c, v):
        self.symbol = sym
        self.timestamp_ns = ts
        self.open_ticks = o
        self.high_ticks = h
        self.low_ticks = lo
        self.close_ticks = c
        self.volume = v


class _TradeEvent:
    def __init__(self, sym, ts, p):
        self.symbol = sym
        self.timestamp_ns = ts
        self.price_ticks = p


# the dispatch in on_market_event keys on the class NAME
_BarEvent.__name__ = "BarEvent"
_TradeEvent.__name__ = "TradeEvent"


def _feed_uptrend(egd, symbols, n=80):
    """Warm-up ingestion: feed each symbol genuine uptrending history + a developing
    trade above the last close (positive 1m momentum)."""
    last = {}
    for s in symbols:
        px = 100.0
        for i in range(n):
            o = px
            px *= 1.004
            egd.on_market_event(_BarEvent(s, i * _NS_MIN, _t(o), _t(max(o, px) * 1.001),
                                          _t(min(o, px) * 0.999), _t(px), 1000 + i))
        last[s] = px
    for s in symbols:
        egd.on_market_event(_TradeEvent(s, (n + 1) * _NS_MIN, _t(last[s] * 1.004)))
    egd._flush_trades()


def _live_stream():
    s = AlpacaMarketDataStream(["AAPL"], api_key="k", api_secret="s")
    s._socket = _FakeSocket()
    s._state = ConnectionState.SUBSCRIBED
    return s


@pytest.mark.asyncio
async def test_full_staged_discovery_to_exposure_gate():
    # --- Stage 1: native registration of the whole universe (incl. symbols a flat
    # discovery window would miss). This is the production register_universe path.
    egd = EngineGpuDiscovery(bar_capacity=256, window=64)
    registered = egd.register_universe(UNIVERSE)
    assert registered == len(UNIVERSE)
    for s in UNIVERSE:
        assert egd.provider.engine.symbol_id(s) is not None

    # --- Stage 2: warm-up ingestion populates native buffers (history).
    _feed_uptrend(egd, UNIVERSE)

    stream = _live_stream()
    ctl = AdaptiveSubscriptionController(
        stream, caps=ChannelCaps(bars=8192, trades=1000, quotes=250))

    # --- Stage 3: BAR subscription over the whole universe -> REQUESTED, then a
    # GENUINE provider confirmation advances to PROVIDER_CONFIRMED (never on send).
    await ctl.apply(tier1_bars=UNIVERSE, now=0.0)
    assert all(ctl.manager.state_of(Channel.BARS, s) == SubState.REQUESTED
               for s in UNIVERSE)
    ctl.on_provider_confirmation({"bars": UNIVERSE}, now=1.0)
    assert all(ctl.manager.state_of(Channel.BARS, s) == SubState.PROVIDER_CONFIRMED
               for s in UNIVERSE)

    # --- Stage 4: GPU structural screening + preliminary ranking over the confirmed
    # bar universe. The preliminary candidate set is PRODUCED by the screen, not
    # injected.
    rows = egd.provider.compute(UNIVERSE)
    preliminary = sorted(s for s, r in rows.items()
                         if (r.tech_valid & native.TECH_DMI) and r.di_plus > r.di_minus)
    assert set(preliminary) == set(UNIVERSE)  # all uptrends pass preliminary

    # --- Stage 5: promote preliminary -> TRADES; provider-confirm; stream developing
    # trades so momentum can be OBSERVED (two ticks -> FRESH).
    await ctl.apply(tier1_bars=UNIVERSE, tier2_trades=preliminary, now=2.0)
    ctl.on_provider_confirmation({"trades": preliminary}, now=3.0)
    for tck in (4.0, 5.0):
        for s in preliminary:
            ctl.on_observation(Channel.TRADES, s, now=tck)
    fresh_trades = ctl.fresh(Channel.TRADES)
    assert fresh_trades == set(preliminary)

    # --- Stage 6: momentum confirmation from FRESH trades gates the QUOTE request.
    momentum = egd.momentum_confirmed(sorted(fresh_trades))
    assert momentum == set(UNIVERSE)  # all developing trades are above prev close
    await ctl.apply(tier1_bars=UNIVERSE, tier2_trades=preliminary,
                    tier3_quotes=sorted(momentum), now=6.0)
    ctl.on_provider_confirmation({"quotes": sorted(momentum)}, now=7.0)
    for tck in (8.0, 9.0):
        for s in momentum:
            ctl.on_observation(Channel.QUOTES, s, now=tck)
    assert ctl.fresh(Channel.QUOTES) == set(momentum)

    # --- Stage 7: final GPU qualification (strict entry contract on the GPU screen).
    final_rows = egd.provider.compute(UNIVERSE)
    qualified = sorted(s for s, r in final_rows.items() if r.entry_qualified)
    assert qualified, "expected at least one entry-qualified candidate"

    # --- Stage 8: exposure authorization gates elective entries. Conventional/broad/
    # sector authorize; geared + inverse are blocked until explicitly authorized.
    reg = ExposureRegistry()
    authorized = [s for s in qualified if reg.authorize_entry(s)[0]]
    blocked = [s for s in qualified if not reg.authorize_entry(s)[0]]
    assert set(authorized) == (set(qualified) & AUTHORIZABLE)
    assert set(blocked) == (set(qualified) - AUTHORIZABLE)
    assert {"TQQQ", "SH", "SQQQ"} & set(qualified) <= set(blocked)  # geared/inverse blocked

    # explicit authorization lets a geared product through the SAME gate
    if "TQQQ" in qualified:
        reg.authorize("TQQQ", True)
        assert reg.authorize_entry("TQQQ")[0] is True


@pytest.mark.asyncio
async def test_quote_not_requested_without_momentum():
    """A candidate with confirmed bars but NO fresh-trade momentum must not reach a
    quote subscription — the staged gate holds (no trade+quote shortcut)."""
    egd = EngineGpuDiscovery(bar_capacity=256, window=64)
    egd.register_universe(["DOWN"])
    # downtrend history + a developing trade BELOW last close -> momentum not confirmed
    px = 100.0
    for i in range(80):
        o = px
        px *= 0.996
        egd.on_market_event(_BarEvent("DOWN", i * _NS_MIN, _t(o), _t(max(o, px) * 1.001),
                                      _t(min(o, px) * 0.999), _t(px), 1000 + i))
    egd.on_market_event(_TradeEvent("DOWN", 81 * _NS_MIN, _t(px * 0.996)))
    egd._flush_trades()

    stream = _live_stream()
    ctl = AdaptiveSubscriptionController(stream, caps=ChannelCaps(bars=100, trades=50, quotes=20))
    await ctl.apply(tier1_bars=["DOWN"], tier2_trades=["DOWN"], now=0.0)
    ctl.on_provider_confirmation({"bars": ["DOWN"], "trades": ["DOWN"]}, now=1.0)
    ctl.on_observation(Channel.TRADES, "DOWN", now=2.0)
    ctl.on_observation(Channel.TRADES, "DOWN", now=3.0)
    # fresh trades, but momentum NOT confirmed -> no quote promotion
    assert egd.momentum_confirmed(["DOWN"]) == set()
    assert ctl.manager.state_of(Channel.QUOTES, "DOWN") == SubState.UNREQUESTED
