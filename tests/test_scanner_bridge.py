"""Tests for the single-pipeline scanner bridge (Milestone 4).

The engine owns the one market-data socket; the scanner consumes the SAME
normalized events through a thread-safe bridge (no second websocket). The bridge
is bounded (drop-oldest) so a slow scanner never blocks the engine, and the
BridgeStreamClient converts engine events (ticks/ns) into the OFE's expected
shape (dollars/epoch-seconds).
"""

from __future__ import annotations

import asyncio

import pytest

from scanner_bridge import BridgeStreamClient, ScannerEventBridge


# --- fake engine events (duck-typed; mirror market_data.alpaca_stream shapes) --
class FakeTrade:
    def __init__(self, symbol, timestamp_ns, price_ticks, size):
        self.symbol = symbol
        self.timestamp_ns = timestamp_ns
        self.price_ticks = price_ticks
        self.size = size


class FakeQuote:
    def __init__(self, symbol, timestamp_ns, bid_ticks, ask_ticks, bsz, asz):
        self.symbol = symbol
        self.timestamp_ns = timestamp_ns
        self.bid_price_ticks = bid_ticks
        self.ask_price_ticks = ask_ticks
        self.bid_size = bsz
        self.ask_size = asz


class FakeBar:
    def __init__(self, symbol):
        self.symbol = symbol
        self.timestamp_ns = 1
        self.close_ticks = 100


def _run(coro):
    return asyncio.run(coro)


# ------------------------------------------------------------------- bridge
def test_bridge_offer_and_drain_fifo():
    b = ScannerEventBridge(maxlen=10)
    for x in ("a", "b", "c"):
        b.offer(x)
    assert b.drain() == ["a", "b", "c"]
    assert b.drain() == []  # drained


def test_bridge_drop_oldest_when_full():
    b = ScannerEventBridge(maxlen=3)
    for x in ["a", "b", "c", "d", "e"]:
        b.offer(x)
    # oldest two dropped; newest three retained in order
    assert b.drain() == ["c", "d", "e"]
    assert b.dropped == 2
    assert b.offered == 5


def test_bridge_drain_respects_max_items():
    b = ScannerEventBridge(maxlen=10)
    for x in range(5):
        b.offer(x)
    assert b.drain(max_items=2) == [0, 1]
    assert b.drain(max_items=2) == [2, 3]
    assert b.drain() == [4]


# ------------------------------------------------------- BridgeStreamClient
def _client(bridge, trades, quotes):
    c = BridgeStreamClient(bridge, price_scale=10000, poll_interval_s=0.0)
    c.bind(on_trade=lambda *a: trades.append(a),
           on_quote=lambda *a: quotes.append(a))
    return c


def test_client_forwards_trade_converted():
    b = ScannerEventBridge()
    trades, quotes = [], []
    c = _client(b, trades, quotes)
    b.offer(FakeTrade("NVDA", 2_000_000_000, 1_250_000, 7))  # 2.0s, $125.00, 7 sh
    c.drain_once()
    assert trades == [("NVDA", 2.0, 125.0, 7.0)]
    assert quotes == []


def test_client_forwards_quote_converted():
    b = ScannerEventBridge()
    trades, quotes = [], []
    c = _client(b, trades, quotes)
    b.offer(FakeQuote("AMD", 1_000_000_000, 1_000_000, 1_001_000, 3, 4))  # $100.00/$100.10
    c.drain_once()
    assert quotes == [("AMD", 1.0, 100.0, 100.1, 3.0, 4.0)]


def test_client_ignores_bars():
    b = ScannerEventBridge()
    trades, quotes = [], []
    c = _client(b, trades, quotes)
    b.offer(FakeBar("SPY"))
    c.drain_once()
    assert trades == [] and quotes == []


def test_client_is_live_tracks_last_message():
    b = ScannerEventBridge()
    trades, quotes = [], []
    c = BridgeStreamClient(b, price_scale=10000, poll_interval_s=0.0,
                           now_fn=lambda: 100.0)
    c.bind(on_trade=lambda *a: trades.append(a), on_quote=lambda *a: quotes.append(a))
    assert c.is_live(now=100.0) is False  # nothing delivered yet
    b.offer(FakeTrade("NVDA", 50_000_000_000, 1_000_000, 1))
    c.drain_once()
    assert c.is_live(now=100.0, max_staleness_s=20.0) is True
    assert c.is_live(now=200.0, max_staleness_s=20.0) is False  # stale


def test_apply_subscriptions_tracks_desired_symbols():
    b = ScannerEventBridge()
    c = BridgeStreamClient(b)
    _run(c.apply_subscriptions(to_add=["nvda", "amd"], to_remove=[]))
    assert c.desired_symbols == frozenset({"NVDA", "AMD"})
    _run(c.apply_subscriptions(to_add=["tsla"], to_remove=["amd"]))
    assert c.desired_symbols == frozenset({"NVDA", "TSLA"})


def test_run_drains_until_stopped():
    b = ScannerEventBridge()
    trades = []
    c = BridgeStreamClient(b, poll_interval_s=0.0)
    c.bind(on_trade=lambda *a: trades.append(a), on_quote=lambda *a: None)
    b.offer(FakeTrade("NVDA", 1_000_000_000, 1_000_000, 1))

    async def driver():
        task = asyncio.create_task(c.run())
        await asyncio.sleep(0.02)
        c.stop()
        await task

    _run(driver())
    assert trades and trades[0][0] == "NVDA"


def test_real_scanner_ofe_fed_through_bridge():
    """End-to-end wiring (as run.MarketEngine does it): a real MarketScanner with
    an injected BridgeStreamClient has its OrderFlowEngine fed from engine events
    offered to the bridge -- no socket involved."""
    from types import SimpleNamespace

    from market_scan.config import MarketScanConfig
    from market_scan.scanner import MarketScanner

    class _Broker:
        api_key = "k"
        secret_key = "s"
        def get_all_open_positions(self):
            return []
        def list_open_orders(self):
            return []

    host = SimpleNamespace(broker=_Broker(), config=SimpleNamespace(paper=True))

    bridge = ScannerEventBridge()
    client = BridgeStreamClient(bridge, poll_interval_s=0.0, now_fn=lambda: 1000.0)
    scanner = MarketScanner(host, MarketScanConfig.from_env(), stream_client=client)
    client.bind(on_trade=scanner.ofe.ingest_trade, on_quote=scanner.ofe.ingest_quote)

    ns = 1_000_000_000_000  # 1000.0s
    bridge.offer(FakeQuote("NVDA", ns, 1_000_000, 1_001_000, 5, 6))  # $100.00/$100.10
    bridge.offer(FakeTrade("NVDA", ns, 1_000_500, 10))              # $100.05 x10
    client.drain_once()

    snap = scanner.ofe.snapshot("NVDA", now=1000.0)
    assert snap is not None
    assert snap.mid == pytest.approx(100.05, abs=1e-6)  # quote reached OFE
    assert snap.last_trade_age_s is not None and snap.last_trade_age_s <= 1.0  # trade too


def test_bad_event_isolated():
    b = ScannerEventBridge()
    quotes = []

    def boom(*a):
        raise RuntimeError("handler boom")

    c = BridgeStreamClient(b, poll_interval_s=0.0)
    c.bind(on_trade=boom, on_quote=lambda *a: quotes.append(a))
    b.offer(FakeTrade("BAD", 1_000_000_000, 1_000_000, 1))
    b.offer(FakeQuote("AMD", 1_000_000_000, 1_000_000, 1_001_000, 1, 1))
    c.drain_once()  # must not raise despite the trade handler blowing up
    assert quotes == [("AMD", 1.0, 100.0, 100.1, 1.0, 1.0)]
