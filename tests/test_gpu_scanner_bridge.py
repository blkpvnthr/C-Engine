"""Phase B.5b — GPU discovery wired into the real MarketScanner.

Proves the native GPU discovery features are OPERATIONALLY CONSUMED by the actual
scanner rank/publish path (no competing scanner/socket/publisher), using injected
fakes for the broker/stream and synthetic bars. Live-session end-to-end
certification remains pending (B.6).
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import _cengine_native as native

from cengine.gpu_discovery import GpuDiscoveryProvider
from cengine.gpu_scanner_bridge import (
    BarAggregator,
    EngineGpuDiscovery,
    make_composed_feature_provider,
)
from market_scan.config import MarketScanConfig
from market_scan.scanner import MarketScanner

SCALE = 10_000
_NS_MIN = 60 * 1_000_000_000


# Lightweight stand-ins whose type names match the dispatch in on_market_event.
class BarEvent:
    def __init__(self, symbol, timestamp_ns, open_ticks, high_ticks, low_ticks,
                 close_ticks, volume):
        self.symbol = symbol
        self.timestamp_ns = timestamp_ns
        self.open_ticks = open_ticks
        self.high_ticks = high_ticks
        self.low_ticks = low_ticks
        self.close_ticks = close_ticks
        self.volume = volume


class TradeEvent:
    def __init__(self, symbol, timestamp_ns, price_ticks):
        self.symbol = symbol
        self.timestamp_ns = timestamp_ns
        self.price_ticks = price_ticks


def _t(x: float) -> int:
    return int(round(x * SCALE))


# ------------------------------------------------------------------ aggregator
def test_bar_aggregator_folds_1m_to_5m():
    agg = BarAggregator(minutes=5)
    span = 60 * 1_000_000_000
    # five 1m bars in bucket 0, then one in bucket 1 -> emits the bucket-0 5m bar
    out = None
    prices = [100, 102, 101, 103, 99]
    for i, p in enumerate(prices):
        out = agg.add_1m("X", i * span, _t(p), _t(p + 1), _t(p - 1), _t(p), 10)
        assert out is None  # still inside bucket 0
    rolled = agg.add_1m("X", 5 * span, _t(105), _t(106), _t(104), _t(105), 10)
    assert rolled is not None
    sym, ts, o, h, lo, c, v = rolled
    assert sym == "X"
    assert o == _t(100)          # first open
    assert h == _t(103 + 1)      # max high across the 5 bars
    assert lo == _t(99 - 1)      # min low
    assert c == _t(99)           # last close in bucket 0
    assert v == 50               # summed volume


# ------------------------------------------------------------------ fakes
class _FakeBroker:
    api_key = "k"
    secret_key = "s"

    def get_all_open_positions(self):
        return []

    def list_open_orders(self):
        return []

    def get_clock_info(self):
        return {"is_open": True}


class _FakeStream:
    def __init__(self):
        self._subscribed = set()

    def connect(self):
        pass

    async def apply_subscriptions(self, *, to_add, to_remove):
        self._subscribed.update(to_add)
        self._subscribed.difference_update(to_remove)

    @property
    def subscribed(self):
        return set(self._subscribed)

    async def run(self, *a, **k):
        return

    def stop(self):
        pass


def _feed_uptrend(gpu: GpuDiscoveryProvider, symbol: str, start=100.0, drift=0.01):
    px = start
    bars5 = []
    for i in range(30):
        o = px
        px = px * (1.0 + drift)
        bars5.append((symbol, i, _t(o), _t(max(o, px) * 1.0015),
                      _t(min(o, px) * 0.9985), _t(px), 1000 + i))
    gpu.ingest_bars(bars5, native.BarTimeframe.FIVE_MINUTE)
    one = px
    bars1 = []
    for j in range(3):
        o = one
        one = one * (1.0 + drift)
        bars1.append((symbol, 1000 + j, _t(o), _t(max(o, one) * 1.001),
                      _t(min(o, one) * 0.999), _t(one), 100 + j))
    gpu.ingest_bars(bars1, native.BarTimeframe.ONE_MINUTE)
    gpu.ingest_trades([(symbol, 2000, _t(one * (1.0 + drift)), 10, 1)])


def _feed_ofe(scanner, symbol, now, price=50.0):
    """Feed a fresh quote+trade into the scanner's order-flow engine so the
    candidate passes the scanner's own freshness gate (mirrors the live stream)."""
    scanner.ofe.ingest_quote(symbol, now, price - 0.01, price + 0.01, 100, 100)
    scanner.ofe.ingest_trade(symbol, now, price, 10)


def _build_scanner(tmp_path, provider):
    cfg = MarketScanConfig(max_subscriptions=30, target_universe_size=5)
    exec_stub = SimpleNamespace(broker=_FakeBroker(),
                                config=SimpleNamespace(paper=True))
    return MarketScanner(
        exec_stub, cfg,
        universe_path=tmp_path / "universe.txt",
        context_path=tmp_path / "live_context.json",
        stream_client=_FakeStream(),
        feature_provider=provider,
        discovery_fn=lambda **k: SimpleNamespace(
            symbols=["WIN"], equities=[], etfs=[], error=None, source_counts={}),
    )


def test_scanner_consumes_gpu_directional_features(tmp_path):
    gpu = GpuDiscoveryProvider(bar_capacity=128, window=64)
    gpu.set_universe(["WIN"])
    _feed_uptrend(gpu, "WIN")

    scanner = _build_scanner(tmp_path, provider=lambda s, n: {})
    # Daily liquidity stats the hard gate needs (the scanner's REST cache); the
    # GPU overlay supplies only the directional features.
    scanner._daily_stats = {"WIN": {"price": 50.0, "adv": 5_000_000,
                                    "addv": 250_000_000, "rel_volume": 1.4}}
    scanner._discovered = ["WIN"]
    _feed_ofe(scanner, "WIN", 1000.0)  # fresh live quote/trade (scanner freshness gate)
    # Compose: base features (incl. daily stats) + GPU directional overlay.
    scanner.feature_provider = make_composed_feature_provider(scanner, gpu)

    # Direct consumption evidence: the overlay produced a bullish directional_row.
    feats = scanner.feature_provider(["WIN"], 1000.0)
    dr = feats["WIN"].directional_row
    assert dr and dr["pdi"] > dr["mdi"]  # GPU DI+ dominant flowed into the scanner
    assert feats["WIN"].vwap_dist is not None

    # End-to-end: the real rank/publish tick consumes it and publishes WIN.
    asyncio.run(scanner._rank_tick(1000.0))
    published = (tmp_path / "universe.txt").read_text()
    assert "WIN" in published  # published via the real UniversePublisher


def test_engine_gpu_discovery_tap_and_provider(tmp_path):
    """The single on_market_event tap + feature_provider_for drive the real
    scanner: streamed 1m bars aggregate to 5m, a trade sets the developing price,
    and the GPU directional features reach the published universe."""
    egd = EngineGpuDiscovery(bar_capacity=256, window=64)
    # 80 uptrending 1m bars -> 16 completed 5m bars (enough for DI).
    px = 100.0
    for i in range(80):
        o = px
        px *= 1.004
        egd.on_market_event(BarEvent("WIN", i * _NS_MIN, _t(o),
                                     _t(max(o, px) * 1.001), _t(min(o, px) * 0.999),
                                     _t(px), 1000 + i))
    # developing trade above the last completed 1m close (positive 1m momentum)
    egd.on_market_event(TradeEvent("WIN", 81 * _NS_MIN, _t(px * 1.004)))

    scanner = _build_scanner(tmp_path, provider=lambda s, n: {})
    scanner._daily_stats = {"WIN": {"price": 50.0, "adv": 5_000_000,
                                    "addv": 250_000_000, "rel_volume": 1.4}}
    scanner._discovered = ["WIN"]
    _feed_ofe(scanner, "WIN", 1000.0)
    scanner.feature_provider = egd.feature_provider_for(scanner)

    feats = scanner.feature_provider(["WIN"], 1000.0)
    dr = feats["WIN"].directional_row
    assert dr and dr["pdi"] > dr["mdi"]   # GPU DI+ dominant reached the scanner
    assert egd.backend in ("metal", "cpu")

    asyncio.run(scanner._rank_tick(1000.0))
    assert "WIN" in (tmp_path / "universe.txt").read_text()


def test_register_universe_full_scale_up_front():
    """Directive item 3: the whole master universe is registered into the native
    DiscoveryEngine up front, so GPU kernels see every member before any tick —
    not just the subset that streamed. Registration grows-only and is idempotent."""
    egd = EngineGpuDiscovery(bar_capacity=64, window=32)
    assert egd.universe_size == 0
    master = [f"SYM{i:04d}" for i in range(500)]
    n = egd.register_universe(master)
    assert n == 500 and egd.universe_size == 500
    # every registered symbol has a real native symbol id (full-scale registration)
    assert egd.provider.engine.symbol_id("SYM0000") is not None
    assert egd.provider.engine.symbol_id("SYM0499") is not None
    # idempotent + grows-only: re-registering the same set adds nothing...
    assert egd.register_universe(master) == 0 and egd.universe_size == 500
    # ...and a lazily-streamed new symbol still extends the universe, never shrinks.
    egd.ensure_symbols(["LATECOMER"])
    assert egd.universe_size == 501
    assert egd.provider.engine.symbol_id("SYM0000") is not None  # not dropped


def test_register_universe_normalizes_and_skips_blanks():
    egd = EngineGpuDiscovery(bar_capacity=32, window=16)
    n = egd.register_universe([" aapl ", "MSFT", "", "   ", "aapl"])
    assert n == 2 and egd.universe_size == 2  # upper+strip+dedup, blanks dropped
    assert egd.provider.engine.symbol_id("AAPL") is not None
    assert egd.provider.engine.symbol_id("MSFT") is not None


def test_market_breadth_operationally_consumed(tmp_path):
    """The engine glue computes + stores market breadth each rank tick (the
    operational consumer of the market_breadth kernel), not just a standalone test."""
    egd = EngineGpuDiscovery(bar_capacity=256, window=64)
    px = 100.0
    for i in range(80):  # 80 1m bars -> 16 completed 5m bars (DI valid)
        o = px
        px *= 1.004
        egd.on_market_event(BarEvent("WIN", i * _NS_MIN, _t(o),
                                     _t(max(o, px) * 1.001), _t(min(o, px) * 0.999),
                                     _t(px), 1000 + i))
    egd.on_market_event(TradeEvent("WIN", 81 * _NS_MIN, _t(px * 1.004)))

    scanner = _build_scanner(tmp_path, provider=lambda s, n: {})
    scanner._daily_stats = {"WIN": {"price": 50.0, "adv": 5_000_000,
                                    "addv": 250_000_000}}
    scanner._discovered = ["WIN"]
    _feed_ofe(scanner, "WIN", 1000.0)
    scanner.feature_provider = egd.feature_provider_for(scanner)

    assert egd.latest_breadth is None  # nothing consumed yet
    scanner.feature_provider(["WIN"], 1000.0)  # drives update_breadth()
    b = egd.latest_breadth
    assert b is not None and b.total >= 1
    assert b.entry_qualified >= 1          # the uptrend symbol qualifies
    assert b.valid + b.data_invalid == b.total
    snap = egd.breadth_snapshot()
    assert snap["available"] and snap["backend"] in ("metal", "cpu")
    assert snap["entry_qualified"] == b.entry_qualified


def test_gpu_rank_matches_python_rank():
    """The GPU rank_fn must produce RankedCandidate output identical to the Python
    ranking.rank it replaces (scores within 1e-4, eligibility + order identical)."""
    import random

    from cengine.gpu_discovery import GpuDiscoveryProvider
    from cengine.gpu_scanner_bridge import make_gpu_rank
    from market_scan import ranking as rk
    from market_scan.config import MarketScanConfig

    prov = GpuDiscoveryProvider()
    gpu_rank = make_gpu_rank(prov)
    cfg = MarketScanConfig()
    mkt = rk.MarketContext(regime_label="BULL", regime_score=40.0, spy_score=25.0,
                           qqq_score=30.0, cfg=None)
    random.seed(11)
    feats = []
    for i in range(25):
        snap = (SimpleNamespace(ok=True, buy_pressure=random.random(),
                                cvd_trend=random.random(),
                                seller_exhaustion=random.random(),
                                seller_exhaustion_l1=random.random())
                if i % 3 else None)
        feats.append(rk.CandidateFeatures(
            symbol=f"S{i}", price=random.uniform(6, 500),
            avg_daily_volume=random.uniform(4e5, 1e7),
            avg_daily_dollar_volume=random.uniform(2e7, 5e8),
            rel_volume=random.uniform(0.3, 3), spread_pct=random.uniform(1e-4, 2e-3),
            directional_row={"pdi": random.uniform(10, 40),
                             "mdi": random.uniform(10, 40),
                             "adx": random.uniform(5, 40),
                             "vwap_dist": random.uniform(-.03, .03),
                             "bb_dist": random.uniform(-.03, .03),
                             "short_mom": random.uniform(-.02, .02)},
            snapshot=snap, vwap_dist=random.uniform(-.03, .03),
            bb_mid_dist=random.uniform(-.03, .03)))
    py = rk.rank(feats, config=cfg, market=mkt)
    gp = gpu_rank(feats, config=cfg, market=mkt)
    assert [r.symbol for r in py] == [r.symbol for r in gp]  # identical order
    pyd = {r.symbol: r for r in py}
    for r in gp:
        assert r.eligible == pyd[r.symbol].eligible
        assert abs(r.score - pyd[r.symbol].score) < 1e-4


def test_scanner_uses_gpu_rank_and_consumes_compaction(tmp_path):
    egd = EngineGpuDiscovery(bar_capacity=256, window=64)
    px = 100.0
    for i in range(80):
        o = px
        px *= 1.004
        egd.on_market_event(BarEvent("WIN", i * _NS_MIN, _t(o),
                                     _t(max(o, px) * 1.001), _t(min(o, px) * 0.999),
                                     _t(px), 1000 + i))
    egd.on_market_event(TradeEvent("WIN", 81 * _NS_MIN, _t(px * 1.004)))

    scanner = _build_scanner(tmp_path, provider=lambda s, n: {})
    scanner._daily_stats = {"WIN": {"price": 50.0, "adv": 5_000_000,
                                    "addv": 250_000_000}}
    scanner._discovered = ["WIN"]
    _feed_ofe(scanner, "WIN", 1000.0)
    scanner.feature_provider = egd.feature_provider_for(scanner)
    scanner.rank_fn = egd.rank_fn()  # GPU-backed ranking

    asyncio.run(scanner._rank_tick(1000.0))
    assert "WIN" in (tmp_path / "universe.txt").read_text()  # ranked+published via GPU
    # compaction consumed during the rank tick (update_breadth)
    ids, total = egd.latest_compaction
    assert total >= 1  # the uptrend symbol is entry-qualified


def test_composed_provider_is_failsafe(tmp_path):
    # A GPU provider with no data must not break the scanner; base features pass
    # through and WIN still qualifies on daily stats alone.
    gpu = GpuDiscoveryProvider(bar_capacity=16, window=16)
    gpu.set_universe(["WIN"])  # no bars fed
    scanner = _build_scanner(tmp_path, provider=lambda s, n: {})
    scanner._daily_stats = {"WIN": {"price": 50.0, "adv": 5_000_000,
                                    "addv": 250_000_000}}
    scanner._discovered = ["WIN"]
    _feed_ofe(scanner, "WIN", 1000.0)
    scanner.feature_provider = make_composed_feature_provider(scanner, gpu)
    feats = scanner.feature_provider(["WIN"], 1000.0)
    # No GPU history -> directional_row left as base (empty), no crash.
    assert "WIN" in feats
    asyncio.run(scanner._rank_tick(1000.0))
    assert "WIN" in (tmp_path / "universe.txt").read_text()
