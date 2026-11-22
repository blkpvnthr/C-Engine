"""Automatic historical warm-up orchestration (directive items 4/7). Mock fetcher
+ real GpuDiscoveryProvider; no network."""

from __future__ import annotations

from cengine.gpu_discovery import GpuDiscoveryProvider
from cengine.historical_warmup import HistoricalWarmup, WarmupStatus, prioritize


def _bars(n, start=100.0):
    out = []
    px = start
    for i in range(n):
        o = px
        px *= 1.004
        out.append({"t": f"2026-11-{(i % 27) + 1:02d}T15:00:00Z",
                    "o": o, "h": max(o, px) * 1.001, "l": min(o, px) * 0.999,
                    "c": px, "v": 1000 + i})
    return out


def _fetch_factory(depth):
    """depth: {symbol: n_bars}. Returns a fetch_fn giving that many bars per symbol."""
    def fetch(symbols, timeframe, start, limit=10000):
        n5 = 40 if timeframe == "5Min" else 6
        return {s: _bars(min(depth.get(s, 0), n5) if timeframe == "5Min"
                         else min(depth.get(s, 0), 6)) for s in symbols}
    return fetch


def test_prioritize_order_and_dedup():
    order = prioritize(positions=["HELD"], pending=["PEND"], references=["SPY"],
                       actionable=["ACT", "HELD"], preliminary=["PRE"],
                       remaining=["R1", "SPY"])
    assert order == ["HELD", "PEND", "SPY", "ACT", "PRE", "R1"]  # priority + dedup


def test_warm_ingests_and_marks_status():
    prov = GpuDiscoveryProvider(bar_capacity=128, window=64)
    prov.set_universe(["AAA", "BBB", "CCC"])
    # AAA full history, BBB partial (few bars -> incomplete), CCC none -> failed
    wu = HistoricalWarmup(prov, _fetch_factory({"AAA": 40, "BBB": 1, "CCC": 0}),
                          batch_size=10, min_5m_bars=30, min_1m_bars=2)
    cov = wu.warm(["AAA", "BBB", "CCC"], start_5m="2026-11-01", start_1m="2026-11-18")
    assert wu.status("AAA") == WarmupStatus.WARMED
    assert wu.status("BBB") == WarmupStatus.INCOMPLETE
    assert wu.status("CCC") == WarmupStatus.FAILED
    assert cov["warmed"] == 1 and cov["incomplete"] == 1 and cov["failed"] == 1
    assert cov["bars_ingested"]["5m"] > 0
    # bars actually reached the native DiscoveryEngine buffers
    aid = prov.engine.symbol_id("AAA")
    assert prov.engine.snapshot(aid).bars_5m >= 30


def test_warm_handles_fetch_error_per_batch():
    prov = GpuDiscoveryProvider(bar_capacity=64, window=32)
    prov.set_universe(["AAA"])

    def broken(symbols, timeframe, start, limit=10000):
        return {"__error__": "HTTP 429"}

    wu = HistoricalWarmup(prov, broken, batch_size=10)
    cov = wu.warm(["AAA"], start_5m="2026-11-01", start_1m="2026-11-18")
    assert wu.status("AAA") == WarmupStatus.FAILED
    assert cov["failed"] == 1 and cov["warmed"] == 0


def test_warm_batches_large_sets():
    prov = GpuDiscoveryProvider(bar_capacity=128, window=64)
    syms = [f"S{i:03d}" for i in range(25)]
    prov.set_universe(syms)
    wu = HistoricalWarmup(prov, _fetch_factory({s: 40 for s in syms}),
                          batch_size=10, min_5m_bars=30)
    cov = wu.warm(syms, start_5m="2026-11-01", start_1m="2026-11-18")
    assert cov["requested"] == 25 and cov["warmed"] == 25


def test_warm_runs_independently_fail_safe():
    # A fetcher that raises must not abort the whole warm-up; symbols -> FAILED.
    prov = GpuDiscoveryProvider(bar_capacity=32, window=16)
    prov.set_universe(["AAA", "BBB"])

    def raiser(symbols, timeframe, start, limit=10000):
        raise RuntimeError("network down")

    wu = HistoricalWarmup(prov, raiser, batch_size=1)
    cov = wu.warm(["AAA", "BBB"], start_5m="2026-11-01", start_1m="2026-11-18")
    assert cov["failed"] == 2  # both batches failed safely, no exception raised
