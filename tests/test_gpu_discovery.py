"""Phase B.5 — GPU-backed discovery feature provider (cengine.gpu_discovery).

Exercises the full native vertical slice end to end: bar ingestion into the
DiscoveryEngine -> technical_features (B.3) -> screen_market (B.1) -> scanner
CandidateFeatures. Synthetic bar inputs; genuine real-time IEX flow is certified
in B.6 and remains pending a live session. These tests run on whichever backend
is available (Metal on Apple Silicon, else the native CPU oracle) and separately
assert the two backends agree.
"""

from __future__ import annotations

import _cengine_native as native
import pytest

from cengine.gpu_discovery import GpuDiscoveryProvider

SCALE = 10_000


def _t(x: float) -> int:
    return int(round(x * SCALE))


def _feed_trend(prov, symbol, *, start, drift, n5=30, n1=3, last_trade=None):
    """Feed n5 5-minute bars (trending by `drift`), n1 1-minute bars, and a trade."""
    px = start
    bars5 = []
    for i in range(n5):
        o = px
        px = px * (1.0 + drift)
        hi = max(o, px) * 1.0015
        lo = min(o, px) * 0.9985
        bars5.append((symbol, i, _t(o), _t(hi), _t(lo), _t(px), 1000 + i))
    prov.ingest_bars(bars5, native.BarTimeframe.FIVE_MINUTE)
    # 1-minute completed bars (last one is the "previous_close_1m" reference)
    one = px
    bars1 = []
    for j in range(n1):
        o = one
        one = one * (1.0 + drift)
        hi = max(o, one) * 1.001
        lo = min(o, one) * 0.999
        bars1.append((symbol, 1000 + j, _t(o), _t(hi), _t(lo), _t(one), 100 + j))
    prov.ingest_bars(bars1, native.BarTimeframe.ONE_MINUTE)
    # latest developing trade price
    tp = last_trade if last_trade is not None else one * (1.0 + drift)
    prov.ingest_trades([(symbol, 2000, _t(tp), 10, 1)])
    return one, tp


@pytest.fixture
def provider():
    p = GpuDiscoveryProvider(bar_capacity=128, window=64)
    p.set_universe(["UP", "DOWN", "PARTIAL"])
    return p


def test_backend_reported():
    p = GpuDiscoveryProvider()
    assert p.backend in ("metal", "cpu")


def test_uptrend_is_entry_qualified(provider):
    _feed_trend(provider, "UP", start=100.0, drift=0.01)  # strong uptrend
    rows = provider.compute(["UP"])
    r = rows["UP"]
    assert r.di_plus > r.di_minus                 # bullish Wilder DMI
    assert r.price > r.previous_close_1m          # positive 1m momentum
    assert r.entry_qualified and not r.technical_exit
    assert not r.data_invalid


def test_downtrend_is_technical_exit(provider):
    # downtrend, last trade below the last completed 1m close
    one, _ = _feed_trend(provider, "DOWN", start=200.0, drift=-0.01,
                         last_trade=None)
    rows = provider.compute(["DOWN"])
    r = rows["DOWN"]
    assert r.di_minus > r.di_plus
    assert r.price < r.previous_close_1m
    assert r.technical_exit and not r.entry_qualified


def test_insufficient_history_is_data_invalid(provider):
    # Only 10 5m bars: DI (needs >=15) invalid -> screen marks DATA_INVALID.
    _feed_trend(provider, "PARTIAL", start=50.0, drift=0.01, n5=10)
    r = provider.compute(["PARTIAL"])["PARTIAL"]
    assert not (r.tech_valid & native.TECH_DMI)
    assert r.data_invalid and not r.entry_qualified


def test_candidate_features_mapping(provider):
    _feed_trend(provider, "UP", start=100.0, drift=0.01)
    feats = provider.candidate_features(
        ["UP"], now=0.0, daily_stats={"UP": {"price": 123.0, "adv": 1e6, "addv": 1e8}})
    cf = feats["UP"]
    assert cf.symbol == "UP"
    assert cf.price == 123.0 and cf.avg_daily_volume == 1e6  # merged from daily_stats
    dr = cf.directional_row
    assert dr["pdi"] is not None and dr["mdi"] is not None
    assert dr["pdi"] > dr["mdi"]                  # bullish
    assert cf.vwap_dist is not None
    # directional_score must accept the row and read it as bullish (> 0)
    import amps_v4_regime
    score = amps_v4_regime.directional_score(dr, amps_v4_regime.RegimeConfig())
    assert score == score and score > 0.0         # not NaN, bullish


def test_unknown_symbol_is_skipped(provider):
    rows = provider.compute(["NOTINUNIVERSE"])
    assert rows == {}
    feats = provider.candidate_features(["NOTINUNIVERSE"], now=0.0)
    assert feats["NOTINUNIVERSE"].directional_row == {}  # default/empty


def test_gpu_and_cpu_backends_agree():
    """When Metal is available, the GPU provider must agree with the CPU one."""
    gpu = GpuDiscoveryProvider(bar_capacity=128, window=64, prefer_metal=True)
    cpu = GpuDiscoveryProvider(bar_capacity=128, window=64, prefer_metal=False)
    if gpu.backend != "metal":
        pytest.skip("Metal unavailable; single-backend host")
    for prov in (gpu, cpu):
        prov.set_universe(["UP", "DOWN"])
        _feed_trend(prov, "UP", start=100.0, drift=0.008)
        _feed_trend(prov, "DOWN", start=180.0, drift=-0.008)
    g = gpu.compute(["UP", "DOWN"])
    c = cpu.compute(["UP", "DOWN"])
    for sym in ("UP", "DOWN"):
        assert g[sym].flags == c[sym].flags
        for attr in ("di_plus", "di_minus", "adx", "sma", "bb_middle", "vwap"):
            assert getattr(g[sym], attr) == pytest.approx(getattr(c[sym], attr), abs=1e-2)
