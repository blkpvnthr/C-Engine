"""Regime→strategy→arbitration→evidence pipeline over REAL native GPU screen rows
(milestone integration). No network; uses the native DiscoveryEngine + Metal/CPU."""

from __future__ import annotations

from cengine.gpu_scanner_bridge import EngineGpuDiscovery
from cengine.regime_strategy_pipeline import (
    RegimeStrategyPipeline,
    screen_row_to_features,
)
from cengine.strategy_families import EligibilityState, StrategyFamily

SCALE = 10_000
_NS_MIN = 60 * 1_000_000_000


def _t(x):
    return int(round(x * SCALE))


class _Bar:
    def __init__(self, sym, ts, o, h, lo, c, v):
        self.symbol = sym
        self.timestamp_ns = ts
        self.open_ticks = o
        self.high_ticks = h
        self.low_ticks = lo
        self.close_ticks = c
        self.volume = v


class _Trade:
    def __init__(self, sym, ts, p):
        self.symbol = sym
        self.timestamp_ns = ts
        self.price_ticks = p


_Bar.__name__ = "BarEvent"
_Trade.__name__ = "TradeEvent"


def _feed(egd, syms, n=80, up=True):
    for s in syms:
        px = 100.0
        for i in range(n):
            o = px
            px = px * 1.004 if up else px * 0.996
            egd.on_market_event(_Bar(s, i * _NS_MIN, _t(o), _t(max(o, px) * 1.001),
                                     _t(min(o, px) * 0.999), _t(px), 1000 + i))
        egd.on_market_event(_Trade(s, (n + 1) * _NS_MIN,
                                   _t(px * (1.004 if up else 0.996))))
    egd._flush_trades()


def test_pipeline_over_native_screen_rows_research_by_default():
    egd = EngineGpuDiscovery(bar_capacity=256, window=64)
    egd.register_universe(["AAA", "BBB"])
    _feed(egd, ["AAA", "BBB"], up=True)
    rows = egd.provider.compute(["AAA", "BBB"])

    pipe = RegimeStrategyPipeline()
    out = pipe.run(rows, now_ns=100 * _NS_MIN,
                   ref_features={"SPY": screen_row_to_features(rows["AAA"])})
    # recommendations produced, decision events emitted
    assert out["recommendations"]
    assert pipe.sink.emitted > 0
    # nothing executed: every family is RESEARCH by default -> empty selection
    assert out["arbitration"]["selected"] == []
    assert out["arbitration"]["research"]
    assert pipe.telemetry()["execution_eligible_families"] == []


def test_pipeline_selects_only_after_explicit_validation():
    egd = EngineGpuDiscovery(bar_capacity=256, window=64)
    egd.register_universe(["AAA"])
    _feed(egd, ["AAA"], up=True)
    rows = egd.provider.compute(["AAA"])
    feats = {s: screen_row_to_features(r) for s, r in rows.items()}

    pipe = RegimeStrategyPipeline()
    pipe.registry.mark_execution_eligible(StrategyFamily.MOMENTUM_CONTINUATION,
                                          validated=True)
    out = pipe.run(feats, now_ns=100 * _NS_MIN, ref_features={"SPY": feats["AAA"]})
    fams = {r.strategy_family for r in out["arbitration"]["selected"]}
    # if momentum conditions pass on the uptrend, it is now executable; others remain research
    if out["arbitration"]["selected"]:
        assert StrategyFamily.MOMENTUM_CONTINUATION in fams
        for r in out["arbitration"]["selected"]:
            assert r.eligibility == EligibilityState.EXECUTION


def test_screen_row_adapter_populates_contract_fields():
    egd = EngineGpuDiscovery(bar_capacity=256, window=64)
    egd.register_universe(["AAA"])
    _feed(egd, ["AAA"], up=True)
    row = egd.provider.compute(["AAA"])["AAA"]
    feats = screen_row_to_features(row)
    for k in ("di_plus_5m", "di_minus_5m", "price", "vwap", "sma20_5m",
              "bb_middle_5m", "developing_1m_price", "previous_completed_1m_close"):
        assert k in feats
