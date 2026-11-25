"""Integration: DynamicUniverseManager <-> scanner publisher authority split.

Verifies the confirmed design: the manager is the sole SELECTION authority (its
active window becomes universe.txt membership) while the existing UniversePublisher
stays the sole WRITER and the entry gate stays intact (entry_eligible_count keeps
reflecting the genuine entry-eligible ranking, not the broader active window).
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import universe
from market_scan.config import MarketScanConfig
from market_scan.health import read_live_context
from market_scan.publisher import UniversePublisher
from market_scan.ranking import CandidateFeatures
from market_scan.scanner import MarketScanner

BASE5 = ["AAA", "BBB", "CCC", "DDD", "EEE"]


# --------------------------------------------------------------------- publisher
def test_publisher_members_vs_ranked_separation(tmp_path):
    cfg = MarketScanConfig(target_universe_size=30)
    pub = UniversePublisher(tmp_path / "universe.txt", tmp_path / "live_context.json", cfg)
    # Active window (members) is broader than the entry-eligible ranking.
    members = BASE5 + ["NVDA", "AMD"]
    res = pub.publish(["NVDA", "AMD"], now=100.0, members=members, health={})
    written = universe.load_canonical_universe(tmp_path / "universe.txt")
    assert set(written) == {s.upper() for s in members}      # universe.txt == active window
    assert res.eligible_count == 2                           # gate sees only genuine eligibles
    ctx = read_live_context(tmp_path / "live_context.json")
    assert {s.upper() for s in ctx["universe"]} == set(written)   # consistency check holds
    assert ctx["entry_eligible_count"] == 2
    assert [s.upper() for s in ctx["ranked"]] == ["NVDA", "AMD"]


def test_publisher_members_none_is_legacy(tmp_path):
    cfg = MarketScanConfig(target_universe_size=30)
    pub = UniversePublisher(tmp_path / "universe.txt", tmp_path / "live_context.json", cfg)
    res = pub.publish(["NVDA", "AMD"], now=100.0, health={})   # members=None -> legacy
    written = universe.load_canonical_universe(tmp_path / "universe.txt")
    assert set(written) == {"NVDA", "AMD"}
    assert res.eligible_count == 2


# ----------------------------------------------------------------------- scanner
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


def _strong_feats(symbols):
    def provider(syms, now):
        out = {}
        for s in syms:
            out[s] = CandidateFeatures(
                symbol=s, price=50.0, avg_daily_volume=5_000_000,
                avg_daily_dollar_volume=250_000_000, rel_volume=1.5,
                spread_pct=0.0004, quote_age_s=1.0,
                directional_row={"pdi": 30, "mdi": 10, "adx": 25,
                                 "vwap_dist": 0.01, "short_mom": 0.02, "bb_dist": 0.01},
                snapshot=SimpleNamespace(ok=True, buy_pressure=0.6, seller_exhaustion=0.4,
                                         seller_exhaustion_l1=0.4, cvd_trend=0.3,
                                         last_trade_age_s=1.0, last_quote_age_s=1.0),
                vwap_dist=0.01, bb_mid_dist=0.01,
            )
        return out
    return provider


def test_rotation_publishes_manager_active_window(tmp_path, monkeypatch):
    # Baseline file (5 synthetic) + rotation enabled + isolated state.
    bp = tmp_path / "baseline_universe.txt"
    bp.write_text(", ".join(f"'{s}'" for s in BASE5) + "\n", encoding="utf-8")
    monkeypatch.setenv("CENGINE_DYNAMIC_UNIVERSE_ROTATION", "1")
    monkeypatch.setenv("CENGINE_BASELINE_UNIVERSE_PATH", str(bp))
    monkeypatch.setenv("CENGINE_DYNAMIC_UNIVERSE_STATE", str(tmp_path / "state.json"))

    disc = ["NVDA", "AMD"]
    cfg = MarketScanConfig(max_subscriptions=30, target_universe_size=10)
    exec_stub = SimpleNamespace(broker=_FakeBroker(), config=SimpleNamespace(paper=True))
    s = MarketScanner(
        exec_stub, cfg,
        universe_path=tmp_path / "universe.txt",
        context_path=tmp_path / "live_context.json",
        stream_client=_FakeStream(),
        feature_provider=_strong_feats(disc),
        discovery_fn=lambda **k: SimpleNamespace(symbols=disc, equities=[], etfs=[],
                                                 error=None, source_counts={}),
    )
    assert s._universe_manager is not None          # rotation wired on
    s._discovered = disc
    asyncio.run(s._rank_tick(1000.0))

    published = set(universe.load_canonical_universe(tmp_path / "universe.txt"))
    # Active window (cap 10) = refs(4) + opportunities(2) + baseline fill(4).
    assert "NVDA" in published and "AMD" in published
    assert len(published & set(BASE5)) >= 1          # baseline permanence in the window
    assert {"SPY", "QQQ", "DIA", "IWM"} <= published  # regime refs subscribed
    assert len(published) <= 10

    ctx = read_live_context(tmp_path / "live_context.json")
    # Entry gate still reflects only the genuine entry-eligible opportunities
    # (proxies/baseline are NOT counted as entry-eligible).
    assert ctx["entry_eligible_count"] == 2
    # Consistency invariant the fail-closed gate relies on.
    assert {x.upper() for x in ctx["universe"]} == published


def test_stalled_manager_surfaces_error_and_falls_back(tmp_path):
    # A manager that raises on rotation must NOT masquerade as healthy: the scanner
    # records the error (-> scanner_status unhealthy -> entry gate blocks) and
    # falls back to its own entry-eligible top-N rather than a stale window.
    disc = ["NVDA", "AMD"]
    cfg = MarketScanConfig(max_subscriptions=30, target_universe_size=10)
    exec_stub = SimpleNamespace(broker=_FakeBroker(), config=SimpleNamespace(paper=True))
    s = MarketScanner(
        exec_stub, cfg,
        universe_path=tmp_path / "universe.txt",
        context_path=tmp_path / "live_context.json",
        stream_client=_FakeStream(),
        feature_provider=_strong_feats(disc),
        discovery_fn=lambda **k: SimpleNamespace(symbols=disc, equities=[], etfs=[],
                                                 error=None, source_counts={}),
    )

    class _StalledManager:
        def maybe_rotate(self, now):
            raise RuntimeError("rotation stalled")

    s._universe_manager = _StalledManager()
    s._discovered = disc
    asyncio.run(s._rank_tick(1000.0))

    assert any("universe-rotation" in e for e in s._errors)   # surfaced, not hidden
    published = set(universe.load_canonical_universe(tmp_path / "universe.txt"))
    assert published == {"NVDA", "AMD"}                       # safe fallback, no stale window


def test_rotation_emits_decision_events_to_sink(tmp_path, monkeypatch):
    from cengine.decision_events import DecisionEventSink, DecisionEventType

    bp = tmp_path / "baseline_universe.txt"
    bp.write_text(", ".join(f"'{s}'" for s in BASE5) + "\n", encoding="utf-8")
    monkeypatch.setenv("CENGINE_DYNAMIC_UNIVERSE_ROTATION", "1")
    monkeypatch.setenv("CENGINE_BASELINE_UNIVERSE_PATH", str(bp))
    monkeypatch.setenv("CENGINE_DYNAMIC_UNIVERSE_STATE", str(tmp_path / "state.json"))

    disc = ["NVDA", "AMD"]
    sink = DecisionEventSink()
    cfg = MarketScanConfig(max_subscriptions=30, target_universe_size=10)
    exec_stub = SimpleNamespace(broker=_FakeBroker(), config=SimpleNamespace(paper=True))
    s = MarketScanner(
        exec_stub, cfg,
        universe_path=tmp_path / "universe.txt",
        context_path=tmp_path / "live_context.json",
        stream_client=_FakeStream(),
        feature_provider=_strong_feats(disc),
        discovery_fn=lambda **k: SimpleNamespace(symbols=disc, equities=[], etfs=[],
                                                 error=None, source_counts={}),
        decision_sink=sink,
    )
    assert s._universe_manager is not None
    s._discovered = disc
    asyncio.run(s._rank_tick(1000.0))
    kinds = {r.event_type for r in sink.peek()}
    assert DecisionEventType.UNIVERSE_ROTATION in kinds       # persisted, not fabricated


def test_rotation_disabled_by_default(tmp_path):
    # No env flag -> legacy behavior, no manager, universe.txt = eligible only.
    disc = ["NVDA", "AMD"]
    cfg = MarketScanConfig(max_subscriptions=30, target_universe_size=5)
    exec_stub = SimpleNamespace(broker=_FakeBroker(), config=SimpleNamespace(paper=True))
    s = MarketScanner(
        exec_stub, cfg,
        universe_path=tmp_path / "universe.txt",
        context_path=tmp_path / "live_context.json",
        stream_client=_FakeStream(),
        feature_provider=_strong_feats(disc),
        discovery_fn=lambda **k: SimpleNamespace(symbols=disc, equities=[], etfs=[],
                                                 error=None, source_counts={}),
    )
    assert s._universe_manager is None
    s._discovered = disc
    asyncio.run(s._rank_tick(1000.0))
    published = set(universe.load_canonical_universe(tmp_path / "universe.txt"))
    assert published == {"NVDA", "AMD"}             # no baseline injected
