"""Tests for the MarketScanner orchestrator (invariants, with injected fakes).

No broker/socket/network. Pins the correctness properties that matter for
production safety:
  * monitoring union always = positions ∪ pending ∪ market refs
  * published entry universe NEVER contains context-only proxies (SPY/QQQ/DIA/IWM)
  * open positions are always in the subscription plan (priority), even over cap
  * zero eligible candidates -> eligible_count 0 (blocks entries) but monitoring continues
  * live_context carries regime + subscription + status telemetry
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import universe
from market_scan.config import MarketScanConfig
from market_scan.health import read_live_context
from market_scan.ranking import CandidateFeatures
from market_scan.scanner import MarketScanner


class _FakeBroker:
    def __init__(self, positions=(), pending=()):
        self._pos = list(positions)
        self._pending = list(pending)
        self.api_key = "k"
        self.secret_key = "s"

    def get_all_open_positions(self):
        return [SimpleNamespace(symbol=s) for s in self._pos]

    def list_open_orders(self):
        return [SimpleNamespace(symbol=s) for s in self._pending]

    def get_clock_info(self):
        return {"is_open": True}


class _FakeStream:
    def __init__(self):
        self.added, self.removed = [], []
        self._subscribed = set()

    def connect(self):
        pass

    async def apply_subscriptions(self, *, to_add, to_remove):
        self.added += to_add
        self.removed += to_remove
        self._subscribed.update(to_add)
        self._subscribed.difference_update(to_remove)

    @property
    def subscribed(self):
        return set(self._subscribed)

    async def run(self, *a, **k):
        return

    def stop(self):
        pass


def _ok_snap(buy=0.6):
    return SimpleNamespace(ok=True, buy_pressure=buy, seller_exhaustion=0.4,
                           seller_exhaustion_l1=0.4, cvd_trend=0.3,
                           last_trade_age_s=1.0, last_quote_age_s=1.0)


def _scanner(tmp_path, *, positions=(), pending=(), discovered=None, provider=None,
             max_subs=30):
    cfg = MarketScanConfig(max_subscriptions=max_subs, target_universe_size=5)
    exec_stub = SimpleNamespace(broker=_FakeBroker(positions, pending),
                               config=SimpleNamespace(paper=True))
    s = MarketScanner(
        exec_stub, cfg,
        universe_path=tmp_path / "universe.txt",
        context_path=tmp_path / "live_context.json",
        stream_client=_FakeStream(),
        feature_provider=provider or (lambda syms, now: {}),
        discovery_fn=lambda **k: SimpleNamespace(
            symbols=list(discovered or []), equities=[], etfs=[], error=None,
            source_counts={}),
    )
    return s


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
                snapshot=_ok_snap(), vwap_dist=0.01, bb_mid_dist=0.01,
            )
        return out
    return provider


def test_monitoring_union_includes_positions_pending_refs(tmp_path):
    s = _scanner(tmp_path, positions=["HELD1"], pending=["PEND1"])
    mon = s._monitoring_union(["HELD1"], ["PEND1"])
    assert {"HELD1", "PEND1", "SPY", "QQQ", "DIA", "IWM"} <= mon


def test_rank_tick_excludes_proxies_from_entry_universe(tmp_path):
    disc = ["NVDA", "AMD", "SPY", "QQQ", "GPN"]  # SPY/QQQ are proxies
    s = _scanner(tmp_path, discovered=disc, provider=_strong_feats(disc))
    s._discovered = disc
    asyncio.run(s._rank_tick(1000.0))
    published = universe.load_canonical_universe(tmp_path / "universe.txt")
    assert "SPY" not in published and "QQQ" not in published
    assert "NVDA" in published and "AMD" in published and "GPN" in published


def test_positions_always_subscribed_even_over_cap(tmp_path):
    disc = [f"C{i}" for i in range(20)]
    s = _scanner(tmp_path, positions=["HELD1", "HELD2"], discovered=disc,
                 provider=_strong_feats(disc + ["HELD1", "HELD2"]), max_subs=5)
    s._discovered = disc
    asyncio.run(s._rank_tick(1000.0))
    assert "HELD1" in s.stream.subscribed and "HELD2" in s.stream.subscribed


def test_zero_eligible_blocks_entries_but_keeps_context(tmp_path):
    disc = ["NVDA", "AMD"]

    def weak_provider(syms, now):
        # price below the $5 floor -> nothing eligible
        return {s: CandidateFeatures(symbol=s, price=2.0, avg_daily_volume=10,
                                     avg_daily_dollar_volume=10, snapshot=None)
                for s in syms}
    s = _scanner(tmp_path, discovered=disc, provider=weak_provider)
    s._discovered = disc
    asyncio.run(s._rank_tick(1000.0))
    ctx = read_live_context(tmp_path / "live_context.json")
    assert ctx["entry_eligible_count"] == 0


def test_health_payload_has_regime_and_subscriptions(tmp_path):
    disc = ["NVDA", "AMD", "GPN"]
    s = _scanner(tmp_path, positions=["HELD1"], discovered=disc,
                 provider=_strong_feats(disc + ["HELD1", "SPY", "QQQ", "DIA", "IWM"]))
    s._discovered = disc
    asyncio.run(s._rank_tick(1000.0))
    ctx = read_live_context(tmp_path / "live_context.json")
    assert "regime" in ctx
    assert "active_subscriptions" in ctx and "available_subscriptions" in ctx
    assert ctx["scanner_status"] in ("healthy", "unhealthy")
    assert ctx["mode"] == "paper"


def test_stop_sets_flag(tmp_path):
    s = _scanner(tmp_path)
    s.stop()
    assert s._stopped is True
