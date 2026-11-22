"""Corrective-review regression tests (post-integration).

Each test reproduces a confirmed/suspected gap from the corrective review BEFORE
the fix, then pins the corrected behavior. No live broker/socket: fakes only.

Covered: #4 candidate-specific freshness, #5 WS readiness, #6 freshness-based
score expiry, #9 subscription-overflow policy, #10 publication consistency.
(#3 broker snapshot, #7 replacement gates, #8 kill-switch live in their own
sections below / in test_market_scan_trade_integration.py.)
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import asyncio

import universe
from market_scan.config import MarketScanConfig
from market_scan.health import (
    STATUS_HEALTHY,
    load_dynamic_universe,
    read_live_context,
    write_live_context,
)
from market_scan.publisher import UniversePublisher
from market_scan.ranking import CandidateFeatures
from market_scan.scanner import MarketScanner
from market_scan.stream_client import StreamClient
from market_scan.subscription_manager import SubscriptionManager


class _FakeStreamLive:
    def __init__(self, live=True):
        self._live = live
        self._subscribed = set()

    async def apply_subscriptions(self, *, to_add, to_remove):
        self._subscribed.update(to_add)
        self._subscribed.difference_update(to_remove)

    @property
    def subscribed(self):
        return set(self._subscribed)

    def is_live(self, now=None, max_staleness_s=20.0):
        return self._live

    def stop(self):
        pass


def _snap(qage=1.0, tage=1.0):
    return SimpleNamespace(ok=True, buy_pressure=0.6, seller_exhaustion=0.4,
                           seller_exhaustion_l1=0.4, cvd_trend=0.3,
                           last_quote_age_s=qage, last_trade_age_s=tage)


def _feat(sym, snap):
    return CandidateFeatures(
        symbol=sym, price=50.0, avg_daily_volume=5_000_000,
        avg_daily_dollar_volume=250_000_000, rel_volume=1.4,
        spread_pct=0.0004,
        quote_age_s=getattr(snap, "last_quote_age_s", 1.0) if snap else None,
        directional_row={"pdi": 30, "mdi": 10, "adx": 25, "vwap_dist": 0.01,
                         "short_mom": 0.02, "bb_dist": 0.01},
        snapshot=snap, vwap_dist=0.01, bb_mid_dist=0.01)


def _scanner(tmp_path, provider, discovered, *, stream=None, cfg=None):
    cfg = cfg or MarketScanConfig(max_subscriptions=30, target_universe_size=10)
    es = SimpleNamespace(
        broker=SimpleNamespace(api_key="k", secret_key="s",
                               get_all_open_positions=lambda: [], list_open_orders=lambda: []),
        config=SimpleNamespace(paper=True))
    ms = MarketScanner(
        es, cfg, universe_path=tmp_path / "universe.txt",
        context_path=tmp_path / "live_context.json",
        stream_client=stream or _FakeStreamLive(True), feature_provider=provider,
        discovery_fn=lambda **k: SimpleNamespace(symbols=discovered, equities=[], etfs=[],
                                                 error=None, source_counts={}))
    ms._discovered = list(discovered)
    return ms


# ====================================================================== #5
def test_stream_client_is_live_requires_actual_message():
    clk = {"t": 1000.0}
    sc = StreamClient(api_key="k", secret_key="s",
                      on_trade=lambda *a: None, on_quote=lambda *a: None,
                      stream_factory=lambda *a, **k: SimpleNamespace(
                          subscribe_trades=lambda *a: None, subscribe_quotes=lambda *a: None))
    sc._now_fn = lambda: clk["t"]
    assert sc.is_live() is False  # subscription registration alone != live
    asyncio.run(sc._handle_trade(SimpleNamespace(
        symbol="SPY", price=1.0, size=1, timestamp=SimpleNamespace(timestamp=lambda: 1.0))))
    assert sc.is_live() is True
    clk["t"] = 1000.0 + 30
    assert sc.is_live(max_staleness_s=20.0) is False  # went stale


def test_scanner_unhealthy_when_stream_not_live_despite_fresh_candidates(tmp_path):
    disc = ["NVDA", "AMD", "SPY", "QQQ"]
    ms = _scanner(tmp_path, lambda syms, now: {s: _feat(s, _snap()) for s in syms},
                  disc, stream=_FakeStreamLive(live=False))
    asyncio.run(ms._rank_tick(1000.0))
    ctx = read_live_context(tmp_path / "live_context.json")
    assert ctx["stream_live"] is False
    assert ctx["scanner_status"] == "unhealthy"
    assert ctx["connection_status"] == "connecting"


# ====================================================================== #4
def test_only_candidates_with_fresh_own_data_published(tmp_path):
    disc = ["FRESH", "STALE", "SPY", "QQQ"]

    def prov(syms, now):
        return {s: _feat(s, _snap(qage=999, tage=999) if s == "STALE" else _snap())
                for s in syms}
    ms = _scanner(tmp_path, prov, disc)
    asyncio.run(ms._rank_tick(1000.0))
    pub = universe.load_canonical_universe(tmp_path / "universe.txt")
    assert "FRESH" in pub and "STALE" not in pub


def test_market_data_fresh_is_reference_based_not_newest_symbol(tmp_path):
    # A fresh candidate must NOT make whole-universe freshness true when the
    # market references are stale.
    disc = ["HOT", "SPY", "QQQ"]

    def prov(syms, now):
        return {s: _feat(s, _snap() if s == "HOT" else _snap(qage=999, tage=999))
                for s in syms}
    ms = _scanner(tmp_path, prov, disc)
    asyncio.run(ms._rank_tick(1000.0))
    ctx = read_live_context(tmp_path / "live_context.json")
    assert ctx["market_data_fresh"] is False


# ====================================================================== #6
def test_candidate_score_expires_when_its_data_goes_stale(tmp_path):
    disc = ["X", "SPY", "QQQ"]
    state = {"age": 1.0}

    def prov(syms, now):
        return {s: _feat(s, _snap(qage=state["age"], tage=state["age"])) for s in syms}
    cfg = MarketScanConfig(max_subscriptions=30, target_universe_size=10,
                           candidate_score_ttl_s=5.0)
    ms = _scanner(tmp_path, prov, disc, cfg=cfg)
    asyncio.run(ms._rank_tick(1000.0))
    assert "X" in ms._scored
    # X's data goes stale and time advances beyond the score TTL
    state["age"] = 999.0
    asyncio.run(ms._rank_tick(1000.0 + 10))
    assert "X" not in ms._scored  # expired, not kept alive by a historical score
    ctx = read_live_context(tmp_path / "live_context.json")
    assert ctx["entry_eligible_count"] == 0  # nothing fresh -> entries blocked


# ====================================================================== #9
# Subscription-overflow policy: the configured cap must NEVER be silently
# exceeded; mandatory symbols are kept by priority but still cap-bounded, and
# any drop (mandatory or candidate) is reported explicitly.
# ======================================================================
def test_cap_is_never_exceeded_even_by_mandatory():
    m = SubscriptionManager(max_subscriptions=3, mandatory_groups=3)
    # positions(2) + pending(2) + refs(2) = 6 mandatory, cap 3
    plan = m.plan([["P1", "P2"], ["O1", "O2"], ["R1", "R2"], ["C1"]])
    assert len(plan.target) == 3            # STRICT cap, not 6
    assert plan.overflow is True
    # highest priority (positions) kept first
    assert "P1" in plan.target and "P2" in plan.target


def test_mandatory_overflow_is_reported_not_silent():
    m = SubscriptionManager(max_subscriptions=3, mandatory_groups=3)
    plan = m.plan([["P1", "P2"], ["O1", "O2"], ["R1"], []])
    assert len(plan.target) == 3
    # the mandatory symbols that didn't fit are reported for a loud warning
    assert set(plan.mandatory_dropped) == {"O2", "R1"}
    assert plan.overflow is True


def test_positions_prioritized_over_pending_over_refs_under_cap():
    m = SubscriptionManager(max_subscriptions=4, mandatory_groups=3)
    plan = m.plan([["P1", "P2"], ["O1"], ["R1", "R2"], ["C1", "C2"]])
    # cap 4: P1,P2,O1 (3 mandatory) + 1 ref; candidates fully dropped
    assert "P1" in plan.target and "P2" in plan.target and "O1" in plan.target
    assert "R1" in plan.target  # first ref fits
    assert "R2" not in plan.target
    assert "C1" not in plan.target and "C2" not in plan.target


def test_no_overflow_when_everything_fits():
    m = SubscriptionManager(max_subscriptions=10, mandatory_groups=3)
    plan = m.plan([["P1"], ["O1"], ["R1", "R2"], ["C1", "C2"]])
    assert plan.overflow is False
    assert not plan.mandatory_dropped
    assert set(plan.target) == {"P1", "O1", "R1", "R2", "C1", "C2"}


def test_candidates_rotate_within_free_budget():
    m = SubscriptionManager(max_subscriptions=5, mandatory_groups=3)
    groups = [["P1"], ["O1"], ["R1"], ["C1", "C2", "C3", "C4"]]  # 3 mandatory + 2 free
    p1 = m.plan(groups)
    chosen1 = [s for s in p1.target if s.startswith("C")]
    assert len(chosen1) == 2
    assert p1.overflow is True  # 2 candidates dropped
    p2 = m.plan(groups)
    chosen2 = [s for s in p2.target if s.startswith("C")]
    assert chosen2 != chosen1   # rotation advanced


# ====================================================================== #10
# Publication consistency: concurrent reads, interrupted publish, restart.
# ======================================================================
def test_concurrent_writes_and_reads_never_tear(tmp_path):
    import threading
    import time
    upath = tmp_path / "universe.txt"
    universe.write_canonical_universe(["A", "B"], upath)
    stop = {"v": False}
    errors = []

    def writer():
        i = 0
        while not stop["v"]:
            try:
                universe.write_canonical_universe([f"S{i % 5}", "Z"], upath)
            except Exception as e:  # noqa: BLE001
                errors.append(("writer", repr(e)))
            i += 1

    def reader():
        while not stop["v"]:
            try:
                got = universe.load_canonical_universe(upath)
                if not got:
                    errors.append(("empty", got))
            except Exception as e:  # noqa: BLE001 - a torn/empty read would raise here
                errors.append(("reader", repr(e)))

    threads = [threading.Thread(target=writer),
               threading.Thread(target=reader), threading.Thread(target=reader)]
    for t in threads:
        t.start()
    time.sleep(0.3)
    stop["v"] = True
    for t in threads:
        t.join()
    assert not errors, errors[:5]   # atomic os.replace -> never torn/empty


def test_interrupted_publish_half_update_blocks_entries(tmp_path):
    # universe.txt updated to new membership but live_context not yet (crash
    # between the two atomic writes) -> mismatch -> fail-safe block (even after
    # the single retry, since the on-disk state is persistently inconsistent).
    import os
    upath = tmp_path / "universe.txt"
    cpath = tmp_path / "live_context.json"
    universe.write_canonical_universe(["A", "B", "C"], upath)
    os.utime(upath, (1000.0, 1000.0))
    write_live_context(cpath, {"scanner_status": STATUS_HEALTHY, "enabled": True,
                               "updated_at": 1000.0, "universe": ["A", "B"],
                               "market_data_fresh": True, "entry_eligible_count": 2})
    res = load_dynamic_universe(universe_path=upath, context_path=cpath,
                                enabled=True, now=1000.5)
    assert res.symbols == []


def test_restart_recovery_reseeds_from_disk_and_republishes(tmp_path):
    cfg = MarketScanConfig()
    up, cp = tmp_path / "universe.txt", tmp_path / "live_context.json"
    p1 = UniversePublisher(up, cp, cfg, mode="paper")
    p1.publish(["A", "B"], now=1000.0, health={"market_data_fresh": True})
    # restart: a brand-new publisher instance on the same paths
    p2 = UniversePublisher(up, cp, cfg, mode="paper")
    p2.publish(["A", "B", "C"], now=1003.0, health={"market_data_fresh": True})
    assert universe.load_canonical_universe(up) == ["A", "B", "C"]
    res = load_dynamic_universe(universe_path=up, context_path=cp, enabled=True, now=1004.0)
    assert res.state == STATUS_HEALTHY and res.symbols == ["A", "B", "C"]


# ====================================================================== #3
# Scanner must not redundantly poll the shared broker on every rank tick.
# ======================================================================
def test_rank_tick_does_not_repoll_broker_each_tick(tmp_path):
    calls = {"pos": 0, "pend": 0}

    class _CountBroker:
        api_key = "k"
        secret_key = "s"

        def get_all_open_positions(self):
            calls["pos"] += 1
            return []

        def list_open_orders(self):
            calls["pend"] += 1
            return []

    es = SimpleNamespace(broker=_CountBroker(), config=SimpleNamespace(paper=True))
    ms = MarketScanner(
        es, MarketScanConfig(), universe_path=tmp_path / "u.txt",
        context_path=tmp_path / "c.json", stream_client=_FakeStreamLive(True),
        feature_provider=lambda syms, now: {s: _feat(s, _snap()) for s in syms},
        discovery_fn=lambda **k: SimpleNamespace(symbols=["NVDA", "SPY", "QQQ"],
                                                 equities=[], etfs=[], error=None, source_counts={}))
    ms._discovered = ["NVDA", "SPY", "QQQ"]
    ms._refresh_positions_snapshot()        # off-loop seed (1 read each)
    base_pos, base_pend = calls["pos"], calls["pend"]
    assert base_pos == 1 and base_pend == 1
    for i in range(5):
        asyncio.run(ms._rank_tick(1000.0 + i))
    assert calls["pos"] == base_pos         # 5 rank ticks -> ZERO extra broker polls
    assert calls["pend"] == base_pend
