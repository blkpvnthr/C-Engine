"""Tests for market_scan.publisher.UniversePublisher.

Publishes universe.txt (atomic, descending rank) + live_context.json. Rewrites
the FILE only on membership change or a forced (discovery) tick — never on every
rank tick — to avoid resetting the engine's scan rotation. Always refreshes the
mtime heartbeat + metadata so a live scanner stays "fresh" and a dead one goes
stale. Zero eligible candidates keeps the last-good file but blocks entries.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import universe
from market_scan.config import MarketScanConfig
from market_scan.health import (
    STATUS_HEALTHY,
    load_dynamic_universe,
    read_live_context,
)
from market_scan.publisher import UniversePublisher


def _pub(tmp_path, **cfg):
    config = MarketScanConfig(**cfg)
    return UniversePublisher(
        universe_path=tmp_path / "universe.txt",
        context_path=tmp_path / "live_context.json",
        config=config,
        mode="paper",
    )


def test_first_publish_writes_file_and_context(tmp_path):
    p = _pub(tmp_path)
    res = p.publish(["NVDA", "AMD", "GPN"], now=1000.0)
    assert res.rewrote_file is True
    assert res.eligible_count == 3
    assert universe.load_canonical_universe(tmp_path / "universe.txt") == ["NVDA", "AMD", "GPN"]
    ctx = read_live_context(tmp_path / "live_context.json")
    assert ctx["scanner_status"] == STATUS_HEALTHY
    assert ctx["entry_eligible_count"] == 3
    assert ctx["mode"] == "paper"
    assert ctx["universe"] == ["NVDA", "AMD", "GPN"]
    assert "pid" in ctx and "updated_at" in ctx


def test_unchanged_membership_does_not_rewrite_file_but_refreshes_heartbeat(tmp_path):
    p = _pub(tmp_path)
    p.publish(["NVDA", "AMD", "GPN"], now=1000.0)
    # same members, different rank order -> file content/order must stay stable
    res = p.publish(["GPN", "NVDA", "AMD"], now=1003.0)
    assert res.rewrote_file is False
    assert universe.load_canonical_universe(tmp_path / "universe.txt") == ["NVDA", "AMD", "GPN"]
    ctx = read_live_context(tmp_path / "live_context.json")
    assert ctx["updated_at"] == 1003.0          # heartbeat advanced
    assert ctx["ranked"] == ["GPN", "NVDA", "AMD"]  # live rank reflected in metadata


def test_membership_change_rewrites_file_with_added_removed(tmp_path):
    p = _pub(tmp_path)
    p.publish(["NVDA", "AMD"], now=1000.0)
    res = p.publish(["NVDA", "TSLA"], now=1003.0)
    assert res.rewrote_file is True
    assert universe.load_canonical_universe(tmp_path / "universe.txt") == ["NVDA", "TSLA"]
    ctx = read_live_context(tmp_path / "live_context.json")
    assert ctx["added"] == ["TSLA"]
    assert ctx["removed"] == ["AMD"]


def test_forced_rewrite_refreshes_rank_order_same_membership(tmp_path):
    p = _pub(tmp_path)
    p.publish(["NVDA", "AMD", "GPN"], now=1000.0)
    res = p.publish(["GPN", "AMD", "NVDA"], now=1300.0, force_rewrite=True)
    assert res.rewrote_file is True
    assert universe.load_canonical_universe(tmp_path / "universe.txt") == ["GPN", "AMD", "NVDA"]


def test_target_size_caps_published_universe(tmp_path):
    p = _pub(tmp_path, target_universe_size=2)
    res = p.publish(["A", "B", "C", "D"], now=1000.0)
    assert universe.load_canonical_universe(tmp_path / "universe.txt") == ["A", "B"]
    assert res.eligible_count == 2


def test_zero_candidates_keeps_last_good_file_but_blocks_entries(tmp_path):
    p = _pub(tmp_path)
    p.publish(["NVDA", "AMD"], now=1000.0)
    res = p.publish([], now=1003.0)
    assert res.eligible_count == 0
    # file untouched (last-good) so legacy importers still succeed
    assert universe.load_canonical_universe(tmp_path / "universe.txt") == ["NVDA", "AMD"]
    ctx = read_live_context(tmp_path / "live_context.json")
    assert ctx["entry_eligible_count"] == 0
    # the health gate blocks entries on eligible==0
    dyn = load_dynamic_universe(
        universe_path=tmp_path / "universe.txt",
        context_path=tmp_path / "live_context.json",
        enabled=True, now=1003.5, max_age_seconds=15.0,
    )
    assert dyn.symbols == []


def test_publish_then_hot_reload_round_trip_healthy(tmp_path):
    p = _pub(tmp_path)
    p.publish(["NVDA", "AMD", "GPN"], now=1000.0, health={"market_data_fresh": True})
    dyn = load_dynamic_universe(
        universe_path=tmp_path / "universe.txt",
        context_path=tmp_path / "live_context.json",
        enabled=True, now=1002.0, max_age_seconds=15.0,
    )
    assert dyn.state == STATUS_HEALTHY
    assert dyn.symbols == ["NVDA", "AMD", "GPN"]


def test_health_fields_are_merged_into_context(tmp_path):
    p = _pub(tmp_path)
    p.publish(["NVDA"], now=1000.0, health={
        "regime": "BULL", "connection_status": "connected",
        "active_subscriptions": 12, "available_subscriptions": 18,
        "discovered_count": 400, "market_session": "rth",
    })
    ctx = read_live_context(tmp_path / "live_context.json")
    assert ctx["regime"] == "BULL"
    assert ctx["connection_status"] == "connected"
    assert ctx["active_subscriptions"] == 12
    assert ctx["discovered_count"] == 400
