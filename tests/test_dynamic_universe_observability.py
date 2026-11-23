"""P5: observability — the [DYNAMIC UNIVERSE MANAGER] block + coverage status."""

from __future__ import annotations

from pathlib import Path

from cengine.dynamic_universe.baseline import BaselineRegistry
from cengine.dynamic_universe.config import UniverseRotationConfig
from cengine.dynamic_universe.manager import DynamicUniverseManager
from cengine.dynamic_universe.observability import render_dynamic_universe_block
from cengine.dynamic_universe.rotation import RankedSymbol

BASE5 = ["AAA", "BBB", "CCC", "DDD", "EEE"]


def test_render_none_is_disabled_stub():
    out = render_dynamic_universe_block(None)
    assert "DYNAMIC UNIVERSE MANAGER" in out and "disabled" in out


def test_render_block_contains_all_spec_lines():
    stats = {
        "baseline": 30, "dynamic_candidates": 7, "master": 42, "active": 30,
        "active_cap": 30, "protected": 3, "active_baseline": 20,
        "active_opportunity": 10, "pending_rotation": 2, "stale_baseline": 0,
        "last_rotation_s": 0.0, "next_rotation_s": 0.0, "coverage": "HEALTHY",
    }
    out = render_dynamic_universe_block(stats)
    for label in ("Baseline universe", "Dynamic candidates", "Master universe",
                  "Active subscriptions", "Protected positions", "Active baseline",
                  "Active opportunities", "Pending rotation", "Last rotation",
                  "Next rotation", "Coverage status"):
        assert label in out
    assert "30/30" in out
    assert "HEALTHY" in out


def test_manager_dashboard_feeds_render(tmp_path):
    bp = tmp_path / "baseline_universe.txt"
    bp.write_text(", ".join(f"'{s}'" for s in BASE5) + "\n", encoding="utf-8")
    mgr = DynamicUniverseManager(
        baseline=BaselineRegistry(bp),
        config=UniverseRotationConfig(active_universe_size=6, refresh_interval_seconds=60.0),
        ranking_source=lambda: [RankedSymbol("NVDA", 0.9)],
        protection_source=lambda: [],
        now_fn=lambda: 1000.0,
    )
    mgr.rotate_once(now_s=1000.0)
    stats = mgr.dashboard_stats()
    assert stats["next_rotation_s"] == 1060.0            # last + interval
    assert stats["coverage"] in ("HEALTHY", "STALE_BASELINE", "DEGRADED")
    out = render_dynamic_universe_block(stats)
    assert "[DYNAMIC UNIVERSE MANAGER]" in out


def test_coverage_degraded_when_protected_exceeds_cap(tmp_path):
    bp = tmp_path / "baseline_universe.txt"
    bp.write_text(", ".join(f"'{s}'" for s in BASE5) + "\n", encoding="utf-8")
    held = [f"POS{i}" for i in range(8)]                 # 8 > cap 5
    mgr = DynamicUniverseManager(
        baseline=BaselineRegistry(bp),
        config=UniverseRotationConfig(active_universe_size=5),
        ranking_source=lambda: [],
        protection_source=lambda: held,
        now_fn=lambda: 0.0,
    )
    mgr.rotate_once(now_s=0.0)
    assert mgr.dashboard_stats()["coverage"] == "DEGRADED"
