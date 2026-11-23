from pathlib import Path

from cengine.dynamic_universe.baseline import BaselineRegistry
from cengine.dynamic_universe.config import UniverseRotationConfig
from cengine.dynamic_universe.manager import ActiveUniverseSnapshot, DynamicUniverseManager
from cengine.dynamic_universe.rotation import RankedSymbol

BASE5 = ["AAA", "BBB", "CCC", "DDD", "EEE"]


def _baseline(tmp_path: Path, syms=BASE5) -> BaselineRegistry:
    p = tmp_path / "baseline_universe.txt"
    p.write_text(", ".join(f"'{s}'" for s in syms) + "\n", encoding="utf-8")
    return BaselineRegistry(p)


def _manager(tmp_path, *, ranking=(), protection=(), references=(), cfg=None, state=None,
             emitter=None, now=1000.0):
    return DynamicUniverseManager(
        baseline=_baseline(tmp_path),
        config=cfg or UniverseRotationConfig(active_universe_size=6,
                                             opportunity_target_slots=3),
        ranking_source=lambda: list(ranking() if callable(ranking) else ranking),
        protection_source=lambda: list(protection() if callable(protection) else protection),
        references=references,
        event_emitter=emitter,
        state_path=state,
        now_fn=lambda: now,
    )


def test_rotate_produces_immutable_snapshot(tmp_path):
    opps = [RankedSymbol("NVDA", 0.9), RankedSymbol("TSLA", 0.8)]
    mgr = _manager(tmp_path, ranking=opps)
    assert mgr.current_active() is None                 # fail-closed before first rotation
    res = mgr.rotate_once(now_s=0.0)
    snap = mgr.current_active()
    assert isinstance(snap, ActiveUniverseSnapshot)
    assert snap.active == res.active
    assert snap.generation == 1
    assert len(snap.active) == 6                         # 2 opps + 4 baseline
    assert "NVDA" in snap.active and "TSLA" in snap.active
    # manager writes NO files: universe.txt must not be created by the manager.
    assert not (tmp_path / "universe.txt").exists()


def test_protection_reserved_in_snapshot(tmp_path):
    held = ["ZZZ"]
    opps = [RankedSymbol(f"O{i}", 0.99) for i in range(10)]
    mgr = _manager(tmp_path, ranking=opps, protection=held)
    res = mgr.rotate_once(now_s=0.0)
    assert "ZZZ" in res.active
    assert "ZZZ" in mgr.current_active().protected
    assert len(res.active) == 6


def test_baseline_membership_never_lost(tmp_path):
    opps = [RankedSymbol(f"O{i}", 0.99) for i in range(50)]
    mgr = _manager(tmp_path, ranking=opps)
    for t in range(0, 2000, 100):
        mgr.rotate_once(now_s=float(t))
    assert set(mgr._baseline.symbols()) == set(BASE5)
    assert set(BASE5).issubset(set(mgr.layers.master))


def test_ranking_source_failure_falls_back_to_baseline(tmp_path):
    def boom():
        raise RuntimeError("scanner down")

    mgr = DynamicUniverseManager(
        baseline=_baseline(tmp_path),
        config=UniverseRotationConfig(active_universe_size=6),
        ranking_source=boom,
        protection_source=lambda: [],
        now_fn=lambda: 0.0,
    )
    res = mgr.rotate_once(now_s=0.0)                     # must not raise
    assert set(res.active) == set(BASE5)
    assert set(mgr.current_active().active) == set(BASE5)


def test_maybe_rotate_rate_limits_but_heartbeat_reads_snapshot(tmp_path):
    cfg = UniverseRotationConfig(active_universe_size=6, refresh_interval_seconds=60.0)
    mgr = _manager(tmp_path, ranking=[RankedSymbol("NVDA", 0.9)], cfg=cfg)
    s0 = mgr.maybe_rotate(now_s=0.0)                     # first -> rotates
    assert s0.generation == 1
    s1 = mgr.maybe_rotate(now_s=30.0)                    # < interval -> cached
    assert s1.generation == 1
    s2 = mgr.maybe_rotate(now_s=61.0)                    # >= interval -> rotates
    assert s2.generation == 2


def test_event_emitter_called(tmp_path):
    seen = {}

    def emit(result, layers, scores, now_s):
        seen["active"] = result.active
        seen["now"] = now_s
        seen["opp"] = layers.opportunity
        seen["scores"] = scores

    mgr = _manager(tmp_path, ranking=[RankedSymbol("NVDA", 0.9)], emitter=emit, now=42.0)
    res = mgr.rotate_once()
    assert seen["active"] == res.active
    assert seen["now"] == 42.0
    assert "NVDA" in seen["opp"]
    assert seen["scores"].get("NVDA") == 0.9


def test_dashboard_stats_shape(tmp_path):
    mgr = _manager(tmp_path, ranking=[RankedSymbol("NVDA", 0.9)])
    assert mgr.dashboard_stats()["coverage"] == "PENDING"
    mgr.rotate_once(now_s=0.0)
    stats = mgr.dashboard_stats()
    assert stats["baseline"] == 5
    assert stats["active"] <= stats["active_cap"] == 6
    assert stats["coverage"] in ("HEALTHY", "DEGRADED")
    assert "active_opportunity" in stats and "active_baseline" in stats
