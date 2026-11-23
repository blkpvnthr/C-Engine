from pathlib import Path

import universe
from cengine.dynamic_universe.baseline import BaselineRegistry
from cengine.dynamic_universe.config import UniverseRotationConfig
from cengine.dynamic_universe.manager import DynamicUniverseManager
from cengine.dynamic_universe.rotation import RankedSymbol

BASE5 = ["AAA", "BBB", "CCC", "DDD", "EEE"]


def _baseline(tmp_path: Path, syms=BASE5) -> BaselineRegistry:
    p = tmp_path / "baseline_universe.txt"
    p.write_text(", ".join(f"'{s}'" for s in syms) + "\n", encoding="utf-8")
    return BaselineRegistry(p)


def _writer(tmp_path: Path):
    uni = tmp_path / "universe.txt"

    def write(active):
        universe.write_canonical_universe(list(active), uni, min_symbols=1)
        return list(active)

    return uni, write


def _manager(tmp_path, *, ranking=(), protection=(), references=(), cfg=None, state=None):
    uni, write = _writer(tmp_path)
    mgr = DynamicUniverseManager(
        baseline=_baseline(tmp_path),
        config=cfg or UniverseRotationConfig(active_universe_size=6,
                                             opportunity_target_slots=3),
        ranking_source=lambda: list(ranking() if callable(ranking) else ranking),
        protection_source=lambda: list(protection() if callable(protection) else protection),
        references=references,
        active_writer=write,
        state_path=state,
        now_fn=lambda: 1000.0,
    )
    return mgr, uni


def test_rotate_writes_active_universe(tmp_path):
    opps = [RankedSymbol("NVDA", 0.9), RankedSymbol("TSLA", 0.8)]
    mgr, uni = _manager(tmp_path, ranking=opps)
    res = mgr.rotate_once(now_s=0.0)
    assert len(res.active) == 6           # cap: 2 opps + 4 baseline
    written = universe.load_canonical_universe(uni)
    assert set(written) == set(res.active)
    assert "NVDA" in res.active and "TSLA" in res.active
    assert len(res.active_baseline) == 4  # 4 of the 5 baseline fill the window
    assert len(res.active_opportunity) == 2


def test_protection_reserved_in_active_and_file(tmp_path):
    held = ["ZZZ"]                          # not baseline, not opp
    opps = [RankedSymbol(f"O{i}", 0.99) for i in range(10)]
    mgr, uni = _manager(tmp_path, ranking=opps, protection=held)
    res = mgr.rotate_once(now_s=0.0)
    assert "ZZZ" in res.active
    assert "ZZZ" in universe.load_canonical_universe(uni)
    assert len(res.active) == 6


def test_baseline_never_written_out_of_existence(tmp_path):
    # Even with a flood of strong opportunities, the BaselineRegistry membership
    # is untouched and baseline symbols remain eligible (master always ⊇ baseline).
    opps = [RankedSymbol(f"O{i}", 0.99) for i in range(50)]
    mgr, uni = _manager(tmp_path, ranking=opps)
    for t in range(0, 2000, 100):
        mgr.rotate_once(now_s=float(t))
    assert set(mgr._baseline.symbols()) == set(BASE5)   # baseline membership intact
    # master always contains the full baseline
    assert set(BASE5).issubset(set(mgr.layers.master))


def test_ranking_source_failure_falls_back_to_baseline(tmp_path):
    def boom():
        raise RuntimeError("scanner down")

    uni, write = _writer(tmp_path)
    mgr = DynamicUniverseManager(
        baseline=_baseline(tmp_path),
        config=UniverseRotationConfig(active_universe_size=6),
        ranking_source=boom,
        protection_source=lambda: [],
        active_writer=write,
        now_fn=lambda: 0.0,
    )
    res = mgr.rotate_once(now_s=0.0)       # must not raise
    assert set(res.active) == set(BASE5)
    assert set(universe.load_canonical_universe(uni)) == set(BASE5)


def test_event_emitter_called(tmp_path):
    seen = {}

    def emit(result, layers, now_s):
        seen["active"] = result.active
        seen["layers"] = layers
        seen["now"] = now_s

    uni, write = _writer(tmp_path)
    mgr = DynamicUniverseManager(
        baseline=_baseline(tmp_path),
        config=UniverseRotationConfig(active_universe_size=6),
        ranking_source=lambda: [RankedSymbol("NVDA", 0.9)],
        protection_source=lambda: [],
        active_writer=write,
        event_emitter=emit,
        now_fn=lambda: 42.0,
    )
    res = mgr.rotate_once()
    assert seen["active"] == res.active
    assert seen["now"] == 42.0
    assert "NVDA" in seen["layers"].opportunity


def test_dashboard_stats_shape(tmp_path):
    mgr, _ = _manager(tmp_path, ranking=[RankedSymbol("NVDA", 0.9)])
    assert mgr.dashboard_stats()["coverage"] == "PENDING"   # before first rotation
    mgr.rotate_once(now_s=0.0)
    stats = mgr.dashboard_stats()
    assert stats["baseline"] == 5
    assert stats["active"] <= stats["active_cap"] == 6
    assert stats["coverage"] in ("HEALTHY", "DEGRADED")
    assert "active_opportunity" in stats and "active_baseline" in stats
