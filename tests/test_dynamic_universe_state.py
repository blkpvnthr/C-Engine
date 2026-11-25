from pathlib import Path

from cengine.dynamic_universe.baseline import BaselineRegistry
from cengine.dynamic_universe.config import UniverseRotationConfig
from cengine.dynamic_universe.lifecycle import LifecycleState, LifecycleTracker
from cengine.dynamic_universe.manager import DynamicUniverseManager
from cengine.dynamic_universe.rotation import RankedSymbol
from cengine.dynamic_universe.state import load_state, save_state

BASE5 = ["AAA", "BBB", "CCC", "DDD", "EEE"]


def test_save_load_roundtrip(tmp_path):
    cfg = UniverseRotationConfig()
    t = LifecycleTracker(cfg)
    t.observe("NVDA", is_baseline=False, eligible=True, ranked=True, score=0.7, now_s=10.0)
    t.mark_active("NVDA", now_s=10.0, reason="promoted")
    sp = tmp_path / "state.json"
    save_state(sp, tracker=t, previous_active=("NVDA", "SPY"), version=5)

    t2, prev, version = load_state(sp, cfg)
    assert prev == ("NVDA", "SPY")
    assert version == 5
    st = t2.get("NVDA")
    assert st is not None
    assert st.state is LifecycleState.ACTIVE
    assert st.entered_active_s == 10.0
    assert st.last_eval_s == 10.0


def test_missing_state_file_is_fresh(tmp_path):
    t, prev, version = load_state(tmp_path / "nope.json", UniverseRotationConfig())
    assert prev == ()
    assert version == 0
    assert t.all() == {}


def test_corrupt_state_file_is_fresh(tmp_path):
    sp = tmp_path / "state.json"
    sp.write_text("{ not json", encoding="utf-8")
    t, prev, version = load_state(sp, UniverseRotationConfig())
    assert prev == () and version == 0 and t.all() == {}


def test_manager_restart_recovers_residence(tmp_path):
    bp = tmp_path / "baseline_universe.txt"
    bp.write_text(", ".join(f"'{s}'" for s in BASE5) + "\n", encoding="utf-8")
    sp = tmp_path / "state.json"

    strong = [RankedSymbol(f"O{i}", 0.99) for i in range(10)]
    cfg = UniverseRotationConfig(active_universe_size=5, opportunity_target_slots=5,
                                 candidate_min_residence_seconds=180.0,
                                 baseline_max_revisit_seconds=1e9)

    mgr1 = DynamicUniverseManager(
        baseline=BaselineRegistry(bp), config=cfg,
        ranking_source=lambda: [], protection_source=lambda: [],
        state_path=sp, now_fn=lambda: 0.0,
    )
    r1 = mgr1.rotate_once(now_s=0.0)          # baseline fills; incumbents entered @ 0
    assert set(r1.active) == set(BASE5)

    # New manager instance (simulated restart) reloads persisted state.
    mgr2 = DynamicUniverseManager(
        baseline=BaselineRegistry(bp), config=cfg,
        ranking_source=lambda: strong, protection_source=lambda: [],
        state_path=sp, now_fn=lambda: 60.0,
    )
    assert set(mgr2._previous_active) == set(BASE5)   # restored
    r2 = mgr2.rotate_once(now_s=60.0)                 # within residence of t=0
    assert set(r2.active) == set(BASE5)               # churn prevented across restart
