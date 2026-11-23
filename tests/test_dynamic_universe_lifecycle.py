import pytest

from cengine.dynamic_universe.config import UniverseRotationConfig
from cengine.dynamic_universe.lifecycle import LifecycleState, LifecycleTracker


def _tracker():
    return LifecycleTracker(UniverseRotationConfig(
        candidate_min_residence_seconds=180.0,
        baseline_max_revisit_seconds=300.0,
    ))


def test_discovery_to_ranked_progression():
    t = _tracker()
    s = t.observe("NVDA", is_baseline=False, eligible=False, ranked=False, score=0.0, now_s=0.0)
    assert s.state is LifecycleState.DISCOVERED
    s = t.observe("NVDA", is_baseline=False, eligible=True, ranked=False, score=0.1, now_s=1.0)
    assert s.state is LifecycleState.ELIGIBLE
    s = t.observe("NVDA", is_baseline=False, eligible=True, ranked=True, score=0.7, now_s=2.0)
    assert s.state is LifecycleState.RANKED
    assert s.last_eval_s == 2.0


def test_min_residence_blocks_demotion():
    t = _tracker()
    t.observe("NVDA", is_baseline=False, eligible=True, ranked=True, score=0.7, now_s=0.0)
    t.mark_active("NVDA", now_s=0.0, reason="promoted")
    assert t.can_demote("NVDA", now_s=100.0) is False   # within 180s
    assert t.can_demote("NVDA", now_s=200.0) is True     # past residence
    assert t.can_demote("NVDA", now_s=100.0, force=True) is True  # emergency bypass


def test_baseline_revisit_deadline():
    t = _tracker()
    t.observe("SPY", is_baseline=True, eligible=True, ranked=True, score=0.5, now_s=0.0)
    assert t.needs_revisit("SPY", now_s=100.0) is False
    assert t.needs_revisit("SPY", now_s=400.0) is True   # > 300s since last eval


def test_baseline_cannot_expire():
    t = _tracker()
    t.observe("SPY", is_baseline=True, eligible=True, ranked=True, score=0.5, now_s=0.0)
    with pytest.raises(ValueError):
        t.expire("SPY", now_s=1000.0, reason="stale")


def test_rotated_out_then_reassess():
    t = _tracker()
    t.observe("NVDA", is_baseline=False, eligible=True, ranked=True, score=0.7, now_s=0.0)
    t.mark_active("NVDA", now_s=0.0, reason="promoted")
    t.mark_rotated_out("NVDA", now_s=300.0, reason="weaker score")
    assert t.get("NVDA").state is LifecycleState.ROTATED_OUT
    t.reassess("NVDA", now_s=600.0)
    assert t.get("NVDA").state is LifecycleState.REASSESSMENT


def test_consecutive_active_cycles_preserve_entry_time():
    # mark_active across cycles must NOT reset entered_active_s (residence clock).
    t = _tracker()
    t.observe("NVDA", is_baseline=False, eligible=True, ranked=True, score=0.7, now_s=0.0)
    t.mark_active("NVDA", now_s=0.0, reason="promoted")
    t.mark_active("NVDA", now_s=60.0, reason="active")     # still active next cycle
    assert t.get("NVDA").entered_active_s == 0.0           # preserved
    assert t.get("NVDA").last_eval_s == 60.0               # refreshed
    assert t.can_demote("NVDA", now_s=100.0) is False      # residence from t=0


def test_observe_without_stamp_preserves_eval_time():
    t = _tracker()
    t.observe("NVDA", is_baseline=True, eligible=True, ranked=True, score=0.5, now_s=0.0)
    t.observe("NVDA", is_baseline=True, eligible=True, ranked=True, score=0.6,
              now_s=500.0, stamp_eval=False)
    assert t.get("NVDA").last_eval_s == 0.0                # NOT advanced
    assert t.needs_revisit("NVDA", now_s=500.0) is True    # stale as of last real eval
