from cengine.dynamic_universe.config import UniverseRotationConfig
from cengine.dynamic_universe.lifecycle import LifecycleTracker
from cengine.dynamic_universe.rotation import (
    RankedSymbol,
    RotationResult,
    UniverseRotationScheduler,
)

SPEC_BASELINE = [f"B{i:02d}" for i in range(30)]   # 30 synthetic baseline symbols


def _ranked(symbols, base_score=0.5):
    return [RankedSymbol(s, base_score - i * 0.001, True) for i, s in enumerate(symbols)]


def _sched(**kw):
    cfg = UniverseRotationConfig(**kw)
    return UniverseRotationScheduler(cfg), LifecycleTracker(cfg)


def test_active_never_exceeds_cap():
    sched, tracker = _sched(active_universe_size=30, baseline_target_slots=20,
                            opportunity_target_slots=10)
    opps = _ranked([f"O{i:02d}" for i in range(50)], base_score=0.9)
    res = sched.rotate(
        baseline=_ranked(SPEC_BASELINE), opportunities=opps,
        protected=[], references=[], previous_active=[], tracker=tracker, now_s=0.0,
    )
    assert isinstance(res, RotationResult)
    assert len(res.active) == 30
    assert len(set(res.active)) == 30          # unique
    assert res.degraded is False


def test_protected_always_retained():
    sched, tracker = _sched(active_universe_size=30)
    held = ["POS1", "POS2", "POS3"]            # not in baseline/opps
    res = sched.rotate(
        baseline=_ranked(SPEC_BASELINE),
        opportunities=_ranked([f"O{i:02d}" for i in range(50)], base_score=0.9),
        protected=held, references=[], previous_active=[], tracker=tracker, now_s=0.0,
    )
    for s in held:
        assert s in res.active
    assert len(res.active) == 30


def test_fills_with_baseline_when_no_opportunities():
    sched, tracker = _sched(active_universe_size=30, opportunity_target_slots=10)
    res = sched.rotate(
        baseline=_ranked(SPEC_BASELINE), opportunities=[],   # none qualified
        protected=[], references=[], previous_active=[], tracker=tracker, now_s=0.0,
    )
    assert len(res.active) == 30
    assert set(res.active) == set(SPEC_BASELINE)   # all 30 baseline fill the window


def test_strong_opportunity_promoted_over_baseline():
    sched, tracker = _sched(active_universe_size=30, baseline_target_slots=20,
                            opportunity_target_slots=10)
    strong = _ranked([f"O{i:02d}" for i in range(10)], base_score=0.99)
    res = sched.rotate(
        baseline=_ranked(SPEC_BASELINE, base_score=0.3), opportunities=strong,
        protected=[], references=[], previous_active=[], tracker=tracker, now_s=0.0,
    )
    assert set(res.active_opportunity) == {s.symbol for s in strong}
    assert len(res.active_baseline) == 20
    assert len(res.active) == 30


def test_starvation_resistance_baseline_revisit():
    # A baseline symbol past its revisit deadline must be scheduled even when
    # opportunities would otherwise fill the window.
    sched, tracker = _sched(active_universe_size=30, baseline_target_slots=20,
                            opportunity_target_slots=10, baseline_max_revisit_seconds=300.0)
    # Seed: everything evaluated at t=0, then B29 evaluated long in the past.
    for s in SPEC_BASELINE:
        tracker.observe(s, is_baseline=True, eligible=True, ranked=True, score=0.5, now_s=0.0)
    tracker.observe("B29", is_baseline=True, eligible=True, ranked=True, score=0.01, now_s=-1000.0)
    strong = _ranked([f"O{i:02d}" for i in range(30)], base_score=0.99)
    res = sched.rotate(
        baseline=_ranked(SPEC_BASELINE, base_score=0.02),  # all baseline weak
        opportunities=strong, protected=[], references=[],
        previous_active=[], tracker=tracker, now_s=400.0,
    )
    assert "B29" in res.active                # revisit guarantee beats weak score
    assert len(res.active) == 30


def test_degraded_when_protected_exceeds_cap():
    sched, tracker = _sched(active_universe_size=5)
    held = [f"POS{i}" for i in range(8)]       # 8 protected > cap of 5
    res = sched.rotate(
        baseline=_ranked(SPEC_BASELINE), opportunities=[], protected=held,
        references=[], previous_active=[], tracker=tracker, now_s=0.0,
    )
    assert res.degraded is True
    assert len(res.active) == 5
    assert len(res.dropped_mandatory) == 3
    # deterministic: first-5 protected kept by input order
    assert list(res.active) == held[:5]


def test_add_remove_delta_vs_previous():
    sched, tracker = _sched(active_universe_size=30)
    prev = SPEC_BASELINE[:30]
    res = sched.rotate(
        baseline=_ranked(SPEC_BASELINE),
        opportunities=_ranked(["NEWA", "NEWB"], base_score=0.99),
        protected=[], references=[], previous_active=prev, tracker=tracker, now_s=0.0,
    )
    assert set(res.added) == set(res.active) - set(prev)
    assert set(res.removed) == set(prev) - set(res.active)


def test_property_cap_and_uniqueness_over_many_pools():
    sched, _ = _sched(active_universe_size=30)
    for seed in range(200):
        tracker = LifecycleTracker(UniverseRotationConfig(active_universe_size=30))
        n_opp = (seed * 7) % 60
        n_prot = (seed * 3) % 6
        opps = _ranked([f"O{i:02d}" for i in range(n_opp)], base_score=0.9)
        prot = [f"P{i}" for i in range(n_prot)]
        res = sched.rotate(
            baseline=_ranked(SPEC_BASELINE), opportunities=opps, protected=prot,
            references=["SPY"], previous_active=[], tracker=tracker, now_s=float(seed),
        )
        assert len(res.active) <= 30
        assert len(set(res.active)) == len(res.active)       # unique
        for p in prot:
            assert p in res.active                           # protection held
