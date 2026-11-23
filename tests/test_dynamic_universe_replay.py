"""P7: static-vs-dynamic universe-selection replay comparison (spec §12).

Synthetic panels where the correct policy is knowable, so the harness's isolation
of the selection variable is verifiable. Results here are labelled 'synthetic' —
no real-market performance is claimed.
"""

from __future__ import annotations

from cengine.dynamic_universe.config import UniverseRotationConfig
from cengine.dynamic_universe.replay_comparison import (
    ReplayPanel,
    block_bootstrap_ci,
    compare_policies,
    policy_static,
    simulate,
    walk_forward_splits,
)

BASE = [f"B{i:02d}" for i in range(30)]


def _panel(n_steps=60, cost_bps=5.0):
    # Baseline returns ~0; candidates O00..O19 with score i/20 have forward return
    # proportional to score (so a selection policy that picks high-score names wins).
    steps = []
    for t in range(n_steps):
        scores = {f"O{i:02d}": i / 20.0 for i in range(20)}
        fwd = {f"O{i:02d}": (i / 20.0) * 0.01 for i in range(20)}  # up to +1%/step
        for b in BASE:
            fwd[b] = 0.0
        steps.append({"scores": scores, "forward_return": fwd})
    return ReplayPanel(baseline=BASE, steps=steps, cost_per_turnover_bps=cost_bps)


def test_dynamic_beats_static_when_candidates_carry_return():
    panel = _panel()
    cfg = UniverseRotationConfig(active_universe_size=30, opportunity_target_slots=10)
    res = compare_policies(panel, cap=30, config=cfg)
    a = res["A_static"].net_return
    d = res["D_adaptive"].net_return
    c = res["C_flexible"].net_return
    assert d > a            # adaptive rotation captures candidate alpha
    assert c > a            # flexible allocation too
    assert d > 0            # the captured alpha is genuinely positive net of cost
    # static holds only the baseline -> full baseline coverage; only the one-time
    # entry cost drags it slightly negative (baseline forward returns are 0).
    assert res["A_static"].baseline_coverage == 1.0
    assert -0.001 < a <= 0


def test_identical_controls_cap_enforced_all_policies():
    panel = _panel()
    res = compare_policies(panel, cap=30)
    for m in res.values():
        assert m.avg_active <= 30
        assert m.steps == 60


def test_cost_penalizes_churn():
    # Higher transaction cost reduces net return for a rotating policy.
    cheap = compare_policies(_panel(cost_bps=0.0), cap=30)["D_adaptive"].net_return
    pricey = compare_policies(_panel(cost_bps=100.0), cap=30)["D_adaptive"].net_return
    assert pricey < cheap


def test_walk_forward_splits_are_chronological():
    splits = walk_forward_splits(100, folds=4)
    assert len(splits) == 4
    for (tr0, tr1), (te0, te1) in splits:
        assert tr0 == 0 and tr1 == te0 and te0 < te1     # expanding, no overlap/shuffle
    assert splits[-1][1][1] == 100                        # last test reaches the end


def test_bootstrap_ci_is_deterministic_and_ordered():
    net = [0.001 * ((i % 7) - 3) for i in range(80)]
    lo1, hi1 = block_bootstrap_ci(net, seed=42)
    lo2, hi2 = block_bootstrap_ci(net, seed=42)
    assert (lo1, hi1) == (lo2, hi2)                       # reproducible
    assert lo1 <= hi1


def test_static_policy_is_rotation_free_after_warmup():
    panel = _panel()
    m = simulate(policy_static, panel, cap=30, name="A_static")
    # Only the initial fill (30 symbols on step 1) changes anything; 0 thereafter.
    assert m.churn < 1.0
    assert m.turnover < 0.02
