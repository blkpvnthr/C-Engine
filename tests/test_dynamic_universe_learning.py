"""P6: shadow-only learned universe-selection policy (contextual bandit).

Tests reward shaping + the LinUCB-backed policy + shadow governance on synthetic
data (no lakehouse / no real market data). The policy NEVER emits orders.
"""

from __future__ import annotations

import numpy as np

from cengine.learning.universe_selection.policy import (
    UniverseSelectionBandit,
    build_shadow_card,
    shadow_compare,
)
from cengine.learning.universe_selection.reward import selection_reward


# ------------------------------------------------------------------- reward
def test_reward_none_when_unobserved():
    assert selection_reward(observed=False) is None


def test_reward_bounded_and_signed():
    assert selection_reward(observed=True, traded=True, net_return_after_cost=0.10) == 1.0
    assert selection_reward(observed=True, traded=True, net_return_after_cost=-0.10) == -1.0
    r = selection_reward(observed=True, traded=True, net_return_after_cost=0.01)
    assert 0.0 < r < 1.0
    # untraded-but-observed is neutral
    assert selection_reward(observed=True, traded=False, net_return_after_cost=0.0) == 0.0


def test_reward_churn_penalty_bounded():
    r = selection_reward(observed=True, traded=True, net_return_after_cost=0.0,
                         churned_within_residence=True, churn_penalty=0.1)
    assert r == -0.1


# ------------------------------------------------------------------- policy
def test_bandit_learns_to_include_profitable_contexts():
    rng = np.random.default_rng(0)
    # 1D context: positive context -> include is profitable, negative -> loss.
    n = 400
    ctx = rng.uniform(-1, 1, size=(n, 1)).astype(np.float32)
    actions = np.ones(n, dtype=np.int64)              # all "include" (logged)
    rewards = np.where(ctx[:, 0] > 0, 1.0, -1.0).astype(np.float32)
    bandit = UniverseSelectionBandit(n_features=1, feature_names=["signal"])
    bandit.fit(ctx, actions, rewards)
    # Also teach the exclude arm a neutral baseline so include<exclude for negatives.
    neg = np.array([[-0.8]], dtype=np.float32)
    pos = np.array([[0.8]], dtype=np.float32)
    bandit.fit(np.vstack([neg, pos]), np.array([0, 0]), np.array([0.0, 0.0], np.float32))

    active = bandit.recommend_active(
        np.array([[0.9], [-0.9]], dtype=np.float32), ["WIN", "LOSE"],
        cap=10, protected=[],
    )
    assert "WIN" in active
    assert "LOSE" not in active


def test_recommend_always_keeps_protected_and_respects_cap():
    bandit = UniverseSelectionBandit(n_features=2)
    ctx = np.zeros((5, 2), dtype=np.float32)
    syms = [f"S{i}" for i in range(5)]
    active = bandit.recommend_active(ctx, syms, cap=3, protected=["HELD1", "HELD2"])
    assert "HELD1" in active and "HELD2" in active
    assert len(active) <= 3


def test_shadow_compare_metrics():
    m = shadow_compare(["A", "B", "C"], ["B", "C", "D"])
    assert m["overlap"] == 2
    assert m["challenger_only"] == ["D"]
    assert m["champion_only"] == ["A"]
    assert 0.0 < m["jaccard"] < 1.0


def test_params_roundtrip():
    b1 = UniverseSelectionBandit(n_features=3)
    ctx = np.ones((4, 3), dtype=np.float32)
    b1.fit(ctx, np.array([1, 1, 0, 0]), np.array([1.0, 1.0, 0.0, 0.0], np.float32))
    b2 = UniverseSelectionBandit(n_features=3)
    b2.load_params(b1.params())
    np.testing.assert_allclose(
        b1._model.predict_reward(ctx), b2._model.predict_reward(ctx), rtol=1e-6)


# --------------------------------------------------------------- governance
def test_build_shadow_card_is_trained_maturity():
    from cengine.learning.registry.maturity import ModelMaturity

    card = build_shadow_card(model_id="usel-1", model_version="1", n_features=4)
    assert card.model_type == "linucb_contextual_bandit"
    assert card.maturity == ModelMaturity.TRAINED.value
    assert card.applicable_strategy_families == ("universe_selection",)


def test_validated_to_shadow_transition_needs_no_approval():
    from cengine.learning.registry.maturity import (
        ModelMaturity, can_transition, requires_explicit_approval,
    )
    assert can_transition(ModelMaturity.VALIDATED, ModelMaturity.SHADOW)
    assert not requires_explicit_approval(ModelMaturity.VALIDATED, ModelMaturity.SHADOW)
    # and SHADOW cannot jump straight to live without approval
    assert requires_explicit_approval(ModelMaturity.LIVE_ELIGIBLE, ModelMaturity.LIVE_APPROVED)


# --------------------------------------------------- data-sufficiency deferral
def test_universe_selection_task_is_registered():
    from cengine.learning.datasets.eligibility import default_task_specs

    tasks = {s.task for s in default_task_specs()}
    assert "universe_selection" in tasks


def test_insufficient_data_defers_not_fabricates(tmp_path, monkeypatch):
    # With an empty lakehouse the universe-selection experiment must be INELIGIBLE
    # and surface a collection requirement — never a fabricated training result.
    monkeypatch.setenv("CENGINE_LAKEHOUSE_ROOT", str(tmp_path / "lake"))
    from cengine.optimization.data_sufficiency import DataSufficiencyController

    ctrl = DataSufficiencyController()
    assert ctrl.is_experiment_eligible("universe_selection") is False
    report = ctrl.assess()
    assert "universe_selection" in report["eligibility"]
    assert "universe_selection" not in report["experiment_eligible_tasks"]
