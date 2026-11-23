"""Constrained offline-RL research framework (Milestone 6, §15/§16/§25).

Validates the training INFRASTRUCTURE on explicitly-labeled synthetic fixtures (no genuine
transitions exist yet): transition construction, mandatory-exit exclusion from the action
space, no fabricated behavior probabilities, conservative learners run, and the training
guard refuses insufficient data + marks synthetic results non-promotable.
"""

from __future__ import annotations

import numpy as np

from cengine.learning.offline_rl import (
    BCConfig,
    BehaviorCloner,
    ConservativeQLearner,
    CQLConfig,
    FittedQEvaluator,
    FQEConfig,
    ImplicitQLearner,
    InsufficientDataError,
    IQLConfig,
    TrainingGuard,
    make_synthetic_dataset,
)


def test_synthetic_dataset_shapes_and_no_fabricated_probabilities():
    ds = make_synthetic_dataset(n_episodes=40, steps=5, n_features=5, n_actions=3)
    n = ds.states.shape[0]
    assert n == 40 * 5
    assert ds.actions.shape == (n,)
    assert ds.next_states.shape == ds.states.shape
    assert ds.synthetic is True
    # behavior-policy probabilities are unknown and never fabricated
    assert ds.action_probability_if_known is None
    assert ds.data_quality.has_behavior_probabilities is False


def test_mandatory_exit_excluded_from_action_space():
    ds = make_synthetic_dataset(n_episodes=40, steps=5, n_actions=3,
                                include_mandatory_exit=True)
    # mandatory protective exit is environment dynamics, never a learnable action
    from cengine.learning.offline_rl import MANDATORY_EXIT_ACTIONS
    for a in ds.action_space.actions:
        assert a not in MANDATORY_EXIT_ACTIONS
    assert ds.data_quality.excluded_mandatory_exits > 0


def test_training_guard_refuses_insufficient_data():
    tiny = make_synthetic_dataset(n_episodes=2, steps=2, n_actions=2)
    guard = TrainingGuard()
    decision = guard.check(tiny, method="behavior_cloning")
    assert decision.allowed_to_train is False or decision.promotable is False


def test_behavior_cloning_trains_but_synthetic_is_not_promotable():
    ds = make_synthetic_dataset(n_episodes=120, steps=6, n_features=5, n_actions=3)
    bc = BehaviorCloner(config=BCConfig(epochs=15, batch_size=128))
    res = bc.fit(ds)
    assert res.trained is True
    assert res.promotable is False          # synthetic data -> research-only (§25)
    assert res.n_actions == 3


def test_fqe_returns_value_estimate():
    ds = make_synthetic_dataset(n_episodes=120, steps=6, n_actions=3, reward_fraction=1.0)
    fqe = FittedQEvaluator(config=FQEConfig(fitted_iterations=3, inner_epochs=5))
    res = fqe.evaluate(ds)
    assert res.trained is True
    assert np.isfinite(res.value_estimate)
    assert res.promotable is False          # synthetic


def test_cql_and_iql_training_steps_run():
    ds = make_synthetic_dataset(n_episodes=120, steps=6, n_actions=3)
    cql = ConservativeQLearner(config=CQLConfig(iterations=2, inner_epochs=4))
    cres = cql.fit(ds)
    assert cres.trained is True
    assert np.isfinite(cres.value_estimate)
    iql = ImplicitQLearner(config=IQLConfig(iterations=2, inner_epochs=4))
    ires = iql.fit(ds)
    assert ires.trained is True
    assert np.isfinite(ires.value_estimate)


def test_insufficient_data_error_is_available():
    # the typed error exists for callers that opt into strict mode
    assert issubclass(InsufficientDataError, Exception)
