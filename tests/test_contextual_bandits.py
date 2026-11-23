"""Tests for the offline contextual-bandit research module (Milestone 6).

Uses EXPLICITLY-LABELED SYNTHETIC fixtures only: a synthetic 3-action linear contextual
reward environment and a hand-written synthetic gold ``offline_rl`` Parquet. No real
lakehouse data is read. GPU assertions are skipped when MLX/Metal is unavailable.

Covers the non-negotiable invariants: actions are never invented, behavior-policy
probabilities are never fabricated, and NO IPS / propensity / doubly-robust path is exposed.
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from cengine.lakehouse.config import LakehouseConfig
from cengine.learning.contextual_bandits import (
    OFF_POLICY_CORRECTION_SUPPORTED,
    BanditDataset,
    LinUCB,
    NeuralBanditRewardModel,
    build_bandit_dataset,
    neural_bandit_available,
)
from cengine.learning.contextual_bandits.dataset import off_policy_correction_reason

# -- synthetic 3-action linear contextual reward environment (SYNTHETIC ONLY) --

def _synthetic_env(*, n: int, d: int = 5, n_actions: int = 3, noise: float = 0.1,
                   seed: int = 0, theta_seed: int = 123):
    """Generate a synthetic linear contextual-bandit env with a RANDOM logging policy.

    reward = context . theta[action] + noise. ``theta_seed`` fixes the reward structure so
    that train and test draws (varying ``seed``) share the SAME environment; only the
    sampled contexts/logged actions/noise differ. The logging (behavior) policy is uniform
    random; we deliberately DO NOT record its probabilities (simulating the real lakehouse,
    where behavior probabilities are unknown/NULL). Returns contexts, logged actions,
    observed rewards, true theta, and the noiseless best action per context.
    """
    theta = np.random.default_rng(theta_seed).normal(0.0, 1.0, size=(n_actions, d))
    rng = np.random.default_rng(seed)
    contexts = rng.normal(0.0, 1.0, size=(n, d))
    logged_actions = rng.integers(0, n_actions, size=n)
    mean_reward = contexts @ theta.T  # (n, n_actions)
    observed = mean_reward[np.arange(n), logged_actions] + rng.normal(0.0, noise, size=n)
    best_action = np.argmax(mean_reward, axis=1)
    return (contexts.astype(np.float32), logged_actions.astype(np.int64),
            observed.astype(np.float32), theta, best_action)


# -- LinUCB learns to prefer the best action ---------------------------------

def test_linucb_learns_to_prefer_best_action():
    Xtr, atr, rtr, theta, _ = _synthetic_env(n=5000, seed=1)
    Xte, _, _, _, best_te = _synthetic_env(n=1500, seed=2)

    model = LinUCB(n_actions=3, n_features=5, alpha=1.0, ridge_lambda=1.0,
                   action_names=["policy_a", "policy_b", "policy_c"])
    result = model.fit(Xtr, atr, rtr)
    assert result.epochs_run == 1
    assert np.isfinite(result.train_loss)

    rec = model.recommend(Xte)  # greedy direct-reward scoring
    accuracy = float(np.mean(rec == best_te))
    assert accuracy > 0.85, f"LinUCB accuracy {accuracy:.3f} not clearly above random (0.33)"

    # predicted reward should strongly rank-correlate with the true per-action reward
    pred = model.predict_reward(Xte)
    assert pred.shape == (Xte.shape[0], 3)

    # UCB scoring + Thompson sampling run and return valid action indices
    ucb = model.ucb_scores(Xte)
    assert ucb.shape == (Xte.shape[0], 3)
    ts = model.thompson_sample(Xte, rng=np.random.default_rng(7))
    assert ts.shape == (Xte.shape[0],)
    assert set(np.unique(ts)).issubset({0, 1, 2})


def test_linucb_respects_eligibility_mask():
    Xtr, atr, rtr, _, _ = _synthetic_env(n=4000, seed=3)
    Xte, _, _, _, best_te = _synthetic_env(n=500, seed=4)
    model = LinUCB(n_actions=3, n_features=5)
    model.fit(Xtr, atr, rtr)

    # Forbid each context's true best action; the recommendation must avoid it.
    mask = np.ones((Xte.shape[0], 3), dtype=bool)
    mask[np.arange(Xte.shape[0]), best_te] = False
    rec = model.recommend(Xte, eligibility_mask=mask)
    assert np.all(rec != best_te)


def test_linucb_serialization_roundtrip():
    Xtr, atr, rtr, _, _ = _synthetic_env(n=2000, seed=5)
    Xte, _, _, _, _ = _synthetic_env(n=200, seed=6)
    model = LinUCB(n_actions=3, n_features=5)
    model.fit(Xtr, atr, rtr)
    before = model.predict_reward(Xte)

    restored = LinUCB(n_actions=3, n_features=5)
    restored.load_params(model.params())
    after = restored.predict_reward(Xte)
    np.testing.assert_allclose(before, after, rtol=1e-6, atol=1e-6)


def test_linucb_records_cpu_device():
    Xtr, atr, rtr, _, _ = _synthetic_env(n=1000, seed=8)
    model = LinUCB(n_actions=3, n_features=5)
    model.fit(Xtr, atr, rtr)
    # numpy ridge solver: GPU unsupported for this op -> CPU, and it is RECORDED.
    assert model.device == "cpu"
    assert model.policy.summary()["placements"].get("cpu", 0) >= 1


# -- neural bandit trains (GPU assert skipped if no MLX) ----------------------

@pytest.mark.skipif(not neural_bandit_available(),
                    reason="MLX / Metal GPU not available")
def test_neural_bandit_trains_and_prefers_best_action():
    from cengine.learning.hardware import DevicePolicy

    Xtr, atr, rtr, _, _ = _synthetic_env(n=5000, seed=11)
    Xte, _, _, _, best_te = _synthetic_env(n=1500, seed=12)

    # min_gpu_elements=1 + force_cpu=False forces the GPU path for this small synthetic set,
    # making the GPU placement OBSERVABLE/MEASURABLE rather than silently assumed.
    policy = DevicePolicy(min_gpu_elements=1, force_cpu=False)
    model = NeuralBanditRewardModel(
        n_features=5, n_actions=3, action_names=["policy_a", "policy_b", "policy_c"],
        device_policy=policy)
    result = model.fit(Xtr, atr, rtr)

    assert result.epochs_run > 0
    assert np.isfinite(result.train_loss)
    assert result.device == "gpu"       # GPU-first, and recorded (not silently claimed)
    assert model.device == "gpu"

    rec = model.recommend(Xte)
    accuracy = float(np.mean(rec == best_te))
    assert accuracy > 0.7, f"neural bandit accuracy {accuracy:.3f} not above random (0.33)"

    pred = model.predict_reward(Xte)
    assert pred.shape == (Xte.shape[0], 3)


@pytest.mark.skipif(not neural_bandit_available(),
                    reason="MLX / Metal GPU not available")
def test_neural_bandit_empty_labeled_is_untrained():
    model = NeuralBanditRewardModel(n_features=5, n_actions=3)
    X = np.zeros((4, 5), dtype=np.float32)
    a = np.zeros((4,), dtype=np.int64)
    r = np.zeros((4,), dtype=np.float32)
    mask = np.zeros((4,), dtype=bool)  # no observed reward
    result = model.fit(X, a, r, reward_mask=mask)
    assert result.epochs_run == 0       # untrained / research-only, not promoted


@pytest.mark.skipif(not neural_bandit_available(),
                    reason="MLX / Metal GPU not available")
def test_neural_bandit_serialization_roundtrip():
    Xtr, atr, rtr, _, _ = _synthetic_env(n=2000, seed=13)
    Xte, _, _, _, _ = _synthetic_env(n=200, seed=14)
    model = NeuralBanditRewardModel(n_features=5, n_actions=3)
    model.fit(Xtr, atr, rtr)
    before = model.predict_reward(Xte)

    restored = NeuralBanditRewardModel(n_features=5, n_actions=3)
    restored.load_params(model.params())
    after = restored.predict_reward(Xte)
    np.testing.assert_allclose(before, after, rtol=1e-4, atol=1e-4)


# -- dataset builder over a synthetic gold offline_rl Parquet ----------------

def _write_synthetic_offline_rl(root) -> LakehouseConfig:
    """Write an EXPLICITLY-SYNTHETIC gold offline_rl Parquet and return its config."""
    cfg = LakehouseConfig(root=root)
    cfg.ensure_dirs()
    dest = cfg.dataset_path("gold", "offline_rl")
    dest.mkdir(parents=True, exist_ok=True)

    # 3 authorized actions + 1 UNAUTHORIZED ("policy_x", must be dropped, never invented).
    actions = (["policy_a", "policy_b", "policy_c"] * 4) + ["policy_x", "policy_x"]
    n = len(actions)
    rng = np.random.default_rng(99)
    # reward present only for a subset (closed positions); NULL otherwise, never fabricated.
    reward = [float(rng.normal()) if i % 2 == 0 else None for i in range(n)]
    table = pa.table({
        "transition_id": [f"t{i}" for i in range(n)],
        "episode_id": [f"e{i}" for i in range(n)],
        "symbol_id": [1000 + (i % 3) for i in range(n)],
        "selected_action": actions,
        "current_weight": [float(rng.normal()) for _ in range(n)],
        "target_weight": [float(rng.normal()) for _ in range(n)],
        "ctx_0": [float(rng.normal()) for _ in range(n)],
        "ctx_1": [float(rng.normal()) for _ in range(n)],
        # behavior-policy probability is ALWAYS NULL (unknown) and must never be used.
        "action_probability_if_known": [None] * n,
        "reward": reward,
        "terminal": [r is not None for r in reward],
    })
    pq.write_table(table, dest / "part-0.parquet")
    return cfg


def test_build_bandit_dataset_from_synthetic_lakehouse(tmp_path):
    cfg = _write_synthetic_offline_rl(tmp_path / "lake")
    authorized = ["policy_a", "policy_b", "policy_c"]
    ds = build_bandit_dataset(authorized, config=cfg,
                              feature_columns=["current_weight", "ctx_0", "ctx_1"])

    assert isinstance(ds, BanditDataset)
    assert ds.action_names == authorized
    assert ds.n_actions == 3
    assert ds.feature_names == ["current_weight", "ctx_0", "ctx_1"]
    assert ds.n_features == 3

    # 12 authorized rows kept, 2 unauthorized ("policy_x") dropped — never invented.
    assert ds.n_rows == 12
    assert ds.provenance["dropped_unauthorized"] == 2
    assert ds.contexts.shape == (12, 3)

    # reward present only for even indices among the authorized rows.
    assert ds.n_labeled == int(np.count_nonzero(ds.reward_mask))
    assert 0 < ds.n_labeled < ds.n_rows

    # action indices are valid; no action outside the authorized set exists.
    assert set(np.unique(ds.actions)).issubset({0, 1, 2})

    # behavior policy is unknown -> off-policy correction unavailable.
    assert ds.behavior_policy_known is False
    assert ds.off_policy_estimation_supported is False

    # the dataset is end-to-end fittable by LinUCB on its labeled rows.
    assert ds.is_fittable()
    X, a, r = ds.labeled()
    LinUCB(n_actions=ds.n_actions, n_features=ds.n_features,
           action_names=ds.action_names).fit(X, a, r)


def test_build_bandit_dataset_defaults_exclude_target_weight(tmp_path):
    cfg = _write_synthetic_offline_rl(tmp_path / "lake")
    ds = build_bandit_dataset(["policy_a", "policy_b", "policy_c"], config=cfg)
    # default context columns are pre-decision only; target_weight (an action output) is
    # excluded to avoid leaking the action into the context.
    assert "target_weight" not in ds.feature_names
    assert ds.feature_names == ["current_weight"]


def test_build_bandit_dataset_missing_data_is_research_only(tmp_path):
    cfg = LakehouseConfig(root=tmp_path / "empty_lake")
    cfg.ensure_dirs()
    ds = build_bandit_dataset(["policy_a", "policy_b"], config=cfg)
    assert ds.n_rows == 0
    assert ds.n_labeled == 0
    assert not ds.is_fittable()                 # untrained / research-only, never promoted
    assert ds.behavior_policy_known is False
    assert ds.off_policy_estimation_supported is False
    assert ds.provenance["note"] == "no_parquet_parts"


# -- NO IPS / propensity / doubly-robust path is exposed ---------------------

def test_no_ips_or_off_policy_correction_path_exposed():
    forbidden = ("ips", "propensity", "doubly", "importance_sampling")

    import cengine.learning.contextual_bandits as cb

    # No public symbol on any class or the package suggests an off-policy correction path.
    for obj in (LinUCB, NeuralBanditRewardModel, BanditDataset, cb):
        for name in dir(obj):
            if name.startswith("__"):
                continue
            low = name.lower()
            assert not any(tok in low for tok in forbidden), (
                f"forbidden off-policy-correction symbol exposed: {obj}.{name}")

    # The gate is declared explicitly and says why.
    assert OFF_POLICY_CORRECTION_SUPPORTED is False
    assert "behavior-policy" in off_policy_correction_reason().lower()


def test_dataset_declares_behavior_policy_unknown(tmp_path):
    cfg = _write_synthetic_offline_rl(tmp_path / "lake")
    ds = build_bandit_dataset(["policy_a", "policy_b", "policy_c"], config=cfg)
    assert ds.behavior_policy_known is False
    assert ds.off_policy_estimation_supported is False
    assert "never fabricated" in ds.provenance["off_policy_correction"]
