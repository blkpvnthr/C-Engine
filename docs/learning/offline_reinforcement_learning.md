# Constrained Offline Reinforcement Learning (Milestone 6)

Recommendation-only offline-RL research infrastructure over the authorized discrete action
space (allocation policies). It loads versioned logged transitions from the gold
`offline_rl_transitions` feed and provides four conservative, data-sized learners —
**Behavior Cloning (BC)**, **Fitted Q Evaluation (FQE)**, **Conservative Q-Learning (CQL)**,
and **Implicit Q-Learning (IQL)** — each GPU-first on MLX with a documented numpy CPU
fallback.

Source: `cengine/learning/offline_rl/` — `transitions.py`, `bc.py`, `fqe.py`, `cql_iql.py`,
`_nets.py`, `__init__.py`. GPU/CPU internals are covered in
`docs/learning/gpu_offline_rl.md`.

## Non-negotiable invariants

- **No orders, no broker, no bypass of financial authority.** `recommend(...)` returns an
  advisory action dict (`advisory_only: True`), never an order.
- **The mandatory -0.5% protective exit is ENVIRONMENT DYNAMICS, excluded from every
  learnable action space.** Reserved logged-action labels in `MANDATORY_EXIT_ACTIONS`
  (`mandatory_exit`, `mandatory_protective_exit`, `protective_exit`, `protective_stop`,
  `stop_loss_exit`, `forced_exit`, `risk_mandatory_exit`) are dropped from the action space
  and counted in the data-quality report. No learner can model, imitate, or recommend the
  protective exit as a policy choice. This exclusion is independent of MLX/GPU/training.
- **Behavior-policy probabilities are unknown and never fabricated**
  (`action_probability_if_known` is ALWAYS `None`), so IPS / weighted doubly-robust
  estimators that need them are not provided. The four shipped methods deliberately do not
  require them (`_BEHAVIOR_PROB_METHODS` is empty).
- **Insufficient or synthetic-as-real data cannot produce a promoted model** — enforced by
  `TrainingGuard` (below), not by convention.

## Honest-data reality

The `offline_rl_execution` task is **INSUFFICIENT_DATA** (see
`cengine/learning/datasets/eligibility.py`): it needs realized reward, action support ≥ 2,
`min_rows_training=5_000`, and `min_rows_validation=20_000` of genuine sequential execution
outcomes, none of which exist yet. All offline-RL learners therefore remain **research-only**
and their infrastructure is validated on **labeled synthetic fixtures**
(`make_synthetic_dataset`, always `synthetic=True`). No economic edge is claimed. The CLI
stub `python -m cengine.learning train-offline-rl`
(`cengine/learning/cli.py::cmd_train_offline_rl`) refuses to train and points the operator at
`profile-datasets`.

## Transition contracts — `transitions.py`

`TRANSITIONS_VERSION = "offline-rl-transitions-1.0.0"`.

**`TransitionDataset`** (immutable, versioned) holds the standard RL tensors: `states`
`(N, d)`, `actions` `(N,)` (index into the action space), `rewards` `(N,)` (NaN where
unknown) with a `reward_known` mask, `next_states` / `next_actions` / `next_action_valid`,
`terminal`, `truncated`, `action_mask` `(N, n_actions)`, `episode_ids`, `is_initial`,
`action_space`, `feature_names`, and a `DataQualityReport`. `action_probability_if_known` is
**always `None`**. Derived properties include `n_transitions`, `n_features`, `n_actions`,
`n_reward_known`, `n_episodes`, `behavior_action_counts()`, `subset(idx)` (for train/val
splits), and `summary()`.

**`ActionSpace`** (frozen) is the discrete authorized-action space with `index(name)` /
`name(idx)` helpers. The mandatory-exit labels are excluded before it is built.

**`DataQualityReport`** is observable provenance: `source`, `synthetic`, `rows_raw`,
`rows_kept`, `excluded_mandatory_exits`, `n_actions`, `n_episodes`, `n_reward_known`,
`n_terminal`, `n_truncated`, `point_in_time_violations`,
`has_behavior_probabilities` (always `False`), `feature_names`, and free-form `notes`.

**`build_dataset_from_records(records, *, synthetic, source, feature_columns=None)`** builds
a dataset from raw logged-transition dict rows:

- The caller MUST label `synthetic` truthfully — `True` for fixtures (never promotable),
  `False` only for genuine lakehouse rows.
- The action space excludes `MANDATORY_EXIT_ACTIONS`; rows whose logged action is a mandatory
  exit (or otherwise not in the kept action set) are dropped and counted in
  `excluded_mandatory_exits`.
- `next_state` / `next_action` are reconstructed by ordering within each `episode_id` by
  `decision_timestamp`; the last logged step of an episode is treated as terminal for
  bootstrap.
- A fabricated `action_probability_if_known` (i.e. non-`None`) is flagged as a
  point-in-time / data-integrity violation (`point_in_time_violations`), which the guard
  treats as `DATA_QUALITY_FAILED`.
- Feature columns are auto-detected from numeric, non-reserved keys (reserved metadata/target
  columns are listed in `_RESERVED_COLUMNS`), falling back to the columns the current gold
  build carries (`current_weight`, `target_weight`).

**`TransitionLoader`** loads genuine logged transitions (`synthetic=False`) from the lakehouse
`offline_rl_transitions` view via `DuckDBWarehouse` (imported lazily). A non-materialized
dataset or a failed query yields an empty, research-only dataset with a `load_failed` note
rather than raising.

**`make_synthetic_dataset(...)`** builds a learnable, explicitly-synthetic transition set:
the logged action depends on state (so BC can beat chance) and closed-transition rewards
depend on state+action (so value learning has signal). It is always `synthetic=True`, and
`include_mandatory_exit=True` injects a reserved exit label to prove it is excluded from the
learnable action space.

## `TrainingGuard` — the data-sufficiency gate

`TrainingGuard` (`transitions.py`) decides whether — and how — a learner may train, reusing
the committed M6 offline-RL eligibility spec so thresholds stay single-sourced. Defaults come
from `default_task_specs()`'s `offline_rl_execution` entry via `_offline_rl_thresholds()`:
`min_research=200`, `min_training=5_000`, `min_validation=20_000`, `min_actions=2` (all
overridable in the constructor).

- `check(dataset, *, method) -> GuardDecision` is **non-raising**; each learner's `fit()`
  calls it and returns an untrained research-only result when training is not allowed, so the
  pipeline keeps working on thin data.
- `require(dataset, *, method, promote=False) -> GuardDecision` is the **hard gate**: it
  raises `InsufficientDataError` unless training is allowed, and `SyntheticDataError` (or
  `InsufficientDataError`) when `promote=True` cannot be satisfied.

Gate logic, in order:

1. Any `point_in_time_violations > 0` → `DATA_QUALITY_FAILED` (not trainable).
2. `n_actions < min_actions` → `INSUFFICIENT_DATA`.
3. Reward-requiring methods with `n_reward_known == 0` → `INSUFFICIENT_DATA`. BC is
   reward-free (`_REWARD_FREE_METHODS = {"bc"}`); FQE/CQL/IQL require realized reward.
4. Effective rows (`n_reward_known` for reward methods, else `n_transitions`) map to an
   `EligibilityState`: `INSUFFICIENT_DATA` → `RESEARCH_ELIGIBLE` → `TRAINING_ELIGIBLE` →
   `VALIDATION_ELIGIBLE`.

`GuardDecision` exposes `allowed_to_train`, `promotable`, `reasons`, and `mode` (the state
value). A result is **`promotable` only when `VALIDATION_ELIGIBLE` AND not synthetic** — so
synthetic fixtures are structurally non-promotable regardless of size.

## The four learners

All four extend `_RLBase` (`_nets.py`): a shared standardizer fit on **TRAIN states only**
(no leakage), `DevicePolicy` bookkeeping, and the GPU-first device choice recorded on every
fit (`self.device`, `self.backend`). Results carry `trained`, `device`, `backend`, `mode`,
`promotable`, and `reasons`. `recommend(state, action_mask=None)` returns an advisory dict
and never the protective exit.

### Behavior Cloning — `bc.py` (`BC_VERSION = "offline-rl-bc-1.0.0"`)

`BehaviorCloner` fits a discrete classifier `pi(a | s)` to `(state, logged_action)` pairs. It
is the simplest conservative learner — it needs **neither rewards nor behavior
probabilities**, so it is always safe to run as a reference / support model and as FQE's
evaluation target. GPU path: MLX MLP + cross-entropy; CPU fallback: multinomial logistic
regression (`np_softmax_regression`). `BCConfig(epochs=120, batch_size=256,
learning_rate=1e-3, hidden=(64,32), weight_decay=1e-4, patience=12, standardize=True,
seed=7)`. Methods: `fit(dataset, *, val_dataset=None) -> BCResult`, `predict_action_probs`,
`predict_action`, `recommend`.

### Fitted Q Evaluation — `fqe.py` (`FQE_VERSION = "offline-rl-fqe-1.0.0"`)

`FittedQEvaluator` estimates the value of a **fixed target policy** by iteratively regressing
Q toward `r + gamma * (1 - terminal) * Q(s', pi(s'))`. It is model-based — it needs rewards +
transitions but **not** behavior probabilities, which is exactly why it is shipped instead of
IPS. The default target policy is the logged policy itself (`pi(s') =` logged next action,
i.e. fitted-SARSA evaluation of the behavior policy, a conservative baseline value); any
object with `predict_action` (e.g. a trained `BehaviorCloner` or `ImplicitQLearner`) or a
callable can be supplied. `evaluate(dataset, *, target_policy=None) -> FQEResult` (fits on
reward-known transitions only) reports `value_estimate`, `bellman_mae`, `iterations`, and
`n_reward_known`. `FQEConfig` defaults `discount=0.99`, `fitted_iterations=8`,
`inner_epochs=40`.

### Conservative Q-Learning — `cql_iql.py` (`CQL_IQL_VERSION = "offline-rl-cql-iql-1.0.0"`)

`ConservativeQLearner` adds a conservative regulariser that pushes Q **down** on
out-of-distribution actions and **up** on the logged action, counteracting the
overestimation that sinks naive offline Q-learning:
`loss = 0.5 * bellman_MSE + alpha * (logsumexp_a Q(s,a) - Q(s, a_logged))`. Reward is
required. `CQLConfig` defaults `discount=0.99`, `alpha=1.0`, `iterations=8`,
`inner_epochs=20`. `fit(dataset) -> CQLResult` reports `td_loss`, `conservative_penalty`,
`value_estimate`. Inference: `q_values`, `greedy_action`, `recommend`.

### Implicit Q-Learning — `cql_iql.py`

`ImplicitQLearner` never queries Q at out-of-distribution actions (the most
data-conservative method): it fits a state value `V` with **expectile regression** toward
`Q(s, a_logged)`, bootstraps `Q` from `V(s')`, and extracts a policy by **advantage-weighted
regression** on the logged actions. `IQLConfig` defaults `discount=0.99`, `expectile=0.7`,
`beta=3.0` (AWR temperature), `weight_clip=100.0`. `fit(dataset) -> IQLResult` reports
`q_loss`, `v_loss`, `policy_loss`, `value_estimate`. Inference: `state_value`,
`policy_action_probs`, `greedy_action`, `predict_action` (alias so an IQL policy can serve as
an FQE target), `recommend`.

## Relationship to the rest of M6

- Entry qualification and the native financial authority are preserved; offline RL only
  produces advisory rankings over already-authorized actions.
- Off-policy value comparison between a challenger and the champion on identical logged
  decisions lives in `cengine/learning/evaluation/ope.py`, where IPS/DR return
  `status='unavailable'` whenever behavior probabilities are unknown.
- Model maturity (`cengine/learning/registry/maturity.py`, `UNTRAINED..RETIRED`) is an
  independent dimension; no learner auto-promotes, and M6 does not promote to live
  (M7 certifies). Rollback to the last approved model or the deterministic baseline is always
  legal.
- Local only: Parquet/PyArrow/DuckDB + MLX. No cloud, no CUDA, no external model APIs.
