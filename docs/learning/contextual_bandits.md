# Contextual Bandits (Milestone 6)

Offline contextual-bandit **research** over a FIXED set of caller-supplied AUTHORIZED
actions (e.g. execution policies or strategy families). Two reward models — a numpy/CPU
`LinUCB` baseline and an MLX GPU-first `NeuralBanditRewardModel` — learn to *rank* eligible
actions from logged `(context, selected_action, reward)` rows. Everything here is
**recommendation-only**: no orders, no live exploration, and no contact with the mandatory
-0.5% protective exit (that exit is environment dynamics, not an action a bandit can pick).

Source: `cengine/learning/contextual_bandits/` — `dataset.py`, `linucb.py`,
`neural_bandit.py`, `__init__.py`.

## Honest-data reality

This is **infrastructure validated on labeled synthetic fixtures**, not a deployed edge.
The `policy_bandit` task over `gold/offline_rl` is currently **INSUFFICIENT_DATA**
(`cengine/learning/datasets/eligibility.py` → `default_task_specs()` requires
`requires_reward=True`, `requires_action_support=2`, `min_rows_training=1_000`, none of
which genuine execution outcomes satisfy yet). The standalone CLI entrypoint
`python -m cengine.learning train-bandit` (`cengine/learning/cli.py::cmd_train_bandit`)
deliberately refuses to fabricate a model and instead points the operator at
`profile-datasets`. No economic edge is claimed.

## No IPS — behavior probabilities are unknown

The single most important invariant: the logging (behavior) policy that produced each
logged decision is **unknown**. In the gold `offline_rl` feed `action_probability_if_known`
is **always NULL and is never fabricated**, so any off-policy value estimator that *requires*
a propensity — inverse-propensity scoring (IPS), weighted-IPS, doubly-robust (DR) — is
**gated off by construction**. This is not a runtime toggle; it is enforced as module-level
state:

- `OFF_POLICY_CORRECTION_SUPPORTED = False` (`dataset.py`).
- `build_bandit_dataset(...)` does not even read `action_probability_if_known`.
- `BanditDataset.behavior_policy_known` is always `False`, and
  `BanditDataset.off_policy_estimation_supported` is therefore always `False`.
- `off_policy_correction_reason()` returns the human-readable explanation surfaced in
  provenance and telemetry.

Only **direct reward-model scoring** and a clearly-labeled **offline Thompson sampling**
ranking (LinUCB only) are exposed. For the engine-wide OPE gate that returns
`status='unavailable'` for IPS/DR on real data, see `cengine/learning/evaluation/ope.py`.

## Dataset builder — `build_bandit_dataset`

`build_bandit_dataset(actions, *, config=None, feature_columns=None, source_layer="gold",
source_dataset="offline_rl") -> BanditDataset`

Reads the lakehouse gold `offline_rl` Parquet directly via DuckDB (the same analytical
surface `DuckDBWarehouse` uses), so profiling, training, and evaluation all see one
authoritative source. `CONTEXTUAL_BANDIT_VERSION = "contextual-bandit-1.0.0"`.

Enforced invariants:

- **Actions are never invented.** The caller supplies the authorized `actions` list; a
  logged `selected_action` outside that list is DROPPED and counted
  (`provenance["dropped_unauthorized"]`), never appended as a new action.
- **Pre-decision context only.** `DEFAULT_CONTEXT_COLUMNS = ("current_weight",)`.
  `target_weight` is *excluded by default* because it is an OUTPUT of the chosen allocation
  policy (the action) and would leak the action into the context; callers may opt in via
  `feature_columns` for ablations only. Missing feature columns are skipped and noted; if
  none resolve, a single constant `bias_intercept` feature yields a degenerate but
  well-formed (research-only) dataset.
- **Reward used only where present.** Gold `reward` is NULL for not-yet-closed positions;
  those rows are retained but masked out of fitting via `reward_mask`. No reward is
  fabricated.
- **Degrades, never crashes.** Absent Parquet, a missing `selected_action` column, or zero
  rows all return a well-formed empty `BanditDataset` (`_empty_dataset`) with a provenance
  `note`, never an exception and never a promoted model.

`BanditDataset` (dataclass) carries `contexts` `(n, d)`, `actions` (index into
`action_names`), `rewards`, `reward_mask`, `action_names`, `feature_names`,
`behavior_policy_known=False`, and a `provenance` dict. Useful accessors: `n_rows`,
`n_actions`, `n_features`, `n_labeled` (rows with observed reward — the only fittable rows),
`labeled()` (returns `(X, actions, rewards)` restricted to observed-reward rows),
`is_fittable()`.

## `LinUCB` — numpy/CPU baseline

`class LinUCB` (`linucb.py`, `framework="numpy"`, `model_type="linucb_contextual_bandit"`).
Disjoint per-action ridge regression with an upper-confidence-bound exploration bonus
(Li et al. 2010). Each authorized action `a` keeps a ridge system
`A_a = lambda*I + sum x xᵀ`, `b_a = sum r x`; its reward estimate is `theta_a · x` with
`theta_a = A_a⁻¹ b_a` and optimism bonus `alpha * sqrt(xᵀ A_a⁻¹ x)`.

Constructor: `LinUCB(n_actions, n_features, *, alpha=1.0, ridge_lambda=1.0,
reward_noise=1.0, action_names=None, device_policy=None)`.

Deliberately numpy/CPU — it is the **independent baseline the GPU neural bandit must beat**,
and the per-action ridge solve is tiny linear algebra dominated by host/device sync if pushed
to the GPU. Even so, every `fit` records a `PlacementDecision` through the shared
`DevicePolicy` with `gpu_supported=False`, so the CPU placement is observable in telemetry
rather than implicit.

Key methods:

- `fit(contexts, actions, rewards, *, reward_mask=None) -> FitResult` — accumulates ridge
  statistics from observed-reward rows only; repeated calls accumulate (online ridge update).
- `predict_reward(contexts) -> (n, n_actions)` — direct expected reward per action.
- `ucb_bonus(contexts)` / `ucb_scores(contexts, *, alpha=None)` — the optimism width and the
  optimistic score `theta_a · x + alpha * bonus`.
- `recommend(contexts, *, eligibility_mask=None, use_ucb=False)` — argmax action per context.
  `eligibility_mask` `(n, n_actions)` restricts the choice to currently
  authorized/eligible actions; default is greedy direct-reward scoring (no exploration).
- `thompson_sample(contexts, *, rng=None, eligibility_mask=None)` — **offline, research-only**
  Thompson sampling: draws `theta_a ~ N(theta_hat_a, reward_noise² · A_a⁻¹)` from the Bayesian
  linear-regression posterior and returns the argmax sampled reward. It ranks offline-provided
  contexts using the posterior **only** — never a behavior-policy probability (none exists) —
  and emits no orders. It is not live or uncontrolled exploration.
- `params()` / `load_params()` — serialize `A`, `b`, and a JSON config blob to numpy arrays.

## `NeuralBanditRewardModel` — MLX GPU-first

`class NeuralBanditRewardModel` (`neural_bandit.py`, `framework="mlx"`,
`model_type="neural_contextual_bandit"`). A shared ReLU MLP trunk (`_BanditNet`) with one
linear reward head per authorized action, trained on logged `(context, selected_action,
reward)` rows with a **per-sample masked loss**: only the head of the action actually taken
receives gradient for that row (`mx.take_along_axis(pred, ab, axis=1)`). It is the learned
counterpart to `LinUCB` and **must beat that baseline on held-out data before any promotion
— it is never preferred merely for running on the GPU**.

Constructor: `NeuralBanditRewardModel(n_features, n_actions, *, config=None,
action_names=None, device_policy=None)`, where `config` is a
`cengine.learning.supervised.base.TrainConfig`.

MLX integration mirrors `cengine/learning/supervised/mlx_models.py`:

- MLX is **import-guarded** — the module imports on a host without MLX, and construction
  raises a clear `RuntimeError` (`_require_mlx`) directing the caller to the numpy `LinUCB`
  baseline. `neural_bandit_available()` is `True` iff MLX imported **and** a Metal GPU is
  actually usable (`mlx_available()`).
- Training uses `nn.value_and_grad` / `opt.update` (AdamW) / `mx.eval` under
  `mx.stream(policy.mlx_device(decision))`, with optional validation-driven early stopping
  (`TrainConfig.patience`) and best-parameter restore.
- Weights serialize to portable numpy (`params()` / `load_params()` with `w::` /
  `std::` / `meta::` key prefixes), so an MLX-trained model can be inspected or re-loaded.
- `fit(...)` on **zero labeled rows** returns an untrained `FitResult`
  (`train_loss=nan`, `epochs_run=0`) — research-only, never promoted.

Methods: `fit(contexts, actions, rewards, *, reward_mask=None, X_val/a_val/r_val=None)`,
`predict_reward(contexts) -> (n, n_actions)`, `recommend(contexts, *,
eligibility_mask=None)` (greedy direct-reward scoring, no exploration, no orders).

## GPU-first with an observable CPU fallback

Both models choose their device through the shared `DevicePolicy`
(`cengine/learning/hardware.py`), which runs on MLX 0.32.4 / Apple Metal (Apple M5 Max):

- GPU-first for sufficiently large learning workloads; **documented, measurable CPU
  fallback** for small, latency-dominated workloads (`n_elements < min_gpu_elements`,
  default `50_000`), unsupported ops (`gpu_supported=False`, LinUCB's path), MLX/Metal
  absence, or `CENGINE_LEARNING_FORCE_CPU` set.
- Every placement is recorded as a `PlacementDecision` (`workload`, `device`, `reason`,
  `gpu_eligible`, `fell_back`) in `policy.decisions`, with `policy.summary()` exposing the
  per-device counts and `fallback_count`. The platform **never silently claims GPU
  acceleration** — small bandit workloads genuinely running on CPU are reported as such.

See `docs/learning/gpu_offline_rl.md` for the MLX ↔ numpy device-policy contract shared with
the offline-RL learners.

## What is and is not guaranteed

- Learning only improves **ranking / eligible-action selection**. The deterministic entry
  qualification (DI+ > DI- AND price > VWAP|SMA20|BB-mid AND developing-1m > prev-close) and
  the native financial authority are untouched; a bandit recommendation can only reorder
  actions that are already authorized and eligible.
- No model submits orders, bypasses the native authority, or touches the mandatory -0.5%
  protective exit.
- No IPS/DR, no fabricated behavior probabilities, no fabricated rewards, no auto-promotion.
  Model maturity is governed separately (`cengine/learning/registry/maturity.py`,
  `UNTRAINED..RETIRED`); M6 does not promote to live.
