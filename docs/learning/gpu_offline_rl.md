# GPU Offline RL: MLX Networks + CPU Fallback (Milestone 6)

How the constrained offline-RL learners (BC / FQE / CQL / IQL) and the neural contextual
bandit actually run on hardware: **GPU-first policy / value / Q networks on MLX (Apple Metal)
with a documented, observable, measurable numpy CPU fallback**. Because the offline-RL data
is INSUFFICIENT_DATA (see `docs/learning/offline_reinforcement_learning.md`), this layer is
exercised and validated on **labeled synthetic fixtures** only — it is proven plumbing, not a
profitability claim.

Source: `cengine/learning/offline_rl/_nets.py` (shared primitives),
`cengine/learning/hardware.py` (`DevicePolicy`), and the per-learner `_fit_mlx` / `_fit_numpy`
pairs in `bc.py`, `fqe.py`, `cql_iql.py`.

Environment: MLX 0.32.4 on an Apple M5 Max (Metal). Local only — no cloud, no CUDA, no
external model APIs.

## Shared primitives — `_nets.py`

`_nets.py` factors out everything the four learners share so the algorithm files do not
duplicate it.

**Import-guarded MLX.** `try: import mlx.core/nn/optimizers ... _MLX = True` / `except: _MLX =
False`. On a non-MLX host the package still imports and every learner degrades to the numpy
CPU path; `_require_mlx()` raises only when a GPU path is *explicitly* requested without MLX.
`mlx_backend_available()` reports whether the MLX package imported (Metal usability is checked
separately by `hardware.mlx_available()`).

**`MLXMLP(n_in, hidden, n_out)`** — a configurable ReLU MLP (`(B, n_in) -> (B, n_out)`)
matching the reference pattern in `cengine/learning/supervised/mlx_models.py`. This single
module is reused as:

- a **policy network** (`n_out = n_actions`) — BC's classifier and IQL's advantage-weighted
  policy head;
- a **value network** (`n_out = 1`) — IQL's expectile state-value `V`;
- a **Q network** (`n_out = n_actions`) — FQE's Q, CQL's conservative Q, and IQL's Q head.

**numpy CPU helpers** (the fallback math, never a silent GPU claim): `np_softmax`,
`np_logsumexp`, `np_one_hot`, and `np_softmax_regression` — full-batch multinomial logistic
regression by gradient descent, used as the CPU behavior-cloning / policy learner (and IQL's
AWR policy with `sample_weight`).

**`_RLBase`** — the common base for all four learners. It holds:

- a `Standardizer` (`cengine/learning/supervised/base.py`) fit on **TRAIN states only** —
  point-in-time correctness, no leakage (`_fit_standardizer` vs `_apply_standardizer`);
- `DevicePolicy` bookkeeping and the seed;
- `_choose_device(workload, *, n_elements)` — the GPU-first decision that records the actual
  device (`self.device`) and backend (`self.backend ∈ {"mlx", "numpy"}`) used on every fit,
  and returns the MLX device object when on GPU.

## Device policy — `DevicePolicy`

`cengine/learning/hardware.py::DevicePolicy` is the single arbiter of GPU vs CPU placement,
shared by the bandit models and all offline-RL learners.

Constructor: `DevicePolicy(*, prefer_gpu=True, min_gpu_elements=50_000, force_cpu=None)`.
`gpu_available = mlx_available() and not force_cpu`. The environment variable
**`CENGINE_LEARNING_FORCE_CPU`** (any value other than empty / `"0"`) forces CPU globally.

`choose(workload, *, n_elements=1e9, gpu_supported=True) -> PlacementDecision` resolves, in
order:

1. `force_cpu` → CPU, reason `forced_cpu (config/env)`.
2. MLX/Metal unavailable → CPU, `fell_back=True`, reason carries the import error.
3. `gpu_supported=False` (e.g. LinUCB's tiny ridge solve) → CPU, `fell_back=True`.
4. `prefer_gpu=False` → CPU, `policy prefers cpu`.
5. `n_elements < min_gpu_elements` → CPU, `small workload ... (sync-dominated)` — small,
   latency-sensitive workloads are dominated by host/device sync, so GPU is counter-productive.
6. Otherwise → **GPU**, reason `gpu-first: {n_elements} elements`.

Every decision is appended to `policy.decisions` (a list of `PlacementDecision` with
`workload`, `device`, `reason`, `gpu_eligible`, `fell_back`), and `fell_back` decisions
increment `fallback_count`. `policy.summary()` returns per-device placement counts and the
fallback count, so telemetry can **prove where computation actually ran** — the platform never
silently claims GPU acceleration when work ran on the CPU. `mlx_device(decision)` maps a
decision to `mx.gpu` / `mx.cpu` (or `None` when MLX is absent).

Each learner sizes `n_elements` to reflect real work, e.g. CQL/IQL use
`n_transitions * n_features * iterations * inner_epochs`, FQE uses
`n_transitions * n_features * fitted_iterations * inner_epochs`, BC uses
`n_transitions * n_features * epochs`. Small synthetic fixtures therefore legitimately land on
CPU, and that is reported, not hidden.

## Hardware introspection

`hardware.inspect_hardware()` probes ACTUAL runtime capabilities (never hardcoded specs) and
returns `HardwareCapabilities`: platform / machine / `is_apple_silicon`, unified memory,
`mlx_installed` / `mlx_metal_available` / `mlx_import_error`, device name, GPU architecture,
working-set / buffer / memory sizes, and `supported_precisions` — the last probed by actually
round-tripping `float32` / `float16` / `bfloat16` through an MLX GPU matmul
(`_validate_precisions`). It also reports `native_metallib_present` (the M1 Metal runtime's
`build-metal/cengine.metallib`), signaling clean coexistence of the native Metal path and MLX.
Exposed via `python -m cengine.learning hardware`.

## MLX training path (GPU)

Each `_fit_mlx` follows the same MLX idiom under `with mx.stream(dev):`
(`dev = policy.mlx_device(decision)`):

- build an `MLXMLP` for each network, seeded by `mx.random.seed`;
- optimize with `optim.AdamW` (learning rate + weight decay from the method's config);
- compute gradients with `nn.value_and_grad`, apply with `opt.update`, and force evaluation
  with `mx.eval(net.parameters(), opt.state)`;
- read the trained weights back to numpy via `tree_flatten`, and rebuild for inference with
  `tree_unflatten` — so GPU-trained parameters serialize to **portable numpy** and inference
  is device-agnostic.

Method-specific GPU objectives:

- **BC** — `nn.losses.cross_entropy` on `MLXMLP(... n_actions)`, with validation-driven early
  stopping and best-parameter restore.
- **FQE** — fitted iteration: bootstrap target `stop_gradient(r + gamma * not_term *
  Q(s', a'))` (a' from the target policy), MSE on the taken action via `take_along_axis`;
  reports Bellman MAE.
- **CQL** — `0.5 * bellman_MSE + alpha * mean(logsumexp_a Q(s,a) - Q(s, a_logged))` using
  `mx.logsumexp`.
- **IQL** — three MLX networks (`qnet`, `vnet`, `pnet`) with separate AdamW optimizers:
  Q-regression to `r + gamma * V(s')`, **expectile** value regression (`where(u<0, 1-tau,
  tau)`), then advantage-weighted policy cross-entropy with clipped weights
  (`minimum(exp(minimum(beta*adv, 30)), weight_clip)`).

## CPU fallback path (numpy)

Each `_fit_numpy` is a faithful, bullet-proof numpy mirror of its GPU counterpart — the
**observable** fallback, not a stub:

- **BC** → `np_softmax_regression` (multinomial logistic regression).
- **FQE** → linear Q `S @ W + b` with fitted-iteration bootstrap targets and an inner
  gradient loop (ridge-regularized), reporting Bellman MAE.
- **CQL** → linear Q with the conservative-penalty gradient `alpha * (softmax(Q) - onehot) +
  TD`, using `np_logsumexp` for the reported penalty.
- **IQL** → linear Q, expectile-weighted value regression, and AWR via
  `np_softmax_regression(sample_weight=weights)`.

Both paths produce the same result dataclasses and the same advisory `recommend(...)` output,
so a model trained on GPU and one trained on CPU are interchangeable downstream. The
numeric difference between paths on the shared synthetic fixtures is what the GPU-vs-CPU
parity checks assert.

## Synthetic-fixture validation (why, and what it does not claim)

Because `offline_rl_execution` and `policy_bandit` are INSUFFICIENT_DATA, there is no genuine
dataset to train on. The GPU/CPU machinery is instead validated on `make_synthetic_dataset`
(`transitions.py`), which is **always labeled `synthetic=True`**. Consequences enforced in
code:

- `TrainingGuard` marks every result on synthetic data research-only and **non-promotable**
  (`promotable` requires `VALIDATION_ELIGIBLE` AND `not synthetic`).
- The fixtures prove the networks *learn signal* (state-dependent logged actions so BC beats
  chance; state+action-dependent rewards so value learning has gradient) and that the
  **mandatory-exit label is excluded** from the learnable action space — but no profitability
  or superiority may be inferred from synthetic data.

## Guarantees this layer upholds

- GPU acceleration is **claimed only when it happened**; every placement is recorded and
  summarizable, and small workloads honestly run on CPU.
- MLX is optional — absence degrades to CPU, never a crash.
- Point-in-time correctness: the standardizer is fit on TRAIN states only.
- The mandatory -0.5% protective exit is environment dynamics, independent of
  MLX/GPU/training/inference, and is never a learnable or recommendable action.
- No orders, no bypass of the native financial authority, no auto-promotion to live; model
  maturity (`cengine/learning/registry/maturity.py`, `UNTRAINED..RETIRED`) is governed
  separately and M6 does not promote to live.
