# Milestone 6 — MLX / Metal Runtime Architecture

The learning layer's primary compute runtime is **MLX 0.32.4** on **Apple M5 Max**
(Metal). MLX gives the learning layer GPU-accelerated autodiff over Apple-silicon unified
memory. This document covers how MLX is wired in, the GPU-first `DevicePolicy` with its
documented CPU fallback, the import-guard pattern that lets every module import on a
non-MLX host, and which workloads run on MLX versus the independent sklearn / numpy
baselines.

> This MLX runtime is **separate from and coexists with** the native Objective-C++ Metal
> runtime documented in [`../metal_runtime_architecture.md`](../metal_runtime_architecture.md).
> The native runtime (loaded `cengine.metallib`) serves operational indicator / covariance
> / discovery kernels; MLX serves the learning layer's neural training and inference. They
> share the same physical Metal GPU but are independent software stacks.
> `inspect_hardware()` records `native_metallib_present` purely as a coexistence signal.

---

## 1. GPU-first with a documented, measurable CPU fallback

The cardinal hardware invariant: **the platform never silently claims GPU acceleration
when work ran on the CPU.** Every placement decision is recorded with a reason and
surfaced to callers / telemetry. Small, latency-sensitive, or unsupported workloads run on
the CPU *on purpose*, and that choice is observable — it is a documented fallback, not a
degraded failure.

This is enforced by `DevicePolicy` in `cengine/learning/hardware.py`.

### `mlx_available()` — the ground-truth probe

`mlx_available()` returns `True` only when MLX imported **and** a Metal GPU is actually
usable (`mx.metal.is_available()`, with a newer-MLX `default_device()` fallback). It is
the single source of truth every module consults — a successful `import mlx` is not
treated as "GPU present".

### `inspect_hardware()` → `HardwareCapabilities`

`inspect_hardware()` inspects the **actual** runtime (never hardcoded specs):
platform / machine / Apple-silicon detection, unified-memory size via `sysctl`,
`mlx_installed` / `mlx_metal_available` / `mlx_import_error`, device name, GPU
architecture, working-set / buffer / memory sizes from `mx.device_info()`, and
`supported_precisions` — which it probes empirically by round-tripping a `float32` /
`float16` / `bfloat16` matmul through the GPU (`_validate_precisions`) rather than
assuming. `report_version = learning-hardware-1.0.0`.

Surfaced on the CLI via `python -m cengine.learning hardware`.

### `DevicePolicy.choose()` → `PlacementDecision`

```python
policy = DevicePolicy(prefer_gpu=True, min_gpu_elements=50_000)
d = policy.choose("train_mlx_quantile_regressor", n_elements=rows * n_features)
# d: PlacementDecision(workload, device, reason, gpu_eligible, fell_back)
mx_dev = policy.mlx_device(d)   # mx.gpu | mx.cpu | None
```

`choose()` resolves placement in this priority order, recording a human-readable `reason`
for each:

| Condition | Device | Reason recorded | `fell_back` |
|---|---|---|---|
| `force_cpu` (config/env) | CPU | `forced_cpu (config/env)` | no |
| MLX/Metal unavailable | CPU | `mlx/metal unavailable: <err>` | **yes** |
| op not GPU-supported | CPU | `op unsupported on gpu` | **yes** |
| `prefer_gpu=False` | CPU | `policy prefers cpu` | no |
| `n_elements < min_gpu_elements` | CPU | `small workload … (sync-dominated)` | no |
| otherwise | **GPU** | `gpu-first: <n> elements` | no |

The `min_gpu_elements` default (**50,000**) encodes the physical reality that tiny
workloads are dominated by host/device synchronization, so GPU is counter-productive
below that size. Every decision is appended to `policy.decisions`, `fallback_count` tracks
genuine fallbacks, and `policy.summary()` reports `{gpu_available, force_cpu,
min_gpu_elements, placements, fallback_count}` — which the inference service exposes in its
`dashboard_stats()`.

`CENGINE_LEARNING_FORCE_CPU` (non-empty, non-`0`) forces every placement to CPU. This is
the explicit, auditable way to disable GPU for a run.

---

## 2. The import-guard pattern

MLX is **optional at import time**. Every module that uses MLX wraps the import so the
module still imports on a non-arm64 / non-Darwin host (or where MLX is simply not
installed), and degrades to CPU / baselines instead of crashing. The canonical form
(`cengine/learning/hardware.py`, `supervised/mlx_models.py`, `contextual_bandits/neural_bandit.py`,
`offline_rl/_nets.py`, `benchmarks/gpu_benchmarks.py`):

```python
try:  # pragma: no cover - import guard
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx.utils import tree_flatten, tree_unflatten
    _MLX = True
except Exception:              # absence must degrade, not crash
    mx = None
    nn = object                # so `class X(nn.Module)` is skipped, not a NameError
    _MLX = False

def _require_mlx() -> None:
    if not _MLX:
        raise RuntimeError("MLX is not available; use a CPU baseline instead")
```

Consequences of the pattern:

- Neural model classes (`_MLP`, `_BanditNet`, `MLXMLP`) are defined only inside
  `if _MLX:`; a stub that calls `_require_mlx()` stands in otherwise. **Importing** the
  module never requires MLX; **constructing** a neural model on a non-MLX host raises a
  clear, typed error.
- `hardware.py` reports `mlx_available = False` and `mlx_import_error` instead of raising,
  so the rest of the engine keeps working.
- The trainer (`supervised/trainer._build_champion`) treats a missing MLX / construction
  failure as a signal to fall back to a CPU baseline *as the champion* rather than
  aborting the daily cycle.

---

## 3. How MLX models train and infer

All MLX models follow the same reference pattern (see `supervised/mlx_models._MLXBase`).

**Architecture.** A configurable ReLU MLP (`_MLP`) with a linear output head. Hidden
layers default to `(64, 32)` via `TrainConfig`. Heads:
- `MLXRegressor` → 1 output, MSE loss.
- `MLXBinaryClassifier` → 1 logit, numerically-stable BCE-with-logits; `predict` returns
  `P(y=1)`.
- `MLXQuantileRegressor` → one output per quantile (default `0.1, 0.5, 0.9`), **pinball
  loss**; yields a point prediction (median), an uncertainty band (half the
  high–low spread via `predict_with_uncertainty`), and `predict_downside` (lowest
  quantile). This is the champion architecture for `forward_return_prediction`.

**Training loop.** `mx.random.seed(cfg.seed)` → device chosen by `DevicePolicy` (workload
sized `n_features * 20_000`) → standardize X (fit on TRAIN only) → `nn.value_and_grad` with
`optim.AdamW(weight_decay)` → per-epoch shuffled mini-batches under `with mx.stream(dev):`
→ `opt.update(net, grads)` then `mx.eval(net.parameters(), opt.state)` to force
completion → early stopping on validation loss with `patience`, keeping best params via
`tree_flatten` / `tree_unflatten`. The `FitResult` records the **actual** `device` used.

**Inference.** `_forward_np` standardizes, re-asks `DevicePolicy` for placement sized
`len(X) * n_features` (so a small single-row request is correctly steered to CPU while a
cross-sectional universe batch goes to GPU), runs under `mx.stream(...)`, and calls
`mx.eval(out)` before converting back to numpy — never reporting async dispatch as
completed work.

**Serialization.** `params()` flattens the net to portable numpy arrays (`w::*` keys),
stores the standardizer (`std::mean` / `std::std`) and a JSON `meta::config` (hidden dims,
n_features, n_out, quantiles). These serialize via `ArtifactStore.save_npz` with
`allow_pickle=False` and a sha256 checksum, so weights are framework-portable (loadable on
CPU or GPU) and tamper-evident. `load_model` (`inference/loader.py`) rebuilds the exact
architecture from `meta::config` before `load_params`.

The contextual-bandit (`NeuralBanditRewardModel`) and offline-RL learners (`BehaviorCloner`,
`FittedQEvaluator`, `ConservativeQLearner`, `ImplicitQLearner`) reuse this same pattern —
`DevicePolicy`-chosen device recorded on every fit, `nn.value_and_grad` / `opt.update` /
`mx.eval` under `mx.stream`, numpy-portable weights — with task-specific losses (e.g. the
bandit's per-action masked MSE where only the taken action's head receives gradient).

---

## 4. Where MLX is used vs. where it is not

A neural MLX model is **never preferred merely for running on the GPU.** It must beat an
independent, deliberately simple CPU baseline out-of-sample before promotion. Some
workloads are intentionally CPU-only because they are the wrong shape for a GPU.

| Workload | Runtime | Why |
|---|---|---|
| Supervised neural regression / classification / quantile (`supervised/mlx_models.py`) | **MLX GPU** (CPU fallback) | Large batched autodiff — GPU-first. Champion for `forward_return_prediction`. |
| Neural contextual bandit (`contextual_bandits/neural_bandit.py`) | **MLX GPU** (CPU fallback) | Batched per-action reward MLP. |
| Offline RL — BC / FQE / CQL / IQL (`offline_rl/`) | **MLX GPU** (numpy CPU fallback) | Batched Q / policy nets; `_RLBase` records the device. |
| Independent baselines (`supervised/baselines.py`) | **CPU** (sklearn HistGradientBoosting, Ridge, logistic, constant / historical-mean) | The adversary the GPU model must beat; `LinUCB` is the bandit baseline. |
| Regime transition model (`regime_models/transition_model.py`) | **CPU** (`gpu_supported=False`) | Tiny sequential transition-counting; sync would dominate. Pure numpy/duckdb. |
| Calibration (`calibration/`) | **CPU** (`CALIBRATION_DEVICE`) | numpy / scikit-learn post-processing; not an MLX kernel. |
| Economic evaluation + OPE (`evaluation/`) | **CPU** | numpy / scipy bootstrap + histogram ops. |
| Drift monitoring (`monitoring/drift.py`) | **CPU** (`gpu_supported=False`) | PSI / KL / JS histogram divergences. |
| Replay environment (`environment/replay.py`) | **CPU** (DuckDB + numpy) | Strictly sequential bar-by-bar simulation. |
| Benchmarks (`benchmarks/gpu_benchmarks.py`) | **Both** (measures GPU vs CPU) | Explicitly times both to report honest crossover. |

The trainer's champion-selection (`supervised/trainer._build_champion`) only picks an MLX
model when `mlx_is_available()`; otherwise a sklearn baseline becomes the champion. Either
way the OOS comparison against the baseline (`_beats`) gates promotion, and the resulting
`ModelCard` records the real `framework`, `framework_version`, `training_device`, and
`inference_device` — so the registry always reflects where compute actually ran.

---

## 5. Honest benchmarking

`benchmarks/gpu_benchmarks.py` certifies performance **correctly**: it forces lazy MLX
computation to complete with `mx.eval()` before stopping the timer, separates
warm-up/compilation from steady-state, and never reports asynchronous dispatch as finished
work. Crucially it reports real acceleration factors **including workloads where the GPU is
slower** — tiny / sync-dominated batches where CPU wins — rather than hiding them
(`benchmark_inference_latency` emits a `faster_device` per batch size and a `crossover_note`).
When MLX is unavailable it emits clearly-marked CPU-only / skipped results. Run via
`python -m cengine.learning benchmark`.

This closes the loop on the GPU-first invariant: the device policy *decides* placement by
workload size, and the benchmarks *prove* that the chosen crossover (`min_gpu_elements`) is
the right call on the actual hardware — never a silent, unverified GPU claim.
