# Metal GPU Training with MLX (Milestone 6)

How C-Engine's learning models train on the Apple Silicon **Metal GPU** via
[MLX](https://github.com/ml-explore/mlx) **0.32.4**, on an **Apple M5 Max** (48 GiB unified
memory, `applegpu_g17s`, float32/float16/bfloat16 all validated). This is a **GPU-first**
design with a **documented, measurable CPU fallback** — the platform never silently claims
GPU acceleration when work actually ran on the CPU.

Training is **recommendation infrastructure only**. It produces model weights that a
detached learning loop uses to *rank* already-qualified candidates. Training and inference
submit no orders, never bypass the native financial authority, and have no relationship to
the mandatory **−0.5%** protective exit (environment dynamics, independent of MLX, the GPU,
training, or inference).

Local-only stack: Parquet / PyArrow / DuckDB + MLX. No cloud, no CUDA, no external model
APIs.

## Device policy: GPU-first, honest fallback (`cengine/learning/hardware.py`)

`DevicePolicy` makes every placement decision explicit and records it, so telemetry can
*prove* where computation ran.

- `mlx_available()` is true only if MLX imported **and** a Metal GPU is actually usable.
- `inspect_hardware()` probes the **actual** runtime (sysctl memory, `mx.device_info`,
  and a live precision round-trip that matmuls in each dtype) — nothing is hardcoded.
  `HARDWARE_REPORT_VERSION = "learning-hardware-1.0.0"`.
- `DevicePolicy.choose(workload, n_elements=..., gpu_supported=...)` returns a
  `PlacementDecision(workload, device, reason, gpu_eligible, fell_back)`. The logic, in
  order:

  | Condition | Device | Reason |
  |---|---|---|
  | `force_cpu` (env `CENGINE_LEARNING_FORCE_CPU`) | CPU | `forced_cpu (config/env)` |
  | MLX/Metal unavailable | CPU | `mlx/metal unavailable: …` (counts as fallback) |
  | op not GPU-supported | CPU | `op unsupported on gpu` (fallback) |
  | `prefer_gpu=False` | CPU | `policy prefers cpu` |
  | `n_elements < min_gpu_elements` (default **50,000**) | CPU | `small workload … (sync-dominated)` |
  | otherwise | GPU | `gpu-first: N elements` |

- `mlx_device(decision)` maps a decision to `mx.gpu` / `mx.cpu` (or `None` if MLX absent).
- `summary()` reports `gpu_available`, `force_cpu`, `min_gpu_elements`, per-device placement
  counts, and `fallback_count`.

The small-workload CPU preference is deliberate and measured: for tiny batches, host/device
synchronization dominates and the GPU is *slower* (see [benchmarks](#the-honest-benchmark)).
`CENGINE_LEARNING_FORCE_CPU` (any value other than empty / `0`) forces the whole learning
subsystem onto the CPU — the documented, reproducible fallback used in CI and on non-Metal
machines.

## The training loop (`cengine/learning/supervised/mlx_models.py`)

All MLX models share `_MLXBase.fit(...)`. The core loop:

1. **Seed + device.** `mx.random.seed(cfg.seed)`; `_mx_device()` asks `DevicePolicy.choose`
   for the training workload (sized `n_features * 20_000`) and records `self.device`
   (`"gpu"` or `"cpu"`) — the device that ends up on the `FitResult` and the `ModelCard`.
2. **Standardize on train only.** The `Standardizer` is fit on the training `X`, then applied
   to train and validation (no leakage).
3. **`value_and_grad` autodiff.** `lvg = nn.value_and_grad(net, loss_fn)` builds the fused
   loss-and-gradient function; the per-model `_loss` (MSE / BCE-with-logits / pinball) defines
   the objective. Optimizer is `AdamW(learning_rate=cfg.learning_rate,
   weight_decay=cfg.weight_decay)`.
4. **Everything runs inside `with mx.stream(dev):`** — the chosen device's stream. Per epoch:
   reshuffle via a per-epoch-seeded permutation, iterate minibatches, and for each batch:
   ```python
   loss, grads = lvg(net, xb, yb)
   opt.update(net, grads)
   mx.eval(net.parameters(), opt.state)   # force the lazy graph to actually compute
   ```
   MLX is lazy: `mx.eval(...)` is what forces the GPU/CPU to *complete* the step rather than
   just enqueue it.
5. **Validation + early stopping.** If a validation slice is given, the validation loss is
   evaluated each epoch; the best parameters are snapshotted with `tree_flatten`, and training
   stops after `cfg.patience` (default **8**) epochs without improvement. On exit the best
   params are restored via `tree_unflatten` + `net.update`.
6. **Return `FitResult`** with `train_loss`, `val_loss`, `epochs_run`, `device`, and the
   per-epoch `history`.

`TrainConfig` defaults (`cengine/learning/supervised/base.py`): `epochs=60`,
`batch_size=1024`, `learning_rate=1e-3`, `hidden=(64, 32)`, `weight_decay=1e-4`, `seed=7`,
`quantiles=(0.1, 0.5, 0.9)`, `patience=8`, `standardize=True`.

Inference (`_forward_np`) likewise asks `DevicePolicy.choose` (sized `len(X) * n_features`),
runs under `mx.stream(...)`, and calls `mx.eval(out)` before converting back to numpy — so
small single-row predictions correctly fall to CPU while large cross-sectional batches use
the GPU.

## Device streams and parity

Both device paths run the *same* code under `mx.stream(mx.gpu)` or `mx.stream(mx.cpu)`; MLX
dispatches identical ops to each backend. Because weights are plain arrays and the math is
identical, a model trained on the GPU and one trained on the CPU produce numerically
consistent results (within float32 tolerance), and a GPU-trained model serves correctly from
either device. This CPU/GPU parity is what makes the fallback *safe*: it is the same model,
not a degraded approximation. The `force_cpu` switch exercises exactly this path.

## Weight serialization: portable npz (`cengine/learning/registry/artifacts.py`)

MLX weights are serialized to **portable numpy `.npz`** so artifacts are framework- and
device-independent and survive round-trips through the checksummed store:

- `_MLXBase.params()` flattens the MLX parameter tree (`tree_flatten`) into
  `{"w::<path>": np.array(...)}`, plus the standardizer (`std::mean` / `std::std`) and a JSON
  `meta::config` (hidden sizes, `n_features`, `n_out`, quantiles) encoded as a uint8 array.
- `ArtifactStore.save_npz(...)` writes atomically (temp file → `flush` → `os.fsync` →
  `os.replace`), names the blob `<version>-<sha256[:16]>.npz`, and is idempotent on identical
  content. Blobs live under `<lakehouse_root>/_learning_models/` and are never committed to
  Git.
- Loading verifies the **sha256 content checksum** before anything is parsed; a corrupted or
  tampered blob raises `ArtifactVerificationError` and is rejected. `load_npz` uses
  `allow_pickle=False` (npz is pure arrays — no code execution on load).
- `_MLXBase.load_params(...)` rebuilds the MLX tree with `tree_unflatten` + `net.update` and
  restores the standardizer.

## The honest benchmark (`cengine/learning/benchmarks/gpu_benchmarks.py`)

The benchmarks are **correctly synchronized**: warm-up/compilation is run once and excluded
from timing, and `mx.eval(...)` forces completion before the clock stops — asynchronous
dispatch is never mistimed as finished work. `BENCHMARK_VERSION = "gpu-benchmarks-1.0.0"`.
Results are reported **including the cases where the GPU is slower**.

### Training throughput (small MLP → GPU is only CPU-comparable)

The learning MLPs are small (13 inputs, hidden `(64, 32)`). On the M5 Max
(`reports/learning_hardware_benchmarks.json`, `reports/mlx_supervised_training_benchmarks.json`):

| Workload | GPU | CPU | GPU speedup |
|---|---|---|---|
| 50,000 × 13, 3 epochs, batch 2048 | 0.0324 s | 0.0386 s | **1.19×** |
| 40,000 × 13, 3 epochs, batch 2048 | 0.0206 s | 0.0259 s | **1.26×** |

So for these small models the GPU is only ~1.2–1.3× faster — **CPU-comparable, not a dramatic
win**. This is reported honestly rather than inflated. The GPU-first design still pays off for
larger cross-sectional inference and the heavier learning workloads (bandits / offline RL),
and the unified-memory headroom (48 GiB; peak training footprint here ≈ 0.02 GiB) leaves
room to grow.

### Inference latency crossover (CPU wins until batch 4096)

For the small MLP, measured p50 latency per batch size shows the CPU faster at every batch
size up to 2048, with the GPU only overtaking at batch **4096**:

| Batch | GPU p50 (ms) | CPU p50 (ms) | faster |
|---|---|---|---|
| 1 | 0.202 | 0.034 | cpu |
| 32 | 0.169 | 0.029 | cpu |
| 256 | 0.162 | 0.040 | cpu |
| 1024 | 0.172 | 0.079 | cpu |
| 2048 | 0.158 | 0.130 | cpu |
| 4096 | 0.158 | 0.237 | **gpu** |

This crossover is exactly why `DevicePolicy` defaults small workloads to the CPU
(`min_gpu_elements = 50,000`). The benchmark's own note: *"GPU wins at large batches; CPU
wins at tiny batches (sync-dominated) — reported honestly, not hidden."*

## Scope honesty

- **GPU-first with a real fallback.** MLX 0.32.4 on Metal (M5 Max) is the default for large
  workloads; `DevicePolicy` gives a documented, measurable CPU fallback, and small MLPs are
  CPU-comparable — the honest benchmark above is not hidden.
- **Device is recorded, never assumed.** Every fit and inference records the actual device on
  the `FitResult` / `ModelCard`; `DevicePolicy.summary()` and `fallback_count` make silent
  GPU claims impossible.
- **Training is advisory infrastructure.** It produces ranking weights, not orders. It never
  bypasses the native financial authority and never touches the −0.5% protective exit.
- **Local only.** Parquet / PyArrow / DuckDB + MLX; no cloud, no CUDA, no external model APIs.
  Weights serialize to portable, checksum-verified npz.
- **No economic claim.** The one trainable task on real data (`forward_return_prediction`)
  showed a marginal OOS edge and was not promoted — see
  [supervised_learning.md](supervised_learning.md).

## Key source references

- `cengine/learning/hardware.py` — `DevicePolicy`, `inspect_hardware`, `mlx_available`, `PlacementDecision`
- `cengine/learning/supervised/mlx_models.py` — `_MLXBase.fit` (`value_and_grad`, `mx.stream`, early stopping, `params`/`load_params`)
- `cengine/learning/supervised/base.py` — `TrainConfig`, `Standardizer`, `FitResult`
- `cengine/learning/registry/artifacts.py` — `ArtifactStore.save_npz` / `load_npz` (atomic, checksummed, `allow_pickle=False`)
- `cengine/learning/benchmarks/gpu_benchmarks.py` — `benchmark_training_throughput`, `benchmark_inference_latency`, `benchmark_memory`
- `reports/learning_hardware_benchmarks.json`, `reports/mlx_supervised_training_benchmarks.json` — the honest numbers
