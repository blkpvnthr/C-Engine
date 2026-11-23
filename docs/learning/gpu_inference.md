# GPU Inference (Milestone 6)

Batched cross-sectional inference on the Apple-silicon Metal GPU via MLX, with a
**documented, measurable** CPU fallback for small workloads. Nothing here is silent: every
device placement is recorded with a reason, and the benchmark harness reports the batch
sizes where the GPU is actually *slower* rather than hiding them. The GPU path is strictly
off the trading-critical path and can never block the mandatory protective exit.

Environment: GPU-first via MLX 0.32.4 on Apple M5 Max (Metal). Local only — Parquet /
PyArrow / DuckDB + MLX; no cloud, no CUDA, no external model APIs.

Source modules:

- `cengine/learning/hardware.py` — `inspect_hardware`, `mlx_available`, `DevicePolicy`.
- `cengine/learning/supervised/mlx_models.py` — MLX models + correctly-synced forward pass.
- `cengine/learning/inference/service.py` — `LearningInferenceService.predict_batch`.
- `cengine/learning/benchmarks/gpu_benchmarks.py` — CPU-vs-GPU benchmark certification.

---

## 1. Device policy: GPU-first with a measurable CPU fallback

`cengine/learning/hardware.py` (`HARDWARE_REPORT_VERSION = "learning-hardware-1.0.0"`).

`DevicePolicy(*, prefer_gpu=True, min_gpu_elements=50_000, force_cpu=None)` decides, per
workload, where computation runs. `gpu_available = mlx_available() and not force_cpu`;
`force_cpu` defaults from the `CENGINE_LEARNING_FORCE_CPU` env var (any value other than
empty/`"0"` forces CPU).

`choose(workload, *, n_elements=10**9, gpu_supported=True)` returns a `PlacementDecision`
(`workload`, `device`, `reason`, `gpu_eligible`, `fell_back`) in this precedence:

1. `force_cpu` → CPU, reason `"forced_cpu (config/env)"`.
2. MLX/Metal unavailable → CPU, reason records the import error (`fell_back=True`).
3. Op unsupported on GPU → CPU (`fell_back=True`).
4. `prefer_gpu=False` → CPU, reason `"policy prefers cpu"`.
5. `n_elements < min_gpu_elements` → CPU, reason
   `"small workload N < 50000 (sync-dominated)"` — small batches are dominated by
   host/device sync, so the GPU is counter-productive.
6. Otherwise → GPU, reason `"gpu-first: N elements"`.

Every decision is appended to `policy.decisions` and fallbacks counted, so `summary()`
proves where work ran: `{gpu_available, force_cpu, min_gpu_elements, placements,
fallback_count}`. This is what the `trade.py` dashboard surfaces — the platform never
claims GPU acceleration for work that ran on CPU.

`inspect_hardware()` probes the **actual** runtime (no hardcoded specs): device name, GPU
architecture, unified memory, and which float precisions actually round-trip through a real
MLX GPU op (`float32`/`float16`/`bfloat16`). If MLX is absent it degrades to CPU-only and
reports `mlx_metal_available = False` rather than raising. Run it with `python -m
cengine.learning hardware`.

---

## 2. Batched cross-sectional inference

The cross-sectional scoring path is `LearningInferenceService.predict_batch` in
`cengine/learning/inference/service.py`:

```python
predict_batch(task, feature_matrix, symbols, feature_schema_version, *,
              now_ns=None, model_id=None) -> list[InferenceResponse]
```

It scores the entire candidate universe in **one model load and one forward pass**:

- Resolve the serving card once (champion for the task, or a pinned `model_id`). If there
  is no serving model (`maturity` not in the serving set), **every** symbol gets an
  `abstain(..., reason="no_serving_model")` response — a uniform, safe result.
- `_get_model(card)` returns the cached model (loaded once; see §3).
- `model.predict_with_uncertainty(X)` runs a single vectorized pass over the stacked
  `float32` feature matrix; the per-symbol `InferenceResponse`s are built from the output
  rows, all carrying the model's actual `device`.
- Any exception during the batch abstains for **all** symbols with
  `batch_inference_error:<ExceptionType>` and increments `fallbacks` — it never raises.

Inside the MLX models (`mlx_models.py`), the forward pass is placement-aware and correctly
synchronized: `_forward_np` calls `self.policy.choose("infer_<model_type>",
n_elements=len(X) * n_features)`, runs under `mx.stream(self.policy.mlx_device(decision))`,
and forces completion with `mx.eval(out)` before returning. The element count
(`rows × features`) is exactly what the `min_gpu_elements` threshold is compared against —
so a small universe scores on CPU and a large one on the GPU, by measured policy.

The quantile regressor's `predict_with_uncertainty` returns the median quantile as the
point estimate and the half-spread between the outer quantiles as the uncertainty, giving
every batched prediction a calibrated uncertainty band.

---

## 3. Persistent model cache

`LearningInferenceService._get_model` caches loaded models in `self._cache`, keyed by
`(model_id, model_version, revision)`. A champion is reconstructed from its verified
artifact (`load_model`, sha256-checked) **once** and reused across every `predict` and
`predict_batch` call. This keeps steady-state inference latency at the forward-pass cost
only — artifact I/O, checksum verification, and MLX weight loading are paid a single time.
`dashboard_stats()` exposes `cached_models` alongside the device-policy summary.

---

## 4. Benchmark certification: measured p50/p95/p99

`cengine/learning/benchmarks/gpu_benchmarks.py`
(`BENCHMARK_VERSION = "gpu-benchmarks-1.0.0"`) measures **honest** performance: warm-up and
compilation are excluded, and `mx.eval()` forces completion before the timer stops so
asynchronous dispatch is never mistaken for finished work. Run with
`python -m cengine.learning benchmark`, or call `run_all_benchmarks()`.

### Inference latency

`benchmark_inference_latency(*, n_features=13, iters=50, seed=0)` sweeps the batch sizes
`(1, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096)`. For each it reports, per device,
a row with:

- `gpu_p50_ms`, `gpu_p95_ms`, `gpu_p99_ms` (GPU latency percentiles),
- `cpu_p50_ms` (CPU median),
- `gpu_throughput_rows_s`,
- `faster_device` — `"gpu"` or `"cpu"` by measured median.

The output carries an explicit `crossover_note`: **CPU wins at tiny batches (sync-
dominated); the GPU wins at large batches** — reported, not hidden. This is the empirical
basis for the `DevicePolicy` `min_gpu_elements` threshold: the crossover is where batched
GPU inference starts beating CPU, and small batches are deliberately routed to CPU.

Percentiles are computed with `numpy.percentile` over `iters` timed runs, after a warm-up
pass that is excluded. Because exact numbers depend on the host (this environment targets
Apple M5 Max / Metal), the harness measures them at run time rather than asserting fixed
values — re-run `python -m cengine.learning benchmark` to regenerate the p50/p95/p99 table
for the current machine.

### Training throughput and memory

`benchmark_training_throughput(...)` reports GPU vs CPU examples/sec with a correctly
separated warm-up and per-epoch `mx.eval()` sync, plus `gpu_speedup_x` and
`gpu_peak_mem_gib`. `benchmark_memory(...)` records peak unified-memory use across matrix
sizes. All three fall back to a clearly-marked `"skipped": "mlx/metal unavailable"` when
MLX is absent, never fabricating numbers.

---

## 5. The GPU path never blocks the protective exit

The protective exit is owned by the independent `ProtectiveExitCoordinator`
(`protective_exit.py`), running on its own periodic backstop task with threshold
`MARKET_SCAN_LOSS_THRESHOLD` (default `-0.005`). The mandatory `-0.5%` protective removal is
environment dynamics — independent of MLX, GPU, training, and inference.

GPU inference is confined to the detached learning loop (`trade.py::_run_learning_cycle`),
which is stop-driven, excluded from the engine's `FIRST_COMPLETED` wait set, and runs any
heavy GPU work in a thread executor. There is no code path by which batching a
cross-sectional GPU inference, warming a model, or running a benchmark can delay, preempt,
or block the protective exit. If MLX is unavailable or a GPU op fails, the `DevicePolicy`
records a CPU fallback and inference continues on CPU — and the protective exit is
unaffected either way, because it never calls the inference service at all.
