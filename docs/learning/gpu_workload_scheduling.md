# GPU Workload Scheduling (Milestone 6)

How GPU (Metal/MLX) learning work is scheduled relative to the live trading engine, and
why it can never slow, block, or gate the order path or the mandatory **−0.5% protective
exit**. The design rule is absolute: learning is **recommendation-only** and runs **off
the trading-critical path**. GPU-first execution on the Apple M5 Max (MLX 0.32.4, Metal)
is a performance choice for training and batched inference — it is never on the latency
path of a protective exit.

## Three facts this document guarantees

1. The learning loop runs **detached** from the asyncio event loop's completion set.
2. GPU **training** runs in a worker thread via `run_in_executor`, never on the event
   loop, and is **off by default in-session**.
3. The **native Metal runtime** (the compiled C++ `cengine.metallib`) and MLX coexist on
   the same device; that coexistence is **measured, not assumed**, and the protective exit
   never waits on GPU work.

## Detached learning loop

The engine (`trade.py`) creates the learning loop as a named asyncio task but deliberately
keeps it **out** of the `FIRST_COMPLETED` wait set that governs engine lifecycle:

```python
# trade.py ~1734 — created, stored on self to avoid GC, cancelled during teardown
self._learning_task = asyncio.create_task(
    self._run_learning_cycle(self._stop), name="learning-cycle")
...
# trade.py ~1749 — the lifecycle wait set; note learning-cycle is NOT included
tasks = [consumer_task, stream_task, reconciliation_task]
if protective_task is not None:
    tasks.append(protective_task)
...
done, pending = await asyncio.wait([*tasks, stop_task],
                                   return_when=asyncio.FIRST_COMPLETED)
```

Consequences, by construction:

- The learning loop **cannot trigger shutdown** — if it errors or finishes, it is not in
  the set that `asyncio.wait(FIRST_COMPLETED)` watches.
- Shutdown **does not wait on it** — it is cancelled during teardown, not awaited as a
  critical task.
- The **protective exit task is in the critical set** (`tasks.append(protective_task)`),
  alongside the market consumer, stream, and reconciliation. Its scheduling is entirely
  independent of the learning loop.

The loop is **stop-driven**: it sleeps on the engine stop event with a timeout rather than
a bare `sleep`, so teardown is immediate:

```python
async def _run_learning_cycle(self, stop):
    interval = float(os.environ.get("CENGINE_LEARNING_CYCLE_S", "3600"))
    train_in_session = os.environ.get("CENGINE_LEARNING_TRAINING", "0") != "0"
    while not stop.is_set():
        try:
            await asyncio.wait_for(stop.wait(), timeout=interval)
        except asyncio.TimeoutError:
            pass
        except asyncio.CancelledError:
            break
        if not train_in_session:
            continue
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._run_learning_cycle_sync)
```

The sibling lakehouse drain loop (`_run_lakehouse_drain`, task `lakehouse-drain`) follows
the exact same detached, stop-driven pattern, and its flush also goes through
`run_in_executor`.

## Training off the event loop, off by default in-session

Heavy GPU training must never occupy the single-threaded asyncio event loop that also
services market events. Two mechanisms enforce this:

1. **Default off in-session.** `CENGINE_LEARNING_TRAINING` defaults to `0`. With it unset,
   the loop wakes on its interval and immediately `continue`s — it keeps the shadow
   inference service resident but runs no training. Heavy training belongs to the
   out-of-session `python -m cengine.learning daily-cycle` (see
   [operations.md](operations.md)).
2. **Thread executor when enabled.** When `CENGINE_LEARNING_TRAINING=1`, the blocking
   daily-cycle body runs in a thread pool via `await loop.run_in_executor(None, ...)`, so
   the MLX/Metal training work (`DailyLearningCycle(...).run()` in `_run_learning_cycle_sync`)
   executes on a worker thread. The event loop stays free to process bars, routing, and —
   critically — the protective exit.

```python
def _run_learning_cycle_sync(self):
    # blocking body; runs in a thread executor, never the event loop
    from cengine.learning.orchestration.daily_cycle import DailyLearningCycle
    DailyLearningCycle(decision_sink=getattr(self, "_decision_sink", None)).run()
```

`CENGINE_LEARNING_CYCLE_S` (default `3600`) sets the wake interval. The whole learning
layer can be disabled with `CENGINE_LEARNING=0`, in which case neither the service nor the
loop is created.

## GPU-first placement with a measured CPU fallback

Where a given workload runs is decided by `DevicePolicy`
(`cengine/learning/hardware.py`), not by assumption. It is GPU-first for large workloads
and falls back to the CPU for small, sync-dominated ones — and records **every** decision
with a reason so telemetry can prove where computation actually ran:

- `min_gpu_elements` (default `50_000`): below this element count, CPU is chosen because
  host/device synchronization dominates and the GPU would be counter-productive. Reason:
  `small workload N < 50000 (sync-dominated)`.
- `CENGINE_LEARNING_FORCE_CPU`: any non-empty value other than `0` forces CPU everywhere
  (reason `forced_cpu`), for deterministic reproduction and tests.
- MLX/Metal unavailable, or an unsupported op: recorded as a fallback
  (`fell_back=True`), never a silent GPU claim.

`DevicePolicy.summary()` exposes placement counts and `fallback_count`; the benchmark
harness (`benchmarks/gpu_benchmarks.py`) reports the crossover honestly, including batch
sizes where the **CPU is faster** than the GPU. The engine never claims GPU acceleration
for work that ran on the CPU.

## Native Metal runtime coexistence — measured, not assumed

C-Engine has two consumers of the same Apple Metal device: the compiled **native** C++
Metal runtime (`build-metal/cengine.metallib`, from the earlier native milestones) and the
**MLX** learning layer. The platform does not assume they coexist cleanly — it measures:

- `inspect_hardware()` records `native_metallib_present` (whether the native metallib is
  compiled and on disk) alongside the live MLX/Metal capability probe, so a report shows
  both runtimes against the same device.
- `_validate_precisions()` round-trips float32/float16/bfloat16 through a real MLX Metal
  matmul and reports only the precisions that actually execute — a live probe, not a spec
  sheet.
- The GPU benchmarks run correctly-synchronized MLX workloads (warm-up/compile excluded,
  `mx.eval()` before timing stops) so observed throughput and peak memory reflect real
  concurrent use, not optimistic dispatch.

Because learning GPU work is detached and, when training, confined to a worker thread, it
contends for the Metal device only with itself and whatever native work the engine
schedules — and that contention is observable in the benchmarks and placement telemetry
rather than hidden.

## The protective exit never waits on GPU work

The mandatory −0.5% protective exit is **environment dynamics**, independent of MLX, Metal,
the GPU, model training/inference, the registry, and the lakehouse. Its independence is
structural, not conventional:

- The protective-exit task is in the engine's critical `FIRST_COMPLETED` set; the learning
  loop is not. The event loop services the exit regardless of learning-loop state.
- When training is enabled it runs on a **worker thread** (`run_in_executor`), so even a
  long GPU training pass cannot starve the event loop that drives the exit.
- The inference service (`inference/service.py`) is explicitly invoked only off the
  trading-critical path (shadow comparison / detached loop) and is **never called from**
  the protective exit. It also always returns a reasoned abstention rather than raising.
- In the replay simulator (`environment/replay.py`) the same −0.5% threshold is enforced
  as environment dynamics every bar, never as a learned or policy action — mirroring the
  production invariant.

No reward, forecast, policy, GPU workload, or scheduling decision can defer, suppress, or
delay the protective exit.

## References

- `trade.py` — `_run_learning_cycle` (~2064), `_run_learning_cycle_sync` (~2090), task
  creation (~1736), the `FIRST_COMPLETED` set (~1749)
- `cengine/learning/orchestration/daily_cycle.py` — the blocking cycle body
- `cengine/learning/hardware.py` — `DevicePolicy`, `inspect_hardware`, precision probe
- `cengine/learning/benchmarks/gpu_benchmarks.py` — synchronized CPU/GPU measurement
- `cengine/learning/inference/service.py` — off-critical-path, always-abstain inference
- `cengine/learning/environment/replay.py` — protective exit as environment dynamics
- [operations.md](operations.md) — CLI and the `CENGINE_LEARNING*` env gates
- [unified_memory_management.md](unified_memory_management.md) — the data-to-Metal path
