# Unified Memory Management (Milestone 6)

How the learning layer moves data from disk to the Metal GPU on Apple silicon, and the
rules it follows to keep memory bounded and honest. The target hardware is an Apple M5
Max with unified memory, driven through **MLX 0.32.4 (Metal)**. Everything is local:
Parquet / PyArrow / DuckDB + MLX — no cloud, no CUDA, no external model APIs.

The guiding principle is **measure, do not assume**. On unified-memory Apple silicon the
CPU and GPU address the same physical RAM, so there is no PCIe host→device transfer
between an `mx.cpu` stream and an `mx.gpu` stream. But that does **not** make every hop
free: converting between Arrow, numpy, and MLX allocators can and does copy. The platform
never assumes zero-copy — it bounds batch sizes, converts explicitly, and benchmarks
**peak** memory with MLX's own counters.

## The data path

```
Parquet (local SSD, zstd, 128k row-groups)
  └─ DuckDB read_parquet(glob, union_by_name=true) + look-ahead-free window SQL
       └─ PyArrow Table            (con.sql(...).to_arrow_table())
            └─ numpy columns        (Table.column(c).to_numpy(zero_copy_only=False))
                 └─ float32 matrix   (np.column_stack → (n, n_features))
                      └─ MLX array    (mx.array(np.asarray(X, dtype=np.float32)))
                           └─ Metal   (with mx.stream(mx.gpu); mx.eval(...))
```

### 1. Parquet at rest
`cengine/lakehouse/config.py` pins the physical layout: `compression="zstd"`
(level 3, snappy fallback), `row_group_size=128_000`, `write_batch_size=10_000`. The
lakehouse root is configurable via `CENGINE_LAKEHOUSE_ROOT` so data can live on an
external local SSD without a code change. Columnar row-groups mean DuckDB only materializes
the columns and ranges it needs.

### 2. DuckDB → PyArrow
`cengine/learning/features/builder.py` builds the training matrix with DuckDB window
functions directly over the Parquet glob (`read_parquet('<glob>', union_by_name=true)`),
partitioned by `(symbol_id, session)` so windows never span instruments or overnight
gaps. The result is handed back as an Arrow table:

```python
tbl = con.sql(sql).to_arrow_table()
```

DuckDB streams the Parquet scan and computes returns / momentum / volatility / SMA ratios
/ RSI / VWAP distance (13 columns, `BAR_FEATURE_COLUMNS` in
`cengine/learning/features/schema.py`) plus the forward-return label in one pass, so the
engine never holds the full raw history as Python objects. The same pattern backs
`gold_training_frame`, the dataset profiler (`datasets/eligibility.py`), and the replay
loader (`environment/replay.py`).

### 3. PyArrow → numpy (copies are explicit)
Columns are pulled into numpy **with copies explicitly allowed**:

```python
data = {c: tbl.column(c).to_numpy(zero_copy_only=False) for c in cols}
```

`zero_copy_only=False` is deliberate and honest: numeric columns may be zero-copy when
Arrow's buffer layout permits, but any column that is chunked, nullable, or non-primitive
(the `symbol_id` object column) **will** copy. The code does not pretend otherwise. The
feature matrix is then assembled as a contiguous `float32` block and filtered for
non-finite rows:

```python
X = np.column_stack([np.asarray(data[c], dtype=np.float32) for c in schema.columns])
finite = np.isfinite(X).all(axis=1) & np.isfinite(y_return)
```

`np.column_stack` produces one more contiguous copy; this is the dominant host-side
allocation and is sized `n × 13 × 4 bytes`.

### 4. numpy → MLX → Metal
Training (`cengine/learning/supervised/mlx_models.py`) moves the host matrix into the MLX
unified-memory allocator once, then indexes minibatches from it:

```python
Xd = mx.array(np.asarray(X, dtype=np.float32))   # into MLX's allocator
yd = mx.array(np.asarray(y, dtype=np.float32))
with mx.stream(dev):                              # dev = mx.gpu or mx.cpu (DevicePolicy)
    ...
    xb, yb = Xs[b:b + bs], ys[b:b + bs]           # bounded minibatch views
    loss, grads = lvg(net, xb, yb)
    opt.update(net, grads)
    mx.eval(net.parameters(), opt.state)          # force materialization, not just dispatch
```

`mx.array(...)` copies the numpy buffer into MLX's allocator — treated as a copy, not
assumed free. Because MLX is lazy, `mx.eval()` is the point at which Metal actually
executes and memory is genuinely resident; the benchmarks rely on this to attribute peak
usage correctly (see below). The `with mx.stream(dev)` block is how the same code path
runs on the GPU or the CPU under `DevicePolicy` without a host↔device transfer on unified
memory.

### 5. Inference
`cengine/learning/inference/service.py` keeps model weights resident in a per-process
cache keyed by `(model_id, model_version, revision)` — one load, reused across requests.
`predict_batch` runs cross-sectional inference over a whole universe as **one model load
and one GPU pass**: the feature matrix is coerced to a single `float32` array and fed in
one shot rather than looped per symbol.

## Bounded batch loading

Peak memory is bounded at every stage rather than being left to chance:

- **Ingest buffer.** The live lakehouse write buffer is a bounded `deque` sized by
  `CENGINE_LAKEHOUSE_MARKET_BUF` (default `200000`) and drained out of band; writes use
  `write_batch_size` (`CENGINE_LAKEHOUSE_BATCH`, default `10000`) row batches.
- **Query.** DuckDB streams the Parquet scan; only the aggregated feature/label frame is
  materialized to Arrow, not the raw bar universe.
- **Training.** Minibatches (`batch_size`, default `4096`) cap the transient
  forward/backward activation footprint; the resident training set is the single `float32`
  matrix plus optimizer state.
- **Model artifacts.** Weights serialize to portable numpy `.npz` via an in-memory
  `BytesIO` buffer (`registry/artifacts.py`), checksummed (sha256) and loaded with
  `allow_pickle=False`; a corrupted or oversized blob is rejected, never streamed blindly
  into the allocator.

## Peak-memory benchmarks

`cengine/learning/benchmarks/gpu_benchmarks.py` (`benchmark_memory`) measures **actual**
peak unified-memory use for matmul workloads across row counts `(1024, 4096, 16384)`:

```python
_reset_peak_mem()                 # mx.reset_peak_memory()
with mx.stream(mx.gpu):
    Y = mx.array(xa) @ mx.array(wa)
    mx.eval(Y)                    # force completion before reading the counter
peak = _peak_mem_gib()            # mx.get_peak_memory() / 1024**3
```

Both counters are probed defensively (`hasattr`) so the benchmark degrades to `None`
rather than raising on an MLX build that lacks them. `benchmark_training_throughput` also
records `gpu_peak_mem_gib` after a reset, around a correctly-synchronized training run
(warm-up/compile excluded; `mx.eval()` before the clock stops). Because measurements are
taken only after `mx.eval()`, asynchronous dispatch is never mistaken for completed,
resident work — the reported peak reflects memory that was really allocated.

`hardware` (`inspect_hardware`, `cengine/learning/hardware.py`) reports the ceilings these
benchmarks run against: unified memory GiB, `max_recommended_working_set_gib`,
`max_buffer_length_gib`, and `memory_size_gib`, all read from the live Metal device —
never hardcoded.

## Why no zero-copy claim

Zero-copy across Arrow → numpy → MLX would require buffer-layout, dtype, alignment, and
lifetime guarantees the platform does not control (chunked Arrow columns, object dtypes,
MLX allocator ownership). Rather than assert it, the code:

1. passes `zero_copy_only=False` so a necessary copy is legal and visible,
2. converts to MLX explicitly with `mx.array(...)`, and
3. verifies the real cost with `reset_peak_memory` / `get_peak_memory` around
   `mx.eval()`.

Unified memory removes the host↔device **transfer** between CPU and GPU streams; it does
not remove **allocator-to-allocator copies**. The distinction is documented and measured,
consistent with the project's GPU-first-but-honest device policy.

## References

- `cengine/lakehouse/config.py` — Parquet layout, compression, batch/row-group sizing
- `cengine/learning/features/builder.py` — DuckDB → Arrow → numpy feature matrix
- `cengine/learning/features/schema.py` — ordered 13-column feature contract
- `cengine/learning/supervised/mlx_models.py` — numpy → MLX → Metal training loop
- `cengine/learning/inference/service.py` — cached weights, single-pass batch inference
- `cengine/learning/registry/artifacts.py` — checksummed npz serialization
- `cengine/learning/benchmarks/gpu_benchmarks.py` — peak-memory + throughput measurement
- `cengine/learning/hardware.py` — live Metal memory ceilings
