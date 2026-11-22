# Native Metal Runtime Architecture (Phase A)

The native Objective-C++ Metal runtime that replaces the PyObjC host for
operational GPU work. One runtime, shared by all callers (directive §4: "There
must not be multiple competing operational Metal runtimes").

## Layers

```
Python supervisor (trade.py, pyenv 3.11.8)
        │  batched calls, GIL released during dispatch
        ▼
_cengine_native  (pybind11)                         bindings/discovery_bindings.cpp
        │
        ▼
trading::metal::MetalRuntime        (pure-C++ PIMPL)  opportunity_discovery/metal_runtime.hpp
        │  owns ↓ (Objective-C++)                      opportunity_discovery/metal_runtime.mm
        ├── id<MTLDevice>, id<MTLCommandQueue>
        ├── id<MTLLibrary>            (loaded cengine.metallib)
        ├── pipeline cache: name → id<MTLComputePipelineState>   (namespace-aware)
        ├── BufferPool               (persistent shared-storage MTLBuffers)
        └── RuntimeTelemetry         (dispatch/cache/buffer/GPU-time counters)
```

CPU oracle (`cpu_reference.{hpp,cpp}`) mirrors each kernel for strict parity and
as the validated fallback. `kernel_registry.{hpp,cpp}` is the native 45-kernel
inventory, reconciled against the live device.

## Key behaviours

- **Pipeline cache.** A pipeline state is built once per resolved kernel name and
  reused; `RuntimeTelemetry.pipeline_cache_hit_rate` reports effectiveness. This
  removes the per-call `newComputePipelineStateWithFunction` cost of the PyObjC path.
- **Namespace-aware resolution.** The 4 indicator kernels are global; covariance /
  monte-carlo / imbalance / reduction kernels live in the shader `namespace gme`.
  `resolve()` tries the bare name, then `gme::<name>`.
- **Persistent buffers.** `BufferPool` recycles shared-storage `MTLBuffer`s by
  best-fit size class; on Apple-Silicon unified memory the allocation is host- and
  GPU-visible with no copy. The hot path reuses buffers instead of allocating per
  observation.
- **Async + completion handlers.** Each dispatch commits a command buffer with a
  real `addCompletedHandler` that records GPU `GPUEndTime − GPUStartTime`,
  completions, and failures on a background queue; a `dispatch_semaphore` makes the
  call deterministic for parity tests while still exercising the async path.
- **CPU fallback.** `create()` never throws on "Metal unavailable" — it returns a
  runtime whose `available()` is false with a `reason()`, so callers degrade to the
  CPU oracle. Mandatory protection never depends on this runtime.

## Buffer ABIs (match the shaders exactly)

| Kernel(s) | Buffers | Grid |
|---|---|---|
| `rolling_sma_ticks` | prices i64@0, out i64@1, count u32@2, window u32@3 | count |
| `exponential_moving_average_ticks` | prices@0, out@1, count@2, num@3, den@4 | **1** (recursive) |
| `average_price_oscillator_ticks` | fast@0, slow@1, out@2, count@3 | count |
| `true_range_ticks` | highs@0, lows@1, prev@2, out@3, count@4 | count |
| `gme::column_means` | values f32@0, CovarianceParams@1, means f32@2 | assets |
| `gme::covariance_matrix` | values@0, means@1, params@2, cov@3 | assets² |
| `gme::ewma_covariance_matrix` | values@0, EwmaParams@1, cov@2 | assets² |

`CovarianceParams{observations, assets, row_stride, reserved}` (4×u32);
`EwmaParams{…4×u32, lambda, minimum_variance, pad0, pad1}` (4×u32 + 4×f32).

## Build + loading

`CENGINE_COMPILE_METAL` defaults **ON** on Apple when `xcrun metal` is present;
CMake enables OBJCXX, compiles the `.mm`, links Metal/Foundation, compiles the
shaders to `cengine.metallib`, and installs it **beside** the module.
`cengine.metal_backend.discover_metallib()` prefers the module-adjacent metallib so
a stale `build/` artifact is never loaded (directive §2). Identity (path, hash,
entry points) is recorded in `reports/shader_library_identity.json`.
