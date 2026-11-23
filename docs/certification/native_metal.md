# Native / Metal Certification (§23)

**Milestone:** M7 Certification
**Verdict:** **CERTIFIED**
**Scope:** The native extension and its authoritative types, the `RiskEngine` as the
sole financial authority, the Metal runtime on Apple M5 Max, exact GPU/CPU parity
(correctness-preserving acceleration), and the documented CPU fallback.

This section certifies the *substrate* — that the native risk gate is correct and
decisive, that GPU acceleration changes speed without changing answers, and that
there is always a validated CPU path. It makes **no** economic claim. Strategy
economics are separately **NOT_CERTIFIED** (see §21 sensitivity, §22 stress, and
the sealed-holdout strategy report).

---

## Platform

| Property | Value |
|---|---|
| Host | Apple M5 Max, 48 GiB unified memory |
| Python | 3.11 |
| GPU stack | MLX / Metal |
| Native extension | `_cengine_native` (pybind11) |
| Metal device (runtime) | `Apple M5 Max` |
| Price scale | `price_scale = 10000` ticks |

---

## 1. Native extension and authoritative types

The compiled extension `_cengine_native` is the single native entry point. At cert
time it exposes the authoritative types consumed by the rest of the stack:

| Export | Role |
|---|---|
| `RiskEngine` | Sole financial authority — accepts or rejects every order (see §2). |
| `DiscoveryEngine` | Opportunity discovery surface (advisory; never an order path). |
| `MetalRuntime` | Compiled-kernel runtime bound to the Metal device (see §3). |
| emergency candidate | Pure native `-3%` emergency-exit threshold candidate — **INACTIVE by default** (see §4). |

These are the *authoritative* types in the sense that financial decisions
(accept/reject, protective thresholds) are resolved in native code, not
re-implemented in Python. Python supervises and batches calls; the native layer
decides.

### Audit note (M7A)

The engineering audit (103 modules, 0 BLOCKED) fixed a real defect in this layer:
`gpu_worker.py` instantiated `MetalRuntime` **at import time** against a hard-coded
`build/` path. It was made **lazy** and switched to `discover_metallib()`, so
importing the module no longer forces runtime construction against a fixed path.

---

## 2. RiskEngine — the sole financial authority

Every order is adjudicated by the native `RiskEngine`. There is no bypass: the
`-0.5%` protective exit and this native risk gate remain the authority across all
operating states, and no learning/discovery module can reach the order path.

### Accept / reject (certified behavior)

| Order | RiskEngine decision | Reason code |
|---|---|---|
| Valid **10-share** order | **ACCEPT** | — |
| Oversized **50,000-share** order | **REJECT** | `QUANTITY_LIMIT_EXCEEDED` |

Illustratively, the gate behaves as:

```
decision = risk_engine.evaluate(order)      # native _cengine_native.RiskEngine

# 10 shares  -> accepted
assert decision.accepted is True

# 50,000 shares -> rejected, with an explicit machine-readable reason
assert decision.accepted is False
assert decision.reason == "QUANTITY_LIMIT_EXCEEDED"
```

The rejection carries an explicit, machine-readable reason code rather than a
silent clamp — the caller is told *why* the order was refused. This is the
behavior that makes the RiskEngine the **sole** financial authority: a strategy
cannot route size the gate has not approved.

---

## 3. Metal runtime on Apple M5 Max

`MetalRuntime` binds to the Metal device reported as **`Apple M5 Max`** and runs
**compiled kernels** from the discovered Metal library. The library is located via
`discover_metallib()` (no hard-coded build path), and the runtime is constructed
lazily so that merely importing a consumer module does not force device
acquisition.

Operational properties certified here:

- The device is the real Apple M5 Max GPU, not a stub.
- Kernels are **compiled** and pipeline-callable on that device.
- GPU dispatch is correctness-preserving (see §4), so running on the GPU is an
  acceleration decision, never a correctness decision.

Under the continuous-runtime supervisor, GPU access is governed by a P0–P9
priority ladder: **P0 protection is never gated**, while research work (≥P6) is
suspended under GPU pressure. The Metal runtime being correctness-preserving is
what makes that scheduling safe — degrading or deferring GPU work can never change
a financial answer.

---

## 4. Exact GPU/CPU parity (correctness-preserving acceleration)

The central guarantee of §23: **the GPU produces exactly the CPU reference
answer.** Acceleration is correctness-preserving.

| Kernel | GPU result vs CPU reference (`metal_cpu_ref`) |
|---|---|
| EMA (exponential moving average) | **EXACT** match |
| APO (absolute price oscillator) | **EXACT** match |

The GPU EMA and APO kernels reproduce the `metal_cpu_ref` CPU reference
**exactly**. Because the GPU and CPU paths agree, the choice of device is purely
about throughput and scheduling — it carries no numerical or decision-level
consequence.

### Emergency-exit candidate: pure and inactive

The native `-3%` emergency-exit threshold candidate is:

- **Pure** — a deterministic candidate computation with no side effects.
- **INACTIVE by default** — it does not fire on its own; the active protective
  authority remains the `-0.5%` exit (see §25).
- **One-directional on risk** — `clamp_emergency_threshold` **never widens risk**:
  its result is always `>= current` (tightening is allowed; loosening is refused).

This mirrors the safety invariant enforced elsewhere in the stack: risk may be
tightened by machinery, never loosened.

---

## 5. Documented CPU fallback

There is always a validated non-GPU path. The fallback is explicit and
documented, not a silent degradation:

- **Forced via policy:** `DevicePolicy(force_cpu=True)` routes execution to the
  CPU path deterministically.
- **Reference kernels:** **4 CPU reference kernels** (`metal_cpu_ref`) mirror the
  GPU kernels and serve as both the parity oracle (§4) and the operational
  fallback.

Because the CPU reference *is* the parity target, falling back to CPU cannot
change results — the GPU was certified to match it exactly. This is what lets the
runtime degrade to CPU under GPU pressure (or on a host without the GPU path)
without any correctness risk, and it is why the continuous-runtime DEGRADED state
can keep protection and execution correct while GPU-dependent research is
suspended.

---

## What this certifies — and what it does not

**Certified (§23):**

- `_cengine_native` exposes the authoritative `RiskEngine` / `DiscoveryEngine` /
  `MetalRuntime` plus the emergency candidate.
- The native `RiskEngine` is the sole financial authority (accept 10 shares;
  reject 50k shares with `QUANTITY_LIMIT_EXCEEDED`).
- The Metal runtime runs compiled kernels on `Apple M5 Max`.
- GPU EMA/APO reproduce the `metal_cpu_ref` CPU reference **exactly**
  (correctness-preserving acceleration).
- A documented CPU fallback exists via `DevicePolicy(force_cpu=True)` plus 4 CPU
  reference kernels.

**Explicitly not claimed here:**

- No economic edge. Strategy economics are **NOT_CERTIFIED** on the sealed holdout
  (0/5 symbols), including at **zero cost** (§21) and under **stress** (§22, >90%
  drawdown in the 2020 and 2022 windows).
- No paper or live readiness. Readiness is **RESEARCH_VALIDATED**; live is
  **INACTIVE** and hard-gated; `live_eligible = False`.

Native/Metal certification is a statement about **correctness of the substrate**,
not about profitability. A correct risk gate and exact GPU/CPU parity are
necessary for deployment but are not, by themselves, an edge.

---

## Reproduction

```
python -m cengine.certification native
```

Related sections:

- `python -m cengine.certification protective-exit` — §25 protective-exit math and independence.
- `python -m cengine.certification failure` — §24 fault isolation (faults in learning/discovery/GPU/data never cascade into protection/execution).
- `python -m cengine.certification full` / `report` — the complete certification chain.

---

## Provenance

All economic results referenced in this document are **historical-replay
simulations on genuine bars** — they are **not live performance** and **not a
profitability claim**. **No live orders were placed.** The native RiskEngine
accept/reject behavior and GPU/CPU parity results are mechanical correctness
checks, not performance measurements.
