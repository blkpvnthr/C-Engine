# Continuous Autonomous Research

**C-Engine Milestone 8 — Autonomous Alpha Discovery**
Scope: §4 (lifecycle in the runtime), §18 (honest GPU-vs-CPU compute), §22 (continuous integration + GPU priority ladder), §24 (dashboard).

This document describes how alpha-discovery research runs off-session inside the
continuous 24/7 runtime without ever interfering with financial protection, the
GPU priority ladder that enforces that separation, the honest completion-measured
Metal-vs-CPU benchmarks, and the alpha-discovery dashboard panel — in which
development results are kept strictly separate from independent certification
results.

---

## 1. Where this sits in the stack

M8 is an **orchestration and research layer** (`cengine/alpha/`) on top of the
existing M1–M7 stack. It builds **no new trading engine, no new learning
platform, no new optimizer, and no new lakehouse**. It composes:

- the certification harness,
- the governed optimizer (`cengine/optimization`),
- the learning stack (`cengine/learning`),
- the continuous runtime (`cengine/runtime`),
- the native / Metal substrate.

Platform: a local-first native quant stack on an Apple **M5 Max (48 GiB unified
memory)**, Python 3.11, with an **MLX / Metal GPU**. Research modules in
`cengine/alpha` are importable and are composed by the lifecycle; the continuous
runtime runs them off-session.

---

## 2. AlphaDiscoveryService in the continuous runtime (§4 / §22)

The alpha-discovery lifecycle is integrated into the continuous 24/7 runtime as
the **`AlphaDiscoveryService`**. Its operating contract:

| Property | Behavior |
|---|---|
| Priority | **P9** (research — the lowest rung on the GPU ladder) |
| GPU access | **Admission-gated** — suspended under GPU pressure, bounded in-session |
| Execution window | **Off-session** |
| Order behavior | **Shadow-only** — no live orders are placed |
| Relationship to protection | **Protection is never gated**; research always yields to it |

The service is a *capital allocator's research tenant*, not an alpha source and
not an execution path. When the GPU is under pressure, or when a session is
live, research is **suspended or bounded** so that it cannot compete with
protective workloads for compute. Research workloads **never interfere with
financial protection**.

---

## 3. The P0–P9 GPU priority ladder (§22)

GPU admission is governed by a monotonic priority ladder. The ordering is the
invariant that matters: **financial protection sits at the top and is never
gated; research sits at the bottom (P9) and is admission-gated**, so it is the
first thing suspended when GPU pressure rises.

| Rung | Tenant class | Gating policy |
|---|---|---|
| **P0 (highest)** | Financial protection | **Never gated** — always admitted, always preempts |
| … | Production / operational workloads | Higher priority than research; preempt P9 |
| **P9 (lowest)** | `AlphaDiscoveryService` research | **Admission-gated** — suspended under GPU pressure, bounded in-session, off-session, shadow-only |

Consequences of the ladder:

- Protection **cannot be starved** by research. A research job never acquires
  GPU in a way that delays or blocks a protective computation.
- Under GPU pressure, the P9 research tenant is **suspended first**.
- In-session, research is **bounded** rather than allowed to run freely.
- The production **−0.5% protective exit**, the independent event-driven path,
  and the 5-second periodic backstop are untouched by anything research does;
  the native C++ `RiskEngine` remains the risk authority.

---

## 4. Honest Metal GPU-vs-CPU benchmarks (§18)

Compute placement is decided by **honest, completion-measured** benchmarks
(`mx.eval` forces the Metal computation to actually complete before timing — no
lazy-evaluation illusions). The result is reported **as measured**, including the
cases where **the CPU wins**:

| Workload | Winner | Measured ratio |
|---|---|---|
| Batched linear inference (small size) | **CPU** | 0.57× (GPU slower) |
| Large feature standardization | **GPU** | 9× (GPU faster) |

Takeaways:

- The GPU is **not** universally faster. At small batch sizes, kernel-launch and
  dispatch overhead make the CPU the better choice for batched linear inference.
- The GPU pays off on **large, wide** numerical work such as feature
  standardization.
- **CPU wins are reported, not hidden.** Compute placement follows the
  measurement, not an assumption that "GPU is always faster."
- MLX/Metal is used where it genuinely helps; for example, the supervised-ML
  work (§10) ran its MLX models on the Metal GPU.

These benchmarks only inform *where* research compute runs. They never change the
priority ladder: even a GPU-favorable research workload stays at P9 and remains
subordinate to protection.

---

## 5. The lifecycle the service runs (§4)

Each research pass is an **`AlphaDiscoveryCycle`** that chains:

```
COLLECT / ATTRIBUTE
  -> IDENTIFY FAILURE MODES
  -> GENERATE HYPOTHESES
  -> CHECK DATA SUFFICIENCY
  -> CONSULT OPTIMIZATION MEMORY
  -> PLAN / RUN EXPERIMENTS
  -> ECONOMIC + ROBUSTNESS EVALUATION
  -> ACCEPT / REJECT / DEFER
  -> PERSIST
```

It reuses existing governed machinery rather than re-inventing it:

- the **M6B persistent `OptimizationMemory`** (deduplication + effectiveness
  tracking),
- the **frozen `ChallengerGates`**,
- the **`ChallengerRegistry`**.

Every generated hypothesis **references evidence** and **defines a falsifiable
test**; `NO_TRADE` is a valid outcome, and negative results are preserved rather
than reinterpreted.

**Genuine run outcome:**

| Metric | Value |
|---|---|
| Hypotheses accepted | **0** |
| Hypotheses rejected | **6** |
| Hypotheses deferred | **2** |
| Incumbent | **Retained** |
| Hypotheses generated | **3** |
| Optimization-memory rows persisted | **14** |

No development challenger reached the frozen gates, so there was nothing to
promote; the incumbent was retained.

---

## 6. The alpha-discovery dashboard (§24)

The runtime dashboard exposes an alpha-discovery panel. Its defining property is
that **development results are kept strictly separate from independent
certification results**, and **no metrics are fabricated**.

Panel fields include:

| Field | Meaning |
|---|---|
| `ALPHA_DISCOVERY_STATUS` | Current state of the research service |
| `EXPERIMENTS_COMPLETED` | Experiments run this cycle |
| `EXPERIMENTS_REJECTED` | Experiments that failed their falsifiable test |
| `EXPERIMENTS_DEFERRED` | Experiments deferred (e.g., insufficient data) |
| `BEST_DEVELOPMENT_NET_EXPECTANCY` | Best net expectancy seen in **development** (not a certified result) |
| `INCUMBENT_NET_EXPECTANCY` | Net expectancy of the retained incumbent |
| `OPTIMIZATION_MEMORY_SIZE` | Rows in the persistent optimization memory |
| `DATA_SUFFICIENCY` | Whether data is adequate for the test at hand |
| `INDEPENDENT_CERTIFICATION_STATUS` | Status of the **independent** certification track (kept separate) |

Why the separation is load-bearing:

- `BEST_DEVELOPMENT_NET_EXPECTANCY` is a **development** figure. It is **not** a
  certified or deployable number, and the panel never presents it as one.
- `INDEPENDENT_CERTIFICATION_STATUS` is sourced from the separate
  certification track. On the tested universe this status is **DEFERRED** —
  there is no positive edge to certify. A *best-of-many development
  configurations* is never described as certified.
- The panel reports only measured values; there are **no fabricated metrics**.

---

## 7. Invariants this subsystem preserves

- Production **−0.5% protective exit** + independent event-driven path +
  5-second periodic backstop are **preserved and never modified**.
- The native C++ **`RiskEngine` remains the risk authority**.
- The **−3% emergency exit candidate stays INACTIVE**.
- **No new engine, optimizer, or lakehouse** is built.
- **No live activation** — research is shadow-only and off-session; no live
  orders were placed.
- **No fabricated performance.**
- The **sealed holdout is protected from optimization** and treated as consumed.
- **Protection is never gated** behind research.

---

## 8. Where the code and evidence live

- Research modules: `cengine/alpha/` — importable, composed by the
  `AlphaDiscoveryCycle`, and run off-session by the continuous runtime.
- Evidence: `reports/alpha_discovery/` — **18 JSON** artifacts plus
  `final_alpha_discovery_report.md`.

---

## 9. Honest standing of the result

On the tested universe, under the mandatory −0.5% protective exit, **no positive,
statistically credible out-of-sample net expectancy was found**. The decision is
to **REJECT the current entries/strategies as a deployable edge** while
**ACCEPTING the alpha-discovery infrastructure as a sound research substrate**.
Independent certification is **DEFERRED** because there is no positive edge to
certify.

---

### Provenance

All economic results referenced here are **historical-replay simulations on
genuine bars** — they are **not live performance** and **not a profitability
claim**. **No live orders were placed.** Negative results are **preserved, never
reinterpreted**.
