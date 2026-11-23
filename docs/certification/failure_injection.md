# Failure-Injection Certification (§24)

**Verdict: CERTIFIED**

**CLI:** `python -m cengine.certification failure`

This document records the failure-injection certification for C-Engine M7. It
describes the six faults that were deliberately injected into the running stack,
the mechanism that isolates each one, and the single invariant the whole section
exists to prove: **faults in learning, discovery, GPU, or data NEVER cascade
into protection or execution.**

---

## 1. The through-line

C-Engine is a layered system. The layers that *discover* and *learn* are
speculative and allowed to fail; the layers that *protect capital* and *execute
or refuse orders* are not. §24 exists to certify that the boundary between those
two groups holds under injected failure.

Two statements summarize the entire certification:

1. **Fault containment** — a fault in learning / discovery / GPU / data is
   isolated to its own layer and never propagates into protection or execution.
2. **Essential-worker survival** — the essential **P0 protection** and **P1
   execution** workers are **never auto-killed**. Whatever the supervisor does
   to shed load or quarantine a fault, it may not take down the workers that
   keep positions safe or that place/refuse orders.

Everything below is a specific, injected test of one of those two statements.

---

## 2. The six injected faults

| # | Injected fault | Layer | Containment mechanism | Certified behavior |
|---|----------------|-------|------------------------|--------------------|
| 1 | Learning worker fault | Learning | Recommendation-only isolation; protection imports no ML | Protection/execution unaffected; essential workers not killed |
| 2 | GPU unavailable | Compute (Metal) | Documented CPU fallback; P0 protection never GPU-gated | Work continues on CPU reference path; protection never gated |
| 3 | Empty lakehouse | Data | Protection imports no lakehouse module | Data fault isolated; protection/execution do not cascade |
| 4 | Broker / venue failure | Execution venue | Per-symbol scan swallows the error, never raises | One symbol's broker failure does not abort the scan |
| 5 | Double-instance | Runtime | Single-instance lock | Second trade instance is blocked |
| 6 | Checkpoint round-trip | Runtime state | Atomic checkpoint write/read | Checkpoint survives a round-trip |

Each fault is detailed below.

---

## 3. Fault 1 — Learning worker fault

**What is injected:** a fault in a learning worker (the speculative,
recommendation-producing layer).

**Why it is contained:**

- Learning is **recommendation-only** (certified separately): no learning module
  imports the order path or calls `.submit()`. A learning worker therefore has
  no handle on execution to corrupt when it fails.
- The `ProtectiveExitCoordinator` **imports no ML / GPU / lakehouse / scanner
  module**. Protection has no code dependency on the learning layer, so a
  learning fault cannot reach it through imports or shared state.
- The essential **P0 protection** and **P1 execution** workers are **never
  auto-killed** when a learning worker faults.

**Certified behavior:** the learning fault is isolated to the learning layer.
Protection and execution continue unaffected, and no essential worker is taken
down as a side effect.

---

## 4. Fault 2 — GPU unavailable

**What is injected:** the GPU/Metal compute path is made unavailable.

**Why it is contained:**

- There is a **documented CPU fallback** via `DevicePolicy(force_cpu=True)`
  backed by **4 CPU reference kernels**. GPU EMA/APO reproduce the CPU reference
  (`metal_cpu_ref`) **exactly**, so falling back to CPU is correctness-preserving,
  not merely a degraded approximation.
- Under the `RuntimeSupervisor` **P0–P9 GPU priority ladder**, **P0 protection is
  never gated**; research priorities (>= P6) are the work that is suspended under
  GPU pressure. Protection does not compete with research for the GPU.
- A real import-time defect was fixed during the engineering audit: `gpu_worker.py`
  instantiated `MetalRuntime` at import time against a hard-coded `build/` path.
  It was made **lazy** and switched to `discover_metallib()`, so the mere absence
  of a GPU/build artifact no longer crashes on import.

**Certified behavior:** when the GPU is unavailable, work proceeds on the CPU
reference path and **P0 protection is never gated**. The GPU fault does not
cascade into protection or execution.

---

## 5. Fault 3 — Empty lakehouse

**What is injected:** the data layer is emptied (an empty lakehouse) to simulate
a missing / unavailable historical data source.

**Why it is contained:**

- The `ProtectiveExitCoordinator` **imports no lakehouse module**. Protection is
  independent of the historical data layer by construction.
- The containment invariant is explicit: faults in **data** never cascade into
  protection or execution.

**Certified behavior:** the data fault is isolated. An empty lakehouse does not
propagate into the protection or execution layers.

---

## 6. Fault 4 — Broker / venue failure

**What is injected:** a per-symbol broker/venue failure during the
protective-exit scan.

**Why it is contained:**

- The protective-exit scan **swallows a per-symbol broker failure and never
  raises**. A failure on one symbol is absorbed locally; it does not throw out of
  the scan loop and it does not abort evaluation of the remaining symbols.

**Certified behavior:** a broker failure on one symbol is contained to that
symbol. The protective-exit scan keeps running and never raises, so a single
venue failure cannot take down protection across the book.

---

## 7. Fault 5 — Double-instance

**What is injected:** a second trade instance is started while one is already
running.

**Why it is contained:**

- The continuous runtime enforces a **single-instance lock**. The second trade
  instance is **blocked** before it can act.

**Certified behavior:** the single-instance lock blocks the second trade
instance, preventing two processes from racing on the same positions or order
path.

---

## 8. Fault 6 — Checkpoint round-trip

**What is injected:** runtime state is written to a checkpoint and read back (a
round-trip), exercising the persistence path used across restarts and recovery.

**Why it is contained:**

- Checkpoints are **atomic**, and the certification verifies that an **atomic
  checkpoint survives a round-trip** — state written out is recovered intact.
- The surrounding continuous-runtime policy is **DEGRADED-on-regression with
  auto-recover**, so checkpoint/restart is a first-class, monitored path rather
  than a best-effort one.

**Certified behavior:** an atomic checkpoint survives a write/read round-trip,
so runtime state can be persisted and restored without corruption.

---

## 9. Architectural invariants that make isolation possible

The per-fault results above are not coincidental; they follow from
structural properties certified elsewhere in M7:

- **Protection has no speculative dependencies.** The
  `ProtectiveExitCoordinator` imports no ML / GPU / lakehouse / scanner module,
  so faults in those layers cannot reach it through code.
- **Learning cannot execute.** Learning is recommendation-only — no learning
  module imports the order path or calls `.submit()`.
- **The native risk gate is the sole financial authority.** The native
  `RiskEngine` accepts/refuses orders independently of the learning and discovery
  layers, so a fault upstream cannot smuggle an order past risk control.
- **Priority ladder protects P0.** The `RuntimeSupervisor` P0–P9 ladder never
  gates P0 protection; it sheds research (>= P6) under GPU pressure instead.
- **Essential workers are never auto-killed.** P0 protection and P1 execution are
  excluded from automatic shutdown during fault handling.
- **Runtime hygiene.** Single-instance lock, atomic checkpoints, and
  DEGRADED-on-regression + auto-recover bound the blast radius of restarts and
  state transitions.

---

## 10. Scope and honest limits

- §24 certifies **fault containment and essential-worker survival** — that
  failures are isolated and protection/execution keep working. It does **not**
  make any economic claim.
- The economic verdicts are unchanged by this section and remain:
  **NOT_CERTIFIED** for strategy economics, capital/cost sensitivity, and stress;
  overall readiness is **RESEARCH_VALIDATED**, **not** ready for paper or live
  capital.
- The strategy's **-0.5% protective exit is a risk backstop, not an economic
  edge.** §24 confirms that backstop stays functional under injected failure; it
  does not confirm the backstop is profitable.

---

## 11. How to reproduce

```
python -m cengine.certification failure
```

The runtime state machine and its single-instance lock / atomic checkpoints can
be inspected with:

```
python -m cengine.runtime status
```

---

*Provenance note: all economic results referenced in this document are
historical-replay simulations on genuine Alpaca SIP bars — not live performance,
and not a profitability claim. No live orders were placed.*
