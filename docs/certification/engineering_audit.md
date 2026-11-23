# C-Engine M7A — Engineering Audit

**Verdict:** `CERTIFIED` · **Modules audited:** 103 · **BLOCKED:** 0
**Command:** `python -m cengine.certification audit`

The M7A engineering audit answers a single, narrow question: *are the components
of C-Engine actually present, importable, tested, and wired into the running
system?* It is a **static, introspective** audit — it inspects the codebase and
its test/consumption indices mechanically. It makes **no economic claim**. Edge,
profitability, and deployment readiness are the concern of other certification
phases (strategy economics, sensitivity, stress, holdout), and those remain
`NOT_CERTIFIED`.

This document describes how the audit derives a status for every module, the
status ladder it climbs, why a static audit is structurally capped at
`OPERATIONALLY_CONSUMED`, the 103-module / 0-BLOCKED result, the component groups
the modules fall into, and the one real bug the audit surfaced and that was
fixed (`gpu_worker.py` instantiating `MetalRuntime` at import time).

---

## 1. Why an introspective audit

A conventional "does it compile" check is too weak for a system where the whole
safety argument depends on specific modules being *live* — the native risk gate,
the protective exit, the failure-isolation workers. The audit therefore does not
trust declarations. It derives each module's status **mechanically** from
observable signals and refuses to award a rung the evidence does not support.

Three signals feed the status matrix:

| Signal | Source | What it proves |
|---|---|---|
| **Import** | Isolated-subprocess import of the module | The module loads cleanly on its own, with no hidden import-time side effects or missing dependencies |
| **Test coverage** | Test-coverage index | The module's behavior is exercised by the test suite |
| **Runtime consumption** | Runtime-consumption index | The module is actually referenced/consumed by the running system, not dead code |

The import check runs **in an isolated subprocess**, one module at a time. This
is deliberate: a module that only imports successfully because some earlier
module already initialized global state is *not* independently importable, and
the isolation exposes exactly that. It is also what caught the `gpu_worker` bug
(§6).

---

## 2. The status ladder: `SPECIFIED` → `LIVE_APPROVED`

Each module is assigned the highest rung its evidence supports. The ladder is
monotonic — a higher rung presupposes every signal below it.

```
SPECIFIED
   ↑  module is declared in the component manifest
IMPORTABLE
   ↑  + imports cleanly in an isolated subprocess
TEST_COVERED
   ↑  + exercised by the test-coverage index
OPERATIONALLY_CONSUMED   ◄── static audit ceiling
   ↑  + referenced/consumed by the running system
───────────────────────────────────────────────
   empirical tier — granted ONLY by certification / live runs
   ↑
LIVE_APPROVED
```

- **SPECIFIED** — the module is named in the component manifest. Existence only;
  no behavioral evidence yet.
- **IMPORTABLE** — adds a clean isolated-subprocess import. No import-time
  explosions, no hard-coded-path landmines, no missing dependencies.
- **TEST_COVERED** — adds coverage in the test-coverage index: its behavior is
  asserted somewhere in the suite.
- **OPERATIONALLY_CONSUMED** — adds a hit in the runtime-consumption index: the
  running system actually depends on it. This is the **highest rung a static
  audit can award.**
- **(empirical tier) … LIVE_APPROVED** — rungs above `OPERATIONALLY_CONSUMED`
  assert *behavior under real conditions* (a module did the right thing during a
  certification run; a module was approved for live capital). These are
  **empirical** claims, and no amount of static inspection can produce the
  evidence for them.

A module that fails the import signal cannot be placed on any behavioral rung and
is reported as **BLOCKED** — it is the audit's only failing state.

---

## 3. Why the static audit caps at `OPERATIONALLY_CONSUMED`

The cap is a design guarantee, not a limitation to be apologized for. The rungs
above `OPERATIONALLY_CONSUMED` require *empirical* evidence that a static,
introspective pass structurally cannot manufacture:

- A static audit can prove a module **is consumed** by the system. It cannot
  prove the module **behaved correctly** when the system ran under genuine data
  and faults — that is what the certification runs (`native`, `failure`,
  `protective-exit`, `entry`, …) exist to measure.
- It cannot prove a module is **approved for live capital** — that requires the
  full certification chain *plus* an external human authorization token the
  software cannot mint.

Honoring the cap is what keeps the audit honest: it refuses to let a module
"look promising" its way into a live-eligible rung. Empirical rungs are granted
only by the cert runs and (ultimately) the hard-gated live-authorization chain —
and as of M7, live is `INACTIVE` and `live_eligible = False`. The audit awarding
`OPERATIONALLY_CONSUMED` and stopping there is the **correct** ceiling, not a
gap.

---

## 4. Result: 103 modules, 0 BLOCKED

| Metric | Value |
|---|---|
| Modules audited | **103** |
| BLOCKED (failed import signal) | **0** |
| Static ceiling reached | `OPERATIONALLY_CONSUMED` |
| Audit verdict | **CERTIFIED** |

Every one of the 103 modules cleared the isolated-subprocess import signal (zero
BLOCKED), and each was placed at the highest rung its three signals support, up
to the static ceiling. The audit is `CERTIFIED` because there are no BLOCKED
modules: nothing in the component manifest fails to load, and the system's wiring
is consistent with its specification.

> The verdict is about **wiring and integrity**, not economics. A `CERTIFIED`
> engineering audit says the parts are all present and connected; it says nothing
> about whether the strategy makes money (it does not — see the strategy,
> sensitivity, and stress phases, all `NOT_CERTIFIED`).

---

## 5. Component groups

The 103 modules organize into the subsystem families that make up C-Engine. The
audit covers each family; the certification CLI exposes a matching verification
command for most of them.

- **Data / lakehouse** — the Parquet + DuckDB bar store and its access layer
  (certified separately in M7B data integrity).
- **Native / Metal GPU** — `_cengine_native` (RiskEngine, DiscoveryEngine,
  MetalRuntime, the emergency candidate), the MLX/Metal kernels, `DevicePolicy`,
  and the CPU reference kernels. *This family contained the fixed bug (§6).*
- **Risk & protective exit** — the native RiskEngine (sole financial authority)
  and the `ProtectiveExitCoordinator` (−0.5% backstop), independent of the
  learning and discovery stacks.
- **Execution** — the order path and intelligent-execution workers.
- **Discovery / scanner** — alpha-discovery and scanning modules.
- **Learning** — learning infrastructure, provenance discipline, and the
  governed (recommendation-only) optimization loop.
- **Portfolio** — portfolio-construction modules.
- **Runtime / supervisor** — the exchange-calendar operating-state machine, the
  `RuntimeSupervisor` with the P0–P9 GPU priority ladder, and the off-session
  services.
- **Certification** — the `cengine.certification` harness itself.

Per-group module counts beyond the audited total of 103 are not asserted here;
the load-bearing facts are the **total of 103** and **0 BLOCKED** across all
groups.

---

## 6. Bug surfaced and fixed: `gpu_worker` import-time `MetalRuntime`

The audit is not a formality — it caught a real defect that a naive import check
would have missed.

**Defect.** `gpu_worker.py` instantiated a `MetalRuntime` **at module import
time**, pointed at a **hard-coded `build/` path** for the compiled Metal library.

**Why the audit caught it.** The import signal runs each module in an **isolated
subprocess**. Under isolation, `gpu_worker`'s import executed the `MetalRuntime`
construction immediately — with no surrounding process to have already prepared
state — and the hard-coded `build/` path made that construction fragile and
environment-dependent. An import-time side effect against a hard-coded path is
exactly the class of defect that keeps a module off `IMPORTABLE`, and the
isolated-subprocess design is what exposed it rather than letting it hide behind
load order.

**Fix.**

1. **Made `MetalRuntime` construction lazy** — the runtime is no longer built as
   a side effect of importing the module; it is constructed when first needed.
2. **Replaced the hard-coded `build/` path with `discover_metallib()`** — the
   compiled Metal library is located by discovery rather than a brittle literal
   path.

After the fix, `gpu_worker` imports cleanly in isolation and the native/Metal
family clears the import signal with the rest of the 103 modules, contributing to
the **0 BLOCKED** result.

This is the intended payoff of an introspective audit: it converts "it probably
loads" into "it provably loads, in isolation, with no import-time landmines," and
it turned up a concrete, fixable bug in the GPU worker in the process.

---

## 7. What this audit does and does not certify

| Claim | Status |
|---|---|
| All 103 modules import cleanly in isolation | **CERTIFIED** (0 BLOCKED) |
| Modules are tested and consumed by the running system | Reflected per-module, up to `OPERATIONALLY_CONSUMED` |
| Component wiring matches specification | **CERTIFIED** |
| `gpu_worker` import-time `MetalRuntime` bug | **FIXED** (lazy + `discover_metallib()`) |
| Modules behaved correctly under real conditions | Out of scope — see `native`, `failure`, `protective-exit`, `entry` cert runs |
| Strategy has an economic edge | **NOT the subject of this audit** — economics are `NOT_CERTIFIED` |
| Live/paper readiness | **NOT the subject of this audit** — live is `INACTIVE`, `live_eligible = False` |

M7A establishes that C-Engine is **engineering-certified**: the components are
present, importable, tested, wired, and free of the import-time defect the audit
surfaced. It does **not** establish an economic edge. Consistent with the overall
M7 conclusion, the platform is a sound research and execution substrate while the
current strategy is **not** a deployable edge.

---

*Provenance note: this document reports a static engineering audit, not economic
results. Any economic results referenced elsewhere in the certification bundle
are historical-replay simulations on genuine bars — not live performance and not
a profitability claim. No live orders were placed.*
