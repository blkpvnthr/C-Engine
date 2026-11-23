# GPU Priority Ladder (P0–P9)

**Scope:** C-Engine M7 certification — continuous-runtime GPU workload admission.
**Component:** `cengine/runtime/supervisor.py` (`RuntimeSupervisor`, `Priority`).
**Platform:** Apple M5 Max, 48 GiB unified memory, MLX/Metal single-GPU.

The C-Engine runtime shares **one** Metal GPU across financial protection,
execution, discovery, analytics, ML, and large research experiments. Because
the GPU is a single shared resource, the `RuntimeSupervisor` imposes a strict
**P0–P9 priority ladder** and an **admission policy** that guarantees financial
protection is *never* starved by learning or research work. This document
describes each rung, the admission rules, and why protection can never be
gated out.

---

## 1. The priority ladder

`Priority` is an `IntEnum` where **lower value = higher priority**. P0 is the
most protected; P9 is the most freely preempted.

| Priority | Name | Workload | Essential? |
|----------|------|----------|:----------:|
| **P0** | `P0_PROTECTION` | Financial protection + broker reconciliation | ✅ yes |
| **P1** | `P1_MARKET_EXEC` | Native market processing + execution | ✅ yes |
| **P2** | `P2_DISCOVERY` | Metal discovery + technical analytics | no |
| **P3** | `P3_ML_INFERENCE` | Production ML inference | no |
| **P4** | `P4_PORTFOLIO` | Portfolio scenario analytics | no |
| **P5** | `P5_FEATURES` | Historical feature construction | no |
| **P6** | `P6_ML_TRAINING` | ML training | no |
| **P7** | `P7_BANDIT` | Bandit optimization | no |
| **P8** | `P8_OFFLINE_RL` | Offline reinforcement learning | no |
| **P9** | `P9_RESEARCH` | Large historical research experiments | no |

### Tiers

- **Protection tier (P0).** Financial protection — the `-0.5%`
  `ProtectiveExitCoordinator` scan and broker reconciliation. Certified
  independent of all learning, GPU, discovery, and lakehouse modules. Always
  on, in every operating state, never gated.
- **Execution tier (P1).** Native market processing and order execution,
  governed by the native `RiskEngine` (the sole financial authority).
  Essential; monitoring/reconciliation runs even off-session.
- **Market-analytics tier (P2–P5).** Discovery, production ML inference,
  portfolio analytics, and feature construction. Active while markets are (or
  recently were) open. Non-essential: isolated on failure, but **not**
  classed as research.
- **Research tier (P6–P9).** ML training, bandit optimization, offline RL, and
  large historical research experiments. The **research cutoff** is
  `_RESEARCH_CUTOFF = P6`: any workload at priority **≥ P6** is "research" and
  is throttled/suspended under GPU pressure and bounded during live sessions.

---

## 2. Essential vs. non-essential

`essential=True` is reserved for **P0 and P1**. The distinction drives failure
isolation (`record_failure`):

- **Essential (P0/P1):** never isolated-killed. If one fails, the supervisor
  restarts it. Financial protection and execution are never auto-terminated.
- **Non-essential (P2–P9):** a learning/analytics/research fault is
  **isolated** — `isolated = not w.essential` — and never cascades into
  protection or execution (`financial_protection_affected: False`).

This is the runtime expression of the M7 **failure-injection** certification
(§24): faults in learning, discovery, GPU, and data never reach P0 protection
or P1 execution workers.

---

## 3. Per-state activation policy

`active_workers(state)` decides *which* workers should be running in a given
exchange-calendar operating state (overnight, pre-market, regular, after-hours,
post-session, weekend, holiday, plus the `DEGRADED`/`RECOVERING` overrides):

| Tier | Activation rule |
|------|-----------------|
| **P0** | Always on, every state. |
| **P1** | On during market-adjacent states; reconciliation/exec monitoring always on. |
| **P2–P5** | On during `PRE_MARKET`, `REGULAR_SESSION`, `AFTER_HOURS`, `POST_SESSION_PROCESSING`. |
| **P6–P9 (research)** | On in research-appropriate off-session states, or (throttled) when execution is permitted — subject to GPU admission. |

Continuous operation does **not** imply continuous execution: only
`REGULAR_SESSION` (and, when explicitly authorized, pre-/after-hours) permits
elective orders. Research runs in the off-session states.

---

## 4. GPU admission policy (§8)

`gpu_admission(workload, state, gpu_pressure)` is the single gate that decides
whether a GPU workload may run **now**. `gpu_pressure` is a normalized
`[0, 1]` utilization signal; `max_gpu_pressure = 0.85` by default.

The decision rules, in order:

1. **P0 is never gated.**
   ```
   workload == P0_PROTECTION  ->  admit=True, reason="protection_never_gated"
   ```
   Protection is admitted unconditionally — regardless of pressure, state, or
   any other workload. This is the first branch and short-circuits everything
   below.

2. **Research suspended under GPU pressure.**
   ```
   gpu_pressure >= 0.85  AND  workload >= P6  ->  admit=False
   ```
   When the GPU is at or above the pressure ceiling, every research-tier
   workload (P6–P9) is refused. Market-analytics (P2–P5), execution (P1), and
   protection (P0) are **not** suspended by this rule.

3. **Research bounded in-session.**
   ```
   execution_permitted(state)  AND  workload >= P6
     ->  admit = (gpu_pressure < 0.5)
   ```
   During a live/execution-permitted session, research may run only as a
   **bounded background** task while pressure is below `0.5`
   (`"bounded_research_in_session"`); at or above `0.5` it is
   `"deferred_to_protect_live_workloads"`. This keeps headroom for P0/P1 even
   before the hard `0.85` ceiling is reached.

4. **Otherwise admitted.**
   ```
   off-session, or non-research (P1–P5)  ->  admit=True
   reason="off_session_or_non_research"
   ```

### Admission summary

| Workload | GPU pressure | State | Admitted? |
|----------|--------------|-------|:---------:|
| P0 protection | any | any | ✅ always |
| P1–P5 (non-research) | any | any | ✅ (subject to per-state activation) |
| P6–P9 research | ≥ 0.85 | any | ❌ suspended |
| P6–P9 research | < 0.85, execution-permitted session | live | ✅ only if pressure < 0.5 |
| P6–P9 research | < 0.85 | off-session | ✅ admitted |

---

## 5. Why protection is never starved

Protection sits at the **top of the ladder (P0)** and is defended by three
independent mechanisms:

1. **Unconditional admission.** The first branch of `gpu_admission` returns
   `admit=True` for P0 before any pressure or state check — there is no code
   path that can refuse protection GPU work.
2. **Essential-worker isolation.** P0 (and P1) are `essential`; non-essential
   failures are isolated and `financial_protection_affected` is always
   `False`. A learning or research crash cannot take protection down.
3. **Headroom reservation via the research gate.** Research is suspended at
   `pressure >= 0.85` and bounded to `pressure < 0.5` in live sessions, so the
   GPU can never be saturated by P6–P9 work at the expense of P0/P1. The
   ceiling exists *to leave capacity for protection and execution*.

Additionally, the certified `ProtectiveExitCoordinator` imports no ML / GPU /
lakehouse / scanner module, so protection correctness does not even depend on
the GPU being available — the native `-3%` emergency candidate has a documented
CPU fallback (`DevicePolicy(force_cpu=True)`), and `clamp_emergency_threshold`
can only tighten risk, never widen it.

---

## 6. Design invariants

- **Lower number = higher priority.** P0 outranks P9; preemption flows from
  the bottom up.
- **Protection and execution are essential and never auto-killed.**
- **Research = priority ≥ P6.** This single cutoff defines everything that is
  throttled under pressure or bounded in-session.
- **Two thresholds govern research:** the hard suspension ceiling
  (`max_gpu_pressure = 0.85`) and the in-session bound (`0.5`).
- **No autonomous widening.** The admission policy only ever *restricts*
  lower-priority work; it has no authority to escalate anything above its
  registered priority.

---

## 7. Operator reference

Inspect runtime health and GPU/worker status via the runtime CLI:

```
python -m cengine.runtime <status|once|run>
```

`RuntimeSupervisor.health()` reports, per worker, its integer priority,
`healthy` flag, `restarts`, and `essential` flag, plus the invariant
`protection_independent_of_learning: True`.

---

## Provenance

All economic results referenced in C-Engine documentation are
**historical-replay simulations on genuine Alpaca SIP bars** — not live
performance, and not a profitability claim. The platform is
**ENGINEERING-CERTIFIED and RESEARCH_VALIDATED** with **NO certified economic
edge** (strategy economics: `NOT_CERTIFIED`), and is **not** paper- or
live-ready. No live orders were placed.
