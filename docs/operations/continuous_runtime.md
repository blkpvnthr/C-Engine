# Continuous 24/7 Runtime (Milestone 7 Addendum)

C-Engine runs as a persistent, exchange-calendar-driven service that stays alive across
sessions, overnight, weekends, and holidays — reusing the existing M1–M6B workers and
adding **no new financial authority**. The governing principle is one sentence:

> **Continuous operation is not continuous execution.**

The process is always running. Elective orders are confined to the regular session (and
only when an authorized trading-engine runner is explicitly wired — absent by default).
Protective monitoring, reconciliation, and safety-invariant governance run in **every**
state; research, learning, and data work run on spare, off-session GPU capacity.

---

## Module map

| Concern | Module | Key symbols |
|---|---|---|
| Exchange calendar / session phase | `cengine/runtime/calendar.py` | `SessionPhase`, `is_regular_session`, `session_phase`, `next_transition` |
| Operating-state machine | `cengine/runtime/state_machine.py` | `OperatingState`, `OperatingStateMachine`, `derive_state`, `execution_permitted`, `research_appropriate` |
| Supervisor / GPU ladder / durability | `cengine/runtime/supervisor.py` | `RuntimeSupervisor`, `Priority`, `SingleInstanceLock`, `read_checkpoint`, `write_checkpoint` |
| Off-session services | `cengine/runtime/continuous.py` | `ContinuousOptimizationService`, `ContinuousAcquisitionService`, `ContinuousGovernanceService` |
| Coordination service | `cengine/runtime/service.py` | `ContinuousRuntimeService`, `ContinuousRuntimeConfig`, `TickPlan` |
| Entrypoint | `cengine/runtime/__main__.py` | `python -m cengine.runtime <status\|once\|run>` |
| macOS install | `scripts/install_runtime_service.sh` | launchd LaunchAgent (`RunAtLoad` + `KeepAlive`) |

---

## Operating-state machine

The engine's operating state is **derived from the exchange calendar** (time-driven),
then adjusted by operational overrides (`DEGRADED` / `RECOVERING`) set by the supervisor.
Only the regular session permits elective orders; the off-session states are when
research and analytics are appropriate.

| State | Trigger | Elective execution | Research appropriate |
|---|---|---|---|
| `INITIALIZING` | process start, pre-first-tick | No | No |
| `PRE_MARKET` | pre-market calendar phase | No (unless separately authorized) | No |
| `REGULAR_SESSION` | regular trading phase | **Yes** (only via authorized runner) | No (research bounded/suspended) |
| `AFTER_HOURS` | after-hours calendar phase | No (unless separately authorized) | No |
| `POST_SESSION_PROCESSING` | post-session window on a trading day | No | Yes |
| `OVERNIGHT_RESEARCH` | overnight on a trading day | No | Yes |
| `WEEKEND_RESEARCH` | non-trading weekend day | No | Yes |
| `HOLIDAY_RESEARCH` | exchange holiday | No | Yes |
| `DEGRADED` | a safety/certification regression detected | No | No |
| `RECOVERING` | regression cleared, returning to normal | No | Bounded |

The nine calendar/operational states above are the FACTS-enumerated set
(overnight / pre-market / regular / after-hours / post-session / weekend / holiday /
degraded / recovering); `INITIALIZING` and a terminal `SHUTTING_DOWN` are lifecycle
transitions around them.

Execution eligibility is a hard function of state: `execution_permitted(state)` returns
true **only** for `REGULAR_SESSION`. This is enforced structurally, independent of any
strategy opinion.

---

## RuntimeSupervisor and the P0–P9 GPU priority ladder

The `RuntimeSupervisor` is a **policy/coordination layer**, not a new execution path. It
decides which reused workers are active per state, whether a GPU research workload may be
admitted under memory/compute pressure, and it isolates recoverable failures so that a
learning or analytics fault can **never** terminate financial protection or broker
reconciliation.

Workloads are ranked on a priority ladder. P0 protection is never gated; research-tier
workloads (priority **≥ P6**) are suspended under GPU pressure and bounded while a live
session is in progress.

| Priority | Workload | Gating |
|---|---|---|
| P0 | Financial protection + broker reconciliation | **Never gated** |
| P1 | Native market processing + execution | Essential (never isolated-killed) |
| P2 | Metal discovery + technical analytics | In-session eligible |
| P3 | Production ML inference | In-session eligible |
| P4 | Portfolio scenario analytics | In-session eligible |
| P5 | Historical feature construction | In-session eligible |
| P6 | ML training | **Research tier** — suspended under GPU pressure, bounded in-session |
| P7 | Bandit | Research tier |
| P8 | Offline RL | Research tier |
| P9 | Large historical research experiments | Research tier |

The supervisor admits GPU research only below a configurable pressure ceiling
(`max_gpu_pressure`, default `0.85`). Essential P0/P1 workers are never auto-killed on
fault; recoverable faults in learning / discovery / GPU / data are isolated and never
cascade into protection or execution.

---

## Off-session services

Three bounded, idempotent services run only when the market is closed. **None** touches
the live order path, expands risk, or fabricates data.

### 1. `ContinuousOptimizationService` — shadow-only (§6)

Runs one **governed** autonomous-optimization cycle per invocation on genuine historical
bars. It is strictly shadow-only: the governed optimization loop **retains the incumbent**
unless a candidate genuinely beats it out-of-sample. Per the economic certification, none
do — so the incumbent is retained, which is the correct falsification-first outcome. The
optimizer cannot expand risk: `OptimizationConfig.validate` refuses autonomous risk
expansion, locks the −0.5% production exit, refuses to loosen the −3% emergency threshold,
and allows only tightening.

### 2. `ContinuousAcquisitionService` — DRY-RUN by default (§9)

Keeps the lakehouse current by computing the per-symbol data **gap** (last stored bar →
now). **By default it is DRY-RUN: it reports the acquisition plan without fetching**, so
it never fabricates bars and requires no network access or credentials. Fetching the
missing window via the existing SIP ingester happens only when explicitly enabled **and**
credentialed.

### 3. `ContinuousGovernanceService` — monitoring only (§10)

Continuously re-checks the **safety invariants** — the −0.5% exit contract, the native
risk gate, and fault isolation — cheaply and frequently, plus the full quick-certification
less often. Any regression flags the runtime `DEGRADED`. Crucially, **governance can only
detect decay; it never grants readiness or upgrades a verdict.** It cannot turn
`NOT_CERTIFIED` into certified, nor make the engine paper- or live-eligible.

---

## Coordination loop and tick plan

`ContinuousRuntimeService` ticks the state machine, builds a `TickPlan` (active workers,
which off-session services are due, whether an execution window is open, current GPU
pressure), dispatches it, verifies the safety invariants, and checkpoints. Default
cadences (`ContinuousRuntimeConfig`) are off-session and bounded — optimization and
acquisition run on multi-hour intervals, the full governance pass less often, with
frequent lightweight invariant checks and atomic checkpoints every few minutes.

---

## Durability and safety guarantees

- **Single-instance lock.** A `SingleInstanceLock` enforces exactly one trading-authority
  instance; a second instance is blocked (`SingleInstanceError`). This is the same
  single-instance guarantee validated under failure injection.
- **Atomic checkpoints.** Runtime state is written atomically and survives a round-trip,
  so the supervisor resumes cleanly after restart.
- **DEGRADED-on-regression + auto-recover.** A detected safety/certification regression
  drives the runtime to `DEGRADED`; once cleared it transitions through `RECOVERING` back
  to normal operation. Elective execution is disabled while degraded.
- **Fault isolation.** Faults in learning, discovery, GPU, or data are swallowed and never
  cascade into P0 protection or P1 execution. The protective-exit scan swallows a
  per-symbol broker failure and never raises.
- **Protection authority is unchanged.** The −0.5% protective exit and the native
  `RiskEngine` gate remain the sole financial authority across every state. The protective
  coordinator imports no ML/GPU/lakehouse/scanner module, so protection is independent of
  learning and discovery.

---

## Execution confinement (why running the service is safe)

Elective order placement is confined to `REGULAR_SESSION` **and** requires an optional,
explicitly-injected authorized trading-engine runner. That runner is **absent by
default**, so:

> Running `python -m cengine.runtime run` with the default configuration never places a
> live order. It drives off-session research and continuous governance only.

Beyond that, live trading remains **INACTIVE** and hard-gated. Staged live deployment
(SHADOW log-only → PILOT_TINY 1% cap → SCALED 10% cap) sits behind the full certification
chain plus an external human authorization token the software cannot mint. There is no
autonomous activation path, and the current strategy carries no certified economic edge
(`NOT_CERTIFIED` economics; readiness `RESEARCH_VALIDATED`; `live_eligible=False`).

---

## CLI

```bash
# Print current operating state, tick plan, worker health, acquisition gap
python -m cengine.runtime status

# Run a single bounded tick (plan + one governance invariant check + due off-session work)
python -m cengine.runtime once

# Run the persistent 24/7 supervisory loop (Ctrl-C / SIGTERM to stop)
python -m cengine.runtime run
```

Related certification entrypoints: `python -m cengine.certification <audit|data|native|
failure|protective-exit|entry|strategy|sensitivity|stress|portfolio|execution|learning|
holdout|full|report>`.

---

## Installation (macOS)

Install the runtime as a user LaunchAgent so it starts at login and is kept alive:

```bash
scripts/install_runtime_service.sh
```

The agent runs `python -m cengine.runtime run` with `RunAtLoad` + `KeepAlive`, restarting
it on crash. Because no trading-engine runner is wired and acquisition stays dry-run,
installing the agent does not place live orders — it provides continuous governance and
off-session research only.

---

## What continuous operation does and does not mean

| Always on (every state) | Off-session only | Regular session only | Never autonomous |
|---|---|---|---|
| P0 protection (−0.5% exit, native risk gate) | Shadow optimization | Elective order placement | Live activation |
| Broker reconciliation | Dry-run acquisition planning | (via authorized runner, absent by default) | Risk expansion |
| Safety-invariant governance | ML training / bandit / RL research | — | Readiness upgrade |
| Single-instance lock + checkpoints | Feature/analytics backfill | — | Widening the −3% emergency |

---

*Provenance: all economic results referenced here (the governed optimizer retaining the
incumbent; the NOT_CERTIFIED economic verdict) are historical-replay simulations on
genuine bars — not live performance and not a profitability claim. No live orders were
placed.*
