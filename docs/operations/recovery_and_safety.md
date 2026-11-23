# Recovery & Operational Safety (Milestone 7 Addendum §2 / §3 / §12)

This document specifies how the C-Engine continuous runtime stays safe under faults
and recovers from crashes. It covers five mechanisms certified under the **Failure
Injection (§24)** suite and the **Continuous Runtime** addendum:

1. **Failure isolation** — non-essential workers can fail without touching protection.
2. **DEGRADED-on-regression + auto-recover** — the state machine reacts to safety decay.
3. **Single-instance trading-authority lock** — only one trading process may ever run.
4. **Atomic checkpoints** — durable, crash-safe runtime state.
5. **Continuous governance monitoring** — detects decay, but **can never grant readiness**.

The governing principle throughout: **faults in learning, discovery, GPU, or data NEVER
cascade into protection or execution.** Protection and the native risk gate remain the
sole financial authority regardless of what fails above them.

Source: `cengine/runtime/supervisor.py`, `cengine/runtime/service.py`,
`cengine/runtime/state_machine.py`, `cengine/runtime/continuous.py`.

---

## 1. Failure Isolation: Non-Essential vs Essential Workers

The `RuntimeSupervisor` registers every worker with a `Priority` on a **P0–P9 GPU
priority ladder** and an `essential` flag. Essential workers (P0/P1) are the financial
core; everything above them is advisory research/analytics that the engine can lose
without consequence to safety.

| Worker | Priority | Essential | Role |
|---|---|---|---|
| `protection` | **P0** PROTECTION | **yes** | Financial protection + broker reconciliation — never gated, never auto-killed |
| `execution` | **P1** MARKET_EXEC | **yes** | Native market processing + order execution |
| `discovery` | P2 DISCOVERY | no | Metal discovery + technical analytics |
| `acquisition` | P5 FEATURES | no | Historical feature / data acquisition (DRY-RUN by default) |
| `optimization` | P9 RESEARCH | no | Large off-session research / optimization (shadow-only) |

### Isolation contract (§3 / §24)

When a worker raises, `RuntimeSupervisor.record_failure(name, error)` decides the blast
radius **purely from the `essential` flag**:

- **Non-essential failure** (learning / discovery / acquisition / optimization) is
  **ISOLATED**. It is logged, the worker is marked unhealthy and its restart counter
  bumped, and the response always reports `financial_protection_affected = False`.
  The fault never propagates to P0/P1.
- **Essential failure** (P0/P1) is **not** silently killed — the supervisor restarts
  it. Essential workers are never auto-terminated by the isolation path.

The per-tick `dispatch()` wraps every non-essential job (optimization, acquisition,
governance full-recert) in its own `try/except` that routes exceptions to
`record_failure(...)` rather than letting them escape the loop. A GPU, data, learning,
or broker fault in any of these is swallowed at the worker boundary.

Certified behaviors under §24 (Failure Injection — **CERTIFIED**):

- Learning / GPU / data / broker faults are isolated.
- Essential **P0 protection** and **P1 execution** workers are never auto-killed.
- The protective-exit scan **swallows a per-symbol broker failure and never raises** —
  a broker error on one symbol cannot stop the protective sweep across the rest.
- Faults in learning / discovery / GPU / data **NEVER cascade** into protection or
  execution.

### Why protection survives everything

The `ProtectiveExitCoordinator` (§25, **CERTIFIED**) imports **no ML / GPU / lakehouse /
scanner module**. Protection is structurally independent of the learning and discovery
subsystems, so there is no import-time or runtime path by which their failures can
reach it. The native `RiskEngine` (§23) remains the **sole financial authority** — it
accepts a valid order and rejects an oversized one with `QUANTITY_LIMIT_EXCEEDED`
independent of any worker health.

---

## 2. DEGRADED-on-Safety-Regression + Auto-Recover

The `OperatingStateMachine` is normally driven by the exchange calendar (overnight /
pre-market / regular / after-hours / post-session / weekend / holiday). On top of that,
the supervisor can impose **override** states: `DEGRADED`, `RECOVERING`, `SHUTTING_DOWN`.

Every tick, `governance_tick(now_ns)` runs a cheap safety-invariant check and reacts:

```
inv = gov.check_invariants()
if not inv["ok"]:
    set_degraded("safety invariant regression: <failures>")   # override -> DEGRADED
elif previously degraded and invariants healthy again:
    clear_override()                                           # auto-recover
```

- **On regression:** the moment a safety invariant fails, the machine is forced into
  **DEGRADED**, the reason is recorded, and the condition is logged at error level. The
  override is materialized immediately onto `.state` (the next `tick()` cannot time-drive
  past it).
- **Auto-recover:** once the invariants read healthy again, the override is cleared and
  the machine returns to its normal **time-driven** state. No manual intervention and no
  operator "all clear" is required to resume normal operation — recovery is a direct
  function of invariants being healthy, not of a human assertion.

DEGRADED is a **containment** state, not a readiness downgrade. It expresses "a safety
property is currently failing"; it never expresses "cleared for capital."

---

## 3. Single-Instance Trading-Authority Lock (§2)

Two trading processes contending for the same account is a correctness and safety
hazard. `SingleInstanceLock` is a PID-file lock acquired at the start of `run()`,
**before** any work:

- On `acquire()`, if the lock file exists and names a **live** PID, it raises
  `SingleInstanceError` — a second trade instance is refused outright.
- A **stale** lock (dead PID, or an unreadable/garbage file) is reclaimed, so a prior
  crash does not permanently wedge the runtime.
- On shutdown, `release()` removes the lock only if it still holds the current PID,
  avoiding races that would delete another owner's lock.

Certified under §24: **the single-instance lock blocks a second trade instance.** Only
one process can ever hold trading authority at a time.

---

## 4. Atomic Checkpoints for Crash Recovery (§12)

Runtime state is persisted durably so a crash or restart resumes cleanly instead of
replaying or double-firing scheduled work.

**Write path — atomic by construction:** `write_checkpoint(path, state)` writes to a
temp file in the destination directory, `flush()` + `os.fsync()` to force the bytes to
disk, then `os.replace(tmp, path)`. Because `os.replace` is an atomic rename, a reader
(or a crash) ever sees **either** the complete old checkpoint **or** the complete new
one — never a torn, half-written file. A failed write unlinks the temp file and
re-raises; it never corrupts the existing checkpoint.

**Read path — fault-tolerant:** `read_checkpoint(path)` returns `None` for a missing
file and swallows `OSError` / `JSONDecodeError`, so a damaged or absent checkpoint
degrades to a clean cold start rather than crashing the service.

**What is checkpointed:** service version, timestamp, current operating state, the
last-run timestamps of each scheduled job, the degraded reason (if any), and supervisor
worker health. On startup, `run()` reads the prior checkpoint and restores the
scheduling cadence so off-session jobs pick up where they left off. Checkpoints are
written on a fixed interval during the loop and once more on shutdown
(`extra={"shutdown": True}`).

Certified under §24: **an atomic checkpoint survives a round-trip** (write → read →
equal state).

---

## 5. Continuous Governance: Detects Decay, Never Grants Readiness

The `ContinuousGovernanceService` runs two cadences of safety monitoring:

- **`check_invariants()`** — fast (sub-second) checks every tick, over the core safety
  certifications: the **mandatory protective exit**, the **native module**, the **native
  risk gate**, and **worker isolation**. Any one failing drives the runtime DEGRADED
  (see §2).
- **`full_recert()`** — a slower periodic quick full-certification that re-runs the
  suite and reports the headline readiness and per-domain status.

### The one-way property

Governance is **strictly a decay detector**. It can only ever move the system *toward*
caution:

- It can flag a regression and force **DEGRADED**.
- It can report that infrastructure is still sound.
- It **cannot** promote the system, lift a gate, or assert economic readiness.

This is enforced by what governance has no code path to do. Readiness (§32) is
**RESEARCH_VALIDATED** with `live_eligible = False`; there is **no certified economic
edge** and the system is **NOT_READY for capital**. Nothing in the continuous loop can
change those verdicts:

- The staged live deployment (SHADOW → PILOT_TINY 1% → SCALED 10%) is **hard-gated**
  behind the full certification chain **plus an external human authorization token the
  software cannot mint**. There is **no autonomous activation path**.
- Execution is confined to `REGULAR_SESSION` and only via an **optional** authorized
  trading-engine runner that is **absent by default** — running the continuous service
  therefore **never places live orders**.
- The off-session services are non-mutating by design: `ContinuousOptimizationService`
  is **shadow-only**, and `ContinuousAcquisitionService` is **DRY-RUN by default** (plan
  only — no fetch, no credentials).
- `FrozenEconomicGates` (§20) are frozen by fingerprint; `verify_frozen()` trips on any
  threshold drift, so governance cannot quietly retrofit easier economic criteria to
  manufacture a pass.

Governance monitoring tightens; it never loosens. Any widening of risk requires a human
token outside the software boundary.

---

## Operating & Verifying

Continuous runtime CLI:

```
python -m cengine.runtime status   # current operating state, worker health, plans
python -m cengine.runtime once     # one planned tick (deterministic)
python -m cengine.runtime run      # persistent supervised loop
```

Certification of these safety mechanisms:

```
python -m cengine.certification failure          # §24 failure injection + isolation
python -m cengine.certification protective-exit   # §25 mandatory-exit independence
python -m cengine.certification native            # §23 native risk gate authority
```

Service installation (macOS): `scripts/install_runtime_service.sh` installs a `launchd`
LaunchAgent with `RunAtLoad` + `KeepAlive`. KeepAlive gives OS-level process
supervision; the single-instance lock guarantees a relaunch cannot double up on trading
authority, and the atomic checkpoint lets the relaunched process resume its cadence.

---

## Summary

| Mechanism | Guarantee | Status |
|---|---|---|
| Worker isolation | Non-essential faults never reach P0/P1; essential workers never auto-killed | §24 CERTIFIED |
| Protective exit independence | Imports no ML/GPU/lakehouse/scanner; swallows per-symbol broker failure | §25 CERTIFIED |
| DEGRADED + auto-recover | Safety regression forces DEGRADED; recovers automatically when healthy | Addendum |
| Single-instance lock | Second trade instance refused; stale locks reclaimed | §24 CERTIFIED |
| Atomic checkpoint | Temp + fsync + `os.replace`; survives write→read round-trip | §24 CERTIFIED |
| Governance monitoring | Detects decay only; cannot grant readiness or mint authorization | Addendum |

The runtime is built so that the worst an upstream fault can do is take itself offline
and (if a safety invariant regresses) drive the system into DEGRADED. It can never
cascade into protection or execution, never place a live order on its own, and never
promote the system toward capital.

---

**Provenance.** All economic results referenced here are **historical-replay simulations
on genuine bars — not live performance, and not a profitability claim.** No live orders
were placed. The continuous runtime detects safety decay and enforces containment; it
does not and cannot assert an economic edge, which remains **NOT_CERTIFIED**.
