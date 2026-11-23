# Running the C-Engine Runtime as a macOS LaunchAgent

This document describes how the C-Engine continuous runtime is operated as a
per-user macOS **LaunchAgent**: the plist template, the install/uninstall
scripts, the `RunAtLoad` + `KeepAlive` restart-on-crash policy, log locations,
and the hard safety boundary that **running the agent alone never places live
orders.**

> **Safety summary (read first).** The LaunchAgent runs
> `python -m cengine.runtime run` 24/7. Continuous operation is **not**
> continuous execution. Acquisition is **dry-run by default** (plan only — no
> fetch, no credentials), and **no trading-engine runner is wired by default**,
> so the agent drives off-session research and continuous governance only. Live
> status is **INACTIVE**; `live_eligible=False`. Nothing in this setup can
> activate trading.

---

## 1. Overview

The continuous runtime is an exchange-calendar-driven operating-state machine
(overnight / pre-market / regular / after-hours / post-session / weekend /
holiday / degraded / recovering) wrapped in a `RuntimeSupervisor` with a P0–P9
GPU priority ladder. P0 protection is never gated; research work (priority ≥ P6)
is suspended under GPU pressure and bounded in-session. Off-session services are:

- **ContinuousOptimizationService** — shadow-only; proposes nothing that widens
  risk and retains the incumbent unless a genuine improvement certifies.
- **ContinuousAcquisitionService** — **DRY-RUN by default**: it plans only, with
  no fetch and no credentials.
- **ContinuousGovernanceService** — safety-invariant monitoring; detects decay
  and never grants readiness.

The runtime holds a single-instance lock, writes atomic checkpoints, enters
`DEGRADED` on regression, and auto-recovers. A LaunchAgent is simply the macOS
mechanism that keeps this loop alive across logins and crashes.

---

## 2. Components

| Path | Role |
| --- | --- |
| `deploy/launchd/com.cengine.runtime.plist.template` | LaunchAgent plist template (`__REPO__` / `__PYTHON__` placeholders) |
| `scripts/install_runtime_service.sh` | Renders the template, installs it to `~/Library/LaunchAgents`, and loads it |
| `scripts/uninstall_runtime_service.sh` | Unloads and removes the installed plist (leaves `data/` and logs intact) |
| `python -m cengine.runtime run` | The persistent 24/7 supervisory loop the agent keeps alive |

The LaunchAgent label is **`com.cengine.runtime`**.

---

## 3. The plist template

The template lives at `deploy/launchd/com.cengine.runtime.plist.template`. It is
not a ready-to-load plist on its own: the two placeholders `__REPO__` and
`__PYTHON__` are substituted at install time by
`scripts/install_runtime_service.sh`.

Key fields and why they are set:

| Key | Value | Purpose |
| --- | --- | --- |
| `Label` | `com.cengine.runtime` | Unique agent identifier used by `launchctl` |
| `ProgramArguments` | `__PYTHON__ -m cengine.runtime run` | Launches the persistent supervisory loop |
| `WorkingDirectory` | `__REPO__` | Runs from the repo root so relative paths resolve |
| `EnvironmentVariables` → `PYTHONUNBUFFERED` | `1` | Line-buffered logs for live `tail -f` |
| `EnvironmentVariables` → `CENGINE_RUNTIME_ENABLE_FETCH` | `0` | Keeps acquisition in dry-run (no fetch, no credentials) |
| `RunAtLoad` | `true` | Start immediately on load / at login |
| `KeepAlive` → `SuccessfulExit` = `false` | restart-on-crash | Relaunch on abnormal exit; a clean exit stays stopped |
| `ThrottleInterval` | `30` | Minimum seconds between relaunches (prevents crash-spin) |
| `StandardOutPath` | `__REPO__/data/runtime/logs/runtime.out.log` | stdout log |
| `StandardErrorPath` | `__REPO__/data/runtime/logs/runtime.err.log` | stderr log |
| `ProcessType` | `Background` | Scheduled as a background workload |

### 3.1 `RunAtLoad` + `KeepAlive` (restart-on-crash)

- **`RunAtLoad = true`** starts the runtime the moment the agent is loaded and on
  every subsequent login.
- **`KeepAlive` with `SuccessfulExit = false`** instructs `launchd` to relaunch
  the process whenever it exits abnormally (a crash), while allowing a clean,
  intentional exit (e.g. during uninstall) to stay stopped.
- **`ThrottleInterval = 30`** caps relaunch frequency so a persistently failing
  process backs off instead of spinning.

Together these give a crash-resilient 24/7 runtime without any external
supervisor. The runtime's own single-instance lock ensures that even if a stale
process and a relaunch briefly overlap, a second trade instance cannot start.

---

## 4. Installing the service

```bash
scripts/install_runtime_service.sh
```

The installer:

1. Resolves the repo root and prefers the project venv Python
   (`.venv/bin/python`), falling back to `python3` on `PATH`.
2. Verifies the template exists at
   `deploy/launchd/com.cengine.runtime.plist.template`.
3. Creates `~/Library/LaunchAgents` and `data/runtime/logs` if missing.
4. Substitutes `__REPO__` and `__PYTHON__` into the template and writes the
   rendered plist to `~/Library/LaunchAgents/com.cengine.runtime.plist`.
5. Unloads any previously installed copy, then `launchctl load`s the new one.

On success it prints the status, log, and stop commands. The install is
idempotent — re-running it re-renders and reloads the agent in place.

### Verify it is running

```bash
launchctl list | grep com.cengine.runtime
```

You can also query runtime state directly without the agent:

```bash
python -m cengine.runtime status   # current operating state, tick plan, worker health, acquisition gap
python -m cengine.runtime once     # one bounded tick (plan + one governance check + due off-session work)
```

---

## 5. Uninstalling the service

```bash
scripts/uninstall_runtime_service.sh
```

This unloads the agent (if present) and removes
`~/Library/LaunchAgents/com.cengine.runtime.plist`. It **does not** touch `data/`
or the logs, so checkpoints and history survive a reinstall. If no plist is
installed, it reports that there is nothing to do.

---

## 6. Logs

| Stream | Path |
| --- | --- |
| stdout | `data/runtime/logs/runtime.out.log` |
| stderr | `data/runtime/logs/runtime.err.log` |

Because `PYTHONUNBUFFERED=1` is set in the plist, output is unbuffered and safe
to follow live:

```bash
tail -f data/runtime/logs/runtime.out.log
```

Related runtime state (not logs) lives alongside the logs under `data/runtime/`:

- `data/runtime/trade.lock` — single-instance lock (blocks a second trade instance).
- `data/runtime/checkpoint.json` — atomic checkpoint, written on an interval and
  on shutdown; survives a round-trip for clean recovery on relaunch.

---

## 7. Safety: the agent alone never places live orders

This is the most important operational property of the LaunchAgent, and it is
enforced in several independent layers:

- **Acquisition is dry-run by default.** `ContinuousAcquisitionService` plans
  only — no fetch, no credentials. The plist pins
  `CENGINE_RUNTIME_ENABLE_FETCH=0`, and the service config default is
  `enable_fetch = False`.
- **No trading-engine runner is wired by default.** Elective order placement is
  confined to `REGULAR_SESSION` **and** requires a separately-authorized
  trading-engine runner that is **absent by default**. The supervisory loop only
  starts a trading task when an execution window is open *and* a runner was
  explicitly injected — with no runner, that branch is never taken. Therefore
  running this agent alone places **no live orders**.
- **Optimization and governance are advisory.** `ContinuousOptimizationService`
  is shadow-only; `ContinuousGovernanceService` monitors invariants and **never
  grants readiness**. No learning or discovery module imports the order path or
  calls `.submit()`.
- **The risk authority is unchanged.** The native `RiskEngine` remains the sole
  financial authority and the mandatory `-0.5%` protective exit remains active;
  fault injection in learning / discovery / GPU / data never cascades into
  protection or execution.
- **Live is hard-gated.** Live deployment is staged
  (`SHADOW` log-only → `PILOT_TINY` 1% cap → `SCALED` 10% cap) behind the full
  certification chain **and** an external human authorization token the software
  cannot mint. There is no autonomous activation path. Current live status is
  **INACTIVE**, `live_eligible=False`.

If you ever intend to run execution, that requires wiring an explicitly
authorized trading-engine runner and clearing the external authorization gate —
neither of which the install script, the plist, or the runtime can do on their
own.

---

## 8. Troubleshooting

| Symptom | Check |
| --- | --- |
| Agent not listed | `launchctl list \| grep com.cengine.runtime`; re-run the installer |
| Immediate re-exit loop | Inspect `data/runtime/logs/runtime.err.log`; `ThrottleInterval=30` paces relaunches |
| "another instance holds" | A `trade.lock` is held (`data/runtime/trade.lock`); ensure no other runtime is running |
| Template not found | Confirm `deploy/launchd/com.cengine.runtime.plist.template` exists before installing |
| Wrong Python | Installer prefers `.venv/bin/python`; create the venv or ensure `python3` is on `PATH` |

---

## 9. Certification context

The continuous runtime and its LaunchAgent packaging are part of the
**engineering-certified, research-validated** C-Engine platform. Failure
injection is certified: essential P0 protection and P1 execution workers are
never auto-killed, the single-instance lock blocks a second trade instance, and
the atomic checkpoint survives a round-trip. The platform has **no certified
economic edge** and is **not ready for paper or live capital** — see the
readiness and strategy-economics certification reports. The LaunchAgent exists to
keep the research/governance substrate alive, not to trade.

---

> **Provenance.** Any economic results referenced here are **historical-replay
> simulations on genuine bars — not live performance and not a profitability
> claim. No live orders were placed.**
