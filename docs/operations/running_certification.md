# Running the C-Engine M7 Certification

Operational runbook for executing the C-Engine certification suite and the
continuous-runtime control commands. This document covers *how to run* the
certification and *where the evidence lands* — it does not restate the economic
verdicts (see the certification report for those).

> **Honest standing.** C-Engine is **ENGINEERING-CERTIFIED** and
> **RESEARCH_VALIDATED**: genuine leak-free data, all components wired, the
> native risk gate and the -0.5% protective exit are correct and independent,
> Metal kernels are correctness-preserving, and injected faults are isolated.
> It has **NO CERTIFIED ECONOMIC EDGE** and is **NOT** ready for paper or live
> capital. Running these commands measures and attests the platform — it does
> not place orders and does not manufacture an edge.

---

## 1. Prerequisites

| Requirement | Value at certification time |
|---|---|
| Host | Apple M5 Max, 48 GiB unified memory |
| Python | 3.11 |
| GPU stack | MLX / Metal (Metal device reports `Apple M5 Max`) |
| Data lakehouse | Local Parquet + DuckDB |
| Market data | 12.2M genuine Alpaca SIP bars |
| 1Min series | 6.76M bars / 5 canonical symbols (AAPL, MSFT, NVDA, SPY, QQQ), 2020-01-01 .. 2026-11-21 |
| 5Min series | 5.45M bars / 31 symbols |
| Price encoding | `price_scale = 10000` ticks |

The native extension `_cengine_native` must be importable (it exposes
`RiskEngine`, `DiscoveryEngine`, `MetalRuntime`, and the emergency-threshold
candidate). A documented CPU fallback exists via `DevicePolicy(force_cpu=True)`
with four CPU reference kernels, so the suite can run on a host without a usable
Metal device.

---

## 2. The certification CLI

All certification sections are driven through a single module entry point:

```
python -m cengine.certification <section>
```

### 2.1 Section subcommands

| Subcommand | Section | What it attests |
|---|---|---|
| `audit` | §M7A | Engineering audit: 103 modules via isolated-subprocess import + test-coverage index + runtime-consumption index. Static audit caps at `OPERATIONALLY_CONSUMED`. |
| `data` | §M7B | Data integrity: duplicates, monotonic timestamps, OHLC validity, non-negative volume, no look-ahead (`availability_ts >= event_ts`), uniform `price_scale`, trading-day coverage, 1m→5m bucket reconciliation. |
| `native` | §23 | Native / Metal: native `RiskEngine` as sole financial authority, Metal device + compiled kernels, GPU EMA/APO reproducing the CPU reference exactly. |
| `failure` | §24 | Failure injection: learning / GPU / data / broker faults isolated; P0 protection and P1 execution workers never auto-killed; single-instance lock; atomic checkpoint round-trip. |
| `protective-exit` | §25 | Protective exit: `ProtectiveExitCoordinator` unrealized-return math (long/short symmetric, fires at `ret <= -0.5%`); native -3% emergency candidate pure and inactive-by-default; `clamp_emergency_threshold` never widens risk. |
| `entry` | §26 | Entry contract: entries deterministic and look-ahead-free (causal) on the raw mask, subsampling disabled. |
| `strategy` | holdout economics | Strategy economics on the sealed holdout (incumbent: mandatory -0.5% protective exit + momentum entry). |
| `sensitivity` | §21 | Capital / cost sensitivity: net expectancy across cost levels, break-even cost, cost-robustness. |
| `stress` | §22 | Stress windows: 2020 COVID crash, 2022 bear market, 2021 calm control. |
| `portfolio` | portfolio | Portfolio-level construction / aggregation. |
| `execution` | execution | Execution-path attestation. |
| `learning` | learning | Learning: infrastructure, provenance discipline, recommendation-only, autonomous-optimization safety, data-eligibility. |
| `holdout` | §5 | Sealed-holdout mechanics: deterministic per-symbol split (train 60% / validation 20% / holdout 20%, embargoed), access-logged seal, scoring only. |
| `full` | — | Runs the complete chain end-to-end. |
| `report` | — | Renders the consolidated certification report from existing section artifacts. |

### 2.2 Section ordering

For a clean-room run, execute sections in dependency order — foundations
first, economics last:

```
audit → data → native → failure → protective-exit → entry
      → strategy → sensitivity → stress
      → portfolio → execution → learning → holdout
      → full → report
```

`full` runs this chain for you; the individual subcommands exist for targeted
re-runs and debugging.

---

## 3. Quick vs full modes

- **Quick mode** runs the focused attestation for a single section — use it to
  iterate on one domain (e.g. re-run `protective-exit` after touching the
  coordinator) without paying for the whole chain.
- **Full mode** (`full`) runs every section end-to-end and is the mode used to
  produce a complete certification pass. Run `full` before `report` so the
  report renders against a complete, current set of section artifacts.

A per-section run attests only that section. A readiness determination (§32)
requires the full chain, and the staged live path is hard-gated behind the full
chain *plus* an external human authorization token (see §6).

---

## 4. Where artifacts are written

Each section writes its evidence as a structured artifact; `report`
consolidates those artifacts into the certification report. Because section
artifacts are the inputs to `report`:

- Run the section (or `full`) **before** `report`, or the report renders stale
  or missing sections.
- Re-running a single section refreshes only that section's artifact; the
  remaining artifacts are untouched until you re-run them.

Determinism aids reproduction: the sealed holdout is derived deterministically
per symbol from the data span and accessed once per symbol via an access-logged
seal (scoring only — no symbol is touched twice and none is ever tuned on), and
the frozen economic gates (§20) are frozen by fingerprint with `verify_frozen()`
tripping on any threshold drift. A re-run on the same data span reproduces the
same splits and the same gate thresholds.

---

## 5. The continuous runtime commands

The continuous runtime is the exchange-calendar-driven operating-state machine
(overnight / pre-market / regular / after-hours / post-session / weekend /
holiday / degraded / recovering) governed by the `RuntimeSupervisor` and its
P0–P9 GPU priority ladder. It is driven through a separate entry point:

```
python -m cengine.runtime <status|once|run>
```

| Subcommand | Behavior |
|---|---|
| `status` | Reports the current operating state and supervisor status. Read-only. |
| `once` | Executes a single supervised cycle (one pass through the state machine) and exits. |
| `run` | Runs the supervisor continuously, transitioning operating state on the exchange calendar. |

### 5.1 Safety posture of the runtime

- **P0 protection is never gated.** Research priorities (`>= P6`) are suspended
  under GPU pressure and bounded in-session.
- **Off-session services are safe by construction:**
  `ContinuousOptimizationService` is shadow-only;
  `ContinuousAcquisitionService` is **DRY-RUN by default** (plan only — no
  fetch, no credentials); `ContinuousGovernanceService` monitors
  safety-invariants and detects decay but **never grants readiness**.
- **Execution is confined to `REGULAR_SESSION`** via an optional authorized
  trading-engine runner that is **absent by default** — so running the service
  never places live orders.
- A single-instance lock and atomic checkpoints protect against double-run and
  partial-write corruption; the supervisor goes **DEGRADED on regression** and
  auto-recovers.

### 5.2 Running as a managed service

The runtime can be installed as a macOS `launchd` LaunchAgent (`RunAtLoad` +
`KeepAlive`):

```
scripts/install_runtime_service.sh
```

Installing the LaunchAgent does **not** change the execution posture: with no
authorized trading-engine runner present (the default), the service plans,
measures, and monitors but places no live orders.

---

## 6. What running the suite does *not* do

- It does **not** make C-Engine profitable or confer an economic edge. On the
  sealed holdout the production incumbent has **negative** net expectancy on
  every canonical symbol; strategy economics (§), cost sensitivity (§21), and
  stress (§22) are all **NOT_CERTIFIED**.
- It does **not** promote the platform to paper or live. Paper certification
  requires `>= 30` genuine elapsed forward sessions plus a research-validated
  edge; none exist. Live is **INACTIVE**.
- It does **not** open a live path. Staged live deployment
  (SHADOW log-only → PILOT_TINY 1% cap → SCALED 10% cap) is hard-gated behind
  the full chain **and** an external human authorization token that the
  software cannot mint. There is no autonomous activation path. The -0.5% exit
  and the native risk gate remain the authority throughout.

---

## 7. Typical workflows

**Attest one domain after a change**

```
python -m cengine.certification protective-exit
```

**Full certification pass + report**

```
python -m cengine.certification full
python -m cengine.certification report
```

**Check the runtime without running it**

```
python -m cengine.runtime status
```

**Single supervised cycle (no continuous loop, no live orders by default)**

```
python -m cengine.runtime once
```

---

*Provenance: all economic results produced by this suite are historical-replay
simulations on genuine bars — not live performance, and not a profitability
claim. No live orders were placed.*
