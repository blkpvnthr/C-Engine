# C-Engine M7 Certification

**Verdict:** `RESEARCH_VALIDATED` — infrastructure sound, **no certified economic edge**, **not** live- or paper-ready.

M7 is an **independent certification layer** placed over the already-built M1–M6B stack. It does **not** rebuild the engine. It attempts to *falsify* every claim the platform makes about itself — data integrity, engineering wiring, the native risk gate, the protective exit, Metal correctness, fault isolation, learning discipline, and above all the *economics* of the production strategy — and it records exactly what survived and what did not.

The honest result: C-Engine is **engineering-certified** and **research-validated** as a sound research + execution substrate, but the production incumbent strategy has **no certified economic edge** and the system is **NOT ready** for paper or live capital. Per the falsification-first mandate, the decision is to **REJECT the current strategy as a deployable edge** while **ACCEPTING the platform** as a sound research + execution foundation.

---

## Platform & data at certification time

| Property | Value |
| --- | --- |
| Compute | Local-first native quant stack on Apple M5 Max (48 GiB unified) |
| Runtime | Python 3.11, MLX / Metal GPU |
| Lakehouse | Local Parquet + DuckDB |
| Total bars | 12.2M genuine Alpaca SIP bars |
| 1-minute | 6.76M bars / 5 canonical symbols (AAPL, MSFT, NVDA, SPY, QQQ), 2020-01-01 .. 2026-11-21 |
| 5-minute | 5.45M bars / 31 symbols |
| Price encoding | `price_scale = 10000` ticks |

The 5 canonical symbols at 1-minute resolution are the universe for all economic certification (sealed holdout, cost sensitivity, stress).

---

## Domain roll-up

Eight engineering/infrastructure domains plus the economics track. Infrastructure domains are certified; the economics track is honestly **not** certified.

| Domain | Status | What it means |
| --- | --- | --- |
| Data integrity (M7B) | **CERTIFIED** | 0 duplicate bars, strictly monotonic timestamps, valid OHLC + non-negative volume, no look-ahead (`availability_ts >= event_ts`), uniform `price_scale=10000`, 100% NYSE trading-day coverage on all 36 series, 1m→5m buckets reconcile 99.99% (AAPL spot-check, 8419 buckets). |
| Engineering audit (M7A) | **CERTIFIED** | 103 modules audited, 0 BLOCKED. Status derived mechanically from isolated-subprocess import + test-coverage index + runtime-consumption index. Static audit caps at `OPERATIONALLY_CONSUMED`; empirical rungs are granted only by cert runs. Fixed a real bug: `gpu_worker.py` instantiated `MetalRuntime` at import time against a hard-coded `build/` path → made lazy + `discover_metallib()`. |
| Native / Metal (§23) | **CERTIFIED** | `_cengine_native` exposes `RiskEngine` / `DiscoveryEngine` / `MetalRuntime` + the emergency candidate. The native `RiskEngine` is the sole financial authority (accepts a valid 10-share order, rejects a 50k-share order with `QUANTITY_LIMIT_EXCEEDED`). Metal device *Apple M5 Max* runs compiled kernels; GPU EMA/APO reproduce the CPU reference (`metal_cpu_ref`) **exactly** (correctness-preserving). Documented CPU fallback via `DevicePolicy(force_cpu=True)` + 4 CPU reference kernels. |
| Protective exit (§25) | **CERTIFIED** | `ProtectiveExitCoordinator` unrealized-return math is long/short symmetric and fires at `ret <= -0.5%` (at −0.5% fires, at −0.4% holds). The coordinator imports no ML/GPU/lakehouse/scanner module (protection is independent of learning + discovery). The native −3% emergency candidate is pure, INACTIVE-by-default, and `clamp_emergency_threshold` never widens risk (result ≥ current; tightening allowed, loosening refused). |
| Entry contract (§26) | **CERTIFIED** | Entries are deterministic and look-ahead-free (causal — truncating the series never changes a past entry), verified on the raw mask with subsampling disabled. |
| Failure injection (§24) | **CERTIFIED** | Learning / GPU / data / broker faults are isolated. Essential P0 protection / P1 execution workers are never auto-killed; the protective-exit scan swallows a per-symbol broker failure and never raises; a single-instance lock blocks a second trade instance; an atomic checkpoint survives a round-trip. Faults in learning/discovery/GPU/data never cascade into protection or execution. |
| Learning | **CERTIFIED** (infra) / **INSUFFICIENT_DATA** (edge) | Infrastructure certified; provenance discipline certified (synthetic & `historical_replay` results can never back a profitability claim — only genuine `real` realized outcomes can); recommendation-only certified (no learning module imports the order path or calls `.submit()`); autonomous-optimization safety certified (`OptimizationConfig.validate` refuses autonomous risk expansion, locks the −0.5% production exit, refuses to loosen the −3% emergency, allows tightening). Data-eligibility is **INSUFFICIENT_DATA**: only a *theoretical* `forward_return` target is validation-eligible; the 6 strategy-relevant / bandit / RL tasks lack materialized realized-outcome layers / live sessions; `bandit_rl` is INSUFFICIENT_DATA. The governed optimization loop **retained the incumbent** — it certified *no* economic improvement (the correct falsification-first outcome). |
| Strategy economics (sealed holdout, §5) | **NOT_CERTIFIED** (0/5 symbols) | The production incumbent (mandatory −0.5% protective exit + momentum entry) has **negative** net expectancy on the sealed holdout for every canonical symbol (expectancy ≈ −0.001/trade, profit factor ≈ 0.6, Sharpe ≈ −0.2, drawdown > 90%). It fails to beat buy-and-hold on every symbol and the random placebo on 3/5. The −0.5% exit is a **risk backstop, not an economic edge**. |
| Capital / cost sensitivity (§21) | **NOT_CERTIFIED** | Negative net expectancy even at **zero** cost (AAPL −0.00033, SPY −0.00001 @ 0 bps); break-even cost = None; 0/N cost-robust. True market-impact capacity is **not modeled** (INSUFFICIENT_DATA — not fabricated). |
| Stress (§22) | **NOT_CERTIFIED** | Negative expectancy with > 90% drawdown in the 2020 COVID crash and 2022 bear-market windows (and the 2021 calm control), for every symbol. |

### Frozen gates & sealed holdout

- **Frozen economic gates (§20):** `FrozenEconomicGates` (min expectancy > 0, PF ≥ 1.10, Sharpe > 0, DD ≤ 25%, must beat buy-and-hold **and** random placebo) are frozen by fingerprint. `verify_frozen()` trips on any threshold drift (anti-retrofitting).
- **Sealed holdout (§5):** derived deterministically per symbol from the data span (train 60% / validation 20% / holdout 20%, embargoed); accessed once per symbol via an access-logged seal for **scoring only** — no symbol touched twice, never tuned on.

### Recovery reproduction (measurement only)

An independent reimplementation on genuine 1-minute bars over **20,572** exit events finds **95.7%** of −0.5% exits recover **above the exit level** within 60 min (moving-block-bootstrap 95% CI [95.4%, 95.9%]), reproducing the ~95.3% figure. **But** only **47.7%** recover **above the entry price** within 60 min — the decision-relevant bar. A high above-exit rate does **not** prove the exit is economically wrong: downside avoided, capital reuse, and opportunity cost are not captured by this measurement.

---

## Readiness, paper, and live

| Gate | State |
| --- | --- |
| Readiness (§32) | `RESEARCH_VALIDATED` — infrastructure sound across 8 domains; **no** certified economic edge; **NOT_READY** for capital; `live_eligible = False`. |
| Paper | `NOT_CERTIFIED` — requires ≥ 30 genuine elapsed forward sessions **and** a research-validated edge; none exist. No prospective paper evidence is fabricated. |
| Live | `INACTIVE` — staged deployment (SHADOW log-only → PILOT_TINY 1% cap → SCALED 10% cap) is hard-gated behind the full chain **and** an external human authorization token the software cannot mint. No autonomous activation path exists. The −0.5% exit + native risk gate remain the authority. |

---

## Continuous runtime (addendum)

An exchange-calendar-driven operating-state machine (overnight / pre-market / regular / after-hours / post-session / weekend / holiday / degraded / recovering), with a `RuntimeSupervisor` running a **P0–P9 GPU priority ladder** (P0 protection is never gated; research ≥ P6 is suspended under GPU pressure and bounded in-session).

Off-session services:
- `ContinuousOptimizationService` — shadow-only.
- `ContinuousAcquisitionService` — DRY-RUN by default (plan only; no fetch, no credentials).
- `ContinuousGovernanceService` — safety-invariant monitoring; detects decay, **never** grants readiness.

Hardening: single-instance lock, atomic checkpoints, DEGRADED-on-regression + auto-recover. Execution is confined to `REGULAR_SESSION` via an optional authorized trading-engine runner (absent by default → running the service never places live orders). Ships as a macOS `launchd` LaunchAgent (`RunAtLoad` + `KeepAlive`); install via `scripts/install_runtime_service.sh`.

---

## How to run it

Two CLIs drive the entire certification and the continuous runtime.

### Certification CLI

```
python -m cengine.certification <command>
```

| Command | Scope |
| --- | --- |
| `audit` | Engineering audit (M7A) |
| `data` | Data integrity (M7B) |
| `native` | Native / Metal (§23) |
| `failure` | Failure injection (§24) |
| `protective-exit` | Protective exit (§25) |
| `entry` | Entry contract (§26) |
| `strategy` | Strategy economics on the sealed holdout (§5) |
| `sensitivity` | Capital / cost sensitivity (§21) |
| `stress` | Stress windows (§22) |
| `portfolio` | Portfolio domain |
| `execution` | Execution domain |
| `learning` | Learning infrastructure + provenance/safety |
| `holdout` | Sealed-holdout manifest |
| `full` | Run the complete chain |
| `report` | Regenerate the consolidated report |

### Runtime CLI

```
python -m cengine.runtime <status|once|run>
```

- `status` — print the current operating state and supervisor health.
- `once` — execute a single tick for the current state.
- `run` — run the supervised continuous loop.

---

## Where the evidence lives

| Location | Contents |
| --- | --- |
| `reports/m7_certification_report.md` | The consolidated, human-readable M7 certification report (the headline + per-domain verdicts). |
| `reports/certification/` | Per-domain machine-readable JSON: `data_integrity.json`, `engineering_audit.json`, `native.json`, `failure_injection.json`, `protective_exit.json`, `entry_contract.json`, `strategy_economics.json`, `capital_sensitivity.json`, `stress.json`, `portfolio.json`, `execution.json`, `learning.json`, `holdout_manifest.json`. |
| `reports/continuous/` | Continuous-runtime evidence: `runtime_status.json`, `supervisor_health.json`, `gpu_priority_ladder.json`, `tick_plans_by_state.json`, `acquisition_plan.json`, `governance_invariants.json`. |

---

## Bottom line

C-Engine is **engineering-certified** and **research-validated**: the data is genuine and leak-free, all components are wired, the native risk gate and −0.5% protective exit are correct and independent of learning + discovery, Metal is correctness-preserving, and faults are isolated. It has **no certified economic edge** and is **not** ready for paper or live capital. The platform is accepted as a sound research + execution substrate; the current strategy is rejected as a deployable edge.

---

> **Provenance.** All economic results above are **historical-replay simulations on genuine bars** — **not** live performance and **not** a profitability claim. No live orders were placed.
