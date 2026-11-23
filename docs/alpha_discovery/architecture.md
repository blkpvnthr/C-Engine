# M8 Alpha-Discovery Architecture

Milestone 8 is an **orchestration and research layer** (`cengine/alpha/`) that sits on top of the existing M1–M7 stack. It builds **no new trading engine, no new learning platform, no new optimizer, and no new lakehouse**. It composes what already exists — the certification harness, the governed optimizer (`cengine/optimization`), the learning stack (`cengine/learning`), the continuous runtime (`cengine/runtime`), and the native/Metal substrate — into a disciplined, falsification-first search for deployable edge.

## Honest result up front

> No positive, statistically credible out-of-sample net expectancy was found on the tested universe under the mandatory **-0.5% protective exit**. The failure is **not** momentum-specific — every strategy family tested is negative under this exit on this universe.
>
> **Decision:** REJECT the current entries/strategies as a deployable edge. ACCEPT the alpha-discovery infrastructure as a sound research substrate.

Every economic number below is a **historical-replay simulation on genuine bars**, not live performance and not a profitability claim. Negative results are preserved and never reinterpreted.

## Platform and data

| Dimension | Value |
|---|---|
| Hardware | Apple M5 Max, 48 GiB unified memory |
| Runtime | Python 3.11, MLX / Metal GPU |
| Data | Genuine Alpaca SIP bars, local Parquet + DuckDB lakehouse |
| 1-minute set | 6.76M bars / 5 canonical symbols (AAPL, MSFT, NVDA, SPY, QQQ), 2020–2026 |
| 5-minute set | 5.45M bars / 31 symbols |
| Price encoding | `price_scale = 10000` ticks |

All M8 development used **only** the 5 canonical 1-minute symbols. The 26 five-minute-only instruments (broad / sector / leveraged / inverse / defensive) are held as a **fresh cohort** untouched by both M7 certification and M8 development.

## What M8 reuses (the substrate)

`cengine/alpha/` is a thin composition layer. It owns no execution authority and introduces no new infrastructure.

| Substrate layer | M8 reuse |
|---|---|
| M7 certification harness | Fill model, cost assumptions, challenger gates, sealed-holdout discipline |
| `cengine/optimization` (M6B) | Persistent `OptimizationMemory` (dedup + effectiveness), `ChallengerGates`, `ChallengerRegistry` |
| `cengine/learning` | MLX/Metal models + sklearn baselines, bandit / offline-RL implementations |
| `cengine/runtime` | Continuous 24/7 service host, GPU admission control, session scheduling |
| Native / Metal | C++ `RiskEngine` authority, Metal kernels, MLX GPU inference |

The production mandatory **-0.5%** protective exit, the native C++ `RiskEngine` authority, and the sealed holdout are all **invariants** M8 preserves and never modifies.

## Module map

All modules under `cengine/alpha/` are importable and composed by the lifecycle; none place live orders.

| Module | Role | Section |
|---|---|---|
| `portfolio_replay.py` | Capital-constrained portfolio accountant (P0 correction) | §3 |
| `accounting_audit.py` | Audits the M7 compound-drawdown artifact vs portfolio accounting | §3 |
| `failure_attribution.py` | MAE/MFE, stop-out rate, directional-edge attribution | §3 / §16 |
| `features.py` | Look-ahead-free, point-in-time entry features | §5 |
| `entry_quality.py` | Entry-filter ablation with TRAIN/VAL splits + multiple-testing caveat | §5 |
| `multi_timeframe.py` | Higher-TF confirmation gates at the executable portfolio level | §6 |
| `regime_research.py` | Regime-proxy abstention policy (direction × volatility terciles) | §7 |
| `family_research.py` | Five-family archetype evaluation under the -0.5% exit | §8 |
| `microstructure_research.py` | L1 / bar-only proxy study; data-sufficiency gating | §9 |
| `supervised_filter.py` | MLX/Metal + sklearn filters scored on economics, not AUC | §10 |
| `bandit_rl_research.py` | Off-policy (refused) + direct-method bandit; offline RL (BC/CQL/IQL/FQE) | §11 / §12 |
| `portfolio_alpha.py` | Portfolio-aware baselines including cash / NO_TRADE | §13 |
| `protective_exit_research.py` | Isolated counterfactual exit studies; production unchanged | §14 |
| `objective.py` | Governed economic objective `J` | §19 |
| `validation.py` | TRAIN / dev-VAL / purged walk-forward / independent certification separation | §20 |
| `lifecycle.py` | `AlphaDiscoveryCycle` orchestrator; `AlphaDiscoveryService` | §4 / §16 / §17 / §22 |
| `dashboard.py` | Alpha-discovery panel fields | §24 |
| `benchmarks.py` | Honest completion-measured GPU-vs-CPU compute benchmarks | §18 |

## The P0 correction — capital-constrained portfolio accountant

The single most important correction in M8 is an **accounting** fix, not a strategy fix.

The M7 report showed a "~90–98% drawdown." That figure is a **compounded independent trade-level artifact**: `economic_metrics` was run with `drawdown_mode="compound"` over a flat per-trade series, which models an all-in, sequential, fully-reinvested **single-position** curve. In reality the strided entries overlap in time and form a **capital-constrained portfolio of concurrent positions**.

`cengine/alpha/portfolio_replay.py` is the corrected accountant. It changes **only the accounting** — it reuses the M7 conservative fill model unchanged:

- Chronological capital with fixed-fraction position sizing and fractional shares
- `max concurrent` and `max gross-exposure` caps
- Conserved cash (cash is never allowed to go negative)
- Mark-to-market, time-indexed equity curve → proper drawdown, CAGR, and annualized Sharpe

Corrected result on **15,000 trades / 5 symbols** (10% per position, max 10 concurrent):

| Measure | M7 compound (artifact) | Capital-constrained portfolio |
|---|---|---|
| Max drawdown | ~100% | 80.3% |
| Final equity | — | -80.2% |
| Annualized Sharpe | — | -5.2 |
| Cash conservation | — | Verified (min cash $10,669, never negative) |

The per-trade economics are **unchanged and still negative**: expectancy **-0.00108**, profit factor **0.64**. M7's "-0.2 Sharpe" was a **per-trade** figure, not annualized. The correction makes the drawdown honest; it **does not rescue the strategy**.

## Failure attribution

With the accounting corrected, `failure_attribution.py` localizes the problem:

- **45.7%** stop-out rate
- Mean **maximum adverse excursion -0.69%**, which **exceeds** the -0.5% stop
- Maximum favorable excursion only **+0.58%** — a near-symmetric swing against a tight stop
- Adverse > favorable ⇒ the momentum entry has **no directional edge** over the hold window

Conclusion: this is an **entry-quality** problem, not an exit-tuning problem.

## Research findings by layer (all negative)

Every layer treats the momentum entry as *minimum eligibility, not profitability*, splits each symbol chronologically into TRAIN (70%) / VAL (30%), fits thresholds on TRAIN only, and measures once on VAL.

| Layer (§) | Best development result | Verdict |
|---|---|---|
| Entry-quality ablation (§5) | baseline -0.00116; best `htf_trend_60` -0.00081; greedy combo overfits (train +0.00009, val insufficient) | No positive OOS filter |
| Multi-timeframe (§6) | confirmation cuts drawdown 80% → 66%, expectancy stays negative | Reduces exposure, no edge |
| Regime (§7) | only `down_midvol` positive on TRAIN, negative on VAL | Abstention helps, stays non-positive |
| Strategy families (§8) | MEAN_REVERSION -0.00087 (PF 0.67), MOMENTUM -0.00108, VOL_BREAKOUT -0.00119, TREND_PULLBACK -0.00102, MULTI_DAY_SWING -0.00091 | Every family negative |
| Microstructure (§9) | genuine L1 = INSUFFICIENT_DATA; bar-only CLV proxy -0.00104 | Deferred to live/paper L1 |
| Supervised ML (§10) | baseline -0.00116; best ridge -0.00071; rank-ICs +0.06…+0.09 | No positive selection |
| Bandit (§11) | off-policy IPS/DR refused (no logged propensities); direct-method -0.00095 VAL vs uniform -0.00109 | INSUFFICIENT_DATA |
| Offline RL (§12) | 0 genuine execution transitions; BC/CQL/IQL/FQE available | INSUFFICIENT_DATA, deferred |
| Portfolio-aware (§13) | all_eligible -80%, selective_htf -70%, **cash +0%** | Cash dominates |

Key qualifications, preserved verbatim from the research:

- **Families (§8):** vectorized, long-only archetypes faithful to `FAMILY_SPECS`, all run through the capital-constrained accountant under the -0.5% exit. `MULTI_DAY_SWING` is structurally incompatible with the intraday stop. The intraday -0.5% exit dominates every long archetype on this universe.
- **Microstructure (§9):** the historical lakehouse holds **bars only** (no trade/quote ticks). Genuine order-flow imbalance, microprice, bid replenishment, spread, and seller-exhaustion are deferred to live/paper L1 capture. No L2 was fabricated.
- **Supervised ML (§10):** models are scored on **economics** (top-decile OOS net expectancy), not AUC. Filtering cannot create an edge the entries do not contain. MLX ran on the Metal GPU.
- **Bandit / RL (§11/§12):** no behavior-policy propensities existed, so off-policy estimators were **refused** rather than fabricated. No uncontrolled live exploration; no unrestricted live RL.

## Protective-exit research (§14)

The production mandatory **-0.5% exit**, the independent event-driven path, and the 5-second periodic backstop are **preserved and never modified**. The **-3% emergency candidate stays INACTIVE**.

Entry selectivity marginally reduces stop-outs (**45.7% → 44.4%**). Wider hard stops improve expectancy **in isolation** (-0.00108 @ 0.5% → -0.00090 @ 2.0%) but **all remain negative**. These are isolated counterfactuals only — production is unchanged and nothing auto-modifies the exit.

## Economic objective and frozen gates

The governed objective (`objective.py`, §19) is:

```
J = net_expectancy
    − drawdown_penalty    · dd
    − downside_penalty     · downside
    − uncertainty_penalty  · unc
```

Penalty weights are calibrated on **development data only**. The objective never optimizes solely for win rate, AUC, Sharpe, or gross return.

The **FROZEN `ChallengerGates`** (§21) carry an anti-retrofitting fingerprint and were **not relaxed to manufacture a pass**:

- Profit factor ≥ **1.30**
- Annualized Sharpe ≥ **1.25**
- Sortino ≥ **1.50**
- Max portfolio drawdown ≤ **10%**
- Positive net expectancy
- Adequate effective sample
- Must beat **cash + incumbent + simple baselines**

## Independent validation discipline (§20)

Training, development-validation, purged walk-forward, and final independent certification are kept separate. Controls against data-snooping:

- Chronological TRAIN/VAL splits; thresholds fit on TRAIN only
- The M7 **sealed holdout is treated as CONSUMED** and never reused
- The 26 five-minute-only instruments form a **fresh cohort**, reserved from both M7 certification and M8 development

Because **no** development challenger reached the frozen gates, the fresh cohort was not consumed by a selection search. **One pre-registered incumbent confirmatory test** on that cohort (**68,853 trades**) independently corroborates the negative finding: expectancy **-0.001**, PF **0.67**, not positive.

**Independent certification is DEFERRED** — there is no positive edge to certify. No best-of-many development configuration is described as certified.

## Discovery lifecycle and runtime integration

`AlphaDiscoveryCycle` (`lifecycle.py`, §4/§16/§17/§22) chains:

```
COLLECT / ATTRIBUTE
  → IDENTIFY FAILURE MODES
  → GENERATE HYPOTHESES (each references evidence + defines a falsifiable test)
  → CHECK DATA SUFFICIENCY
  → CONSULT OPTIMIZATION MEMORY
  → PLAN / RUN EXPERIMENTS
  → ECONOMIC + ROBUSTNESS EVALUATION
  → ACCEPT / REJECT / DEFER
  → PERSIST
```

It reuses the M6B persistent `OptimizationMemory`, the frozen `ChallengerGates`, and the `ChallengerRegistry`. The genuine run produced:

- **0 accepted / 6 rejected / 2 deferred**, incumbent retained
- **3 hypotheses** generated
- **14 optimization-memory rows** persisted

The cycle is embedded in the continuous 24/7 runtime as **`AlphaDiscoveryService`** at research priority **P9**: GPU-admission-gated (suspended under GPU pressure, bounded in-session), shadow-only, and off-session. **Protection is never gated.**

## Compute profile (§18)

Benchmarks are honest, completion-measured (`mx.eval`) GPU-vs-CPU comparisons:

- Batched linear inference at small size favors **CPU** (0.57×)
- Large feature standardization favors **GPU** (9×)

CPU wins are reported, not hidden. Research workloads never interfere with financial protection.

## Dashboard (§24)

`dashboard.py` exposes an alpha-discovery panel with development results **strictly separated** from independent certification results, and no fabricated metrics. Fields include: `ALPHA_DISCOVERY_STATUS`, `EXPERIMENTS_COMPLETED` / `REJECTED` / `DEFERRED`, `BEST_DEVELOPMENT_NET_EXPECTANCY`, `INCUMBENT_NET_EXPECTANCY`, `OPTIMIZATION_MEMORY_SIZE`, `DATA_SUFFICIENCY`, and `INDEPENDENT_CERTIFICATION_STATUS`.

## Invariants

- Production **-0.5% exit** and native C++ `RiskEngine` authority preserved
- **-3% emergency candidate remains inactive**
- No new engine, optimizer, or lakehouse
- No live activation — **no live orders were placed**
- No fabricated performance
- Sealed holdout protected from optimization

## Evidence and entrypoints

- `cengine/alpha/` — research modules, importable and composed by the lifecycle
- `cengine/runtime` — runs the modules off-session via `AlphaDiscoveryService`
- `reports/alpha_discovery/` — **18 JSON** evidence files + `final_alpha_discovery_report.md`

---

**Provenance.** All economic results in this document are **historical-replay simulations on genuine Alpaca SIP bars** — they are **not live performance** and **not a profitability claim**. **No live orders were placed.** Negative results are preserved and are never reinterpreted as success.
