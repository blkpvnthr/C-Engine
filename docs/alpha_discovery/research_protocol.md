# Alpha Discovery Research Protocol (Milestone 8)

The continuous research lifecycle that governs how C-Engine proposes, tests, and
disposes of candidate trading edges. M8 is an **orchestration + research layer**
(`cengine/alpha/`) over the existing M1–M7 stack. It builds no new trading
engine, learning platform, optimizer, or lakehouse — it composes the
certification harness, the governed optimizer (`cengine/optimization`), the
learning stack (`cengine/learning`), the continuous runtime (`cengine/runtime`),
and the native/Metal substrate.

---

## 1. First principles

The protocol is deliberately hostile to its own candidates. Three rules bind
every stage.

- **Falsification first.** Every strategy starts as false. The burden of proof
  is on the candidate. Each stage tries to destroy the signal; only a survivor
  earns further attention. A valid, well-supported `NO_TRADE` / `REJECT` is a
  first-class outcome, not a failure of the process.
- **Net performance only.** Gross performance is irrelevant. All economics are
  measured net of the M7 conservative fill model and under the production
  mandatory **-0.5% protective exit**. Per-trade net expectancy and profit
  factor are the primary ledger.
- **Out-of-sample is king.** Thresholds are fit on chronological TRAIN splits
  and measured once on held-out VAL. A sealed holdout and a fresh 5-minute
  cohort are reserved and never consumed by a selection search.

> Honest overall result: **no positive, statistically credible out-of-sample
> net expectancy** was found on the tested universe under the -0.5% exit. The
> failure is **not momentum-specific**. Decision: **REJECT** the current
> entries/strategies as a deployable edge; **ACCEPT** the alpha-discovery
> infrastructure as a sound research substrate.

---

## 2. The research lifecycle

`AlphaDiscoveryCycle` chains the stages below. It reuses the M6B persistent
`OptimizationMemory` (dedup + effectiveness tracking), the frozen
`ChallengerGates`, and the `ChallengerRegistry`.

```
COLLECT / ATTRIBUTE
      ↓
IDENTIFY FAILURE MODES
      ↓
GENERATE HYPOTHESES  ───────────┐ (each references evidence + a falsifiable test)
      ↓                         │
CHECK DATA SUFFICIENCY          │
      ↓                         │
CONSULT OPTIMIZATION MEMORY  ───┘ (dedup, prior effectiveness)
      ↓
PLAN / RUN EXPERIMENTS
      ↓
ECONOMIC + ROBUSTNESS EVALUATION
      ↓
ACCEPT / REJECT / DEFER
      ↓
PERSIST
```

| Stage | Purpose | Key guarantee |
|-------|---------|---------------|
| COLLECT / ATTRIBUTE | Gather net per-trade economics and attribute P&L to entry, exit, and regime | Genuine Alpaca SIP bars only |
| IDENTIFY FAILURE MODES | Localize where capital is lost | Adverse vs favorable excursion, stop-out rate |
| GENERATE HYPOTHESES | Turn each failure into a falsifiable claim | Must cite evidence + define a test |
| CHECK DATA SUFFICIENCY | Refuse experiments the data cannot support | Returns `INSUFFICIENT_DATA` honestly |
| CONSULT OPTIMIZATION MEMORY | Avoid re-running dead ends | Dedup + effectiveness lookup |
| PLAN / RUN EXPERIMENTS | Execute on the corrected accountant | TRAIN-fit, VAL-measured |
| ECONOMIC + ROBUSTNESS EVALUATION | Score on economics, not AUC | Governed objective `J`, frozen gates |
| ACCEPT / REJECT / DEFER | Dispose of each candidate | Gates not relaxed |
| PERSIST | Record outcome + rows for future cycles | Negative results preserved |

---

## 3. COLLECT / ATTRIBUTE — the corrected accountant

A **P0 accounting audit** preceded all entry research. The M7 "~90–98%
drawdown" was a **compounded independent trade-level artifact**: an
`economic_metrics` run with `drawdown_mode="compound"` over a flat per-trade
series behaves as an all-in, sequential, fully-reinvested single-position curve.
The strided entries actually overlap in time and form a **capital-constrained
portfolio of concurrent positions**.

`cengine/alpha/portfolio_replay.py` is the corrected accountant: chronological
capital, fixed-fraction sizing, fractional shares, max-concurrent and
max-gross-exposure caps, conserved cash, and mark-to-market time-indexed equity
→ proper drawdown / CAGR / annualized Sharpe. It reuses the M7 conservative fill
model — **only the accounting changes.**

| Measure | M7 compound view | Capital-constrained portfolio |
|---------|------------------|-------------------------------|
| Max drawdown | ~100% | 80.3% (10%/position, max 10 concurrent) |
| Final equity | — | -80.2% |
| Annualized Sharpe | — | -5.2 |
| Cash floor | — | never negative (min $10,669) |

Per-trade economics are **unchanged and still negative** (expectancy -0.00108,
PF 0.64). M7's "-0.2 Sharpe" was a per-trade figure, not annualized. **The
correction does not rescue the strategy** — it only tells the truth about the
equity path.

---

## 4. IDENTIFY FAILURE MODES — where the capital goes

Attribution on the genuine trade series localizes the loss to the **entry**, not
the exit.

- **45.7%** stop-out rate under the -0.5% exit.
- Mean **max-adverse excursion -0.69%** exceeds the -0.5% stop.
- Mean **max-favorable excursion only +0.58%** — a near-symmetric swing against
  a tight stop.
- Adverse > favorable ⇒ the momentum entry has **no directional edge** over the
  hold window.

Diagnosis: an **entry-quality problem**, not an exit-tuning problem.

---

## 5. GENERATE HYPOTHESES — from failure finding to falsifiable test

Every hypothesis references the evidence above and defines a test that can
return negative. The genuine cycle generated **3 hypotheses**. The pattern:

| Failure finding | Hypothesis | Falsifiable test | Outcome |
|-----------------|-----------|------------------|---------|
| No directional edge in the entry | A point-in-time entry filter isolates profitable momentum | Fit threshold on TRAIN, measure net expectancy on VAL; must be positive | No filter positive OOS (best `htf_trend_60` -0.00081 vs baseline -0.00116) |
| 45.7% stop-outs, adverse > favorable | Higher-timeframe confirmation removes the losing entries | Executable portfolio replay with HTF gate; per-trade expectancy must turn positive | Trades + damage cut (DD 80%→66%), expectancy stays negative |
| Losses concentrate in a regime | Abstaining in the worst regime restores positive expectancy | Abstention policy chosen on TRAIN, measured once on VAL | Only positive-TRAIN regime (`down_midvol`) overfits; VAL non-positive |

Hypotheses are **minimum-eligibility reframings**: the momentum entry is treated
as eligibility, never as evidence of profitability. Complexity is penalized; a
greedy combination that looked good in TRAIN (+0.00009) failed to generalize.

---

## 6. CHECK DATA SUFFICIENCY — honest refusal

The protocol refuses experiments the data cannot support and labels them
`INSUFFICIENT_DATA` rather than fabricating a result.

- **Microstructure (L1).** Genuine quote/order-flow imbalance, microprice, bid
  replenishment, spread, and seller-exhaustion signals require trade/quote
  ticks. The historical lakehouse holds **bars only**. A clearly-labeled
  bar-only close-location-value proxy did not help (-0.00104). Genuine
  microstructure is **deferred** to live/paper L1 capture; **no L2 fabricated.**
- **Bandit (off-policy).** IPS/DR estimators were **refused** — there are no
  logged behavior-policy propensities, and none were fabricated. The
  direct-method best-family-per-regime (-0.00095 VAL vs uniform -0.00109) stays
  negative → certification `INSUFFICIENT_DATA`. No uncontrolled live exploration.
- **Offline RL.** Zero genuine execution transitions exist (only single-step
  replay). BC / CQL / IQL / FQE are implemented and unit-tested, but
  sequential-policy certification is `INSUFFICIENT_DATA`, deferred until
  paper/live transitions accrue. No unrestricted live RL.

---

## 7. CONSULT OPTIMIZATION MEMORY

Before running an experiment, the cycle queries the M6B persistent
`OptimizationMemory` for deduplication and prior effectiveness, so dead ends are
not re-paid for. The genuine run **persisted 14 optimization-memory rows** for
future cycles.

---

## 8. PLAN / RUN EXPERIMENTS — the experiment surface

All experiments run on the corrected accountant with chronological TRAIN(70%)/
VAL(30%) splits, thresholds fit on TRAIN only, and look-ahead-free,
point-in-time features (momentum/accel, rel-volume, displacement-from-SMA,
ATR-normalized displacement, BB width, 15m/60m higher-TF trend, trend
persistence, time-of-day, directional separation). Causality is verified by
test; developing candles are never used to qualify an earlier decision.

| Experiment | Result | Verdict |
|------------|--------|---------|
| Entry-quality ablation (§5) | Best `htf_trend_60` -0.00081; baseline -0.00116; greedy combo overfits | No OOS edge |
| Multi-timeframe confirmation | DD 80%→66%, per-trade expectancy still negative | Reduces exposure, no edge |
| Regime abstention | Up-vol regimes bleed worst; abstention improves but non-positive | NO_TRADE valid |
| Strategy families (all five) | Every family negative | Exit dominates all long archetypes |
| Supervised ML (MLX/Metal + sklearn) | Best ridge -0.00071; rank-ICs +0.06..+0.09 | Filtering cannot create edge |
| Portfolio-aware baselines | all_eligible -80%, selective_htf -70%, **cash +0%** | Cash dominates |

**Strategy families** are decisive: all five, evaluated independently under the
-0.5% exit via the capital-constrained accountant (vectorized archetypes
faithful to `FAMILY_SPECS`, long-only), are negative —
`MEAN_REVERSION` least-bad (-0.00087, PF 0.67), `MOMENTUM` -0.00108,
`VOL_BREAKOUT` -0.00119, `TREND_PULLBACK` -0.00102, `MULTI_DAY_SWING` -0.00091
(and structurally incompatible with the intraday stop). **The intraday -0.5%
exit dominates every long archetype on this universe.**

**Supervised ML** is scored on economics (top-decile OOS net expectancy), **not
AUC**; no model's selection is positive. MLX ran on the Metal GPU.

---

## 9. ECONOMIC + ROBUSTNESS EVALUATION — objective and gates

Candidates are scored against a **governed economic objective**, never win rate /
AUC / Sharpe / gross alone:

```
J = net_expectancy
      − drawdown_penalty    · drawdown
      − downside_penalty    · downside
      − uncertainty_penalty · uncertainty
```

Penalty weights are calibrated on **development data only**. Promotion is
gated by the **frozen `ChallengerGates`** (anti-retrofitting fingerprint):

- profit_factor ≥ **1.30**
- annualized Sharpe ≥ **1.25**
- Sortino ≥ **1.50**
- max portfolio drawdown ≤ **10%**
- positive net expectancy
- adequate effective sample
- must beat **cash + incumbent + simple baselines**

The gates were **not relaxed** to manufacture a pass.

### Independent validation layers

Training / development-validation / purged walk-forward / final independent
certification are kept separate. The M7 sealed holdout is treated as
**consumed and never reused**. All M8 development used only the **5 canonical
1-minute symbols** (AAPL, MSFT, NVDA, SPY, QQQ). The **26 five-minute-only
instruments** form a fresh cohort untouched by M7 certification and M8
development. Because no development challenger reached the frozen gates, the
fresh cohort was **not** consumed by a selection search; **one pre-registered
incumbent confirmatory test** on it (**68,853 trades**) independently
corroborates the negative finding (expectancy -0.001, PF 0.67, not positive).
**Independent certification: DEFERRED** — there is no positive edge to certify.

---

## 10. ACCEPT / REJECT / DEFER — the genuine cycle result

The decision stage disposes of each candidate against the gates. The genuine run:

| Disposition | Count |
|-------------|-------|
| **ACCEPT** | **0** |
| **REJECT** | **6** |
| **DEFER** | **2** |
| Incumbent retained | yes |
| Hypotheses generated | 3 |
| Optimization-memory rows persisted | 14 |

`DEFER` is reserved for candidates blocked by data sufficiency (genuine L1
microstructure; sequential-policy RL) that may be revisited once paper/live
capture provides the missing transitions/ticks.

---

## 11. PERSIST — and runtime integration

Outcomes, evidence, and optimization-memory rows are written back so future
cycles inherit them. Evidence lives in `reports/alpha_discovery/` (**18 JSON +
`final_alpha_discovery_report.md`**). Dashboard fields keep **development results
strictly separated** from independent certification results
(`ALPHA_DISCOVERY_STATUS`, `EXPERIMENTS_COMPLETED/REJECTED/DEFERRED`,
`BEST_DEVELOPMENT_NET_EXPECTANCY`, `INCUMBENT_NET_EXPECTANCY`,
`OPTIMIZATION_MEMORY_SIZE`, `DATA_SUFFICIENCY`,
`INDEPENDENT_CERTIFICATION_STATUS`, …); no fabricated metrics.

The cycle is wired into the continuous 24/7 runtime as `AlphaDiscoveryService`
at **research priority P9** — GPU-admission-gated (suspended under GPU pressure,
bounded in-session), **shadow-only**, and **off-session**. Financial protection
is never gated.

---

## 12. Invariants

- Production mandatory **-0.5% exit** + independent event-driven path + 5-second
  periodic backstop **preserved and never modified**; the -3% emergency
  candidate stays **inactive**. Alternative wider hard stops improve expectancy
  in isolation (-0.00108 @ 0.5% → -0.00090 @ 2.0%) but **all remain negative** —
  isolated counterfactuals only; production unchanged; no auto-modification.
- Native C++ `RiskEngine` authority and the `MarketRegimeEngine` production
  authority preserved.
- No new engine / optimizer / lakehouse; no live activation; no fabricated
  performance; sealed holdout protected from optimization.
- Compute reporting is honest (completion-measured via `mx.eval`): batched
  linear inference favors CPU at small size (0.57×); large feature
  standardization favors GPU (9×). CPU wins are reported, not hidden.

---

*Provenance: all economic results above are historical-replay simulations on
genuine bars — not live performance, and not a profitability claim. No live
orders were placed. Negative results are preserved and never reinterpreted.*
