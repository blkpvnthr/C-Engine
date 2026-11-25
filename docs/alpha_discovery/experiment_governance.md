# Experiment Governance (Milestone 8, §16 / §17 / §22)

The M8 alpha-discovery layer (`cengine/alpha/`) is an **orchestration + research
layer** over the existing M1–M7 stack. It builds no new trading engine, learning
platform, optimizer or lakehouse. Governance here means: hypotheses must earn their
existence from evidence, every experiment must define how it could fail, search is
bounded and versioned, redundant work is refused, and the whole loop runs inside the
continuous runtime without ever competing with financial protection.

The governed lifecycle is `AlphaDiscoveryCycle`:

```
COLLECT / ATTRIBUTE
  → IDENTIFY FAILURE MODES
  → GENERATE HYPOTHESES          (evidence-referenced, falsifiable)
  → CHECK DATA SUFFICIENCY
  → CONSULT OPTIMIZATION MEMORY  (reuse M6B, dedup, effectiveness)
  → PLAN / RUN EXPERIMENTS       (bounded, versioned)
  → ECONOMIC + ROBUSTNESS EVALUATION
  → ACCEPT / REJECT / DEFER      (frozen ChallengerGates → ChallengerRegistry)
  → PERSIST
```

It composes existing components — the certification harness, the governed optimizer
(`cengine/optimization`), the learning stack (`cengine/learning`), the continuous
runtime (`cengine/runtime`), and the native/Metal substrate. No step is skipped.

---

## §16 — Governed hypothesis generation

Hypotheses are not free-form ideas. Each one is generated **from recorded evidence**
(failure attribution and prior experiment results) and must:

- **Reference the evidence** that motivates it (e.g. the entry-quality failure mode,
  the stop-out rate, the adverse-vs-favorable asymmetry).
- **Define a falsifiable test** — a concrete experiment whose outcome can reject it.

This reflects the project-wide falsification-first stance: every strategy starts as
false and must survive an attempt to destroy it. The governing failure attribution
for the tested universe is decisive and directs hypothesis generation:

| Evidence (§3 / §16) | Value |
| --- | --- |
| Stop-out rate under mandatory −0.5% exit | 45.7% |
| Mean max-adverse-excursion | −0.69% (exceeds the −0.5% stop) |
| Mean max-favorable-excursion | +0.58% |
| Conclusion | Near-symmetric swing against a tight stop; adverse > favorable → **no directional edge** over the hold window |

The attribution identifies this as an **entry-quality problem, not an exit-tuning
problem**, and the failure is **not momentum-specific**. Hypotheses are therefore
aimed at entry quality, confirmation, regime abstention and family selection — each
tested as a falsifiable experiment rather than tuned for backtest appearance.

In the genuine governed run, **3 hypotheses were generated**, each referencing
evidence and defining a falsifiable test.

---

## §17 — Bounded versioned search, deduplication, memory-informed selection

Search is deliberately constrained to resist overfitting and multiple-testing
inflation.

- **Bounded + versioned.** Experiment plans are finite and versioned; the lifecycle
  reuses the **governed optimizer** (`cengine/optimization`) rather than an unbounded
  parameter sweep. Any strategy that needs heavy parameter tuning is treated as
  suspect.
- **Deduplication.** The lifecycle reuses the **M6B persistent `OptimizationMemory`**
  for dedup and effectiveness scoring, so previously-explored configurations are not
  re-run. In the genuine run, **14 optimization-memory rows were persisted**.
- **Memory-informed selection.** Before planning, the cycle **consults the
  optimization memory** to prioritize experiments by recorded effectiveness and to
  avoid repeating known-ineffective configurations.
- **Data sufficiency gate.** The cycle checks data sufficiency before running; where
  evidence is inadequate the correct output is `INSUFFICIENT_DATA` (as happened for
  genuine microstructure, off-policy bandit, and sequential offline RL), not a
  manufactured result.

### Multiple-testing / data-snooping controls

- Per-symbol **chronological TRAIN (70%) / VAL (30%)** splits.
- Thresholds and filters **fit on TRAIN only**, measured on VAL.
- Greedy, complexity-penalized combination with an explicit multiple-testing caveat.
- The **M7 sealed holdout is treated as CONSUMED** and never reused.
- A **fresh cohort** — 26 five-minute-only instruments untouched by both M7
  certification and M8 development — is reserved. Because no development challenger
  reached the frozen gates, the fresh cohort was **not** consumed by a selection
  search; one pre-registered incumbent confirmatory test on it (68,853 trades)
  independently corroborated the negative finding (expectancy −0.001, PF 0.67, not
  positive).

---

## Economic objective and frozen gates

Experiments are evaluated against a governed economic objective, never backtest
metrics alone.

- **Objective (§19):**
  `J = net_expectancy − drawdown_penalty·dd − downside_penalty·downside − uncertainty_penalty·unc`.
  Penalty weights are calibrated on **development data only**. The objective is never
  optimized solely for win rate, AUC, Sharpe or gross return.
- **Frozen `ChallengerGates` (§21)** — an anti-retrofitting fingerprint, not relaxed
  to manufacture a pass:

| Gate | Threshold |
| --- | --- |
| Profit factor | ≥ 1.30 |
| Annualized Sharpe | ≥ 1.25 |
| Sortino | ≥ 1.50 |
| Max portfolio drawdown | ≤ 10% |
| Net expectancy | positive |
| Effective sample | adequate |
| Must beat | cash + incumbent + simple baselines |

---

## §22 — Challenger registry: accept / reject / defer

The `ChallengerRegistry` records every experiment outcome as one of **ACCEPT /
REJECT / DEFER**, with the incumbent retained unless a challenger clears the frozen
gates. The genuine governed run:

| Outcome | Count |
| --- | --- |
| Accepted | 0 |
| Rejected | 6 |
| Deferred | 2 |
| Incumbent | retained |
| Hypotheses generated | 3 |
| Optimization-memory rows persisted | 14 |

No challenger reached the frozen gates, so **independent certification is DEFERRED**
(there is no positive edge to certify). No best-of-many development configuration is
described as certified. `DEFER` is a first-class, honest outcome — used when a result
is non-positive but not conclusively rejected, or when evidence is insufficient.

### Defer redundant experiments when no new evidence

Governance explicitly **defers redundant experiments when no new evidence exists**.
An experiment is not re-run if:

- the optimization memory already records an equivalent (deduplicated) configuration,
- data sufficiency has not changed since the last attempt, or
- no new failure-mode evidence has been collected that would alter the hypothesis.

This conserves the bounded compute budget and prevents repeated testing of the same
idea from inflating apparent significance.

---

## Continuous scheduling in the 24/7 runtime

The lifecycle is integrated into the continuous runtime as **`AlphaDiscoveryService`**:

- **P9 research priority** — the lowest-priority class; financial protection is never
  gated behind it.
- **GPU-admission-gated** — suspended under GPU pressure and bounded in-session, so
  research never starves the Metal GPU needed by the live path.
- **Shadow-only, off-session** — it runs off-session and places **no live orders**.
- **Protection is never gated.** The production mandatory −0.5% exit, its independent
  event-driven path and 5-second periodic backstop, and the native C++ `RiskEngine`
  authority are preserved and never deprioritized by research scheduling.

### Cooldowns and resource budgets

- **GPU admission control** suspends research under GPU pressure and bounds each
  research session, acting as the primary resource budget.
- **In-session bounding** limits how much work a single scheduled pass performs.
- **Deduplication + defer-on-no-new-evidence** act as a logical cooldown: the same
  experiment is not repeatedly re-scheduled absent new evidence.
- Compute is spent honestly — measured completion (`mx.eval`) showed CPU wins at
  small batched linear inference (0.57×) and GPU wins at large feature
  standardization (9×); the honest CPU-wins result is reported, not hidden. Research
  workloads never interfere with financial protection.

---

## Invariants preserved by governance

- Production mandatory **−0.5% protective exit** and native C++ `RiskEngine`
  authority preserved; the −3% emergency candidate stays **inactive**.
- **No new engine / optimizer / lakehouse** is built — M8 only composes M1–M7.
- **No live activation**; `AlphaDiscoveryService` is shadow-only and off-session.
- **No fabricated performance**; the sealed holdout is protected from optimization.
- Negative results are preserved and never reinterpreted.

---

## Honest result

No positive, statistically credible out-of-sample net expectancy was found on the
tested universe under the mandatory −0.5% protective exit. Governance **REJECTs the
current entries/strategies as a deployable edge** and **ACCEPTs the alpha-discovery
infrastructure as a sound research substrate**. The challenger registry recorded
0 accepted / 6 rejected / 2 deferred with the incumbent retained; independent
certification remains deferred.

Evidence: `reports/alpha_discovery/` (18 JSON + `final_alpha_discovery_report.md`).

---

> **Provenance.** All economic results referenced here are historical-replay
> simulations on genuine Alpaca SIP bars — not live performance, and not a
> profitability claim. No live orders were placed. Negative results are preserved and
> never reinterpreted.
