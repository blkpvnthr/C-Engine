# Readiness Classification & Deployment Frameworks

**C-Engine M7 Certification — §32 Readiness, Paper Framework, Staged-Live Framework**

This document records the final readiness classification for C-Engine and the two
forward-facing deployment frameworks it gates: the **paper framework** and the
**staged-live framework**. It explains why the honest verdict separates a sound
*infrastructure* result from an uncertified *economic* result, and why neither paper
nor live capital is authorized today.

---

## 1. The Infrastructure vs Economic Split

C-Engine is certified along two independent axes that must never be conflated.

| Axis | Question it answers | M7 verdict |
|------|---------------------|------------|
| **Infrastructure** | Is the platform correct, leak-free, isolated, and wired end-to-end? | **RESEARCH_VALIDATED** across 8 domains |
| **Economic** | Does the strategy have a certified, deployable edge? | **NOT_CERTIFIED** (0/5 symbols) |

A system can be engineered perfectly and still have nothing profitable to say. That is
exactly C-Engine's situation, and the readiness classification is built to express it
without overstating either side.

### Infrastructure: what is certified

| Domain | Result |
|--------|--------|
| Data integrity (M7B) | **CERTIFIED** — 12.2M genuine Alpaca SIP bars; 0 duplicates, strictly monotonic timestamps, valid OHLC, no look-ahead, 100% NYSE trading-day coverage, 1m→5m buckets reconcile 99.99% |
| Engineering audit (M7A) | **CERTIFIED** — 103 modules audited, 0 BLOCKED |
| Native / Metal (§23) | **CERTIFIED** — native `RiskEngine` is sole financial authority; GPU EMA/APO reproduce the CPU reference exactly |
| Failure injection (§24) | **CERTIFIED** — learning/GPU/data/broker faults isolated; P0 protection / P1 execution never auto-killed |
| Protective exit (§25) | **CERTIFIED** — `-0.5%` exit math long/short symmetric; coordinator imports no ML/GPU/lakehouse/scanner module |
| Entry contract (§26) | **CERTIFIED** — entries deterministic and look-ahead-free (causal) |
| Recovery reproduction | **CERTIFIED (measurement only)** — see §4 below |
| Learning infrastructure | **CERTIFIED** — provenance discipline, recommendation-only, autonomous-optimization safety all certified |

### Economics: what is NOT certified

| Dimension | Result |
|-----------|--------|
| Strategy economics (sealed holdout, §5) | **NOT_CERTIFIED**, 0/5 symbols — negative net expectancy on every canonical symbol (expectancy ≈ −0.001/trade, profit factor ≈ 0.6, Sharpe ≈ −0.2, drawdown >90%); fails to beat buy-and-hold on every symbol and the random placebo on 3/5 |
| Capital / cost sensitivity (§21) | **NOT_CERTIFIED** — negative net expectancy even at **zero cost** (AAPL −0.00033, SPY −0.00001 @ 0 bps); break-even cost = None; 0/N cost-robust; true market-impact capacity not modeled (INSUFFICIENT_DATA) |
| Stress (§22) | **NOT_CERTIFIED** — negative expectancy with >90% drawdown in the 2020 COVID crash, 2022 bear market, and 2021 calm control, for every symbol |

> The `-0.5%` protective exit is a **RISK BACKSTOP, not an economic edge.** The production
> incumbent (mandatory `-0.5%` protective exit + momentum entry) has negative net
> expectancy on the sealed holdout. The governed optimization loop certified **no economic
> improvement** and **retained the incumbent** — the correct falsification-first outcome.

---

## 2. Readiness Classification (§32)

**Readiness: `RESEARCH_VALIDATED`.**

| Field | Value |
|-------|-------|
| Classification | `RESEARCH_VALIDATED` |
| Infrastructure | Sound across 8 domains |
| Certified economic edge | **None** |
| `NOT_READY` for capital | **True** |
| `live_eligible` | **False** |

`RESEARCH_VALIDATED` is deliberately *below* any capital-bearing tier. It asserts only that
the platform is a trustworthy substrate for research and execution — genuine leak-free data,
all components wired, native risk gate and `-0.5%` exit correct and independent, Metal
kernels correctness-preserving, faults isolated. It asserts **nothing** about profitability.

Because there is no certified edge, every downstream gate (paper, then live) fails at its
first precondition. The classification cannot be upgraded by tuning, retrofitting, or
additional backtesting — only by producing a research-validated edge *and* the genuine
forward evidence the frameworks below require.

---

## 3. The Paper Framework

Paper certification is the first forward-facing gate above `RESEARCH_VALIDATED`. It exists
to convert in-sample / historical-replay research into *prospective* evidence that did not
exist when the hypothesis was formed.

### Paper gate preconditions

Paper certification requires **both**:

1. **≥ 30 genuine elapsed forward sessions** — real wall-clock trading sessions observed
   going forward, not replayed, not simulated, not backfilled.
2. **A research-validated edge** — an economic result that has already cleared the frozen
   economic gates (§20) and sealed holdout (§5) scoring.

### Current status: `NOT_CERTIFIED`

| Precondition | Status |
|--------------|--------|
| ≥ 30 genuine forward sessions | **Not met** — none exist |
| Research-validated edge | **Not met** — economics NOT_CERTIFIED (§1) |
| Fabricated prospective evidence | **None** — no paper evidence is invented |

Both preconditions currently fail. There are zero genuine forward sessions, and there is no
certified edge to forward-test. Critically, **no prospective paper evidence is fabricated**
to close the gap: an absent forward record is reported as absent, never synthesized.

> Elapsed forward sessions cannot be manufactured. They accrue only with the passage of real
> trading days under observation. This is the structural reason paper certification cannot be
> "completed early" by any amount of computation.

---

## 4. Why a High Recovery Rate Does Not Rescue the Economics

A recurring temptation is to read the protective exit's recovery statistics as evidence the
exit is economically wrong. The certification explicitly refuses that inference.

An independent reimplementation on genuine 1m bars over **20,572 exit events** found:

- **95.7%** of `-0.5%` exits recover **above the exit level** within 60 min
  (moving-block-bootstrap 95% CI **[95.4%, 95.9%]**), reproducing the ~95.3% figure.
- **Only 47.7%** recover **above the entry price** within 60 min — the decision-relevant bar.

A high above-exit recovery rate does **not** prove the exit is economically wrong. It does
not capture downside avoided, capital reuse, or opportunity cost. This is a **measurement**,
certified as such, and it does not upgrade the economic verdict or substitute for forward
evidence in the paper framework.

---

## 5. The Staged-Live Framework

Live deployment is **INACTIVE** and hard-gated. The framework is defined so that it *can*
exist safely, not because anything is authorized to use it.

### The three stages

| Stage | Capital cap | Behavior |
|-------|-------------|----------|
| `SHADOW` | — | Log-only; no orders placed |
| `PILOT_TINY` | 1% cap | Tightly capped live exposure |
| `SCALED` | 10% cap | Capped scaled exposure |

Progression is strictly ordered: `SHADOW → PILOT_TINY → SCALED`. No stage may be skipped.

### Hard gating

Staged live deployment is gated behind **the full certification chain PLUS an external human
authorization token that the software cannot mint.**

- **No autonomous activation path exists.** The system cannot promote itself from
  `RESEARCH_VALIDATED` into any live stage.
- **The human token is required and un-forgeable by the software.** Absent a valid external
  token, every stage transition is refused.
- **The authority remains fixed.** The `-0.5%` protective exit and the native risk gate
  remain the financial authority at every stage. The native `RiskEngine` accepts a valid
  10-share order and rejects a 50k-share order with `QUANTITY_LIMIT_EXCEEDED`.

### Why it is inactive today

Live eligibility (`live_eligible`) is **False** because the chain is broken at its very first
economic link: economics are NOT_CERTIFIED (§1), so paper certification cannot be reached
(§3), so no live stage is reachable. The staged framework is wired and testable but cannot
be entered.

---

## 6. Runtime Confinement Supporting the Frameworks

The continuous-runtime addendum enforces the frameworks operationally, not just on paper:

- **Execution is confined to `REGULAR_SESSION`** via an *optional* authorized trading-engine
  runner that is **absent by default** — running the continuous service never places live
  orders.
- Off-session services are non-authoritative: `ContinuousOptimizationService` is shadow-only,
  `ContinuousAcquisitionService` is DRY-RUN by default (plan only, no fetch, no credentials),
  and `ContinuousGovernanceService` monitors safety invariants and **detects decay but never
  grants readiness.**
- A `RuntimeSupervisor` P0–P9 GPU priority ladder keeps **P0 protection never gated**;
  research (≥P6) is suspended under GPU pressure.
- A single-instance lock blocks a second trade instance; atomic checkpoints survive a
  round-trip.

No service in the continuous runtime can grant readiness, mint a live authorization, or
place an order outside the authorized, human-gated path.

---

## 7. Anti-Retrofitting Guarantees

The frameworks are protected against after-the-fact threshold relaxation:

- **Frozen economic gates (§20):** `FrozenEconomicGates` (min expectancy > 0, PF ≥ 1.10,
  Sharpe > 0, DD ≤ 25%, must beat buy-and-hold + random placebo) are frozen by fingerprint;
  `verify_frozen()` trips on **any** threshold drift.
- **Sealed holdout (§5):** derived deterministically per symbol (train 60% / validation 20%
  / holdout 20%, embargoed); accessed once per symbol via an access-logged seal for **scoring
  only** — no symbol touched twice, never tuned on.
- **Learning provenance discipline:** synthetic and `historical_replay` results can **never**
  back a profitability claim; only genuine `real` realized outcomes can. The bandit/RL and
  strategy-relevant tasks are INSUFFICIENT_DATA — no materialized realized-outcome layers, no
  live sessions.

These guarantees mean the only way to advance readiness is to produce genuine evidence that
clears *fixed* bars — the bars cannot be moved to fit the result.

---

## 8. Summary

| Framework | Gate | Status |
|-----------|------|--------|
| Readiness (§32) | Infrastructure sound, no certified edge | **`RESEARCH_VALIDATED`** — NOT_READY for capital, `live_eligible=False` |
| Paper | ≥ 30 genuine forward sessions + research-validated edge | **`NOT_CERTIFIED`** — zero sessions, no edge, nothing fabricated |
| Staged live | Full chain + external human token; no autonomous activation | **INACTIVE** — SHADOW → PILOT_TINY → SCALED unreachable |

C-Engine is **engineering-certified and research-validated** but has **no certified economic
edge** and is **not ready for paper or live capital.** Per the project's falsification-first
standard, the decision is to **REJECT the current strategy as a deployable edge** while
**ACCEPTING the platform as a sound research + execution substrate.**

---

> **Provenance note.** All economic results referenced in this document are
> historical-replay simulations on genuine bars — **not** live performance and **not** a
> profitability claim. No live orders were placed.
