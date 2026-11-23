# C-Engine M7 Certification — Methodology

This document defines the certification methodology applied to C-Engine at the M7
milestone. It is the standard against which every economic and engineering claim
in the certification suite is tested. The methodology is deliberately hostile to
the strategy: its purpose is not to confirm that an edge exists, but to give any
alleged edge every opportunity to fail.

The platform under test is a local-first native quant stack running on an Apple
M5 Max (48 GiB unified memory), Python 3.11, with MLX/Metal GPU acceleration.

---

## 1. Core Philosophy

Three principles govern every verdict. They are inherited directly from the
project's research charter and are non-negotiable.

### 1.1 Falsification first — the burden of proof is on the strategy

Every strategy begins its life as **false**. A signal is presumed to have no
edge until it has survived deliberate attempts to destroy it. We do not search
for evidence that confirms a strategy; we search for evidence that kills it, and
we promote a strategy only when it refuses to die.

Practically, this means certification runs are structured as falsification
attempts: walk-forward and holdout scoring, placebo baselines, cost-sensitivity
sweeps, and stress windows. A strategy that cannot be falsified on the evidence
available is **not** thereby certified — it may simply be `INSUFFICIENT_DATA`.

### 1.2 Net performance only

Gross performance is irrelevant. Every economic figure is evaluated **after**
realistic frictions. Fills are modeled conservatively (see §6), and costs are
charged on both sides of every round trip. A strategy that is profitable gross
and unprofitable net is unprofitable.

### 1.3 Out-of-sample is king

In-sample optimization carries no weight. The methodology prefers, in order:
sealed-holdout scoring, walk-forward results, and — when it eventually exists —
genuine forward paper evidence. Any result that required parameter tuning on the
data it is scored against is treated as suspect and does not support a verdict.

---

## 2. Verdict Vocabulary

Certification does not reduce to pass/fail. Each check resolves to one of:

| Verdict | Meaning |
| --- | --- |
| `CERTIFIED` | The property was tested and held under the frozen criteria. |
| `NOT_CERTIFIED` | The property was tested and **failed** the frozen criteria. |
| `INSUFFICIENT_DATA` | The evidence required to render a verdict does not exist. This is a **first-class outcome**, never silently upgraded to a pass and never fabricated into a fail. |
| `RESEARCH_VALIDATED` | Reserved for overall readiness: infrastructure is sound, but no economic edge is certified. |

`INSUFFICIENT_DATA` deserves emphasis. When the data to decide a question is
absent — no materialized realized-outcome layers, no live sessions, no modeled
market-impact curve — the honest answer is "we do not know." The methodology
forbids converting that silence into either confidence or condemnation. For
example, true market-impact capacity (§21) and the bandit/RL learning tasks are
reported as `INSUFFICIENT_DATA`, not as zero and not as failure.

---

## 3. Frozen Economic Gates (§20)

Economic acceptance is governed by **FrozenEconomicGates**, a fixed set of
thresholds that a strategy must clear to be considered a deployable edge:

- Minimum expectancy **> 0** per trade
- Profit factor **≥ 1.10**
- Sharpe **> 0**
- Maximum drawdown **≤ 25%**
- Must **beat buy-and-hold**
- Must **beat the random placebo**

### 3.1 Fingerprint anti-retrofitting

The gates are **frozen by fingerprint**. A `verify_frozen()` check computes a
fingerprint over the threshold set and trips if any value has drifted. This is
an explicit anti-retrofitting control: it prevents the common failure mode in
which thresholds are quietly relaxed after the fact so that a losing strategy
appears to clear the bar. The gates must be the same gates that were committed
before scoring; if they are not, the certification is invalid by construction.

---

## 4. Provenance Discipline

Not all results are equal, and the methodology enforces this at the data layer.

- **`synthetic`** results (generated data) can **never** back a profitability
  claim.
- **`historical_replay`** results — simulations over genuine bars — can **never**
  back a profitability claim either. They are evidence about mechanics and
  falsification, not about live edge.
- Only genuine **`real`** realized outcomes can support a claim that a strategy
  made money.

This discipline is certified in the learning subsystem: no learning path is
permitted to launder synthetic or replayed results into an economic claim. It is
also why every economic section of this certification carries the provenance note
reproduced at the end of this document. **All economic results in M7 are
historical-replay simulations on genuine bars.** None are live performance.

---

## 5. The Sealed Holdout (§5)

Economic scoring is performed against a **sealed holdout** to prevent the
out-of-sample set from leaking into the research loop.

- Splits are derived **deterministically per symbol** from the data span:
  **train 60% / validation 20% / holdout 20%**, with an embargo between segments.
- The holdout is accessed through an **access-logged seal**, and only **once per
  symbol**, for **scoring only**.
- **No symbol is touched twice**, and the holdout is **never tuned on**.

The seal is the mechanism that makes "out-of-sample is king" enforceable rather
than aspirational. If a symbol's holdout had already been consumed, a second
access would be visible in the log; the one-shot discipline is auditable, not a
matter of trust.

---

## 6. Conservative Fill Model

Simulated executions resolve ambiguity **against** the strategy. Three rules
define the conservative fill model used throughout the economic sections:

1. **Stop-before-target.** When a bar's range spans both a protective/stop level
   and a profit target, the stop is assumed to fill first. Ambiguous bars are
   resolved as losses, never as wins.
2. **Gap-through-at-open.** When price gaps through a level between bars, the fill
   is taken at the **open** on the far side of the gap, not at the level — the
   strategy eats the gap.
3. **Costs on both sides.** Frictions are charged on entry and on exit of every
   round trip; there is no one-sided accounting.

These rules make the simulated P&L a *lower* bound on an optimistic fill model,
consistent with the net-performance-only principle.

---

## 7. Protective Exit as Backstop, Not Edge

A mandatory **-0.5% protective exit** is part of the production incumbent. The
methodology treats this exit as a **risk backstop**, not as a source of economic
edge. The distinction matters for how recovery statistics are interpreted
(§9.3): a high rate of price recovery *above the exit level* does not prove the
exit is economically wrong, because downside avoided, capital reuse, and
opportunity cost are not captured by that single statistic. The exit is
certified for correctness and independence (it imports no ML/GPU/lakehouse/
scanner module), which is a separate question from whether the entry+exit
combination has an edge — it does not (§9).

---

## 8. Baseline Hierarchy (§8)

A strategy is never scored in a vacuum. It is scored **against baselines**, and
it must beat the economically meaningful ones. The baseline hierarchy is:

| Baseline | Role |
| --- | --- |
| `hold_to_horizon` | Enter on signal, hold to a fixed horizon — isolates the contribution of the exit logic. |
| `buy_and_hold` | The passive benchmark the strategy must beat to justify its existence. |
| `random_placebo` | A randomized-entry control. Beating this is the minimum bar for "the signal is doing something real." |
| `always_flat` | Hold no position — the zero-risk, zero-return floor. |

Under the frozen gates (§3), a strategy must beat **both** `buy_and_hold` **and**
`random_placebo`. A strategy that cannot beat a coin flip has not demonstrated a
signal; a strategy that cannot beat passive exposure has not demonstrated a
reason to trade.

---

## 9. What the Methodology Found at M7

The methodology was applied in full. The economic verdict is negative, and the
methodology reports it honestly rather than massaging it.

### 9.1 Strategy economics (sealed holdout) — `NOT_CERTIFIED`, 0/5 symbols

The production incumbent (mandatory -0.5% protective exit + momentum entry) has
**negative net expectancy** on the sealed holdout for **every** canonical symbol
(AAPL, MSFT, NVDA, SPY, QQQ): expectancy ≈ -0.001/trade, profit factor ≈ 0.6,
Sharpe ≈ -0.2, drawdown > 90%. It **fails to beat buy-and-hold on every symbol**
and fails to beat the **random placebo on 3/5**.

### 9.2 Capital / cost sensitivity (§21) — `NOT_CERTIFIED`

Net expectancy is negative **even at zero cost** (AAPL -0.00033, SPY -0.00001 at
0 bps). Break-even cost is `None`; the strategy is cost-robust on 0 of N
configurations. True market-impact capacity is **not modeled** and is reported as
`INSUFFICIENT_DATA` — not fabricated.

### 9.3 Stress (§22) — `NOT_CERTIFIED`

Negative expectancy with > 90% drawdown in the 2020 COVID-crash and 2022
bear-market windows, and in the 2021 calm control, for every symbol.

### 9.4 Recovery reproduction (measurement only) — `CERTIFIED`

An independent reimplementation over 20,572 exit events on genuine 1-minute bars
finds **95.7%** of -0.5% exits recover **above the exit level** within 60 min
(moving-block-bootstrap 95% CI [95.4%, 95.9%]), reproducing the ~95.3% figure.
But only **47.7%** recover **above the entry price** within 60 min — the
decision-relevant bar. Per §7, the high above-exit rate does **not** prove the
exit is economically wrong.

### 9.5 Governed optimization — retains the incumbent

The governed optimization loop certified **no economic improvement** and
**retained the incumbent**. This is the correct falsification-first outcome: the
loop was unable to manufacture an edge, and it did not pretend to.

---

## 10. Scope: Economics vs. Infrastructure

The methodology keeps two questions strictly separate.

- **Does the strategy have an edge?** Answer: **No.** Economics are
  `NOT_CERTIFIED` across strategy (§9.1), sensitivity (§9.2), and stress (§9.3).
- **Is the platform a sound research and execution substrate?** Answer: **Yes.**
  Data integrity, engineering, native/Metal correctness, failure isolation, the
  protective exit, the entry contract, and learning infrastructure are each
  certified on their own terms.

The overall readiness verdict (§32) is therefore **RESEARCH_VALIDATED**:
infrastructure is sound across its domains, there is **no certified economic
edge**, the system is **NOT_READY** for capital, and `live_eligible = False`.
Paper certification is `NOT_CERTIFIED` because it requires ≥ 30 genuine elapsed
forward sessions **and** a research-validated edge, and neither exists; no
prospective paper evidence is fabricated. Live deployment is **INACTIVE**, hard-
gated behind the full chain plus an external human authorization token the
software cannot mint.

Per the project's decision framework, the conclusion is to **REJECT** the current
strategy as a deployable edge while **ACCEPTING** the platform as a sound
research and execution substrate.

---

## Provenance Note

All economic results referenced in this document are **historical-replay
simulations on genuine bars** — they are **not** live performance and **not** a
profitability claim. **No live orders were placed.**
