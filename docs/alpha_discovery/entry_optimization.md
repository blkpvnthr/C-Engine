# Entry-Quality Optimization, Multi-Timeframe & Regime Conditioning (Milestone 8, §5–§7)

This document covers three branches of the M8 alpha-discovery investigation into
the momentum entry: entry-quality filtering (§5), higher-timeframe confirmation
(§6), and regime conditioning (§7). All three ask the same question from
different angles — *can we select a better subset of momentum entries?* — and all
three return the same honest answer: **no filter, timeframe gate, or regime
condition produces positive out-of-sample net expectancy on the tested
universe.** The directional insight that survives is diagnostic, not deployable:
less-extended entries and higher-timeframe confirmation *reduce damage*, but they
create *no edge*.

M8 builds no new engine. These experiments are an orchestration/research layer
(`cengine/alpha/`) composing the existing certification harness, governed
optimizer, learning stack, and the capital-constrained portfolio accountant. All
economic numbers are historical-replay simulations on genuine Alpaca SIP bars
under the production mandatory **−0.5% protective exit**.

---

## 1. Framing: momentum is minimum eligibility, not profitability

The M7 accounting audit (§3) established that the raw momentum entry has **no
directional edge** over its hold window:

| Diagnostic | Value |
|---|---|
| Stop-out rate | 45.7% |
| Mean max-adverse-excursion (MAE) | −0.69% |
| Mean max-favorable-excursion (MFE) | +0.58% |
| Protective stop | −0.5% |

Mean adverse excursion (−0.69%) **exceeds** the −0.5% stop, while favorable
excursion is only +0.58% — a near-symmetric swing against a tight stop. Adverse
dominates favorable, so the entry carries no directional advantage over the hold
window. This reframes the entire investigation:

> The momentum trigger is treated as a **minimum eligibility condition** (the set
> of bars we are *allowed* to consider), **not** as a profitability signal. The
> §5–§7 question is whether a point-in-time filter can carve a profitable subset
> out of the eligible set.

Because the damage comes from entry selection and not from where the exit sits,
this is an **entry-quality problem, not an exit-tuning problem** (see §9).

---

## 2. The look-ahead-free point-in-time feature set (§5)

Every candidate filter is computed from information available **strictly at or
before** the decision bar. Developing (not-yet-closed) candles are **never** used
to qualify an earlier decision. Causality is enforced by a dedicated test, not
left to convention.

| Feature | Description |
|---|---|
| `momentum` / `accel` | Price momentum and its rate of change |
| `rel_volume` | Relative volume vs recent baseline |
| `displacement_from_sma` | Distance of price from its SMA (extension) |
| `atr_norm_displacement` | ATR-normalized displacement (extension in volatility units) |
| `bb_width` | Bollinger-band width (volatility state) |
| `htf_trend_15` / `htf_trend_60` | 15-minute / 60-minute higher-timeframe trend |
| `trend_persistence` | Duration/consistency of the prevailing trend |
| `time_of_day` | Intraday session position |
| `directional_separation` | Directional-indicator separation |

These are the same families used downstream by the ML (§10) and regime (§7)
experiments, so the ablation, model, and regime results share one causal feature
contract.

---

## 3. Per-filter OOS ablation methodology

The ablation is designed to be hostile to false positives:

- **Chronological split, per symbol.** Each symbol is split TRAIN (first 70%) /
  VAL (last 30%) in time order. No shuffling — the validation period always
  follows the training period, so a filter cannot borrow future structure.
- **Threshold fit on TRAIN only.** Each filter's cut threshold is fit on TRAIN
  and then measured, unchanged, on VAL. VAL is never used to choose a threshold.
- **Greedy, complexity-penalized combination.** Filters are combined greedily
  with an explicit complexity penalty so that adding a filter must *earn* its
  marginal degrees of freedom rather than merely reduce the trade count.
- **Explicit multiple-testing caveat.** Many filters are screened; the more
  thresholds tested, the more likely one looks good on VAL by chance. The best
  single-filter result is interpreted *with* this caveat, not as a discovery.

The objective is **net per-trade expectancy** under the −0.5% exit — not win
rate, not AUC, not gross.

---

## 4. Ablation results (§5): no filter earns positive OOS expectancy

| Configuration | OOS net expectancy | Verdict |
|---|---|---|
| Baseline (all eligible momentum entries) | −0.00116 | negative |
| Best single filter — `htf_trend_60` ≤ threshold | −0.00081 | **least-bad, still negative** |
| Greedy complexity-penalized combo | train +0.00009 / val insufficient | **overfits** |

Findings:

- **No entry filter produces positive OOS net expectancy.** The best single
  filter (`htf_trend_60`) improves expectancy from −0.00116 to −0.00081 — a
  meaningful reduction in damage, but still **negative**.
- **The greedy combination overfits.** It reaches a marginally positive
  expectancy on TRAIN (+0.00009) but has insufficient VAL support to confirm —
  exactly the pattern the complexity penalty and chronological split exist to
  catch.
- **Directional insight (diagnostic only):** *less-extended* entries do
  marginally better. The best filters lean against extension (lower
  displacement / higher-TF trend not already stretched). This narrows damage; it
  does not manufacture an edge.

---

## 5. Multi-timeframe confirmation (§6)

Higher-timeframe (15m / 60m) trend confirmation is evaluated **at the executable
portfolio level** — through the capital-constrained accountant, not on an
idealized per-trade series — so the result reflects real concurrency and sizing.

| Configuration | Portfolio drawdown | Per-trade expectancy |
|---|---|---|
| All eligible entries | 80% | negative |
| Higher-TF confirmation gate | 66% | **still negative** |

- Confirmation **cuts both trades and damage**: drawdown falls from ~80% to ~66%.
- Per-trade expectancy **remains negative**. The gate reduces *exposure* to a
  losing entry; it does not turn it into a winning one.
- Causality preserved: developing candles are never used to qualify an earlier
  decision — confirmation is read only from closed higher-TF bars.

Multi-timeframe confirmation is therefore a **risk-reduction lever, not an alpha
source**.

---

## 6. Regime conditioning (§7)

A lightweight regime proxy is used for research only — **the production
`MarketRegimeEngine` remains the sole authority**. The proxy partitions bars by
*direction × volatility terciles*, and an **abstention policy** (which regimes to
sit out) is chosen on TRAIN and measured **once** on VAL.

| Finding | Detail |
|---|---|
| Worst regime for momentum | **up-vol** regimes — momentum bleeds worst here |
| Only positive-TRAIN regime | `down_midvol` — but **negative on VAL** (overfits) |
| Abstention policy effect | improves results but stays **non-positive** |
| Valid outcome | **NO_TRADE** is an accepted result |

- The only regime that looks profitable in-sample (`down_midvol`) fails
  out-of-sample — the same overfitting signature as the greedy filter combo.
- Abstaining from the worst regimes improves the aggregate but does not cross
  into positive territory.
- Choosing to **not trade** a regime is an explicit, supported outcome, not a
  failure of the harness.

---

## 7. Failure attribution: an entry-quality problem, not an exit problem

Across §5–§7 the pattern is consistent and the attribution is unambiguous:

- Filtering by entry feature (§5) → least-bad −0.00081, still negative.
- Gating by higher timeframe (§6) → drawdown 80% → 66%, expectancy still
  negative.
- Conditioning by regime (§7) → abstention improves but stays non-positive.

Every lever that **reduces exposure** reduces *damage* without creating *edge*.
That is the signature of an entry with no directional advantage — consistent with
the MAE/MFE diagnostic in §1 where adverse excursion (−0.69%) exceeds favorable
(+0.58%). The constraint is the **quality of the entry decision**, not the
placement of the −0.5% exit. Widening the exit (researched in isolation in §14)
improves expectancy in counterfactual but every variant remains negative;
production exits are never modified by these experiments.

---

## 8. Conclusion

| Branch | Best OOS result | Verdict |
|---|---|---|
| Entry-quality filters (§5) | −0.00081 (`htf_trend_60`) | **no positive OOS expectancy** |
| Multi-timeframe (§6) | drawdown 80%→66%, expectancy negative | **risk lever, not edge** |
| Regime conditioning (§7) | abstention improves, stays non-positive | **NO_TRADE valid** |

**Decision: REJECT** the filtered/conditioned momentum entries as a deployable
edge. No entry-quality, timeframe, or regime conditioning produced positive,
statistically credible out-of-sample net expectancy. The investigation **ACCEPTS
the ablation/attribution methodology** — chronological splits, TRAIN-only
threshold fitting, complexity penalties, an executable-portfolio evaluation, and
an explicit multiple-testing caveat — as a sound research substrate. Negative
results are preserved and are not reinterpreted as success.

The actionable research conclusion is a *redirection*: filtering a directionless
entry cannot create an edge the entry does not contain. The problem is entry
quality, and it must be solved at the entry, not downstream.

---

*Provenance: all economic results above are historical-replay simulations on
genuine Alpaca SIP bars, not live performance and not a profitability claim. No
live orders were placed. The production mandatory −0.5% protective exit and the
native risk authority were preserved and never modified by these experiments.
Negative results are preserved and never reinterpreted.*
