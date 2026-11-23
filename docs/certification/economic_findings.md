# C-Engine M7 Certification — Economic Findings

**Verdict (economics): NOT_CERTIFIED — 0/5 symbols.**
**Decision: REJECT the current strategy as a deployable edge.**
**Infrastructure verdict (separate): RESEARCH_VALIDATED.**

This document records the central, honest economic finding of the M7
certification. The platform is engineering-certified and research-validated;
the *strategy it executes* is not. These are two different claims and this
file is about the second one.

---

## 1. The central finding

The production incumbent — a **mandatory -0.5% protective exit paired with a
momentum entry** — has **negative net expectancy on the sealed holdout for
every one of the 5 canonical symbols** (AAPL, MSFT, NVDA, SPY, QQQ).

Three independent facts make the rejection unambiguous:

1. **Negative net expectancy after costs.** Expectancy ≈ **-0.001 per trade**,
   profit factor ≈ **0.6**, Sharpe ≈ **-0.2**, max drawdown **> 90%** — on
   every symbol.
2. **Negative even at zero cost.** With all frictions set to 0 bps the
   expectancy is still negative (AAPL **-0.00033**, SPY **-0.00001** @ 0 bps).
   There is no cost level at which the strategy breaks even
   (break-even cost = **None**).
3. **Negative under stress.** Expectancy stays negative with > 90% drawdown in
   the 2020 COVID crash window, the 2022 bear-market window, and even the 2021
   calm control — for every symbol.

It fails to beat buy-and-hold on **all 5** symbols and fails the random
placebo on **3 of 5**. The frozen economic gates (expectancy > 0, PF ≥ 1.10,
Sharpe > 0, DD ≤ 25%, must beat buy-and-hold and random placebo) are **not
met on any symbol**.

> The -0.5% exit is a **risk backstop, not an economic edge.** It is certified
> as a correct, independent protective mechanism (§25) — this document is about
> whether trading *around* it produces positive expectancy. It does not.

---

## 2. Per-symbol results (sealed holdout, net of costs)

Scored once per symbol through the access-logged seal (train 60% / validation
20% / holdout 20%, embargoed; no symbol touched twice; never tuned on).

| Symbol | Net expectancy / trade | Profit factor | Sharpe | Max drawdown | Beats buy-and-hold? | Gate result |
|--------|------------------------|---------------|--------|--------------|---------------------|-------------|
| AAPL   | ≈ -0.001              | ≈ 0.6         | ≈ -0.2 | > 90%        | No                  | FAIL        |
| MSFT   | ≈ -0.001              | ≈ 0.6         | ≈ -0.2 | > 90%        | No                  | FAIL        |
| NVDA   | ≈ -0.001              | ≈ 0.6         | ≈ -0.2 | > 90%        | No                  | FAIL        |
| SPY    | ≈ -0.001              | ≈ 0.6         | ≈ -0.2 | > 90%        | No                  | FAIL        |
| QQQ    | ≈ -0.001              | ≈ 0.6         | ≈ -0.2 | > 90%        | No                  | FAIL        |

**Summary: 0 / 5 symbols certified.**

Notes on the figures:
- Values are the certified per-symbol *characterization* of the holdout
  scoring; expectancy, PF, Sharpe and drawdown all cluster at the levels
  shown for every symbol.
- **Random placebo:** passed on 2 of 5, failed on 3 of 5. A strategy that
  cannot reliably beat a random entry of matched frequency has not
  demonstrated information in its entries.
- **Zero-cost expectancy** (frictionless, from §21) is reported exactly for
  AAPL (**-0.00033**) and SPY (**-0.00001**); both remain negative, confirming
  the deficit is structural rather than cost-driven.

---

## 3. Why it loses — the mechanism

The loss is not noise and not a cost artifact. It comes from the **shape of the
exit policy** interacting with the entry:

- **The tight protective stop exits positions that frequently recover.**
  The mandatory -0.5% exit fires on small adverse moves. Many of those moves
  reverse shortly afterward — so the strategy realizes the small loss and then
  watches the name climb back without being in it.
- **The time-stop caps the winners.** Positions that *do* work are closed by
  the holding-period cap before the favorable move is fully captured. Upside is
  truncated while downside is realized in full at -0.5%.
- **The payoff is asymmetric in the wrong direction.** Frequent small realized
  losses + occasional truncated wins → profit factor ≈ 0.6 (losses outweigh
  gains ~5:3) and negative expectancy *before any cost is applied*.
- **Costs compound an already-negative edge.** Commissions, slippage and
  spread turn a small structural deficit into a larger one and push
  break-even cost to **None** — there is no friction regime in which the
  strategy clears zero.

In short: the entry does not identify moves large enough to overcome a -0.5%
stop plus a time cap. The exit does its job as *protection* (bounding loss per
trade), but bounding loss is not the same as producing gain.

---

## 4. Tie to the recovery reproduction

A common intuition is: "if prices usually recover after the -0.5% exit, the
exit is throwing away money." The independent recovery reproduction
(measurement only, on genuine 1-minute bars, **20,572 exit events**) shows why
that intuition is **the wrong benchmark**:

| Recovery benchmark (within 60 min) | Rate | 95% CI | Decision-relevant? |
|------------------------------------|------|--------|--------------------|
| Price recovers **above the exit level** | **95.7%** | [95.4%, 95.9%] | No |
| Price recovers **above the entry price** | **47.7%** | — | **Yes** |

- The **95.7% above-exit** figure (reproducing the ~95.3% prior result)
  measures how often the market ticks back above the -0.5% level. It says
  nothing about whether *re-entering or holding* would have been profitable —
  it ignores downside avoided, capital reuse, and opportunity cost.
- The **47.7% above-entry** figure is the decision-relevant bar: after being
  stopped at -0.5%, the position gets back above the original **entry** price
  within the hour **less than half the time**. A coin-flip of recovery to
  break-even is exactly consistent with a strategy that cannot generate
  positive expectancy.

**A high above-exit recovery rate does NOT prove the exit is economically
wrong.** The exit is certified-correct protection; the entries, not the stop,
are what fail to pay.

---

## 5. Cost-sensitivity curve (§21)

Capital / cost sensitivity is **NOT_CERTIFIED: 0/N cost-robust.**

- **Shape:** expectancy as a function of per-trade cost is a monotonically
  decreasing line that is **already below zero at the left edge (0 bps)**.
  Adding cost only moves it further negative.
- **Zero-cost anchors:** AAPL **-0.00033**, SPY **-0.00001** at 0 bps — both
  negative.
- **Break-even cost:** **None.** There is no cost level, including zero, at
  which net expectancy reaches 0. The curve never crosses the x-axis within the
  feasible region.
- **Market-impact capacity:** **NOT modeled — INSUFFICIENT_DATA.** True
  market-impact capacity was not fabricated; it is explicitly reported as not
  measurable with available data. The cost-sensitivity result therefore does
  not even depend on an impact assumption to fail.

Interpretation: because the strategy is underwater before a single basis point
of friction, no realistic improvement in execution cost, sizing, or capacity
rescues it. The problem is the signal, not the trading cost.

---

## 6. Governed-optimization corroboration

The governed optimization loop was run under falsification-first discipline and
**retained the incumbent** — it certified **no economic improvement** over the
-0.5% exit + momentum entry. `OptimizationConfig.validate` refuses autonomous
risk expansion, locks the -0.5% production exit, and refuses to loosen the -3%
emergency threshold. The loop's inability to find a better configuration is
itself the correct falsification-first outcome: it looked, and there was no edge
to retain other than the risk backstop.

Strategy-relevant, bandit and RL learning tasks are **INSUFFICIENT_DATA** (no
materialized realized-outcome layers, no live sessions). No profitability claim
can be or is made from synthetic or historical-replay results.

---

## 7. Conclusion

| Question | Answer |
|----------|--------|
| Positive net expectancy on holdout? | **No** (0/5) |
| Positive at zero cost? | **No** |
| Positive under crash / bear stress? | **No** |
| Beats buy-and-hold? | **No** (0/5) |
| Beats random placebo? | **No** (fails 3/5) |
| Any break-even cost? | **None** |

Per CLAUDE.md (falsification first; net performance only; rejection over false
confidence):

> **REJECT the current strategy as a deployable edge.**
> Accept the platform as a sound research and execution substrate.

Economic readiness (§32) is **RESEARCH_VALIDATED**, **NOT_READY for capital**,
`live_eligible = False`. Paper is **NOT_CERTIFIED** (requires ≥ 30 genuine
forward sessions and a research-validated edge; none exist). Live is
**INACTIVE** and hard-gated behind an external human authorization token the
software cannot mint.

---

### Provenance note

All economic results in this document are **historical-replay simulations on
genuine Alpaca SIP bars** — they are **not live performance** and **not a
profitability claim**. **No live orders were placed.**
