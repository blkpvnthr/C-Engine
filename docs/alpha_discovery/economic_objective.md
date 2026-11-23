# Economic Objective & Challenger-Acceptance Gates (Milestone 8, §13/§19/§21)

Milestone 8 is an orchestration + research layer (`cengine/alpha/`) over the
existing M1–M7 stack. It builds no new trading engine, optimizer, learning
platform, or lakehouse — it composes the certification harness, the governed
optimizer (`cengine/optimization`), the learning stack (`cengine/learning`),
the continuous runtime (`cengine/runtime`), and the native/Metal substrate.

This document specifies **what M8 is allowed to call "good"**: the governed
economic objective `J` (§19), the capital-constrained portfolio accountant and
the cash / `NO_TRADE` baseline it is scored against (§13), and the **frozen**
challenger-acceptance gates that any candidate must clear before it can be
promoted (§21).

> **Honest standing.** No positive, statistically credible out-of-sample net
> expectancy was found on the tested universe under the mandatory −0.5%
> protective exit. The current entries/strategies are **REJECTED** as a
> deployable edge; the alpha-discovery infrastructure is **ACCEPTED** as a sound
> research substrate. Everything below describes the scoring machinery — not a
> profitable result.

---

## 1. The governed economic objective `J` (§19)

Candidates are never scored on win rate, AUC, gross return, or Sharpe in
isolation. They are scored on a single governed objective that nets realistic
costs and then charges explicit penalties for the ways a flattering mean can lie:

```
J = net_expectancy
    − drawdown_penalty    · dd
    − downside_penalty    · downside
    − uncertainty_penalty · unc
```

| Term | Meaning |
| --- | --- |
| `net_expectancy` | Per-unit net return **after** the M7 conservative fill model (commissions, slippage, spread). Gross is irrelevant. |
| `dd` | Realized portfolio drawdown from the capital-constrained accountant (§2). |
| `downside` | Downside-deviation term — penalizes left-tail / negative dispersion, not symmetric volatility. |
| `unc` | Estimation uncertainty of the edge (small / unstable effective sample is charged). |

**Calibration discipline.** The three penalty weights
(`drawdown_penalty`, `downside_penalty`, `uncertainty_penalty`) are calibrated on
**development data only**. They are never fit on the sealed holdout or the fresh
cohort (§20). The objective is explicitly forbidden from being reduced to any
single flattering metric:

- Never optimize solely for **win rate**.
- Never optimize solely for **AUC**.
- Never optimize solely for **Sharpe**.
- Never optimize solely for **gross** return.

`J` is a necessary, not a sufficient, condition for promotion — a candidate that
scores well on `J` still has to clear every frozen gate in §4.

---

## 2. The capital-constrained portfolio accountant (§13)

`J` is only meaningful if the drawdown and return it consumes are computed
**correctly**. The P0 accounting audit (§3) found that the M7 "~90–98% drawdown"
headline was a **compounded independent trade-level artifact**: running
`economic_metrics` with `drawdown_mode="compound"` over a flat per-trade series
models an all-in, sequential, fully-reinvested **single-position** curve. The
strided entries actually overlap in time and form a **capital-constrained
portfolio of concurrent positions**.

`cengine/alpha/portfolio_replay.py` is the corrected accountant. It:

- walks capital **chronologically**;
- uses **fixed-fraction** position sizing with **fractional shares**;
- enforces a **max concurrent positions** cap and a **max gross-exposure** cap;
- **conserves cash** (cash is never allowed to go negative);
- marks to market on a **time-indexed** equity curve, yielding a proper
  drawdown, CAGR, and **annualized** Sharpe;
- **reuses the M7 conservative fill model unchanged** — only the *accounting*
  changes, never the per-trade economics.

### Corrected vs. compounded result (15,000 trades / 5 symbols)

| Measure | M7 compound (artifact) | Capital-constrained portfolio (corrected) |
| --- | --- | --- |
| Max drawdown | ~100% | **80.3%** (10% / position, max 10 concurrent) |
| Final equity | — | **−80.2%** |
| Annualized Sharpe | — | **−5.2** |
| Cash conservation | — | Verified; cash never negative (min **$10,669**) |

The per-trade economics are **unchanged and still negative** after the
correction: expectancy **−0.00108**, profit factor **0.64**. (M7's "−0.2 Sharpe"
was a *per-trade* figure, not annualized.) **The accounting correction does not
rescue the strategy** — it only makes the drawdown and Sharpe that feed `J`
honest.

---

## 3. The cash / `NO_TRADE` baseline — and why it dominates (§13)

Cash (`NO_TRADE`) is not a fallback; it is an **explicit, first-class baseline**
that every challenger must beat. On the tested universe it **dominates** every
deployed-entry portfolio:

| Portfolio | Terminal result |
| --- | --- |
| `all_eligible` (deploy every eligible entry) | **−80%** |
| `selective_htf` (higher-timeframe confirmation) | **−70%** |
| **cash / `NO_TRADE`** | **+0%** |

Higher-timeframe confirmation cuts trades and cuts *damage* (drawdown
80% → 66% at the executable portfolio level, §6) but never turns per-trade
expectancy positive — it reduces exposure, it does not create edge. Deploying
negative-edge entries **destroys capital relative to holding cash**, so a valid
`NO_TRADE` decision is fully supported by the accountant and is the correct
economic outcome here.

---

## 4. FROZEN challenger-acceptance gates (§21)

`ChallengerGates` is the promotion filter. Clearing `J` (§1) is table stakes; a
candidate is promotable **only if it clears every gate below**. The gate set
carries an **anti-retrofitting fingerprint** so that any quiet change to the
thresholds is detectable, and the gates are **not relaxed to manufacture a
pass**.

| # | Gate | Threshold |
| --- | --- | --- |
| 1 | Profit factor | **≥ 1.30** |
| 2 | Annualized Sharpe | **≥ 1.25** |
| 3 | Sortino ratio | **≥ 1.50** |
| 4 | Max portfolio drawdown | **≤ 10%** |
| 5 | Net expectancy | **positive** |
| 6 | Effective sample | **adequate** |
| 7 | Baseline dominance | must **beat cash + incumbent + simple baselines** |

Notes on the gate design:

- **Drawdown** (gate 4) is measured by the capital-constrained accountant
  (§2) — the honest ≤ 10% portfolio drawdown, not the compounded artifact.
- **Baseline dominance** (gate 7) encodes §3 directly: beating cash /
  `NO_TRADE`, the current incumbent, and simple baselines is mandatory, not
  optional.
- **Adequate effective sample** (gate 6) pairs with the `uncertainty_penalty`
  term in `J` — a lucky edge on a thin sample fails here even if the mean looks
  good.
- **Anti-retrofitting fingerprint.** The frozen gate configuration is
  fingerprinted so the thresholds cannot be silently loosened to force a
  candidate through. Gates are never relaxed to produce a pass.

### How the gates performed in the genuine run

No development challenger reached the frozen gates. Across the alpha-discovery
cycle the honest tally was **0 accepted / 6 rejected / 2 deferred**, with the
incumbent retained. Because no development challenger cleared the gates, the
fresh 26-symbol five-minute cohort was **not consumed by a selection search**;
one pre-registered confirmatory incumbent test on it (68,853 trades)
independently corroborated the negative finding (expectancy **−0.001**, profit
factor **0.67** — not positive). Independent certification is therefore
**DEFERRED**: there is no positive edge to certify, and no best-of-many
development configuration is described as certified.

---

## 5. Invariants this layer preserves

- The production **mandatory −0.5% protective exit** and the native C++
  `RiskEngine` authority are preserved; protection is never gated by research.
- The **−3% emergency exit candidate remains INACTIVE**.
- No new engine, optimizer, or lakehouse is introduced.
- No live activation; **no live orders were placed**.
- The **sealed M7 holdout is treated as consumed** and is never reused for
  optimization; the fresh cohort is reserved.
- Penalty weights (§1) are calibrated on development data only.
- Negative results are preserved and never reinterpreted.

---

## Provenance

All economic results reported here are **historical-replay simulations on
genuine Alpaca SIP bars** — they are **not live performance** and **not a
profitability claim**. **No live orders were placed.** Negative results are
preserved as-is and are never reinterpreted. The honest conclusion stands:
**no deployable edge was found; the strategy is REJECTED and the research
infrastructure is ACCEPTED.**
