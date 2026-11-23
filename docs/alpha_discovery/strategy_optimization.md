# Strategy Optimization — Independent Family, Microstructure, and Learning-Based Research

**C-Engine Milestone 8 — Autonomous Alpha Discovery**
Covers §8 strategy families · §9 microstructure · §10 supervised ML · §11 bandit · §12 offline RL

---

## 1. Scope

Milestone 8 is an orchestration and research layer (`cengine/alpha/`) over the
existing M1–M7 stack. It builds **no** new trading engine, learning platform,
optimizer, or lakehouse — it composes the certification harness, the governed
optimizer (`cengine/optimization`), the learning stack (`cengine/learning`), the
continuous runtime (`cengine/runtime`), and the native/Metal substrate.

This document records five **independent** attempts to find a deployable edge
beyond the original momentum entry:

| § | Research track | Verdict |
|---|----------------|---------|
| 8 | Five strategy families (independent archetypes) | **REJECT** — every family negative |
| 9 | Microstructure (genuine L1) | **INSUFFICIENT_DATA** — bars only; proxy does not help |
| 10 | Supervised ML selection (MLX/Metal + baselines) | **REJECT** — no positive selection |
| 11 | Contextual bandit allocation | **INSUFFICIENT_DATA** — IPS refused; direct-method negative |
| 12 | Offline RL (sequential policy) | **INSUFFICIENT_DATA** — 0 genuine execution transitions |

The headline finding across all five tracks: **the failure is not
momentum-specific.** No positive, statistically credible out-of-sample net
expectancy was found on the tested universe under the mandatory **−0.5%**
protective exit.

---

## 2. Shared methodology

All tracks share the same honest accounting and anti-snooping discipline:

- **Capital-constrained accountant** (`cengine/alpha/portfolio_replay.py`) —
  chronological capital, fixed-fraction sizing, fractional shares, max-concurrent
  and max-gross-exposure caps, conserved cash, mark-to-market time-indexed equity.
  It reuses the M7 conservative fill model; only the accounting changes.
- **Mandatory protective exit** — the production **−0.5%** hard stop is applied to
  every archetype and is never modified during research. The **−3%** emergency
  candidate remains inactive.
- **Data** — genuine Alpaca SIP bars in a local Parquet + DuckDB lakehouse. All
  M8 development used only the five canonical 1-minute symbols (AAPL, MSFT, NVDA,
  SPY, QQQ; 2020–2026; 6.76M 1Min bars). `price_scale = 10000` ticks.
- **Splits** — per-symbol chronological TRAIN (70%) / VAL (30%); thresholds and
  model parameters are fit on TRAIN only and measured on VAL. The M7 sealed
  holdout is treated as **consumed** and never reused.
- **Economic scoring** — net expectancy per trade and profit factor (PF), not
  win rate, AUC, or gross Sharpe.
- **Frozen gates** (anti-retrofitting) — profit_factor ≥ 1.30, annualized
  Sharpe ≥ 1.25, Sortino ≥ 1.50, max portfolio drawdown ≤ 10%, positive net
  expectancy, adequate effective sample, and must beat cash, incumbent, and
  simple baselines. Gates were **not** relaxed to manufacture a pass.

---

## 3. §8 — Independent strategy families

The momentum entry was treated as *minimum eligibility*, not profitability. To
test whether the problem was momentum-specific, five families were evaluated
**independently** through the capital-constrained accountant under the −0.5%
exit. Each archetype is a vectorized, long-only implementation faithful to
`FAMILY_SPECS`.

### Result — every family is negative

| Family | Net expectancy / trade | Profit factor | Notes |
|--------|-----------------------:|:-------------:|-------|
| MEAN_REVERSION | **−0.00087** | 0.67 | Least-bad, still negative |
| MULTI_DAY_SWING | −0.00091 | — | Structurally incompatible with the intraday −0.5% stop |
| TREND_PULLBACK | −0.00102 | — | — |
| MOMENTUM | −0.00108 | — | Original incumbent entry |
| VOL_BREAKOUT | −0.00119 | — | Worst of the five |

### Interpretation

- **Decisive and not momentum-specific.** Mean-reversion, trend-pullback,
  volatility breakout, and multi-day swing archetypes all lose money alongside
  momentum. The deficiency is not in the choice of entry logic.
- **The intraday −0.5% exit dominates every long archetype** on this universe.
  The tight protective stop is hit too often relative to the favorable excursion
  any of these entries can generate within the hold window.
- MULTI_DAY_SWING is **structurally** incompatible with an intraday −0.5% stop:
  a multi-day thesis cannot survive a sub-1% intraday protective exit.

**Verdict: REJECT.** No family produces positive OOS net expectancy. The
protective exit is a hard constraint that none of the tested long archetypes
can overcome on this data.

---

## 4. §9 — Microstructure

Genuine Level-1 microstructure was proposed as a possible source of edge:
quote/order-flow imbalance, microprice, bid replenishment, spread dynamics, and
seller-exhaustion signals.

### Data reality

- The historical lakehouse holds **bars only** — there are **no trade or quote
  ticks** available for the development period.
- Genuine L1 signals therefore cannot be computed, scored, or certified on the
  historical data. **No L2 book was fabricated.**

### Bar-only proxy

- A clearly-labeled, bar-only **close-location-value** proxy was constructed as a
  transparent substitute for intrabar flow.
- It **does not help**: net expectancy **−0.00104** — negative, consistent with
  the family results.

### Interpretation

Genuine microstructure is **deferred to live/paper L1 capture**, where
quote-level data can be recorded honestly. The bar-only proxy is retained only
as a labeled placeholder, not as evidence of a microstructure edge.

**Verdict: INSUFFICIENT_DATA.** Genuine L1 cannot be evaluated on bars; the
honest proxy adds nothing.

---

## 5. §10 — Supervised ML filter

A supervised selection layer was tested: can a model filter the incumbent
entries down to a profitable subset?

### Setup

- **Primary model:** MLX/Metal network (ran on the Metal GPU).
- **Baselines:** sklearn decision tree, ridge regression, and a constant
  predictor.
- **Labels:** realized **net** return per candidate trade.
- **Scoring:** economics — top-decile OOS net expectancy — **not** AUC.

### Result — no positive selection

| Model | Top-decile OOS net expectancy |
|-------|------------------------------:|
| Baseline (no filter) | −0.00116 |
| Best (ridge) | **−0.00071** |

- Every model's selected subset remains **negative**.
- Rank information coefficients are **tiny** (+0.06 … +0.09) — weak and
  economically immaterial.

### Interpretation

**Filtering cannot create an edge the entries do not contain.** A selection
model can only reweight existing candidates; when the underlying expectancy is
negative and the predictive signal is near-zero, no decile is profitable net of
costs. The MLX/Metal path is validated as working compute, not as alpha.

**Verdict: REJECT.** No model produces positive OOS selection.

---

## 6. §11 — Contextual bandit allocation

A contextual bandit was considered for allocating capital across families per
regime.

### Off-policy evaluation refused

- Off-policy **IPS / DR** estimators were **refused**: the logged history
  contains **no behavior-policy propensities**, and **none were fabricated**.
  Importance weights without genuine logging propensities would be fiction.

### Direct method

- A direct-method **best-family-per-regime** policy was evaluated as a
  transparent (biased) alternative:

| Policy | VAL net expectancy |
|--------|-------------------:|
| Best-family-per-regime (direct method) | −0.00095 |
| Uniform allocation | −0.00109 |

- The direct-method policy is marginally better than uniform but **still
  negative**.

### Interpretation

Selecting the least-bad family per regime reduces damage but does not create
positive expectancy. With no logged propensities, there is no sound off-policy
certification path, and **no uncontrolled live exploration** was performed to
manufacture one.

**Verdict: INSUFFICIENT_DATA.** IPS/DR unavailable; direct-method negative.

---

## 7. §12 — Offline RL (sequential policy)

An offline RL track was scoped for a sequential execution/allocation policy.

### Data reality

- There are **0 genuine execution transitions** — the available data is
  single-step replay only, not a trajectory of state → action → reward → next
  state suitable for sequential-policy learning.
- Implementations of **BC, CQL, IQL, and FQE** are available and unit-tested, so
  the machinery is ready, but there is nothing genuine to train or certify on.

### Interpretation

Sequential-policy certification is **deferred** until genuine paper/live
transitions accrue. **No unrestricted live RL** was run to generate transitions.

**Verdict: INSUFFICIENT_DATA.** 0 genuine execution transitions.

---

## 8. Cross-cutting conclusion

- **The failure is not momentum-specific.** Five independent families, a
  microstructure proxy, a supervised filter, a bandit allocator, and an offline
  RL scope all fail to produce positive OOS net expectancy under the −0.5% exit.
- **The protective exit is the binding constraint**, not the entry logic. Every
  long archetype on this universe is dominated by the mandatory −0.5% stop.
- **Honesty over optics.** Where data was absent (genuine L1, logged
  propensities, execution transitions), the verdict is INSUFFICIENT_DATA — not a
  fabricated metric and not a relaxed gate. Negative results are preserved and
  never reinterpreted.
- **Decision:** REJECT the current entries and strategy archetypes as a
  deployable edge; ACCEPT the alpha-discovery infrastructure as a sound research
  substrate. Holding cash / NO_TRADE remains a valid, supported outcome.

### Verdict summary

| § | Track | Best observed | Verdict |
|---|-------|--------------:|---------|
| 8 | Strategy families | −0.00087 (MEAN_REVERSION) | REJECT |
| 9 | Microstructure | −0.00104 (bar-only proxy) | INSUFFICIENT_DATA |
| 10 | Supervised ML | −0.00071 (ridge) | REJECT |
| 11 | Bandit | −0.00095 (direct method) | INSUFFICIENT_DATA |
| 12 | Offline RL | — (0 transitions) | INSUFFICIENT_DATA |

---

## 9. Invariants preserved

- Production mandatory **−0.5%** protective exit and the native C++ RiskEngine
  authority are preserved and unmodified.
- The **−3%** emergency candidate stays inactive.
- No new engine, optimizer, or lakehouse was built; no live activation occurred.
- The M7 sealed holdout was protected from optimization; no L2 book was
  fabricated; no fabricated performance of any kind was reported.

---

## Provenance

All economic results in this document are **historical-replay simulations on
genuine bars** — not live performance, and not a profitability claim. **No live
orders were placed.** Negative results are preserved exactly as measured and are
never reinterpreted.
