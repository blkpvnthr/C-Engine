# Capital / Cost Sensitivity (§21) and Stress (§22)

**Milestone:** M7 Certification
**Verdict:** `NOT_CERTIFIED` (both §21 and §22)
**CLI:** `python -m cengine.certification sensitivity` · `python -m cengine.certification stress`

---

## 1. Purpose

These two certification suites interrogate the production incumbent strategy
(the mandatory `-0.5%` protective exit plus the momentum entry) along two axes
that a backtest headline can never reveal on its own:

- **§21 — Capital / cost sensitivity.** Does the strategy's net expectancy
  survive realistic trading frictions? At what cost level does it break even?
  How much capital can it absorb before market impact erases the edge?
- **§22 — Stress.** Does the strategy hold up when the market regime turns
  hostile — the 2020 COVID crash, the 2022 bear market — and does it behave
  sanely in a calm control window (2021)?

Per CLAUDE.md, **gross performance is irrelevant** and **every strategy starts
as false.** These suites exist to attempt destruction of the signal, not to
decorate it. Both suites returned `NOT_CERTIFIED`, which is the correct
falsification-first outcome and is fully consistent with the sealed-holdout
strategy economics (`NOT_CERTIFIED`, 0/5 symbols).

---

## 2. §21 — Capital / Cost Sensitivity

### 2.1 Cost sweep (0 → 16 bps)

The cost sweep re-runs the strategy's realized outcomes across a ladder of
per-side transaction-cost assumptions, from a frictionless `0 bps` floor up to
`16 bps`, and reports net expectancy at each rung. The decisive observation is
at the **zero-cost floor**: a strategy that cannot earn a positive expectancy
when trading is *free* has no economic edge to protect, and no cost model can
rescue it.

| Cost level (per side) | Finding |
|---|---|
| **0 bps (frictionless floor)** | **Negative net expectancy** — AAPL `-0.00033/trade`, SPY `-0.00001/trade` |
| 2 – 16 bps | Monotonically worse; expectancy remains negative across the entire sweep |
| Break-even cost | **`None`** — there is no cost level at which the strategy turns profitable |
| Cost-robust symbols | **0 / N** |

### 2.2 The negative-even-at-zero-cost finding

This is the load-bearing result of §21:

> At `0 bps` — no commission, no slippage, no spread, no borrow — the
> incumbent's net expectancy is **already negative** on the measured symbols
> (AAPL `-0.00033`, SPY `-0.00001`).

The implication is unambiguous. Trading costs are **not** the reason the
strategy loses money. The signal itself has negative expectancy before any
friction is applied. Removing every cost assumption does not move the verdict.
A cost-sensitivity suite normally asks *"how much friction can the edge
tolerate?"* — here there is no edge to erode in the first place.

### 2.3 Break-even cost = `None`

Because expectancy is negative at the frictionless floor and only degrades as
costs rise, **there is no crossover point** between gross profit and cumulative
cost. The break-even cost is reported honestly as `None` rather than
back-solved, extrapolated, or fabricated. A `None` break-even is the
mathematically correct answer when the gross-of-cost curve never clears zero.

### 2.4 Market-impact capacity — NOT modeled (`INSUFFICIENT_DATA`)

True market-impact capacity (how much capital the strategy can deploy before
its own order flow moves prices and destroys the signal) is **not assessable**
from the available data and is reported as `INSUFFICIENT_DATA`.

- The lakehouse holds genuine OHLCV bars, **not** the full depth-of-book / order-flow
  record required to estimate a defensible impact curve.
- Rather than invent a capacity number from a toy impact model, the suite
  declines to assess it. **No capacity figure is fabricated.**
- This is moot for the verdict in any case: capacity analysis is only
  meaningful for a strategy that *has* a positive edge to scale. With negative
  expectancy at zero cost, capacity is not the binding constraint.

### 2.5 §21 verdict

**`NOT_CERTIFIED`** — negative net expectancy even at zero cost; break-even
cost = `None`; `0/N` cost-robust; market-impact capacity not modeled
(`INSUFFICIENT_DATA`).

---

## 3. §22 — Stress

### 3.1 Regime windows

The stress suite replays the strategy through three historical regime windows
drawn from the genuine 1-minute bar record, including a calm control to guard
against a one-sided "only tested in crashes" critique:

| Window | Regime | Role |
|---|---|---|
| **2020 COVID crash** | Extreme volatility / drawdown | Adverse stress |
| **2022 bear market** | Sustained downtrend | Adverse stress |
| **2021** | Calm / trending | Control |

### 3.2 Results

Across **every** window and **every** symbol:

- **Negative expectancy.**
- **Drawdown > 90%.**

The calm 2021 control is the most damning data point: the strategy does **not**
recover its footing even when the regime is benign. The failure is not a
crash-only artifact that a regime filter could sidestep — it is present in calm
conditions too, which rules out "wrong regime" as the explanation and points
back at the signal itself.

| Window | Expectancy | Max drawdown | Coverage |
|---|---|---|---|
| 2020 COVID crash | Negative | > 90% | all symbols |
| 2022 bear market | Negative | > 90% | all symbols |
| 2021 (calm control) | Negative | > 90% | all symbols |

### 3.3 §22 verdict

**`NOT_CERTIFIED`** — negative expectancy with `> 90%` drawdown in the 2020
COVID crash and 2022 bear-market windows, and in the 2021 calm control, for
every symbol.

---

## 4. Consistency with the sealed-holdout economics

These two suites do not contradict the rest of M7 — they corroborate it. The
sealed-holdout strategy-economics result (§, `NOT_CERTIFIED`, `0/5` symbols)
already established that the incumbent has negative net expectancy on the
embargoed holdout for every canonical symbol (expectancy `~ -0.001/trade`,
profit factor `~0.6`, Sharpe `~ -0.2`, drawdown `> 90%`), failing to beat
buy-and-hold on every symbol and the random placebo on `3/5`.

§21 and §22 approach the same conclusion from two independent directions:

- **§21** removes all costs and still finds negative expectancy → the loss is
  structural, not frictional.
- **§22** varies the regime (crash, bear, calm) and still finds negative
  expectancy with catastrophic drawdown → the loss is not regime-specific.

Three orthogonal falsification attempts — holdout scoring, cost sweep, regime
stress — all return the same verdict. That convergence is exactly what a
falsification-first process is designed to surface.

---

## 5. What this does NOT impeach

The `-0.5%` protective exit remains a **certified risk backstop**, not an
economic edge, and nothing in §21/§22 changes that:

- The `-0.5%` exit is a **risk control** (§25 `CERTIFIED`): its unrealized-return
  math is long/short symmetric and it fires correctly at `ret <= -0.5%`.
- Entries are deterministic and look-ahead-free (§26 `CERTIFIED`).
- The negative expectancy is a property of the **entry edge**, measured net of
  (and even before) costs — it is not a defect in the protection layer.

The honest reading: the platform correctly measured that the incumbent has no
deployable edge. The protection and execution substrate works; the strategy
does not earn.

---

## 6. Reproduction

```bash
# Capital / cost sensitivity (§21)
python -m cengine.certification sensitivity

# Stress (§22)
python -m cengine.certification stress

# Full chain + consolidated report
python -m cengine.certification full
python -m cengine.certification report
```

---

## 7. Summary

| Item | Result |
|---|---|
| §21 cost sweep range | `0 → 16 bps` per side |
| Net expectancy at `0 bps` | Negative (AAPL `-0.00033`, SPY `-0.00001`) |
| Break-even cost | `None` |
| Cost-robust symbols | `0 / N` |
| Market-impact capacity | Not modeled — `INSUFFICIENT_DATA` (not fabricated) |
| §21 verdict | **`NOT_CERTIFIED`** |
| §22 regime windows | 2020 COVID crash · 2022 bear · 2021 calm control |
| §22 expectancy / drawdown | Negative / `> 90%`, every window, every symbol |
| §22 verdict | **`NOT_CERTIFIED`** |

---

> **Provenance.** All economic results reported here are historical-replay
> simulations on genuine Alpaca SIP bars — they are **not** live performance and
> **not** a profitability claim. No live orders were placed. Market-impact
> capacity is reported as `INSUFFICIENT_DATA` and is not fabricated.
