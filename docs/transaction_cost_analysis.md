# Transaction Cost Estimation & Analysis (Milestone 4, §7/§18)

`cengine/transaction_cost.py` keeps **expected** (pre-trade) and **realized**
(post-trade) costs strictly separate.

## Expected — `TransactionCostEstimator.estimate(...)`

Component-separated, in basis points of notional (`CostEstimate`):

- `spread_cost_bps` — half the quoted spread when crossing, a quarter when resting.
- `market_impact_bps` — prefers **Kyle's lambda** (price impact per unit signed
  $volume) from microstructure; else a size-vs-liquidity √(participation) model;
  participation uses observed arrival-intensity × mean trade size (never L2 depth).
- `fee_bps` — 0 by default (Alpaca US equities are commission-free).
- `basis` — `"microstructure"` or `"conservative_fallback"`.

When microstructure is unavailable or stale it returns a **deterministic, explicitly
conservative** fallback (pessimistic fixed spread + impact) flagged
`is_conservative_fallback = True`.

## Realized — `analyze_execution(...)`

Computes `ExecutionOutcome` from a terminal/managed order + genuine timestamps.
Side-aware (`_bps`: positive = worse — paid more on a BUY, received less on a SELL):

- `implementation_shortfall_bps` — average fill vs **decision** price.
- `realized_slippage_bps` — average fill vs **arrival** price.
- `adverse_price_movement_bps` — arrival vs decision drift.
- latencies — `acknowledgement_latency_ns`, `fill_latency_ns`.
- `filled_quantity` / `unfilled_quantity`, `cancellation_count`, `repricing_count`,
  `quoted_spread_bps`, `expected_slippage_bps`.

Unknown prices/timestamps stay `None`; **no counterfactual fills are fabricated**.
Unfilled opportunity is recorded via `unfilled_quantity` rather than imagined.

## No double-counting (§7)

The estimator is the single **execution-time** cost model. It is kept separate from
the portfolio optimizer's pre-trade per-family bps estimate (used only for ranking):
callers that already folded a cost into expected returns do not add it again.

## Evidence

`DecisionEventSink.record_transaction_cost_analysis(outcome, ...)` emits a
`TRANSACTION_COST_ANALYSIS` event linked by decision id; plans emit `EXECUTION_PLAN`
with `estimated_transaction_cost_bps` + `cost_components`. See
`reports/transaction_cost_validation.json`.
