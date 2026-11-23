# Level 1 Microstructure Intelligence (Milestone 4 addendum)

Alpaca Algo Trader Plus provides consolidated SIP trades + top-of-book quotes but
**no Level-2 depth**. `cengine/microstructure.py` derives L1 microstructure features
from genuine SIP observations, **reusing** `order_flow_engine.SymbolOrderFlow` and
adding only what it lacks.

## Features (18)

Reused from `order_flow_engine` (quote imbalance, OFI, microprice tilt, trade
classification, CVD, relative spread, liquidity, seller/buyer exhaustion, absorption/
replenishment) **plus** new: microprice displacement (bps), **trade arrival
intensity**, aggressive imbalance, **trade-size distribution** (mean/p95/large
fraction), **Kyle's lambda** (price impact per unit signed $volume), **Amihud
illiquidity** (|ret|/$volume), **adverse selection** (post-trade mid drift in the
aggressor's favor), liquidity resilience, and a deterministic execution-cost input
(via the `TransactionCostEstimator`). A `fill_probability_features()` contract is
prepared for a future model (no model is fit here).

## Observed vs inferred (data integrity)

`MicrostructureFeatures` separates **observed** (bid/ask/sizes/spread/last trade) from
**inferred** (OFI, CVD, aggressor side, Kyle, Amihud, exhaustion, adverse selection)
and **quality** (`is_stale`, `quote_age_s`, `trade_age_s`, `crossed_or_locked`,
`n_trades_window`, `confidence`, `quality`). It **never fabricates** L2 depth, hidden
orders, queue position, liquidity walls, participant identity, cancellations, or an
aggressor side that cannot be determined — **unknown stays `None`**. Crossed/locked,
stale, and insufficient-data conditions are flagged and lower `confidence`; nothing
raises. Kyle's lambda / Amihud use **trade-to-trade** mid changes (not the just-
updated quote mid).

## Architecture & subscriptions

C++-style incremental rolling state (deques) per symbol; computations are tiny (a few
hundred candidate symbols) so they run on CPU — a GPU batch is not justified at this
scale (measured, consistent with §20). The engine consumes the **same** SIP quote/
trade tap (`trade.py._publish_market_event` → `MicrostructureEngine.on_quote_event/
on_trade_event`); **no new provider connection**. Compute is applied primarily to
actionable candidates / held / pending symbols. The independent protective exit keeps
its own monitoring.

## Integration

Features feed the ExecutionPlanner (policy + limit price), TransactionCostEstimator,
adaptive position management, and TCA; they are available to regime/strategy/ranking/
optimizer/replacement as advisory inputs. Microstructure signals **never override**
strict entry qualification, strategy maturity, exposure authorization, or native
financial risk controls, and never the mandatory −0.5% exit.

## Future Level 2 (design only)

A future depth-feed interface can populate the existing (unbound, unfed)
`MarketDepthBook` without fabricating depth from L1. No additional subscription is
purchased or required. See
[microstructure_feature_contract.md](microstructure_feature_contract.md),
`reports/microstructure_validation.json`, `reports/microstructure_gpu_benchmarks.json`.
