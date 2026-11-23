# Dataset & View Catalog

All datasets are partitioned Parquet (`feed=/trade_date=`) and exposed as DuckDB views
(`warehouse.py`). Monetary fields are in `DEFAULT_PRICE_SCALE = 10000` ticks.

## Bronze (immutable raw capture)

Decision/execution events route by type (`bronze.py:_EVENT_DATASET`):

| Dataset | Source event types |
|---------|--------------------|
| `engine_decisions` | `regime_transition`, `strategy_recommendation`, `rejected_opportunity`, `ranking_components`, `arbitration_decision` |
| `risk_decisions` | `risk_approval`, `risk_rejection` |
| `portfolio_decisions` | `portfolio_state/optimization/recommendation/constraint_binding`, `capital_reservation`, `replacement_decision` |
| `execution_plans` | `execution_plan`, `execution_plan_rejected`, `microstructure_snapshot`, `transaction_cost_analysis` |
| `order_events` | `order_intent`, `broker_ack` |
| `fill_events` | `partial_fill`, `complete_fill` |
| `position_events` | `exit_decision`, `position_management`, `profit_protection`, `realized_outcome` |

Raw market/reference datasets (`BRONZE_DATASETS`): `market_bars`, `market_trades`,
`market_quotes`, `market_corrections`, `instrument_metadata`, `universe_membership`,
`subscription_events`, `market_data_quality_events`, `reconciliation_events`,
`system_health_events`.

## Silver (validated, typed, deduplicated)

Each silver dataset is a typed projection of a bronze dataset
(`silver.py:_TRANSFORMS`), deduplicated by `event_id`, incremental by availability
high-water:

| Silver dataset | From bronze | Key columns |
|----------------|-------------|-------------|
| `strategy_recommendations` | engine_decisions | decision_id, strategy_family, eligibility, confidence, rank_score, expected_net_return |
| `strategy_rejections` | engine_decisions | decision_id, rejection_reason |
| `market_regimes` | engine_decisions | regime_version, level, direction, trend, volatility, liquidity |
| `portfolio_decisions` | portfolio_decisions | portfolio_decision_id, source_strategy_decision_id, target_weight, allocation_policy, constraint_status |
| `execution_plans` | execution_plans | execution_plan_id, side, execution_policy, limit_price_ticks, spread_bps |
| `microstructure_features` | execution_plans | spread_bps, quote_imbalance, order_flow_imbalance, kyle_lambda, amihud_illiquidity, adverse_selection_bps |
| `transaction_costs` | execution_plans | implementation_shortfall_bps, realized_slippage_bps, average_fill_price_ticks |
| `orders` | order_events | client_order_id, venue_order_id, side, quantity, status |
| `fills` | fill_events | venue_order_id, last_fill_quantity, last_fill_price_ticks, average_fill_price_ticks |

## Gold (learning datasets)

| Gold dataset | Built by | Contents |
|--------------|----------|----------|
| `position_outcomes` | `build_position_outcomes` | FIFO-matched closed positions: entry/exit price, holding duration, gross/net return |
| `open_positions` | `build_position_outcomes` | still-open quantity, `OPEN_NO_REALIZED_OUTCOME` (never labeled closed) |
| `supervised` | `build_supervised` | recommendation features → realized label or censored flag |
| `offline_rl` | `build_offline_rl` | logged transitions: action, reward (closed only), terminal, NULL action probability |
| `daily_performance` | `attribution.py` | daily attribution record + loss-source taxonomy |
| `data_quality_results` | `pipeline.py` | persisted quality profile |

## DuckDB warehouse views (`warehouse.py:_VIEWS`)

`register_views()` creates one view per materialized dataset (missing datasets are
skipped). Highlights: `strategy_recommendations`, `portfolio_decisions`, `fills`,
`position_outcomes`, `supervised_training_datasets`, `offline_rl_transitions`,
`market_bars`, `daily_performance`. Views are `read_parquet(glob, union_by_name=true)` —
DuckDB stores no bulk copy of the lakehouse.

Example lineage query (strategy → portfolio):
```sql
SELECT r.decision_id, p.portfolio_decision_id
FROM strategy_recommendations r
JOIN portfolio_decisions p ON p.source_strategy_decision_id = r.decision_id;
```
