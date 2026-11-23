# Execution Policies & Planner (Milestone 4, §4/§5/§8/§9)

`cengine/execution_planner.py` turns an authorized recommendation + fresh L1
microstructure into a versioned `ExecutionPlan` with a broker-compatible policy and
a tick-rounded limit price, then hands it to the **existing** `CandidateRouter` via a
`PlannedExecutionPolicy`.

## Policies (§8)

| Policy | When | Order |
|---|---|---|
| `PASSIVE_LIMIT` | low urgency, spread ≥ 8 bps | join the bid/ask (rest) |
| `MARKETABLE_LIMIT` | moderate urgency / momentum / expiring | cross at ask/bid, limit-capped |
| `IMMEDIATE_EXECUTION` | stale quote or no computable limit | MARKET |
| `BOUNDED_REPRICE` | wide spread + patience | passive, N repricing attempts |
| `TIME_SLICED_EXECUTION` | order large vs observed top-of-book | sliced |
| `URGENT_RISK_REDUCTION` | urgent risk reduction | MARKET |
| `MANDATORY_PROTECTIVE_EXIT` | the −0.5% exit | **independent** authority (not produced here) |

`select_policy(...)` inputs: urgency, spread, expected cost, opportunity expiration,
momentum, order size vs liquidity, staleness. No policy unsupported by the broker is
selected; a limit policy with no fresh quote downgrades to `IMMEDIATE_EXECUTION`
rather than fabricating a price.

## Limit-price selection (§9)

`compute_limit_price_ticks(...)` uses fresh observed bid/ask, rounds to the broker
tick size (penny = 100 ticks at `DEFAULT_PRICE_SCALE` 10_000 for ≥ $1 equities), and
bounds elective slippage: a BUY limit never exceeds `decision_price × (1 +
max_slippage_bps/1e4)`; a SELL limit never falls below the symmetric floor. This
prevents uncontrolled chasing of a deteriorating opportunity. Returns `None` (→
market) for immediate/urgent/mandatory policies. Prices are never invented without a
fresh quote.

## ExecutionPlan contract (§5)

Versioned dataclass with explicit unknowns (`None`): `execution_plan_id`,
`strategy_decision_id`, `portfolio_decision_id`, `symbol_id`,
`instrument_generation`, `side`, `requested_quantity`, `requested_notional`,
`reference_price`, `decision_price`, `arrival_price`, `observed_bid`, `observed_ask`,
`spread_bps`, `estimated_transaction_cost_bps`, `estimated_slippage_bps`, `urgency`,
`execution_policy`, `limit_price_ticks`, `expiration_timestamp`,
`maximum_repricing_attempts`, `capital_reservation_id`, `portfolio_version`,
`market_observation_version`, `execution_authorization`, `cost_components`,
`rejection_reason`.

## Router adapter

`PlannedExecutionPolicy` implements the existing `ExecutionPolicy` protocol
(`instruction_for(candidate) → ExecutionInstruction`). The pipeline registers each
plan; the router builds the `OrderIntent` (LIMIT with the plan's price, or MARKET),
then the native RiskEngine + OrderManager remain authoritative. Mandatory exits are
never produced by the planner — they keep their independent broker-compatible path.
