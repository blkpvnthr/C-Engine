# Intelligent Execution Engine (Milestone 4)

Converts authorized strategy/portfolio recommendations into cost-aware, liquidity-
aware, risk-controlled broker orders — **reusing** the existing financial authority
(`OrderManager`, native C++ `RiskEngine`, `CandidateRouter`, `AlpacaExecutionVenue`),
never a parallel execution path.

## Lifecycle

```
StrategyRecommendation → PortfolioRecommendation → CapitalReservation
  → ExecutionPlan → RiskAuthorization → OrderIntent → BrokerSubmission
  → BrokerAcknowledgement → PartialFill → CompleteFill → BrokerReconciliation
  → ExecutionOutcome
```

Every transition carries a stable decision id (`execution_plan_id`,
`portfolio_decision_id`, `strategy_decision_id`, `correlation_id`). Only the existing
authorized path submits orders.

## Components (all reuse-first)

| Concern | Module | Notes |
|---|---|---|
| L1 microstructure | `cengine/microstructure.py` | reuses `order_flow_engine`; see [level1_microstructure_intelligence.md](level1_microstructure_intelligence.md) |
| Expected/realized cost | `cengine/transaction_cost.py` | see [transaction_cost_analysis.md](transaction_cost_analysis.md) |
| Plan + policy + limit price | `cengine/execution_planner.py` | see [execution_policy.md](execution_policy.md) |
| Position mgmt + exit arbitration | `cengine/position_management.py` | see [position_management.md](position_management.md) |
| Order lifecycle / idempotency | `order_manager.py` | see [order_lifecycle.md](order_lifecycle.md) |
| GPU scenario risk | `cengine/portfolio_analytics.py` | see [gpu_portfolio_analytics.md](gpu_portfolio_analytics.md) |

Integration: the portfolio pipeline (`cengine/portfolio_pipeline.py`) builds an
`ExecutionPlan` per routed recommendation (fresh-quote limit price + cost estimate +
evidence) via a `PlannedExecutionPolicy` the existing router consumes.

## Default non-live

Execution routing is gated by `CENGINE_PORTFOLIO_EXECUTION_AUTH` (default `none` →
advisory: emits plans + evidence, routes nothing). New execution policies default to
a non-live maturity until separately validated. Mock, paper, and live authorization
boundaries are preserved.

## Mandatory protective exit — preserved

The independent **−0.5%** protective exit (`protective_exit.py`) keeps its own
broker-compatible authority and 5-second backstop. No discretionary execution
policy, optimizer, microstructure signal, or GPU workload can defer, suppress, or
veto it. Exit arbitration (§14) always ranks it above discretionary decisions.

## Tick-scale unification

M4 is the first milestone to populate `limit_price_ticks`. It exposed a pre-existing
mismatch: market data + native risk used `DEFAULT_PRICE_SCALE = 10_000`
($0.0001/tick) while `AlpacaExecutionVenue` and `_fmt_usd` assumed cents (×100). Both
now use `DEFAULT_PRICE_SCALE`, unifying the whole system so computed limit prices,
native-risk collars, and dashboards agree.

## Scope honesty

Implementation and **mock-execution** are certified (full path through the real
native `RiskEngine` + an isolated mock broker). This is **not** economic validation,
paper certification, or live certification — those remain separate, gated milestones.
No live orders during implementation or validation.
