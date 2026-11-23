# GPU-Accelerated Portfolio Intelligence (Milestone 3)

This is the overview of the M3 portfolio layer: a GPU-assisted, exposure-aware,
risk-constrained capital allocation path that routes through the **existing**
execution authority. It reuses existing components wherever possible and creates
no parallel execution engine.

## The path

```
strategy recommendations (M2)  +  observed returns  +  broker-reconciled state
  → PortfolioAnalytics          (GPU covariance / EWMA; cengine/portfolio_analytics.py)
  → PortfolioOptimizer          (CVXPY mean-variance OR native risk-based; cengine/portfolio_optimizer.py)
  → PortfolioConstraints        (hard constraints; cengine/portfolio_constraints.py)
  → CapitalReservationLedger    (atomic; cengine/capital_reservation.py)
  → CandidateRouter             (EXISTING — exposure gate)
  → native C++ RiskEngine       (EXISTING — authoritative)
  → OrderManager                (EXISTING)
  → broker / mock broker        → fill reconciliation
  → DecisionEventSink           (portfolio decision evidence)
```

Composition lives in `cengine/portfolio_pipeline.py` (`PortfolioPipeline`); the
authoritative, versioned `PortfolioState` is assembled by
`cengine/portfolio_state.py`.

## Default-advisory

The pipeline is wired into `trade.py` behind `CENGINE_PORTFOLIO_EXECUTION_AUTH`
(default `none`). In the default state it emits portfolio state + recommendations
+ decision evidence but **routes nothing**. Routing additionally requires
sufficient observed return history to accrue (no fabricated returns) and an
explicit authorization (`mock` / `paper` / `live`). Mock authorization is isolated
and confers no paper/live eligibility (shared with the M2 maturity model).

## Reuse, not duplicate

M3 reuses: the CVXPY `RegimeMarkowitzAllocator`, the Metal covariance kernels, the
`RiskReservationBook`, `PortfolioMetricsCollector`, the exposure registry's
effective-exposure math, `market_scan/replacement.py`, the M2
`StrategyExecutionBridge` audit-evidence handler, and the `CandidateRouter` /
native `RiskEngine` / `OrderManager` execution authority. The only new native code
is `risk/portfolio_allocator.cpp` (deterministic risk-based allocation + portfolio
risk math), which complements — does not duplicate — the CVXPY optimizer.

## Preserved safety

The independent **−0.5% mandatory protective exit** (`protective_exit.py`) is
untouched: it runs on its own 5-second task, is never arbitrated, and fires
regardless of optimizer / covariance / GPU availability. The optimizer cannot
defer, suppress, or veto it, and no replacement is required before an exit. See
`reports/protective_exit_validation.json`.

## Scope honesty

The portfolio path is **mock-execution certified** (`reports/portfolio_mock_execution_validation.json`)
through the real native RiskEngine + an isolated mock broker. It is **NOT**
economically validated, paper-certified, or live-certified — those are separate,
gated milestones. The economic comparison (`reports/portfolio_economic_comparison.json`)
uses explicitly-labeled synthetic data; genuine historical / walk-forward / OOS /
SIP-vs-IEX validation is deferred (data/session-gated).

## Component docs

- [portfolio_optimization.md](portfolio_optimization.md) — optimizer, objective, recommendation contract, constraints
- [capital_reservation.md](capital_reservation.md) — capital reservation + portfolio state
- [portfolio_replacement.md](portfolio_replacement.md) — replacement workflow
- [gpu_portfolio_analytics.md](gpu_portfolio_analytics.md) — GPU covariance consumption + crossover
