# Portfolio Optimization (Milestone 3, §7/§8/§10/§11)

`cengine/portfolio_optimizer.py` (`PortfolioOptimizer`) turns a candidate set +
covariance + portfolio state into constrained, sized, versioned
`PortfolioRecommendation` objects.

## Policy selection

- **Mean-variance** — reuses the CVXPY `RegimeMarkowitzAllocator`
  (`cengine/portfolio_optimization.py`) **only** when reliable expected returns are
  supplied AND CVXPY is importable (installed: 1.9.3). Objective (§7):
  `maximize wᵀμ − λ·wᵀΣw` with a regime tilt, gross + per-asset limits.
  `return_basis = "calibrated_mean_variance"`.
- **Risk-based (default fallback)** — the native C++ deterministic allocator
  `_cengine_native.allocate_risk_based` (`risk/portfolio_allocator.cpp`):
  inverse-volatility weights projected onto the long-only / per-asset / gross
  constraint set, with portfolio variance + risk contributions. Used when reliable
  expected returns are unavailable (§7). `return_basis = "deterministic_risk_based"`.
- A pure-numpy inverse-vol path backs up the native allocator if `_cengine_native`
  is absent.

Native parity vs numpy: portfolio variance ≈1e-19, inverse-vol exact, risk
contributions sum to volatility (≈1e-18).

## No fabricated returns

Heuristic strategy scores are **never** converted into expected returns. When no
calibrated μ is available, allocation is explicitly risk-based and
`expected_incremental_net_return` stays `None`. Transaction costs are counted once.
Cash is a valid allocation (the long-only budget residual).

## PortfolioRecommendation contract (§11)

`portfolio_decision_id`, `source_strategy_decision_id`, `symbol_id`,
`instrument_generation`, `portfolio_version`, `strategy_id`, `strategy_family`,
`regime_version`, `feature_version`, `optimizer_version`, `configuration_version`,
`observation_timestamp`, `decision_timestamp`, `expiration_timestamp`,
`current_weight`, `target_weight`, `requested_quantity`, `requested_notional_ticks`,
`capital_required_ticks`, `expected_incremental_net_return` (None if unknown),
`expected_risk_contribution`, `estimated_transaction_cost`,
`effective_exposure_delta_ticks`, `constraint_status`, `binding_constraints`,
`allocation_policy`, `return_basis`, `eligibility`, `rejection_reason`.

## Sizing (§10)

`requested_quantity = target_weight · equity / price` (integer shares; fractional
only when `allow_fractional`). Price/quantity/capital are revalidated by the native
RiskEngine immediately before authorization — the optimizer only proposes.

## Hard constraints (§8) — never silently relaxed

`cengine/portfolio_constraints.py` (`enforce`) enforces: max position weight, max
family weight, max sector weight, max gross/net exposure, max effective leverage,
max correlated-cluster exposure, max portfolio volatility, max drawdown (circuit
breaker), max turnover, instrument authorization, min trade notional, min liquidity.
Scalable breaches (gross/net/leverage/vol/correlation/group) are scaled **down**
proportionally and recorded as binding; non-scalable violations (unauthorized
instrument, drawdown breach, excessive turnover) reject with weights cleared.
Infeasible optimization produces a safe scale-down or an explicit rejection — no
objective overrides a hard constraint.

Evidence: `reports/portfolio_optimizer_validation.json`,
`reports/portfolio_constraint_validation.json`.
