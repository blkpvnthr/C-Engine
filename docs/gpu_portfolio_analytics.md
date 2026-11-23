# GPU Portfolio Analytics (Milestone 3 §5/§20, extended in Milestone 4 §19)

`cengine/portfolio_analytics.py` (`PortfolioAnalytics`) computes the portfolio risk
inputs, operationally **consuming** the existing Metal covariance kernels.

## Operationally-consumed covariance kernels

Via the native `_cengine_native.MetalRuntime`:

- `column_means` + `covariance_matrix` — sample covariance (divisor obs−1)
- `ewma_covariance_matrix` — EWMA covariance (λ weighting, diagonal variance floor)

These were previously compiled + host-callable but **not** operationally consumed.
M3 wires them into the live portfolio path (`PortfolioAnalytics → PortfolioOptimizer
→ PortfolioPipeline`, into `trade.py`'s portfolio tick), and `tools/metal_evidence.py`
now marks them `operationally_consumed` (8 consumed total: 5 discovery + 3
covariance). Parity vs numpy is **<7e-11** on-device (Apple M5 Max). A missing/failed
runtime transparently falls back to numpy (`source == "cpu"`).

## Honest data handling

- **Timestamp-aligned returns, no fabricated zeros** (`build_return_matrix`):
  complete-case alignment keeps only timestamps shared by all retained symbols;
  symbols with insufficient coverage are excluded and reported in `excluded`.
  Unavailable observations are **never** replaced with zero returns (§5).
- **Regularization + PSD check**: symmetrize, optional diagonal variance floor,
  ridge; `eigvalsh` confirms positive-semidefinite / numerical validity.

## Derived analytics (CPU / native)

Correlation, portfolio volatility, risk contributions, and scenario losses are
cheap linear algebra over the small `assets × assets` matrix. Volatility and risk
contributions use the native allocator (`_cengine_native.portfolio_variance` /
`portfolio_risk_contributions`). Scenario losses are a **deterministic, seeded**
CPU/numpy Gaussian Monte-Carlo (VaR / ES), explicitly labeled `source = "cpu_numpy"`.

## Measured CPU/GPU crossover (§20)

`tools/portfolio_gpu_benchmark.py` → `reports/gpu_portfolio_benchmarks.json`
(Apple M5 Max, candidate sizes 1…4096). The covariance kernel is correctness-first
(one thread per matrix cell, no parallel reductions, for deterministic parity) and
is **not faster than numpy's BLAS** covariance at the tested sizes — so there is no
crossover in range, and covariance stays on CPU for typical candidate sets while the
GPU kernel remains available + parity-validated and genuinely consumed when enabled.
Covariance is sized to the **candidate set** (instruments that can enter the
portfolio), never the full discovery universe.

## Milestone 4 — Monte-Carlo integration (§19)

The Monte-Carlo scenario path is now **bound and consumed**. `MetalRuntime.
portfolio_scenario_returns(normals, cholesky, weights, scenarios, assets,
return_model)` chains `gme::correlate_normals` + `gme::portfolio_scenario_pnl`.
Normals are **host-seeded** so a CPU oracle reproduces results exactly
(`correlated = normals @ cholesky^T`; `portfolio_return = weights · correlated`);
on-device parity is **< 2e-7** (Apple M5 Max). `PortfolioAnalytics.scenario_losses`
consumes it (VaR/ES, `source = "metal_monte_carlo"`) with numpy fallback; these are
Gaussian **simulations under explicit assumptions, not calibrated forecasts**.
Operationally-consumed kernels are now **10** (8 + `correlate_normals` +
`portfolio_scenario_pnl`).

## Milestone 4 — covariance optimization conclusion (§19)

`tools/gpu_portfolio_m4.py` → `reports/gpu_covariance_optimization.json` benchmarks
the current kernel vs numpy/BLAS across candidate sizes. The correctness-first
one-thread-per-cell kernel is not faster than BLAS at realistic candidate-set sizes
(≤ a few hundred names), so covariance stays on CPU per the measured crossover. A
tiled/threadgroup-reduction covariance kernel is the documented optimization path,
**deferred with measured justification** (it would not change the operational decision
at realistic sizes); the existing kernel remains consumed + parity-validated.

## Reductions — compiled-but-unbound (measured justification, §19)

The two-stage reduction kernels remain compiled + pipeline-callable but **not bound**
(`reports/gpu_reduction_integration.json`): the portfolio/execution workloads operate
on small candidate arrays and trivial numpy-quantile tail risk, so binding them would
be an artificial/redundant operational call (explicitly disallowed by §19). Deferred
until a genuine large-N reduction workload exists.

Evidence: `reports/gpu_portfolio_benchmarks.json`,
`reports/gpu_covariance_optimization.json`, `reports/gpu_monte_carlo_validation.json`,
`reports/gpu_reduction_integration.json`,
`reports/portfolio_architecture_validation.json`.
