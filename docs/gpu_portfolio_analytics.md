# GPU Portfolio Analytics (Milestone 3, §5/§20)

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

## Deferred (honest)

The Monte-Carlo and reductions Metal kernels remain **compiled-but-unbound** (no
pybind binding; the generic `MetalRuntime.dispatch()` is C++-only). GPU scenario /
tail-risk consumption and a parallel-reduction covariance kernel (which could move
the crossover down) are tracked future increments — not claimed as consumed.

Evidence: `reports/gpu_portfolio_benchmarks.json`,
`reports/portfolio_architecture_validation.json`.
