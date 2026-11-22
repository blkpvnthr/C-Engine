# Metal Kernel Integration Inventory

Device: **Apple M5 Max**  ·  Library: `/Users/blkpvnthr/sustenance/build/cengine.metallib`

- Kernel entry points discovered: **45** across 5 shaders
- Compiled + pipeline-callable on device: **45/45** (100.0%)
- Host-interfaced + hardware-executed + CPU-parity-validated: **7**
- Consumer-connected (declared): **21**
- Blocked (not yet consumer-wired): **38**

| Shader | Kernel | Compiled | Host interface | Consumer(s) | HW-exec | Parity | Status |
|---|---|:--:|---|---|:--:|:--:|---|
| indicators.metal | `rolling_sma_ticks` | ✓ | MetalIndicators.sma | technical_features, strategies, scanner | ✓ | ✓ | VALIDATED |
| indicators.metal | `exponential_moving_average_ticks` | ✓ | MetalIndicators.ema | technical_features, strategies | ✓ | ✓ | VALIDATED |
| indicators.metal | `average_price_oscillator_ticks` | ✓ | MetalIndicators.apo | technical_features, apo_strategy | ✓ | ✓ | VALIDATED |
| indicators.metal | `true_range_ticks` | ✓ | MetalIndicators.true_range | technical_features, atr, di_bb | ✓ | ✓ | VALIDATED |
| covariance.metal | `column_means` | ✓ | MetalCovariance.covariance_matrix | portfolio_optimizer | ✓ | ✓ | VALIDATED |
| covariance.metal | `covariance_matrix` | ✓ | MetalCovariance.covariance_matrix | portfolio_optimizer, risk_manager | ✓ | ✓ | VALIDATED |
| covariance.metal | `ewma_covariance_matrix` | ✓ | MetalCovariance.ewma_covariance_matrix | portfolio_optimizer, regime | ✓ | ✓ | VALIDATED |
| covariance.metal | `statarb_pair_regression` | ✓ | — | — | — | — | BLOCKED — structured StatArb ABI; research-only, not yet wired |
| covariance.metal | `statarb_residual_series` | ✓ | — | — | — | — | BLOCKED — structured StatArb ABI; research-only, not yet wired |
| covariance.metal | `volatility_factor_features` | ✓ | — | — | — | — | BLOCKED — structured factor ABI; not yet wired |
| covariance.metal | `statarb_normalized_weights` | ✓ | — | — | — | — | BLOCKED — long-only engine; statarb weights research-only, not wired |
| monte-carlo.metal | `generate_standard_normals` | ✓ | — | research, risk_scenarios | — | — | BLOCKED — MonteCarloParams ulong-aligned ABI + stochastic validation; wiring pending |
| monte-carlo.metal | `correlate_normals` | ✓ | — | — | — | — | BLOCKED — structured Cholesky ABI; research pipeline not yet wired |
| monte-carlo.metal | `generate_correlated_returns` | ✓ | — | — | — | — | BLOCKED — depends on correlate_normals pipeline; not yet wired |
| monte-carlo.metal | `portfolio_scenario_pnl` | ✓ | — | portfolio_risk | — | — | BLOCKED — multi-stage MC pipeline; not yet wired |
| monte-carlo.metal | `statarb_pair_monte_carlo` | ✓ | — | — | — | — | BLOCKED — research-only; not yet wired |
| monte-carlo.metal | `simulate_portfolio_paths` | ✓ | — | research | — | — | BLOCKED — multi-stage MC pipeline; not yet wired |
| monte-carlo.metal | `pnl_to_loss` | ✓ | — | — | — | — | BLOCKED — MC aggregation stage; not yet wired |
| monte-carlo.metal | `extract_scenario_pnl` | ✓ | — | — | — | — | BLOCKED — MC aggregation stage; not yet wired |
| monte-carlo.metal | `extract_pair_pnl` | ✓ | — | — | — | — | BLOCKED — MC aggregation stage; not yet wired |
| monte-carlo.metal | `tail_moment_partials` | ✓ | — | — | — | — | BLOCKED — feeds reduce_* combine stage; not yet wired |
| monte-carlo.metal | `threshold_tail_partials` | ✓ | — | risk_scenarios | — | — | BLOCKED — feeds reduce_threshold_tail; not yet wired |
| monte-carlo.metal | `apply_stress_shocks` | ✓ | — | stress_testing | — | — | BLOCKED — structured stress ABI; not yet wired |
| imbalance.metal | `validate_depth_snapshots` | ✓ | — | — | — | — | BLOCKED — requires genuine L2 depth; IEX provides L1 only |
| imbalance.metal | `compute_level_imbalance` | ✓ | — | — | — | — | BLOCKED — requires genuine L2 depth; IEX provides L1 only |
| imbalance.metal | `compute_imbalance_features` | ✓ | — | — | — | — | BLOCKED — requires genuine L2 depth; IEX provides L1 only |
| imbalance.metal | `compute_top_level_imbalance` | ✓ | — | order_flow, scanner | — | — | BLOCKED — L1-compatible; structured depth ABI wiring pending |
| imbalance.metal | `compute_weighted_depth_imbalance` | ✓ | — | — | — | — | BLOCKED — requires genuine L2 depth; IEX provides L1 only |
| imbalance.metal | `compute_depth_shape` | ✓ | — | — | — | — | BLOCKED — requires genuine L2 depth; IEX provides L1 only |
| imbalance.metal | `compute_pair_pressure_features` | ✓ | — | — | — | — | BLOCKED — requires genuine L2 depth; IEX provides L1 only |
| reductions.metal | `reduce_sum_partials` | ✓ | — | aggregation, portfolio_stats | — | — | BLOCKED — two-stage partial/combine pipeline wiring pending |
| reductions.metal | `reduce_moment_partials` | ✓ | — | statistics, volatility | — | — | BLOCKED — two-stage partial/combine pipeline wiring pending |
| reductions.metal | `reduce_minmax_partials` | ✓ | — | statistics | — | — | BLOCKED — two-stage partial/combine pipeline wiring pending |
| reductions.metal | `reduce_dot_partials` | ✓ | — | — | — | — | BLOCKED — combine stage wiring pending |
| reductions.metal | `reduce_covariance_sufficient_partials` | ✓ | — | portfolio_optimizer | — | — | BLOCKED — pairs with combine_covariance_partials; pending |
| reductions.metal | `reduce_matrix_column_partials` | ✓ | — | — | — | — | BLOCKED — combine stage wiring pending |
| reductions.metal | `reduce_pnl_partials` | ✓ | — | portfolio_stats | — | — | BLOCKED — pairs with combine_pnl_partials; pending |
| reductions.metal | `reduce_threshold_tail_partials` | ✓ | — | risk_scenarios | — | — | BLOCKED — pairs with combine_threshold_tail_partials; pending |
| reductions.metal | `reduce_validation_counts` | ✓ | — | — | — | — | BLOCKED — combine stage wiring pending |
| reductions.metal | `reduce_histogram_partials` | ✓ | — | — | — | — | BLOCKED — combine stage wiring pending |
| reductions.metal | `combine_sum_partials` | ✓ | — | aggregation | — | — | BLOCKED — second stage of reduce_sum; pending full pipeline |
| reductions.metal | `combine_covariance_partials` | ✓ | — | portfolio_optimizer | — | — | BLOCKED — second stage; pending full pipeline |
| reductions.metal | `combine_pnl_partials` | ✓ | — | — | — | — | BLOCKED — second stage; pending full pipeline |
| reductions.metal | `combine_threshold_tail_partials` | ✓ | — | — | — | — | BLOCKED — second stage; pending full pipeline |
| reductions.metal | `normalize_zscores` | ✓ | — | — | — | — | BLOCKED — combine stage wiring pending |
