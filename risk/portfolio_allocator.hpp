#pragma once

// ============================================================================
// portfolio_allocator.hpp
// ============================================================================
//
// Native deterministic portfolio allocation + risk math (Milestone 3, §3/§7/§25).
//
// This is NOT a duplicate of the CVXPY regime mean-variance optimizer
// (cengine/portfolio_optimization.py). It provides the native, deterministic
// RISK-BASED allocation policy the directive mandates when reliable expected
// returns are unavailable (§7), plus the portfolio-risk primitives consumed by
// the constraint layer and dashboard: portfolio variance/volatility, marginal
// and component risk contributions, inverse-volatility weights, and long-only
// capped-weight projection with a gross budget.
//
// It contains NO order placement, NO broker connectivity, and NO risk approval.
// The native RiskEngine + OrderManager remain the execution authority; these are
// advisory allocation inputs only.
//
// Covariance is passed in row-major, assets*assets (produced by the GPU/Metal
// covariance kernel or its CPU oracle). All math is double precision for
// deterministic CPU parity; covariance floats are upcast on entry.
// ============================================================================

#include <cstddef>
#include <cstdint>
#include <vector>

namespace trading::portfolio {

// Result of a deterministic risk-based allocation.
struct AllocationResult {
    std::vector<double> weights;              // per-asset target weights (long-only)
    std::vector<double> risk_contributions;   // component risk contribution (sums to portfolio vol)
    double portfolio_variance = 0.0;          // wᵀΣw
    double portfolio_volatility = 0.0;        // sqrt(max(0, wᵀΣw))
    double gross_weight = 0.0;                // Σ|w|
    double cash_weight = 0.0;                 // 1 - gross (long-only budget residual)
    bool per_asset_cap_binding = false;
    bool gross_cap_binding = false;
    bool valid = false;                       // false if inputs were invalid
};

// wᵀΣw. covariance is assets*assets row-major. Returns NaN on shape mismatch.
[[nodiscard]] double portfolio_variance(
    const std::vector<double>& weights,
    const std::vector<float>& covariance,
    std::uint32_t assets);

// Σw (the marginal variance contributions). Returns empty on shape mismatch.
[[nodiscard]] std::vector<double> marginal_variance(
    const std::vector<double>& weights,
    const std::vector<float>& covariance,
    std::uint32_t assets);

// Component risk contributions c_i = w_i * (Σw)_i / portfolio_vol; Σ c_i = vol.
// If portfolio_vol == 0 all contributions are 0.
[[nodiscard]] std::vector<double> risk_contributions(
    const std::vector<double>& weights,
    const std::vector<float>& covariance,
    std::uint32_t assets);

// Inverse-volatility raw weights from per-asset variances: w_i ∝ 1/sqrt(var_i),
// normalized to sum to 1 over assets with positive, finite variance. Assets with
// non-positive or non-finite variance receive weight 0.
[[nodiscard]] std::vector<double> inverse_volatility_weights(
    const std::vector<double>& variances);

// Project raw (non-negative) weights to a long-only, per-asset-capped allocation
// with a gross budget. Negative inputs are clamped to 0 (long-only). Each weight
// is capped at per_asset_cap; the vector is scaled so Σw ≤ gross_cap. Never
// relaxes a cap: excess beyond gross_cap stays in cash. Sets the binding flags.
[[nodiscard]] AllocationResult project_long_only_capped(
    const std::vector<double>& raw_weights,
    double per_asset_cap,
    double gross_cap);

// Deterministic risk-based allocation: inverse-volatility weights projected onto
// the long-only/per-asset/gross constraint set, with portfolio variance and risk
// contributions computed against the supplied covariance. This is the fallback
// allocation policy (§7) and the native "valid constrained allocation" (§25).
[[nodiscard]] AllocationResult allocate_risk_based(
    const std::vector<double>& variances,
    const std::vector<float>& covariance,
    std::uint32_t assets,
    double per_asset_cap,
    double gross_cap);

}  // namespace trading::portfolio
