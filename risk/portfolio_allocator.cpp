// ============================================================================
// portfolio_allocator.cpp — native deterministic portfolio allocation (§3/§7/§25)
// ============================================================================

#include "risk/portfolio_allocator.hpp"

#include <cmath>
#include <limits>

namespace trading::portfolio {

namespace {

[[nodiscard]] bool finite(double v) { return std::isfinite(v); }

[[nodiscard]] bool shapes_ok(
    const std::vector<double>& weights,
    const std::vector<float>& covariance,
    std::uint32_t assets) {
    if (assets == 0) {
        return false;
    }
    if (weights.size() != assets) {
        return false;
    }
    return covariance.size() == static_cast<std::size_t>(assets) * assets;
}

}  // namespace

std::vector<double> marginal_variance(
    const std::vector<double>& weights,
    const std::vector<float>& covariance,
    std::uint32_t assets) {
    if (!shapes_ok(weights, covariance, assets)) {
        return {};
    }
    std::vector<double> out(assets, 0.0);
    for (std::uint32_t i = 0; i < assets; ++i) {
        double acc = 0.0;
        for (std::uint32_t j = 0; j < assets; ++j) {
            const double cov =
                static_cast<double>(covariance[static_cast<std::size_t>(i) * assets + j]);
            acc += cov * weights[j];
        }
        out[i] = acc;
    }
    return out;
}

double portfolio_variance(
    const std::vector<double>& weights,
    const std::vector<float>& covariance,
    std::uint32_t assets) {
    if (!shapes_ok(weights, covariance, assets)) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    const std::vector<double> sigma_w = marginal_variance(weights, covariance, assets);
    double variance = 0.0;
    for (std::uint32_t i = 0; i < assets; ++i) {
        variance += weights[i] * sigma_w[i];
    }
    return variance;
}

std::vector<double> risk_contributions(
    const std::vector<double>& weights,
    const std::vector<float>& covariance,
    std::uint32_t assets) {
    if (!shapes_ok(weights, covariance, assets)) {
        return {};
    }
    const std::vector<double> sigma_w = marginal_variance(weights, covariance, assets);
    double variance = 0.0;
    for (std::uint32_t i = 0; i < assets; ++i) {
        variance += weights[i] * sigma_w[i];
    }
    std::vector<double> contributions(assets, 0.0);
    const double vol = std::sqrt(variance > 0.0 ? variance : 0.0);
    if (!(vol > 0.0)) {
        return contributions;  // zero vol -> zero contributions (never divide by 0)
    }
    for (std::uint32_t i = 0; i < assets; ++i) {
        contributions[i] = weights[i] * sigma_w[i] / vol;
    }
    return contributions;
}

std::vector<double> inverse_volatility_weights(
    const std::vector<double>& variances) {
    const std::size_t n = variances.size();
    std::vector<double> raw(n, 0.0);
    double total = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        const double var = variances[i];
        if (finite(var) && var > 0.0) {
            const double inv_vol = 1.0 / std::sqrt(var);
            raw[i] = inv_vol;
            total += inv_vol;
        }
    }
    if (!(total > 0.0)) {
        return std::vector<double>(n, 0.0);  // no valid variance -> all cash
    }
    for (std::size_t i = 0; i < n; ++i) {
        raw[i] /= total;
    }
    return raw;
}

AllocationResult project_long_only_capped(
    const std::vector<double>& raw_weights,
    double per_asset_cap,
    double gross_cap) {
    AllocationResult result;
    const std::size_t n = raw_weights.size();
    result.weights.assign(n, 0.0);
    result.risk_contributions.assign(n, 0.0);

    if (!(per_asset_cap > 0.0) || !(gross_cap > 0.0) || !finite(per_asset_cap) ||
        !finite(gross_cap)) {
        return result;  // invalid caps -> all cash, valid=false
    }

    // Long-only clamp + per-asset cap.
    std::vector<double> capped(n, 0.0);
    double gross = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        double w = raw_weights[i];
        if (!finite(w) || w <= 0.0) {
            w = 0.0;  // long-only
        }
        if (w > per_asset_cap) {
            w = per_asset_cap;
            result.per_asset_cap_binding = true;
        }
        capped[i] = w;
        gross += w;
    }

    // Gross budget: scale down proportionally if over the gross cap. Never scale
    // UP (excess budget stays in cash; we do not invent exposure).
    if (gross > gross_cap && gross > 0.0) {
        const double scale = gross_cap / gross;
        for (std::size_t i = 0; i < n; ++i) {
            capped[i] *= scale;
        }
        gross = gross_cap;
        result.gross_cap_binding = true;
    }

    result.weights = capped;
    result.gross_weight = gross;
    result.cash_weight = 1.0 - gross;
    result.valid = true;
    return result;
}

AllocationResult allocate_risk_based(
    const std::vector<double>& variances,
    const std::vector<float>& covariance,
    std::uint32_t assets,
    double per_asset_cap,
    double gross_cap) {
    if (variances.size() != assets || assets == 0) {
        AllocationResult bad;
        bad.weights.assign(assets, 0.0);
        bad.risk_contributions.assign(assets, 0.0);
        bad.cash_weight = 1.0;
        return bad;
    }
    const std::vector<double> raw = inverse_volatility_weights(variances);
    AllocationResult result = project_long_only_capped(raw, per_asset_cap, gross_cap);
    if (!result.valid) {
        return result;
    }
    if (covariance.size() == static_cast<std::size_t>(assets) * assets) {
        result.portfolio_variance = portfolio_variance(result.weights, covariance, assets);
        result.portfolio_volatility =
            std::sqrt(result.portfolio_variance > 0.0 ? result.portfolio_variance : 0.0);
        result.risk_contributions = risk_contributions(result.weights, covariance, assets);
    }
    return result;
}

}  // namespace trading::portfolio
