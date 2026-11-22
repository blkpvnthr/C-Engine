// ============================================================================
// cpu_reference.cpp — native CPU oracles (see cpu_reference.hpp).
// Mirrors cengine/metal_cpu_ref.py semantics exactly.
// ============================================================================
#include "opportunity_discovery/cpu_reference.hpp"

#include <cmath>
#include <cstdlib>

namespace trading::metal::cpu_ref {

std::int64_t trunc_div(std::int64_t a, std::int64_t b) noexcept {
    if (b == 0) return 0;
    // C++ integer division already truncates toward zero for signed operands,
    // matching Metal's `long /`. Implemented explicitly for clarity/parity.
    const std::int64_t q = std::llabs(a) / std::llabs(b);
    return ((a < 0) != (b < 0)) ? -q : q;
}

std::vector<std::int64_t> sma_ticks(const std::vector<std::int64_t>& prices,
                                    std::uint32_t window) {
    const std::size_t n = prices.size();
    std::vector<std::int64_t> out(n, 0);
    if (window == 0) return out;
    for (std::size_t i = 0; i < n; ++i) {
        if (i + 1 < window) continue;
        std::int64_t sum = 0;
        for (std::size_t j = i + 1 - window; j <= i; ++j) sum += prices[j];
        out[i] = trunc_div(sum, static_cast<std::int64_t>(window));
    }
    return out;
}

std::vector<std::int64_t> ema_ticks(const std::vector<std::int64_t>& prices,
                                    std::uint32_t alpha_numerator,
                                    std::uint32_t alpha_denominator) {
    const std::size_t n = prices.size();
    if (n == 0 || alpha_denominator == 0 ||
        alpha_numerator > alpha_denominator) {
        return std::vector<std::int64_t>(n, 0);
    }
    std::vector<std::int64_t> out(n, 0);
    out[0] = prices[0];
    const std::int64_t num = static_cast<std::int64_t>(alpha_numerator);
    const std::int64_t den = static_cast<std::int64_t>(alpha_denominator);
    for (std::size_t i = 1; i < n; ++i) {
        out[i] = trunc_div(num * prices[i] + (den - num) * out[i - 1], den);
    }
    return out;
}

std::vector<std::int64_t> apo_ticks(const std::vector<std::int64_t>& fast_ema,
                                    const std::vector<std::int64_t>& slow_ema) {
    const std::size_t n = fast_ema.size();
    std::vector<std::int64_t> out(n, 0);
    for (std::size_t i = 0; i < n && i < slow_ema.size(); ++i) {
        out[i] = fast_ema[i] - slow_ema[i];
    }
    return out;
}

std::vector<std::int64_t> true_range_ticks(
    const std::vector<std::int64_t>& highs,
    const std::vector<std::int64_t>& lows,
    const std::vector<std::int64_t>& previous_closes) {
    const std::size_t n = highs.size();
    std::vector<std::int64_t> out(n, 0);
    for (std::size_t i = 0; i < n; ++i) {
        const std::int64_t a = highs[i] - lows[i];
        const std::int64_t b = std::llabs(highs[i] - previous_closes[i]);
        const std::int64_t c = std::llabs(lows[i] - previous_closes[i]);
        out[i] = std::max(a, std::max(b, c));
    }
    return out;
}

std::vector<float> column_means(const std::vector<float>& values,
                                std::uint32_t observations,
                                std::uint32_t assets) {
    std::vector<float> means(assets, 0.0f);
    if (observations == 0 || assets == 0) return means;
    for (std::uint32_t a = 0; a < assets; ++a) {
        double acc = 0.0;
        for (std::uint32_t o = 0; o < observations; ++o) {
            acc += values[static_cast<std::size_t>(o) * assets + a];
        }
        means[a] = static_cast<float>(acc / observations);
    }
    return means;
}

std::vector<float> covariance_matrix(const std::vector<float>& values,
                                     std::uint32_t observations,
                                     std::uint32_t assets) {
    std::vector<float> cov(static_cast<std::size_t>(assets) * assets, 0.0f);
    if (observations < 2 || assets == 0) {
        for (auto& v : cov) v = std::nanf("");
        return cov;
    }
    const std::vector<float> means = column_means(values, observations, assets);
    for (std::uint32_t i = 0; i < assets; ++i) {
        for (std::uint32_t j = 0; j < assets; ++j) {
            double acc = 0.0;
            for (std::uint32_t o = 0; o < observations; ++o) {
                const double di =
                    values[static_cast<std::size_t>(o) * assets + i] - means[i];
                const double dj =
                    values[static_cast<std::size_t>(o) * assets + j] - means[j];
                acc += di * dj;
            }
            cov[static_cast<std::size_t>(i) * assets + j] =
                static_cast<float>(acc / (observations - 1));
        }
    }
    return cov;
}

std::vector<float> ewma_covariance_matrix(const std::vector<float>& values,
                                          std::uint32_t observations,
                                          std::uint32_t assets, float lambda) {
    std::vector<float> cov(static_cast<std::size_t>(assets) * assets, 0.0f);
    if (observations == 0 || assets == 0 || !(lambda > 0.0f && lambda <= 1.0f)) {
        for (auto& v : cov) v = std::nanf("");
        return cov;
    }
    // Weighted means (newest weight 1, older lambda, lambda^2, ...).
    std::vector<double> wmean(assets, 0.0);
    double wsum = 0.0;
    double w = 1.0;
    for (std::uint32_t reverse = 0; reverse < observations; ++reverse) {
        const std::uint32_t o = observations - 1 - reverse;
        for (std::uint32_t a = 0; a < assets; ++a) {
            wmean[a] += w * values[static_cast<std::size_t>(o) * assets + a];
        }
        wsum += w;
        w *= lambda;
    }
    for (std::uint32_t a = 0; a < assets; ++a) wmean[a] /= wsum;

    for (std::uint32_t i = 0; i < assets; ++i) {
        for (std::uint32_t j = 0; j < assets; ++j) {
            double acc = 0.0;
            w = 1.0;
            for (std::uint32_t reverse = 0; reverse < observations; ++reverse) {
                const std::uint32_t o = observations - 1 - reverse;
                const double di =
                    values[static_cast<std::size_t>(o) * assets + i] - wmean[i];
                const double dj =
                    values[static_cast<std::size_t>(o) * assets + j] - wmean[j];
                acc += w * di * dj;
                w *= lambda;
            }
            cov[static_cast<std::size_t>(i) * assets + j] =
                static_cast<float>(acc / wsum);
        }
    }
    return cov;
}

}  // namespace trading::metal::cpu_ref
