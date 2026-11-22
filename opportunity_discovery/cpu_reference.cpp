// ============================================================================
// cpu_reference.cpp — native CPU oracles (see cpu_reference.hpp).
// Mirrors cengine/metal_cpu_ref.py semantics exactly.
// ============================================================================
#include "opportunity_discovery/cpu_reference.hpp"

#include <algorithm>
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

std::vector<ScreeningOutput> screen_market(
    const std::vector<ScreeningInput>& inputs) {
    std::vector<ScreeningOutput> out(inputs.size());
    for (std::size_t i = 0; i < inputs.size(); ++i) {
        const ScreeningInput& s = inputs[i];
        ScreeningOutput r;  // zero-initialised by member defaults

        // Validity gates — identical to opportunity_screening.metal. A missing
        // or non-finite reference can never satisfy a qualification (§29/§30).
        const bool price_valid =
            s.valid_market && std::isfinite(s.price) && s.price > 0.0f;
        const bool direction_valid = s.valid_direction &&
                                     std::isfinite(s.di_plus_5m) &&
                                     std::isfinite(s.di_minus_5m);
        const bool momentum_valid = price_valid && s.valid_1m &&
                                    std::isfinite(s.previous_close_1m) &&
                                    s.previous_close_1m > 0.0f;
        const bool vwap_valid =
            s.valid_vwap && std::isfinite(s.session_vwap) && s.session_vwap > 0.0f;
        const bool sma_valid =
            s.valid_sma && std::isfinite(s.sma20_5m) && s.sma20_5m > 0.0f;
        const bool bb_valid =
            s.valid_bb && std::isfinite(s.bb_middle_5m) && s.bb_middle_5m > 0.0f;

        const bool bullish = direction_valid && s.di_plus_5m > s.di_minus_5m;
        const bool bearish = direction_valid && s.di_minus_5m > s.di_plus_5m;

        const bool above_vwap = price_valid && vwap_valid && s.price > s.session_vwap;
        const bool below_vwap = price_valid && vwap_valid && s.price < s.session_vwap;

        const bool above_reference =
            price_valid &&
            (above_vwap || (sma_valid && s.price > s.sma20_5m) ||
             (bb_valid && s.price > s.bb_middle_5m));
        const bool below_reference =
            price_valid &&
            (below_vwap || (sma_valid && s.price < s.sma20_5m) ||
             (bb_valid && s.price < s.bb_middle_5m));

        const bool positive_1m = momentum_valid && s.price > s.previous_close_1m;
        const bool negative_1m = momentum_valid && s.price < s.previous_close_1m;

        std::uint32_t flags = 0;
        if (bullish) flags |= kBullishDirection;
        if (bearish) flags |= kBearishDirection;
        if (positive_1m) flags |= kPositiveMomentum;
        if (negative_1m) flags |= kNegativeMomentum;
        if (above_vwap) flags |= kAboveVwap;
        if (below_vwap) flags |= kBelowVwap;

        // The mandatory shared entry contract (directive §28). Necessary, not
        // sufficient, for an order — portfolio/risk gates still apply downstream.
        if (bullish && above_reference && positive_1m) flags |= kEntryQualified;
        if (bearish && below_reference && negative_1m) flags |= kTechnicalExit;
        if (!price_valid || !direction_valid || !momentum_valid ||
            !(vwap_valid || sma_valid || bb_valid)) {
            flags |= kDataInvalid;
        }
        r.flags = flags;

        r.momentum_1m_pct = momentum_valid
                                ? (s.price / s.previous_close_1m - 1.0f) * 100.0f
                                : std::nanf("");
        r.directional_strength =
            direction_valid ? s.di_plus_5m - s.di_minus_5m : std::nanf("");
        r.vwap_distance_pct =
            (price_valid && vwap_valid)
                ? (s.price / s.session_vwap - 1.0f) * 100.0f
                : std::nanf("");

        out[i] = r;
    }
    return out;
}

std::vector<TechFeatures> technical_features(
    const std::vector<float>& closes, const std::vector<float>& highs,
    const std::vector<float>& lows, const std::vector<float>& volumes,
    const std::vector<std::uint32_t>& counts, const TechWindowParams& params) {
    const std::uint32_t S = params.symbol_count;
    const std::uint32_t W = params.max_window;
    const std::uint32_t p = params.dmi_period == 0 ? 14u : params.dmi_period;
    const std::uint32_t sp = params.sma_period == 0 ? 20u : params.sma_period;
    std::vector<TechFeatures> out(S);

    for (std::uint32_t s = 0; s < S; ++s) {
        TechFeatures f;
        const std::size_t base = static_cast<std::size_t>(s) * W;
        std::uint32_t n = s < counts.size() ? counts[s] : 0;
        if (n > W) n = W;

        if (n >= 1) {
            f.last_close = closes[base + n - 1];
            f.valid |= kTechClose;
        }
        if (n >= 2) {
            f.prev_close = closes[base + n - 2];
            f.valid |= kTechPrevClose;
        }

        // SMA / Bollinger over the last `sp` closes (population stddev).
        if (n >= sp) {
            float sum = 0.0f;
            for (std::uint32_t i = n - sp; i < n; ++i) sum += closes[base + i];
            const float sma = sum / static_cast<float>(sp);
            float var = 0.0f;
            for (std::uint32_t i = n - sp; i < n; ++i) {
                const float d = closes[base + i] - sma;
                var += d * d;
            }
            const float sd = std::sqrt(var / static_cast<float>(sp));
            f.sma = sma;
            f.bb_middle = sma;
            f.bb_upper = sma + 2.0f * sd;
            f.bb_lower = sma - 2.0f * sd;
            f.valid |= kTechSma | kTechBb;
        }

        // Session VWAP over the window (typical price * volume).
        float vsum = 0.0f;
        float pvsum = 0.0f;
        for (std::uint32_t i = 0; i < n; ++i) {
            const float tp =
                (highs[base + i] + lows[base + i] + closes[base + i]) / 3.0f;
            vsum += volumes[base + i];
            pvsum += tp * volumes[base + i];
        }
        if (vsum > 0.0f) {
            f.vwap = pvsum / vsum;
            f.valid |= kTechVwap;
        }

        // Wilder DMI / ATR / ADX (sum-form smoothing, matches the kernel).
        if (n >= p + 1) {
            const float fp = static_cast<float>(p);
            float s_tr = 0.0f;
            float s_dmp = 0.0f;
            float s_dmm = 0.0f;
            auto step = [&](std::uint32_t i, float& tr, float& dmp, float& dmm) {
                const float up = highs[base + i] - highs[base + i - 1];
                const float dn = lows[base + i - 1] - lows[base + i];
                dmp = (up > dn && up > 0.0f) ? up : 0.0f;
                dmm = (dn > up && dn > 0.0f) ? dn : 0.0f;
                const float hl = highs[base + i] - lows[base + i];
                const float hc = std::fabs(highs[base + i] - closes[base + i - 1]);
                const float lc = std::fabs(lows[base + i] - closes[base + i - 1]);
                tr = std::max(hl, std::max(hc, lc));
            };
            for (std::uint32_t i = 1; i <= p; ++i) {
                float tr, dmp, dmm;
                step(i, tr, dmp, dmm);
                s_tr += tr;
                s_dmp += dmp;
                s_dmm += dmm;
            }
            float di_p = s_tr > 0.0f ? 100.0f * s_dmp / s_tr : 0.0f;
            float di_m = s_tr > 0.0f ? 100.0f * s_dmm / s_tr : 0.0f;
            float dx = (di_p + di_m) > 0.0f
                           ? 100.0f * std::fabs(di_p - di_m) / (di_p + di_m)
                           : 0.0f;
            float dx_sum = dx;
            std::uint32_t dx_count = 1;
            bool adx_started = false;
            float adx = 0.0f;
            if (p == 1) {
                adx = dx;
                adx_started = true;
            }
            for (std::uint32_t i = p + 1; i < n; ++i) {
                float tr, dmp, dmm;
                step(i, tr, dmp, dmm);
                s_tr = s_tr - s_tr / fp + tr;
                s_dmp = s_dmp - s_dmp / fp + dmp;
                s_dmm = s_dmm - s_dmm / fp + dmm;
                di_p = s_tr > 0.0f ? 100.0f * s_dmp / s_tr : 0.0f;
                di_m = s_tr > 0.0f ? 100.0f * s_dmm / s_tr : 0.0f;
                dx = (di_p + di_m) > 0.0f
                         ? 100.0f * std::fabs(di_p - di_m) / (di_p + di_m)
                         : 0.0f;
                if (!adx_started) {
                    dx_sum += dx;
                    ++dx_count;
                    if (dx_count == p) {
                        adx = dx_sum / fp;
                        adx_started = true;
                    }
                } else {
                    adx = (adx * (fp - 1.0f) + dx) / fp;
                }
            }
            f.di_plus = di_p;
            f.di_minus = di_m;
            f.atr = s_tr / fp;
            f.valid |= kTechDmi;
            if (adx_started) {
                f.adx = adx;
                f.valid |= kTechAdx;
            }
        }
        out[s] = f;
    }
    return out;
}

}  // namespace trading::metal::cpu_ref
