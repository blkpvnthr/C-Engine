// ============================================================================
// cpu_reference.hpp
// ----------------------------------------------------------------------------
// Native CPU oracles for the Metal kernels. These mirror the exact kernel
// semantics (and the existing cengine/metal_cpu_ref.py reference) so GPU output
// can be validated for strict numerical parity, and so the application has a
// validated CPU fallback when Metal is unavailable or below the GPU crossover
// (directive §4/§13). Integer indicators use C-style truncating division to
// match the Metal `long /` operator exactly.
// ============================================================================
#pragma once

#include <cstdint>
#include <vector>

namespace trading::metal::cpu_ref {

// Integer division truncating toward zero (matches Metal/C `long` /).
[[nodiscard]] std::int64_t trunc_div(std::int64_t a, std::int64_t b) noexcept;

// --- indicators (exact integer parity with indicators.metal) ---------------
[[nodiscard]] std::vector<std::int64_t> sma_ticks(
    const std::vector<std::int64_t>& prices, std::uint32_t window);
[[nodiscard]] std::vector<std::int64_t> ema_ticks(
    const std::vector<std::int64_t>& prices,
    std::uint32_t alpha_numerator, std::uint32_t alpha_denominator);
[[nodiscard]] std::vector<std::int64_t> apo_ticks(
    const std::vector<std::int64_t>& fast_ema,
    const std::vector<std::int64_t>& slow_ema);
[[nodiscard]] std::vector<std::int64_t> true_range_ticks(
    const std::vector<std::int64_t>& highs,
    const std::vector<std::int64_t>& lows,
    const std::vector<std::int64_t>& previous_closes);

// --- covariance (float; compared within float32 tolerance) -----------------
// values is observations*assets row-major. Results are flattened row-major.
[[nodiscard]] std::vector<float> column_means(
    const std::vector<float>& values, std::uint32_t observations,
    std::uint32_t assets);
[[nodiscard]] std::vector<float> covariance_matrix(
    const std::vector<float>& values, std::uint32_t observations,
    std::uint32_t assets);
[[nodiscard]] std::vector<float> ewma_covariance_matrix(
    const std::vector<float>& values, std::uint32_t observations,
    std::uint32_t assets, float lambda);

}  // namespace trading::metal::cpu_ref
