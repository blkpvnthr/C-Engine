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

#include "opportunity_discovery/screening_types.hpp"

namespace trading::metal::cpu_ref {

// --- opportunity screening (exact branch-for-branch parity with -----------
// metal/opportunity_screening.metal's screen_market). This is the CPU oracle
// for GPU-parity validation AND the validated CPU fallback / crossover path for
// the mandatory shared entry contract (directive §12/§28). One output per input.
[[nodiscard]] std::vector<ScreeningOutput> screen_market(
    const std::vector<ScreeningInput>& inputs);

// --- technical features (exact float32 parity with ------------------------
// metal/technical_indicators.metal's technical_features). Computes Wilder DMI
// (DI+/DI-/ADX), SMA20, Bollinger bands, ATR and session VWAP per symbol from a
// row-major (symbol_count * max_window) bar window. Oracle + CPU fallback for
// the Phase B.3 feature pipeline (directive §27).
[[nodiscard]] std::vector<TechFeatures> technical_features(
    const std::vector<float>& closes, const std::vector<float>& highs,
    const std::vector<float>& lows, const std::vector<float>& volumes,
    const std::vector<std::uint32_t>& counts, const TechWindowParams& params);

// --- market breadth / candidate compaction (exact parity with -------------
// metal/market_breadth.metal). `flags` is one ScreeningOutput.flags per symbol.
[[nodiscard]] BreadthCounters market_breadth(const std::vector<std::uint32_t>& flags);
// Returns the entry-qualified symbol indices (ascending). `total_qualified` is
// set to the full count even if it exceeds `capacity` (overflow = count>capacity).
[[nodiscard]] std::vector<std::uint32_t> compact_entry_candidates(
    const std::vector<std::uint32_t>& flags, std::uint32_t capacity,
    std::uint32_t& total_qualified);

// --- cross-sectional ranking (exact float32 parity with -------------------
// metal/opportunity_ranking.metal, which mirrors market_scan.ranking.
// score_candidate). One RankOutput per RankInput; sorting/top-K is done by the
// host after scoring.
[[nodiscard]] std::vector<RankOutput> rank_scores(
    const std::vector<RankInput>& inputs, const RankParams& params);

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
