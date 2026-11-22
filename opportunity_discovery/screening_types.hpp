// ============================================================================
// screening_types.hpp
// ----------------------------------------------------------------------------
// Pure-C++ POD contracts shared across the native opportunity-discovery layer.
//
// This header is deliberately free of Objective-C / Metal headers so it can be
// included by both plain C++ translation units and the Objective-C++ (.mm)
// Metal host. The .mm files own the MTLDevice / command queues / pipelines;
// everything here is host-side, framework-agnostic data.
//
// Phase A scope: the native Metal runtime, its telemetry, and the kernel
// registry. Phase B extends this header with the per-symbol screening inputs,
// discovery records, and membership snapshots.
// ============================================================================
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace trading::metal {

// Which backend actually produced a result. Telemetry and discovery records
// must never conflate a measured GPU execution with a CPU fallback.
enum class ComputeBackend : std::uint8_t {
    Metal = 0,
    Cpu = 1,
};

[[nodiscard]] inline const char* to_string(ComputeBackend b) noexcept {
    return b == ComputeBackend::Metal ? "metal" : "cpu";
}

// Integration lifecycle for a single Metal kernel. Mirrors the states the
// architectural-correction directive (§10) requires us to distinguish so that
// "it compiled" is never confused with "it ran on hardware and matched the CPU
// oracle" or "a real consumer uses it".
enum class KernelStatus : std::uint8_t {
    Discovered = 0,            // present in a .metal source file
    Compiled = 1,              // present in the loaded .metallib
    Registered = 2,            // described in this registry
    PipelineCached = 3,        // an MTLComputePipelineState was built + cached
    HostCallable = 4,          // a native host interface can dispatch it
    HardwareExecuted = 5,      // dispatched on a real device at least once
    ConsumerConnected = 6,     // a legitimate operational/research consumer uses it
    NumericallyValidated = 7,  // GPU output matched the CPU reference oracle
    Benchmarked = 8,           // GPU vs CPU timing measured on device
    Blocked = 9,               // cannot be wired yet (documented reason)
};

[[nodiscard]] const char* to_string(KernelStatus s) noexcept;

// Whether a kernel is wired for live operational use, kept for research only,
// or blocked outright (e.g. needs L2 depth that IEX does not provide).
enum class KernelMode : std::uint8_t {
    Operational = 0,
    Research = 1,
    Blocked = 2,
};

[[nodiscard]] const char* to_string(KernelMode m) noexcept;

// One row of the native kernel inventory. Field names intentionally match the
// directive's required inventory schema (§10/§14) so the emitted JSON is a
// faithful, auditable mirror of the registry.
struct KernelInfo {
    std::string shader_file;        // e.g. "indicators.metal"
    std::string kernel_name;        // bare name, e.g. "rolling_sma_ticks"
    std::string resolved_name;      // name used to look it up (may add "gme::")
    std::string host_interface;     // dotted native/py host, or "" if none
    std::string buffer_contract;    // short human description of the ABI
    std::string consumer;           // INTENDED downstream consumer, or ""
    std::string required_data;      // e.g. "l1", "l2_depth", "returns_matrix"
    std::string blocker;            // reason if blocked, else ""
    KernelMode mode{KernelMode::Blocked};
    bool pipeline_cached{false};
    bool hardware_executed{false};
    bool numerically_validated{false};
    bool benchmarked{false};
    // A kernel is operationally_consumed ONLY when a real operational consumer
    // actually dispatches it on the live path. `consumer` names the *intended*
    // consumer; this flag records the *actual* wiring. A hardware-validated
    // kernel is NOT automatically operationally integrated (directive §80/§86).
    bool operationally_consumed{false};
};

// Monotonic runtime counters. Everything the live dashboard and the
// live_gpu_execution_trace evidence need in order to prove the GPU is actually
// doing operational work (directive §11/§20). All counts are cumulative.
// ---------------------------------------------------------------------------
// Individual-ticker screening contract (Phase B).
//
// The host-side mirror of metal/opportunity_screening.metal. Layout MUST match
// the Metal struct byte-for-byte (all fields are 4-byte float/uint, no padding).
// This encodes the mandatory shared qualification contract (directive §11/§12):
//   entry_eligible = valid_market && valid_5m_direction && valid_1m &&
//                    di_plus_5m > di_minus_5m &&
//                    (above_vwap || above_sma20_5m || above_bb_middle_5m) &&
//                    current_price_1m > previous_close_1m
// evaluated identically on GPU (Metal) and CPU (the oracle / fallback).
// ---------------------------------------------------------------------------
enum ScreenFlag : std::uint32_t {
    kEntryQualified = 1u << 0,
    kTechnicalExit = 1u << 1,
    kBullishDirection = 1u << 2,
    kBearishDirection = 1u << 3,
    kPositiveMomentum = 1u << 4,
    kNegativeMomentum = 1u << 5,
    kAboveVwap = 1u << 6,
    kBelowVwap = 1u << 7,
    kDataInvalid = 1u << 8,
};

struct ScreeningInput {
    float price{0.0f};
    float previous_close_1m{0.0f};
    float di_plus_5m{0.0f};
    float di_minus_5m{0.0f};
    float session_vwap{0.0f};
    float sma20_5m{0.0f};
    float bb_middle_5m{0.0f};
    std::uint32_t valid_market{0};
    std::uint32_t valid_direction{0};
    std::uint32_t valid_1m{0};
    std::uint32_t valid_vwap{0};
    std::uint32_t valid_sma{0};
    std::uint32_t valid_bb{0};
};

struct ScreeningOutput {
    std::uint32_t flags{0};
    float momentum_1m_pct{0.0f};
    float directional_strength{0.0f};
    float vwap_distance_pct{0.0f};
};

// ---------------------------------------------------------------------------
// Technical-feature contract (Phase B.3).
//
// metal/technical_indicators.metal computes, for a batch of symbols (one GPU
// thread per symbol) over each symbol's rolling 5-minute bar window, the
// directional/technical features that feed ScreeningInput: Wilder DMI (DI+/DI-/
// ADX, period 14), SMA20, the 20-period Bollinger middle/upper/lower band,
// session VWAP, ATR, and the last/previous completed closes. The CPU oracle
// cpu_ref::technical_features mirrors it in float32 in the SAME order so GPU and
// CPU agree within a justified tolerance (directive §27/§124). SMA20 and the
// Bollinger middle are the same measurement and are reported once (§29).
//
// Input layout: closes/highs/lows/volumes are symbol_count*max_window row-major
// float32 (row = symbol, bars oldest->newest in [0, count)); counts[symbol] is
// the number of valid bars in that row.
// ---------------------------------------------------------------------------
struct TechWindowParams {
    std::uint32_t symbol_count{0};
    std::uint32_t max_window{0};   // row stride (bar slots per symbol)
    std::uint32_t dmi_period{14};  // Wilder DMI/ADX/ATR period
    std::uint32_t sma_period{20};  // SMA / Bollinger period
};

enum TechValid : std::uint32_t {
    kTechClose = 1u << 0,      // >=1 bar
    kTechPrevClose = 1u << 1,  // >=2 bars
    kTechSma = 1u << 2,        // >=sma_period bars
    kTechBb = 1u << 3,         // >=sma_period bars
    kTechDmi = 1u << 4,        // >=dmi_period+1 bars (DI+/DI-)
    kTechAdx = 1u << 5,        // >=2*dmi_period+1 bars
    kTechVwap = 1u << 6,       // cumulative volume > 0
};

struct TechFeatures {
    float last_close{0.0f};
    float prev_close{0.0f};
    float sma{0.0f};
    float bb_middle{0.0f};
    float bb_upper{0.0f};
    float bb_lower{0.0f};
    float di_plus{0.0f};
    float di_minus{0.0f};
    float adx{0.0f};
    float atr{0.0f};
    float vwap{0.0f};
    std::uint32_t valid{0};
};

// ---------------------------------------------------------------------------
// Market breadth + candidate compaction contract (Phase B.4).
//
// metal/market_breadth.metal reduces an array of per-symbol screen flags (the
// ScreeningOutput.flags bitmasks) into breadth counters via atomics (one thread
// per symbol), and compacts the entry-qualified symbol indices. The CPU oracle
// cpu_ref::market_breadth / cpu_ref::compact_entry_candidates mirror both. Counts
// are order-independent so GPU and CPU agree exactly; compaction order is
// non-deterministic on GPU, so parity compares the id SET (and the count).
// ---------------------------------------------------------------------------
struct BreadthCounters {
    std::uint32_t total{0};
    std::uint32_t valid{0};              // not DATA_INVALID
    std::uint32_t entry_qualified{0};
    std::uint32_t technical_exit{0};
    std::uint32_t bullish{0};
    std::uint32_t bearish{0};
    std::uint32_t positive_momentum{0};
    std::uint32_t negative_momentum{0};
    std::uint32_t above_vwap{0};
    std::uint32_t below_vwap{0};
    std::uint32_t data_invalid{0};
};

// ---------------------------------------------------------------------------
// Cross-sectional ranking contract (Phase B GPU ranking).
//
// metal/opportunity_ranking.metal computes the composite candidate score
// (one GPU thread per candidate) that previously ran in Python
// market_scan.ranking.score_candidate. The per-candidate raw components
// (directional_score rs, order_flow, direction sign, passes_liquidity) are
// precomputed host-side by REUSING the exact scanner functions, so the GPU path
// cannot drift from validated ranking semantics; the GPU does the normalization
// + weighted composite (the cross-sectional, market-relative part). The CPU
// oracle cpu_ref::rank_scores mirrors the kernel float-for-float. Sorting /
// top-K is a deterministic host step (trivial + stable for N<=4096; directive §4).
// ---------------------------------------------------------------------------
struct RankInput {
    float rs{0.0f};            // directional_score (valid iff has_rs)
    float short_mom{0.0f};
    float vwap_dist{0.0f};
    float bb_mid_dist{0.0f};
    float rel_volume{1.0f};    // host defaults None -> 1.0
    float addv{0.0f};          // avg_daily_dollar_volume
    float spread_pct{0.0f};    // valid iff has_spread
    float order_flow{0.0f};    // [0,1], precomputed from the OFE snapshot
    float dir_sign{1.0f};      // +1 long / -1 inverse ETF
    std::uint32_t has_rs{0};
    std::uint32_t has_spread{0};
    std::uint32_t eligible{0}; // passes_liquidity hard filter (precomputed)
};

struct RankParams {
    float w_rel_strength{0.30f};
    float w_momentum{0.15f};
    float w_volume_expansion{0.10f};
    float w_liquidity{0.10f};
    float w_order_flow{0.20f};
    float w_regime_align{0.15f};
    float spy_score{0.0f};
    float qqq_score{0.0f};
    float regime_score{0.0f};
    std::uint32_t count{0};
};

struct RankOutput {
    float total_score{0.0f};
    float rel_strength{0.0f};
    float momentum{0.0f};
    float volume_expansion{0.0f};
    float liquidity{0.0f};
    float order_flow{0.0f};
    float regime_align{0.0f};
    std::uint32_t eligible{0};
};

struct RuntimeTelemetry {
    std::uint64_t dispatches_submitted{0};
    std::uint64_t dispatches_completed{0};
    std::uint64_t dispatch_failures{0};
    std::uint64_t pipeline_cache_hits{0};
    std::uint64_t pipeline_cache_misses{0};
    std::uint64_t buffers_allocated{0};
    std::uint64_t buffers_reused{0};
    std::uint64_t bytes_uploaded{0};
    std::uint64_t bytes_downloaded{0};
    std::uint64_t cpu_fallbacks{0};
    std::uint64_t last_dispatch_duration_ns{0};
    std::uint64_t total_dispatch_duration_ns{0};

    [[nodiscard]] double pipeline_cache_hit_rate() const noexcept {
        const std::uint64_t total = pipeline_cache_hits + pipeline_cache_misses;
        return total == 0 ? 0.0
                          : static_cast<double>(pipeline_cache_hits) /
                                static_cast<double>(total);
    }
    [[nodiscard]] double mean_dispatch_ms() const noexcept {
        return dispatches_completed == 0
                   ? 0.0
                   : (static_cast<double>(total_dispatch_duration_ns) /
                      static_cast<double>(dispatches_completed)) /
                         1.0e6;
    }
};

}  // namespace trading::metal
