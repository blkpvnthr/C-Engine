#pragma once

#include "orderbook/orderbook.hpp"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace trading::analytics {

// ============================================================================
// CONSTANTS
// ============================================================================

inline constexpr float kNaN =
    std::numeric_limits<float>::quiet_NaN();


// ============================================================================
// FEATURE STATUS
// ============================================================================

enum class FeatureStatus : std::uint8_t {
    Valid,
    Warmup,
    MissingData,
    InvalidInput
};


// ============================================================================
// RAW MARKET SAMPLE
// ============================================================================
//
// Compact, contiguous representation intended to be easy to batch for CPU
// vectorization and later copy/map into Apple Metal buffers.
//
// Prices are integer ticks at the execution boundary. Floating-point
// conversion should happen only inside analytics where approximation is
// acceptable.
//

struct MarketSample {
    std::uint64_t timestamp_ns{0};

    std::int64_t bid_ticks{0};
    std::int64_t ask_ticks{0};

    std::uint64_t bid_quantity{0};
    std::uint64_t ask_quantity{0};

    std::optional<std::int64_t> last_ticks;
    std::optional<std::uint64_t> last_quantity;

    std::uint64_t cumulative_volume{0};

    [[nodiscard]]
    bool valid() const noexcept {
        return timestamp_ns != 0 &&
               bid_ticks > 0 &&
               ask_ticks > 0 &&
               bid_ticks < ask_ticks;
    }
};


// ============================================================================
// L2 DEPTH SAMPLE
// ============================================================================
//
// Structure-of-arrays is preferred for GPU work. Each vector index represents
// the same depth level.
//
// bids:
//      level 0 = best bid
//      level 1 = next bid
//
// asks:
//      level 0 = best ask
//      level 1 = next ask
//

struct DepthSample {
    std::uint64_t timestamp_ns{0};

    std::vector<std::int64_t> bid_price_ticks;
    std::vector<std::uint64_t> bid_quantities;

    std::vector<std::int64_t> ask_price_ticks;
    std::vector<std::uint64_t> ask_quantities;

    [[nodiscard]]
    bool valid() const noexcept {
        if (timestamp_ns == 0) {
            return false;
        }

        if (
            bid_price_ticks.size() !=
            bid_quantities.size()
        ) {
            return false;
        }

        if (
            ask_price_ticks.size() !=
            ask_quantities.size()
        ) {
            return false;
        }

        if (
            bid_price_ticks.empty() ||
            ask_price_ticks.empty()
        ) {
            return false;
        }

        return true;
    }

    [[nodiscard]]
    std::size_t bid_levels() const noexcept {
        return bid_price_ticks.size();
    }

    [[nodiscard]]
    std::size_t ask_levels() const noexcept {
        return ask_price_ticks.size();
    }
};


// ============================================================================
// BAR
// ============================================================================

struct Bar {
    std::uint64_t timestamp_ns{0};

    float open{kNaN};
    float high{kNaN};
    float low{kNaN};
    float close{kNaN};

    float volume{0.0F};

    [[nodiscard]]
    bool valid() const noexcept;
};


// ============================================================================
// FEATURE CONFIGURATION
// ============================================================================

struct FeatureConfig {
    // Moving averages.
    std::size_t sma_fast_period{5};
    std::size_t sma_slow_period{20};
    std::size_t sma_50_period{50};
    std::size_t sma_200_period{200};

    // EMA / MACD.
    std::size_t ema_fast_period{12};
    std::size_t ema_slow_period{26};
    std::size_t macd_signal_period{9};

    // Oscillators.
    std::size_t rsi_period{14};
    std::size_t roc_period{12};
    std::size_t cci_period{20};
    std::size_t stoch_rsi_period{14};

    // Volatility.
    std::size_t volatility_period{20};

    // L2.
    std::size_t depth_levels{10};

    // Exponential weighting applied by level:
    //
    //      weight(level) = depth_decay ^ level
    //
    // Must be in (0, 1].
    float depth_decay{0.85F};

    [[nodiscard]]
    bool valid() const noexcept;
};


// ============================================================================
// L1 FEATURES
// ============================================================================

struct L1Features {
    FeatureStatus status{FeatureStatus::MissingData};

    std::uint64_t timestamp_ns{0};

    float bid{kNaN};
    float ask{kNaN};

    float midpoint{kNaN};
    float spread{kNaN};
    float spread_bps{kNaN};

    float bid_quantity{kNaN};
    float ask_quantity{kNaN};

    // (bid_qty - ask_qty) / (bid_qty + ask_qty)
    float top_level_imbalance{kNaN};

    // Quantity-weighted top-of-book fair-price estimate:
    //
    // (ask * bid_qty + bid * ask_qty) /
    // (bid_qty + ask_qty)
    float microprice{kNaN};

    // microprice - midpoint
    float microprice_offset{kNaN};

    // Offset expressed relative to spread.
    float microprice_offset_spreads{kNaN};
};


// ============================================================================
// L2 FEATURES
// ============================================================================

struct L2Features {
    FeatureStatus status{FeatureStatus::MissingData};

    std::uint64_t timestamp_ns{0};

    std::size_t levels_used{0};

    float total_bid_depth{0.0F};
    float total_ask_depth{0.0F};

    float weighted_bid_depth{0.0F};
    float weighted_ask_depth{0.0F};

    // Raw depth imbalance.
    float depth_imbalance{kNaN};

    // Distance-weighted depth imbalance.
    float weighted_depth_imbalance{kNaN};

    // Approximate weighted average depth prices.
    float weighted_bid_price{kNaN};
    float weighted_ask_price{kNaN};

    // Difference between weighted bid/ask liquidity pressure.
    float liquidity_pressure{kNaN};
};


// ============================================================================
// TECHNICAL FEATURES
// ============================================================================

struct TechnicalFeatures {
    FeatureStatus status{FeatureStatus::Warmup};

    std::uint64_t timestamp_ns{0};

    float close{kNaN};

    float sma_fast{kNaN};
    float sma_slow{kNaN};
    float sma_50{kNaN};
    float sma_200{kNaN};

    float ema_fast{kNaN};
    float ema_slow{kNaN};

    float macd{kNaN};
    float macd_signal{kNaN};
    float macd_histogram{kNaN};

    float rsi{kNaN};
    float roc{kNaN};
    float cci{kNaN};
    float stoch_rsi{kNaN};

    float simple_return{kNaN};
    float log_return{kNaN};

    float realized_volatility{kNaN};

    // Typical-price/volume VWAP over the supplied window.
    float vwap{kNaN};

    // close - VWAP
    float vwap_distance{kNaN};

    // (close / VWAP - 1)
    float vwap_distance_pct{kNaN};
};


// ============================================================================
// COMBINED FEATURE VECTOR
// ============================================================================
//
// This is deliberately plain-data oriented so that the later Metal boundary
// can flatten large batches without exposing the mutable order book.
//

struct FeatureVector {
    std::uint64_t timestamp_ns{0};

    L1Features l1;
    L2Features l2;
    TechnicalFeatures technical;
};


// ============================================================================
// FLAT GPU FEATURE ROW
// ============================================================================
//
// Fixed-width POD-style representation suitable for bulk transfer to Metal.
//
// No std::string, std::vector, std::optional, pointers, or iterators are
// present in this structure.
//

struct GpuFeatureRow {
    std::uint64_t timestamp_ns{0};

    float midpoint{kNaN};
    float spread{kNaN};
    float spread_bps{kNaN};
    float top_level_imbalance{kNaN};
    float microprice{kNaN};

    float depth_imbalance{kNaN};
    float weighted_depth_imbalance{kNaN};
    float liquidity_pressure{kNaN};

    float sma_fast{kNaN};
    float sma_slow{kNaN};
    float sma_50{kNaN};
    float sma_200{kNaN};

    float ema_fast{kNaN};
    float ema_slow{kNaN};

    float macd{kNaN};
    float macd_signal{kNaN};
    float macd_histogram{kNaN};

    float rsi{kNaN};
    float roc{kNaN};
    float cci{kNaN};
    float stoch_rsi{kNaN};

    float simple_return{kNaN};
    float log_return{kNaN};
    float realized_volatility{kNaN};

    float vwap{kNaN};
    float vwap_distance_pct{kNaN};
};


// ============================================================================
// GPU DEPTH BATCH
// ============================================================================
//
// Flattened structure-of-arrays representation.
//
// Layout for prices/quantities:
//
//      index = sample_index * levels + level_index
//
// Example with levels == 3:
//
//      sample 0: [L0, L1, L2]
//      sample 1: [L0, L1, L2]
//      sample 2: [L0, L1, L2]
//
// This format maps naturally onto Metal compute kernels.
//

struct GpuDepthBatch {
    std::size_t sample_count{0};
    std::size_t levels{0};

    std::vector<std::uint64_t> timestamps_ns;

    std::vector<float> bid_prices;
    std::vector<float> ask_prices;

    std::vector<float> bid_quantities;
    std::vector<float> ask_quantities;

    [[nodiscard]]
    bool valid() const noexcept;

    [[nodiscard]]
    std::size_t flattened_size() const noexcept {
        return sample_count * levels;
    }
};


// ============================================================================
// FEATURE BATCH
// ============================================================================

struct FeatureBatch {
    std::vector<GpuFeatureRow> rows;

    [[nodiscard]]
    bool empty() const noexcept {
        return rows.empty();
    }

    [[nodiscard]]
    std::size_t size() const noexcept {
        return rows.size();
    }

    [[nodiscard]]
    const GpuFeatureRow* data() const noexcept {
        return rows.data();
    }

    [[nodiscard]]
    GpuFeatureRow* data() noexcept {
        return rows.data();
    }

    [[nodiscard]]
    std::size_t size_bytes() const noexcept {
        return rows.size() * sizeof(GpuFeatureRow);
    }
};


// ============================================================================
// BOOK SNAPSHOT ADAPTER
// ============================================================================
//
// Converts the execution-oriented BookSnapshot into an analytics-oriented
// DepthSample. The order book remains authoritative; analytics receives only
// immutable snapshots.
//

class BookFeatureAdapter {
public:
    [[nodiscard]]
    static DepthSample from_book_snapshot(
        const BookSnapshot& snapshot,
        std::uint64_t timestamp_ns,
        std::size_t max_levels = 0
    );

    [[nodiscard]]
    static MarketSample top_of_book(
        const BookSnapshot& snapshot,
        std::uint64_t timestamp_ns
    );
};


// ============================================================================
// FEATURE ENGINE
// ============================================================================
//
// CPU reference implementation interface.
//
// features.cpp should implement these routines first and serve as the
// correctness oracle for Metal kernels.
//
// Later architecture:
//
//      LimitOrderBook
//           |
//           v
//      BookSnapshot
//           |
//           v
//      BookFeatureAdapter
//           |
//           +--------------------+
//           |                    |
//           v                    v
//      CPU FeatureEngine     MetalFeatureEngine
//      reference/oracle      parallel backend
//           |                    |
//           +---------+----------+
//                     |
//                     v
//                FeatureVector
//
// Neither backend is permitted to mutate order state.
//

class FeatureEngine {
public:
    explicit FeatureEngine(
        FeatureConfig config = {}
    );

    [[nodiscard]]
    const FeatureConfig& config() const noexcept {
        return config_;
    }

    // ------------------------------------------------------------------------
    // L1
    // ------------------------------------------------------------------------

    [[nodiscard]]
    L1Features compute_l1(
        const MarketSample& sample
    ) const;

    // ------------------------------------------------------------------------
    // L2
    // ------------------------------------------------------------------------

    [[nodiscard]]
    L2Features compute_l2(
        const DepthSample& depth
    ) const;

    // ------------------------------------------------------------------------
    // TECHNICALS
    // ------------------------------------------------------------------------

    [[nodiscard]]
    TechnicalFeatures compute_technicals(
        std::span<const Bar> bars
    ) const;

    // ------------------------------------------------------------------------
    // COMBINED
    // ------------------------------------------------------------------------

    [[nodiscard]]
    FeatureVector compute(
        const MarketSample& market,
        const DepthSample& depth,
        std::span<const Bar> bars
    ) const;

    // ------------------------------------------------------------------------
    // BATCH
    // ------------------------------------------------------------------------

    [[nodiscard]]
    FeatureBatch compute_batch(
        std::span<const MarketSample> market,
        std::span<const DepthSample> depth,
        std::span<const std::vector<Bar>> bar_windows
    ) const;

    // ------------------------------------------------------------------------
    // GPU FLATTENING
    // ------------------------------------------------------------------------

    [[nodiscard]]
    static GpuFeatureRow flatten(
        const FeatureVector& features
    ) noexcept;

    [[nodiscard]]
    static GpuDepthBatch flatten_depth(
        std::span<const DepthSample> samples,
        std::size_t levels
    );

private:
    FeatureConfig config_;

    // ========================================================================
    // NUMERICAL HELPERS
    // ========================================================================

    [[nodiscard]]
    static float ticks_to_float(
        std::int64_t ticks
    ) noexcept;

    [[nodiscard]]
    static float safe_ratio(
        float numerator,
        float denominator
    ) noexcept;

    [[nodiscard]]
    static float sma(
        std::span<const Bar> bars,
        std::size_t period
    ) noexcept;

    [[nodiscard]]
    static float ema(
        std::span<const Bar> bars,
        std::size_t period
    ) noexcept;

    [[nodiscard]]
    static float ema_values(
        std::span<const float> values,
        std::size_t period
    ) noexcept;

    [[nodiscard]]
    static float rsi(
        std::span<const Bar> bars,
        std::size_t period
    ) noexcept;

    [[nodiscard]]
    static float roc(
        std::span<const Bar> bars,
        std::size_t period
    ) noexcept;

    [[nodiscard]]
    static float cci(
        std::span<const Bar> bars,
        std::size_t period
    ) noexcept;

    [[nodiscard]]
    static float stoch_rsi(
        std::span<const Bar> bars,
        std::size_t rsi_period,
        std::size_t stoch_period
    );

    [[nodiscard]]
    static float realized_volatility(
        std::span<const Bar> bars,
        std::size_t period
    ) noexcept;

    [[nodiscard]]
    static float vwap(
        std::span<const Bar> bars
    ) noexcept;

    [[nodiscard]]
    static float simple_return(
        std::span<const Bar> bars
    ) noexcept;

    [[nodiscard]]
    static float log_return(
        std::span<const Bar> bars
    ) noexcept;

    [[nodiscard]]
    static float macd_line(
        std::span<const Bar> bars,
        std::size_t fast_period,
        std::size_t slow_period
    ) noexcept;

    [[nodiscard]]
    static float macd_signal(
        std::span<const Bar> bars,
        std::size_t fast_period,
        std::size_t slow_period,
        std::size_t signal_period
    );

    [[nodiscard]]
    static std::vector<float> rsi_series(
        std::span<const Bar> bars,
        std::size_t period
    );

    [[nodiscard]]
    static std::vector<float> macd_series(
        std::span<const Bar> bars,
        std::size_t fast_period,
        std::size_t slow_period
    );

    // ========================================================================
    // VALIDATION
    // ========================================================================

    [[nodiscard]]
    static bool bars_valid(
        std::span<const Bar> bars
    ) noexcept;

    [[nodiscard]]
    std::size_t required_warmup() const noexcept;
};


// ============================================================================
// OPTIONAL METAL BACKEND CONTRACT
// ============================================================================
//
// This interface lets the orchestration layer use a Metal implementation
// without coupling callers to Objective-C++/Metal headers.
//
// A later .mm implementation can own MTLDevice, command queues, pipelines,
// and unified-memory buffers behind a PIMPL.
//
// CPU FeatureEngine remains the numerical reference.
//

class MetalFeatureEngine {
public:
    explicit MetalFeatureEngine(
        FeatureConfig config = {}
    );

    ~MetalFeatureEngine();

    MetalFeatureEngine(
        const MetalFeatureEngine&
    ) = delete;

    MetalFeatureEngine& operator=(
        const MetalFeatureEngine&
    ) = delete;

    MetalFeatureEngine(
        MetalFeatureEngine&&
    ) noexcept;

    MetalFeatureEngine& operator=(
        MetalFeatureEngine&&
    ) noexcept;

    [[nodiscard]]
    bool available() const noexcept;

    [[nodiscard]]
    std::string device_name() const;

    // Bulk L2 analytics. Intended as one of the first Metal kernels because
    // each sample/depth calculation can be parallelized efficiently.
    [[nodiscard]]
    std::vector<L2Features> compute_l2_batch(
        const GpuDepthBatch& batch
    );

    // Bulk flattened feature processing. The exact kernel implementation can
    // expand as more indicators move from CPU reference code to Metal.
    [[nodiscard]]
    FeatureBatch compute_batch(
        const FeatureBatch& input
    );

private:
    struct Impl;
    Impl* impl_{nullptr};
};


// ============================================================================
// HUMAN-READABLE ENUM TEXT
// ============================================================================

[[nodiscard]]
std::string_view to_string(
    FeatureStatus status
) noexcept;

} // namespace trading::analytics
