#include "features.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace trading::analytics {

namespace {

[[nodiscard]]
bool finite(const float value) noexcept {
    return std::isfinite(value);
}

[[nodiscard]]
float quiet_nan() noexcept {
    return std::numeric_limits<float>::quiet_NaN();
}

[[nodiscard]]
float typical_price(const Bar& bar) noexcept {
    return (bar.high + bar.low + bar.close) / 3.0F;
}

[[nodiscard]]
std::size_t min_depth_levels(
    const DepthSample& depth,
    const std::size_t configured
) noexcept {
    const auto available =
        std::min(depth.bid_levels(), depth.ask_levels());

    return configured == 0
        ? available
        : std::min(available, configured);
}

} // namespace


// ============================================================================
// BAR / CONFIG / BATCH VALIDATION
// ============================================================================

bool Bar::valid() const noexcept {
    return timestamp_ns != 0 &&
           finite(open) &&
           finite(high) &&
           finite(low) &&
           finite(close) &&
           finite(volume) &&
           volume >= 0.0F &&
           high >= low &&
           high >= open &&
           high >= close &&
           low <= open &&
           low <= close &&
           open > 0.0F &&
           high > 0.0F &&
           low > 0.0F &&
           close > 0.0F;
}


bool FeatureConfig::valid() const noexcept {
    return sma_fast_period > 0 &&
           sma_slow_period > 0 &&
           sma_50_period > 0 &&
           sma_200_period > 0 &&
           ema_fast_period > 0 &&
           ema_slow_period > 0 &&
           macd_signal_period > 0 &&
           rsi_period > 0 &&
           roc_period > 0 &&
           cci_period > 0 &&
           stoch_rsi_period > 0 &&
           volatility_period > 0 &&
           depth_levels > 0 &&
           finite(depth_decay) &&
           depth_decay > 0.0F &&
           depth_decay <= 1.0F;
}


bool GpuDepthBatch::valid() const noexcept {
    if (sample_count == 0 || levels == 0) {
        return false;
    }

    if (timestamps_ns.size() != sample_count) {
        return false;
    }

    if (
        sample_count >
        std::numeric_limits<std::size_t>::max() / levels
    ) {
        return false;
    }

    const auto expected = sample_count * levels;

    return bid_prices.size() == expected &&
           ask_prices.size() == expected &&
           bid_quantities.size() == expected &&
           ask_quantities.size() == expected;
}


// ============================================================================
// BOOK SNAPSHOT ADAPTER
// ============================================================================

DepthSample BookFeatureAdapter::from_book_snapshot(
    const BookSnapshot& snapshot,
    const std::uint64_t timestamp_ns,
    const std::size_t max_levels
) {
    DepthSample result;
    result.timestamp_ns = timestamp_ns;

    const auto bid_count =
        max_levels == 0
            ? snapshot.bids.size()
            : std::min(max_levels, snapshot.bids.size());

    const auto ask_count =
        max_levels == 0
            ? snapshot.asks.size()
            : std::min(max_levels, snapshot.asks.size());

    result.bid_price_ticks.reserve(bid_count);
    result.bid_quantities.reserve(bid_count);
    result.ask_price_ticks.reserve(ask_count);
    result.ask_quantities.reserve(ask_count);

    for (std::size_t i = 0; i < bid_count; ++i) {
        const auto& level = snapshot.bids[i];

        if (level.price_ticks <= 0) {
            throw std::invalid_argument(
                "BookFeatureAdapter: invalid bid price"
            );
        }

        result.bid_price_ticks.push_back(level.price_ticks);
        result.bid_quantities.push_back(level.total_quantity);
    }

    for (std::size_t i = 0; i < ask_count; ++i) {
        const auto& level = snapshot.asks[i];

        if (level.price_ticks <= 0) {
            throw std::invalid_argument(
                "BookFeatureAdapter: invalid ask price"
            );
        }

        result.ask_price_ticks.push_back(level.price_ticks);
        result.ask_quantities.push_back(level.total_quantity);
    }

    return result;
}


MarketSample BookFeatureAdapter::top_of_book(
    const BookSnapshot& snapshot,
    const std::uint64_t timestamp_ns
) {
    MarketSample result;
    result.timestamp_ns = timestamp_ns;

    if (snapshot.bids.empty() || snapshot.asks.empty()) {
        return result;
    }

    const auto& bid = snapshot.bids.front();
    const auto& ask = snapshot.asks.front();

    result.bid_ticks = bid.price_ticks;
    result.ask_ticks = ask.price_ticks;
    result.bid_quantity = bid.total_quantity;
    result.ask_quantity = ask.total_quantity;

    return result;
}


// ============================================================================
// FEATURE ENGINE CONSTRUCTION
// ============================================================================

FeatureEngine::FeatureEngine(
    FeatureConfig config
)
    : config_(std::move(config)) {

    if (!config_.valid()) {
        throw std::invalid_argument(
            "FeatureEngine: invalid FeatureConfig"
        );
    }
}


// ============================================================================
// L1 FEATURES
// ============================================================================

L1Features FeatureEngine::compute_l1(
    const MarketSample& sample
) const {
    L1Features result;
    result.timestamp_ns = sample.timestamp_ns;

    if (!sample.valid()) {
        result.status = FeatureStatus::InvalidInput;
        return result;
    }

    const float bid = ticks_to_float(sample.bid_ticks);
    const float ask = ticks_to_float(sample.ask_ticks);

    const float bid_qty =
        static_cast<float>(sample.bid_quantity);

    const float ask_qty =
        static_cast<float>(sample.ask_quantity);

    const float total_qty = bid_qty + ask_qty;
    const float midpoint = (bid + ask) * 0.5F;
    const float spread = ask - bid;

    result.bid = bid;
    result.ask = ask;
    result.midpoint = midpoint;
    result.spread = spread;

    result.spread_bps =
        midpoint > 0.0F
            ? (spread / midpoint) * 10'000.0F
            : quiet_nan();

    result.bid_quantity = bid_qty;
    result.ask_quantity = ask_qty;

    result.top_level_imbalance =
        total_qty > 0.0F
            ? (bid_qty - ask_qty) / total_qty
            : 0.0F;

    if (total_qty > 0.0F) {
        result.microprice =
            ((ask * bid_qty) + (bid * ask_qty)) /
            total_qty;

        result.microprice_offset =
            result.microprice - midpoint;

        result.microprice_offset_spreads =
            spread > 0.0F
                ? result.microprice_offset / spread
                : quiet_nan();
    }

    result.status = FeatureStatus::Valid;
    return result;
}


// ============================================================================
// L2 FEATURES
// ============================================================================

L2Features FeatureEngine::compute_l2(
    const DepthSample& depth
) const {
    L2Features result;
    result.timestamp_ns = depth.timestamp_ns;

    if (!depth.valid()) {
        result.status = FeatureStatus::InvalidInput;
        return result;
    }

    const std::size_t levels =
        min_depth_levels(depth, config_.depth_levels);

    if (levels == 0) {
        result.status = FeatureStatus::MissingData;
        return result;
    }

    result.levels_used = levels;

    double total_bid = 0.0;
    double total_ask = 0.0;

    double weighted_bid = 0.0;
    double weighted_ask = 0.0;

    double weighted_bid_px_num = 0.0;
    double weighted_ask_px_num = 0.0;

    double weighted_bid_qty_for_price = 0.0;
    double weighted_ask_qty_for_price = 0.0;

    double weight = 1.0;

    for (std::size_t i = 0; i < levels; ++i) {
        const auto bid_ticks = depth.bid_price_ticks[i];
        const auto ask_ticks = depth.ask_price_ticks[i];

        if (bid_ticks <= 0 || ask_ticks <= 0) {
            result.status = FeatureStatus::InvalidInput;
            return result;
        }

        if (i > 0) {
            if (
                depth.bid_price_ticks[i] >
                depth.bid_price_ticks[i - 1]
            ) {
                result.status = FeatureStatus::InvalidInput;
                return result;
            }

            if (
                depth.ask_price_ticks[i] <
                depth.ask_price_ticks[i - 1]
            ) {
                result.status = FeatureStatus::InvalidInput;
                return result;
            }
        }

        const double bid_qty =
            static_cast<double>(depth.bid_quantities[i]);

        const double ask_qty =
            static_cast<double>(depth.ask_quantities[i]);

        const double bid_px =
            static_cast<double>(bid_ticks);

        const double ask_px =
            static_cast<double>(ask_ticks);

        total_bid += bid_qty;
        total_ask += ask_qty;

        const double weighted_bid_qty = bid_qty * weight;
        const double weighted_ask_qty = ask_qty * weight;

        weighted_bid += weighted_bid_qty;
        weighted_ask += weighted_ask_qty;

        weighted_bid_px_num +=
            bid_px * weighted_bid_qty;

        weighted_ask_px_num +=
            ask_px * weighted_ask_qty;

        weighted_bid_qty_for_price += weighted_bid_qty;
        weighted_ask_qty_for_price += weighted_ask_qty;

        weight *= static_cast<double>(config_.depth_decay);
    }

    result.total_bid_depth = static_cast<float>(total_bid);
    result.total_ask_depth = static_cast<float>(total_ask);

    result.weighted_bid_depth =
        static_cast<float>(weighted_bid);

    result.weighted_ask_depth =
        static_cast<float>(weighted_ask);

    const double total_depth = total_bid + total_ask;

    result.depth_imbalance =
        total_depth > 0.0
            ? static_cast<float>(
                (total_bid - total_ask) / total_depth
              )
            : 0.0F;

    const double weighted_total =
        weighted_bid + weighted_ask;

    result.weighted_depth_imbalance =
        weighted_total > 0.0
            ? static_cast<float>(
                (weighted_bid - weighted_ask) /
                weighted_total
              )
            : 0.0F;

    if (weighted_bid_qty_for_price > 0.0) {
        result.weighted_bid_price =
            static_cast<float>(
                weighted_bid_px_num /
                weighted_bid_qty_for_price
            );
    }

    if (weighted_ask_qty_for_price > 0.0) {
        result.weighted_ask_price =
            static_cast<float>(
                weighted_ask_px_num /
                weighted_ask_qty_for_price
            );
    }

    // Normalized weighted depth pressure in [-1, 1].
    result.liquidity_pressure =
        result.weighted_depth_imbalance;

    result.status = FeatureStatus::Valid;
    return result;
}


// ============================================================================
// TECHNICAL FEATURES
// ============================================================================

TechnicalFeatures FeatureEngine::compute_technicals(
    const std::span<const Bar> bars
) const {
    TechnicalFeatures result;

    if (bars.empty()) {
        result.status = FeatureStatus::MissingData;
        return result;
    }

    result.timestamp_ns = bars.back().timestamp_ns;

    if (!bars_valid(bars)) {
        result.status = FeatureStatus::InvalidInput;
        return result;
    }

    result.close = bars.back().close;

    result.sma_fast = sma(bars, config_.sma_fast_period);
    result.sma_slow = sma(bars, config_.sma_slow_period);
    result.sma_50 = sma(bars, config_.sma_50_period);
    result.sma_200 = sma(bars, config_.sma_200_period);

    result.ema_fast = ema(bars, config_.ema_fast_period);
    result.ema_slow = ema(bars, config_.ema_slow_period);

    result.macd = macd_line(
        bars,
        config_.ema_fast_period,
        config_.ema_slow_period
    );

    result.macd_signal = macd_signal(
        bars,
        config_.ema_fast_period,
        config_.ema_slow_period,
        config_.macd_signal_period
    );

    if (
        finite(result.macd) &&
        finite(result.macd_signal)
    ) {
        result.macd_histogram =
            result.macd - result.macd_signal;
    }

    result.rsi = rsi(bars, config_.rsi_period);
    result.roc = roc(bars, config_.roc_period);
    result.cci = cci(bars, config_.cci_period);

    result.stoch_rsi = stoch_rsi(
        bars,
        config_.rsi_period,
        config_.stoch_rsi_period
    );

    result.simple_return =
        simple_return(bars);

    result.log_return =
        log_return(bars);

    result.realized_volatility =
        realized_volatility(
            bars,
            config_.volatility_period
        );

    result.vwap = vwap(bars);

    if (finite(result.vwap) && result.vwap > 0.0F) {
        result.vwap_distance =
            result.close - result.vwap;

        result.vwap_distance_pct =
            (result.close / result.vwap) - 1.0F;
    }

    result.status =
        bars.size() >= required_warmup()
            ? FeatureStatus::Valid
            : FeatureStatus::Warmup;

    return result;
}


// ============================================================================
// COMBINED FEATURES
// ============================================================================

FeatureVector FeatureEngine::compute(
    const MarketSample& market,
    const DepthSample& depth,
    const std::span<const Bar> bars
) const {
    FeatureVector result;

    result.timestamp_ns =
        market.timestamp_ns != 0
            ? market.timestamp_ns
            : depth.timestamp_ns;

    result.l1 = compute_l1(market);
    result.l2 = compute_l2(depth);
    result.technical = compute_technicals(bars);

    if (!bars.empty()) {
        result.timestamp_ns =
            std::max(
                result.timestamp_ns,
                bars.back().timestamp_ns
            );
    }

    return result;
}


// ============================================================================
// BATCH FEATURES
// ============================================================================

FeatureBatch FeatureEngine::compute_batch(
    const std::span<const MarketSample> market,
    const std::span<const DepthSample> depth,
    const std::span<const std::vector<Bar>> bar_windows
) const {
    if (
        market.size() != depth.size() ||
        market.size() != bar_windows.size()
    ) {
        throw std::invalid_argument(
            "FeatureEngine::compute_batch: input batch sizes differ"
        );
    }

    FeatureBatch result;
    result.rows.reserve(market.size());

    for (std::size_t i = 0; i < market.size(); ++i) {
        const auto features = compute(
            market[i],
            depth[i],
            std::span<const Bar>(
                bar_windows[i].data(),
                bar_windows[i].size()
            )
        );

        result.rows.push_back(flatten(features));
    }

    return result;
}


// ============================================================================
// GPU FLATTENING
// ============================================================================

GpuFeatureRow FeatureEngine::flatten(
    const FeatureVector& features
) noexcept {
    GpuFeatureRow row;

    row.timestamp_ns = features.timestamp_ns;

    row.midpoint = features.l1.midpoint;
    row.spread = features.l1.spread;
    row.spread_bps = features.l1.spread_bps;
    row.top_level_imbalance =
        features.l1.top_level_imbalance;
    row.microprice = features.l1.microprice;

    row.depth_imbalance =
        features.l2.depth_imbalance;
    row.weighted_depth_imbalance =
        features.l2.weighted_depth_imbalance;
    row.liquidity_pressure =
        features.l2.liquidity_pressure;

    row.sma_fast = features.technical.sma_fast;
    row.sma_slow = features.technical.sma_slow;
    row.sma_50 = features.technical.sma_50;
    row.sma_200 = features.technical.sma_200;

    row.ema_fast = features.technical.ema_fast;
    row.ema_slow = features.technical.ema_slow;

    row.macd = features.technical.macd;
    row.macd_signal = features.technical.macd_signal;
    row.macd_histogram =
        features.technical.macd_histogram;

    row.rsi = features.technical.rsi;
    row.roc = features.technical.roc;
    row.cci = features.technical.cci;
    row.stoch_rsi = features.technical.stoch_rsi;

    row.simple_return =
        features.technical.simple_return;
    row.log_return =
        features.technical.log_return;
    row.realized_volatility =
        features.technical.realized_volatility;

    row.vwap = features.technical.vwap;
    row.vwap_distance_pct =
        features.technical.vwap_distance_pct;

    return row;
}


GpuDepthBatch FeatureEngine::flatten_depth(
    const std::span<const DepthSample> samples,
    const std::size_t levels
) {
    if (samples.empty()) {
        throw std::invalid_argument(
            "FeatureEngine::flatten_depth: samples cannot be empty"
        );
    }

    if (levels == 0) {
        throw std::invalid_argument(
            "FeatureEngine::flatten_depth: levels must be positive"
        );
    }

    if (
        samples.size() >
        std::numeric_limits<std::size_t>::max() / levels
    ) {
        throw std::overflow_error(
            "FeatureEngine::flatten_depth: flattened size overflow"
        );
    }

    GpuDepthBatch result;
    result.sample_count = samples.size();
    result.levels = levels;

    const std::size_t flat_size =
        samples.size() * levels;

    result.timestamps_ns.reserve(samples.size());

    result.bid_prices.assign(flat_size, kNaN);
    result.ask_prices.assign(flat_size, kNaN);
    result.bid_quantities.assign(flat_size, 0.0F);
    result.ask_quantities.assign(flat_size, 0.0F);

    for (std::size_t sample_i = 0;
         sample_i < samples.size();
         ++sample_i) {

        const auto& sample = samples[sample_i];

        if (!sample.valid()) {
            throw std::invalid_argument(
                "FeatureEngine::flatten_depth: invalid depth sample"
            );
        }

        result.timestamps_ns.push_back(
            sample.timestamp_ns
        );

        const auto bid_levels =
            std::min(levels, sample.bid_levels());

        const auto ask_levels =
            std::min(levels, sample.ask_levels());

        const std::size_t base =
            sample_i * levels;

        for (std::size_t level = 0;
             level < bid_levels;
             ++level) {

            if (sample.bid_price_ticks[level] <= 0) {
                throw std::invalid_argument(
                    "FeatureEngine::flatten_depth: invalid bid price"
                );
            }

            result.bid_prices[base + level] =
                ticks_to_float(
                    sample.bid_price_ticks[level]
                );

            result.bid_quantities[base + level] =
                static_cast<float>(
                    sample.bid_quantities[level]
                );
        }

        for (std::size_t level = 0;
             level < ask_levels;
             ++level) {

            if (sample.ask_price_ticks[level] <= 0) {
                throw std::invalid_argument(
                    "FeatureEngine::flatten_depth: invalid ask price"
                );
            }

            result.ask_prices[base + level] =
                ticks_to_float(
                    sample.ask_price_ticks[level]
                );

            result.ask_quantities[base + level] =
                static_cast<float>(
                    sample.ask_quantities[level]
                );
        }
    }

    return result;
}


// ============================================================================
// BASIC NUMERICAL HELPERS
// ============================================================================

float FeatureEngine::ticks_to_float(
    const std::int64_t ticks
) noexcept {
    return static_cast<float>(ticks);
}


float FeatureEngine::safe_ratio(
    const float numerator,
    const float denominator
) noexcept {
    if (
        !finite(numerator) ||
        !finite(denominator) ||
        denominator == 0.0F
    ) {
        return quiet_nan();
    }

    return numerator / denominator;
}


// ============================================================================
// SMA
// ============================================================================

float FeatureEngine::sma(
    const std::span<const Bar> bars,
    const std::size_t period
) noexcept {
    if (period == 0 || bars.size() < period) {
        return quiet_nan();
    }

    double sum = 0.0;

    const std::size_t start =
        bars.size() - period;

    for (std::size_t i = start; i < bars.size(); ++i) {
        sum += static_cast<double>(bars[i].close);
    }

    return static_cast<float>(
        sum / static_cast<double>(period)
    );
}


// ============================================================================
// EMA
// ============================================================================

float FeatureEngine::ema(
    const std::span<const Bar> bars,
    const std::size_t period
) noexcept {
    if (period == 0 || bars.size() < period) {
        return quiet_nan();
    }

    double initial = 0.0;

    for (std::size_t i = 0; i < period; ++i) {
        initial += static_cast<double>(bars[i].close);
    }

    double value =
        initial / static_cast<double>(period);

    const double alpha =
        2.0 / (static_cast<double>(period) + 1.0);

    for (std::size_t i = period; i < bars.size(); ++i) {
        const double close =
            static_cast<double>(bars[i].close);

        value =
            alpha * close +
            (1.0 - alpha) * value;
    }

    return static_cast<float>(value);
}


float FeatureEngine::ema_values(
    const std::span<const float> values,
    const std::size_t period
) noexcept {
    if (period == 0 || values.size() < period) {
        return quiet_nan();
    }

    double initial = 0.0;

    for (std::size_t i = 0; i < period; ++i) {
        if (!finite(values[i])) {
            return quiet_nan();
        }

        initial += static_cast<double>(values[i]);
    }

    double value =
        initial / static_cast<double>(period);

    const double alpha =
        2.0 / (static_cast<double>(period) + 1.0);

    for (std::size_t i = period; i < values.size(); ++i) {
        if (!finite(values[i])) {
            return quiet_nan();
        }

        value =
            alpha * static_cast<double>(values[i]) +
            (1.0 - alpha) * value;
    }

    return static_cast<float>(value);
}


// ============================================================================
// RSI
// ============================================================================

float FeatureEngine::rsi(
    const std::span<const Bar> bars,
    const std::size_t period
) noexcept {
    if (
        period == 0 ||
        bars.size() < period + 1
    ) {
        return quiet_nan();
    }

    double gain_sum = 0.0;
    double loss_sum = 0.0;

    for (std::size_t i = 1; i <= period; ++i) {
        const double delta =
            static_cast<double>(bars[i].close) -
            static_cast<double>(bars[i - 1].close);

        if (delta > 0.0) {
            gain_sum += delta;
        } else {
            loss_sum -= delta;
        }
    }

    double avg_gain =
        gain_sum / static_cast<double>(period);

    double avg_loss =
        loss_sum / static_cast<double>(period);

    for (std::size_t i = period + 1;
         i < bars.size();
         ++i) {

        const double delta =
            static_cast<double>(bars[i].close) -
            static_cast<double>(bars[i - 1].close);

        const double gain =
            delta > 0.0 ? delta : 0.0;

        const double loss =
            delta < 0.0 ? -delta : 0.0;

        avg_gain =
            ((avg_gain * static_cast<double>(period - 1)) +
             gain) /
            static_cast<double>(period);

        avg_loss =
            ((avg_loss * static_cast<double>(period - 1)) +
             loss) /
            static_cast<double>(period);
    }

    if (avg_loss == 0.0) {
        return avg_gain == 0.0
            ? 50.0F
            : 100.0F;
    }

    if (avg_gain == 0.0) {
        return 0.0F;
    }

    const double rs = avg_gain / avg_loss;

    return static_cast<float>(
        100.0 - (100.0 / (1.0 + rs))
    );
}


// ============================================================================
// ROC
// ============================================================================

float FeatureEngine::roc(
    const std::span<const Bar> bars,
    const std::size_t period
) noexcept {
    if (
        period == 0 ||
        bars.size() <= period
    ) {
        return quiet_nan();
    }

    const float previous =
        bars[bars.size() - 1 - period].close;

    const float current =
        bars.back().close;

    if (previous <= 0.0F) {
        return quiet_nan();
    }

    return ((current / previous) - 1.0F) * 100.0F;
}


// ============================================================================
// CCI
// ============================================================================

float FeatureEngine::cci(
    const std::span<const Bar> bars,
    const std::size_t period
) noexcept {
    if (period == 0 || bars.size() < period) {
        return quiet_nan();
    }

    const std::size_t start =
        bars.size() - period;

    double mean = 0.0;

    for (std::size_t i = start; i < bars.size(); ++i) {
        mean += static_cast<double>(
            typical_price(bars[i])
        );
    }

    mean /= static_cast<double>(period);

    double mean_deviation = 0.0;

    for (std::size_t i = start; i < bars.size(); ++i) {
        mean_deviation += std::abs(
            static_cast<double>(
                typical_price(bars[i])
            ) - mean
        );
    }

    mean_deviation /=
        static_cast<double>(period);

    if (mean_deviation == 0.0) {
        return 0.0F;
    }

    const double current =
        static_cast<double>(
            typical_price(bars.back())
        );

    return static_cast<float>(
        (current - mean) /
        (0.015 * mean_deviation)
    );
}


// ============================================================================
// STOCHASTIC RSI
// ============================================================================

float FeatureEngine::stoch_rsi(
    const std::span<const Bar> bars,
    const std::size_t rsi_period,
    const std::size_t stoch_period
) {
    if (
        rsi_period == 0 ||
        stoch_period == 0
    ) {
        return quiet_nan();
    }

    const auto series =
        rsi_series(bars, rsi_period);

    if (series.size() < stoch_period) {
        return quiet_nan();
    }

    const auto start =
        series.end() -
        static_cast<std::ptrdiff_t>(stoch_period);

    float minimum =
        std::numeric_limits<float>::infinity();

    float maximum =
        -std::numeric_limits<float>::infinity();

    for (auto it = start; it != series.end(); ++it) {
        if (!finite(*it)) {
            return quiet_nan();
        }

        minimum = std::min(minimum, *it);
        maximum = std::max(maximum, *it);
    }

    const float current = series.back();

    if (maximum == minimum) {
        return 0.0F;
    }

    return
        ((current - minimum) /
         (maximum - minimum)) *
        100.0F;
}


// ============================================================================
// REALIZED VOLATILITY
// ============================================================================

float FeatureEngine::realized_volatility(
    const std::span<const Bar> bars,
    const std::size_t period
) noexcept {
    if (
        period < 2 ||
        bars.size() < period + 1
    ) {
        return quiet_nan();
    }

    const std::size_t first_return =
        bars.size() - period;

    double mean = 0.0;

    for (std::size_t i = first_return;
         i < bars.size();
         ++i) {

        const double previous =
            static_cast<double>(bars[i - 1].close);

        const double current =
            static_cast<double>(bars[i].close);

        if (previous <= 0.0 || current <= 0.0) {
            return quiet_nan();
        }

        mean += std::log(current / previous);
    }

    mean /= static_cast<double>(period);

    double variance_sum = 0.0;

    for (std::size_t i = first_return;
         i < bars.size();
         ++i) {

        const double value =
            std::log(
                static_cast<double>(bars[i].close) /
                static_cast<double>(bars[i - 1].close)
            );

        const double deviation = value - mean;
        variance_sum += deviation * deviation;
    }

    const double variance =
        variance_sum /
        static_cast<double>(period - 1);

    return static_cast<float>(
        std::sqrt(std::max(0.0, variance))
    );
}


// ============================================================================
// VWAP
// ============================================================================

float FeatureEngine::vwap(
    const std::span<const Bar> bars
) noexcept {
    if (bars.empty()) {
        return quiet_nan();
    }

    double price_volume = 0.0;
    double volume_sum = 0.0;

    for (const auto& bar : bars) {
        if (
            !finite(bar.volume) ||
            bar.volume < 0.0F
        ) {
            return quiet_nan();
        }

        const double volume =
            static_cast<double>(bar.volume);

        price_volume +=
            static_cast<double>(
                typical_price(bar)
            ) * volume;

        volume_sum += volume;
    }

    if (volume_sum == 0.0) {
        return quiet_nan();
    }

    return static_cast<float>(
        price_volume / volume_sum
    );
}


// ============================================================================
// RETURNS
// ============================================================================

float FeatureEngine::simple_return(
    const std::span<const Bar> bars
) noexcept {
    if (bars.size() < 2) {
        return quiet_nan();
    }

    const float previous =
        bars[bars.size() - 2].close;

    const float current =
        bars.back().close;

    if (previous <= 0.0F) {
        return quiet_nan();
    }

    return (current / previous) - 1.0F;
}


float FeatureEngine::log_return(
    const std::span<const Bar> bars
) noexcept {
    if (bars.size() < 2) {
        return quiet_nan();
    }

    const double previous =
        static_cast<double>(
            bars[bars.size() - 2].close
        );

    const double current =
        static_cast<double>(
            bars.back().close
        );

    if (previous <= 0.0 || current <= 0.0) {
        return quiet_nan();
    }

    return static_cast<float>(
        std::log(current / previous)
    );
}


// ============================================================================
// MACD
// ============================================================================

float FeatureEngine::macd_line(
    const std::span<const Bar> bars,
    const std::size_t fast_period,
    const std::size_t slow_period
) noexcept {
    if (
        fast_period == 0 ||
        slow_period == 0 ||
        bars.size() < std::max(
            fast_period,
            slow_period
        )
    ) {
        return quiet_nan();
    }

    const float fast =
        ema(bars, fast_period);

    const float slow =
        ema(bars, slow_period);

    if (!finite(fast) || !finite(slow)) {
        return quiet_nan();
    }

    return fast - slow;
}


float FeatureEngine::macd_signal(
    const std::span<const Bar> bars,
    const std::size_t fast_period,
    const std::size_t slow_period,
    const std::size_t signal_period
) {
    if (
        fast_period == 0 ||
        slow_period == 0 ||
        signal_period == 0
    ) {
        return quiet_nan();
    }

    const auto series =
        macd_series(
            bars,
            fast_period,
            slow_period
        );

    if (series.size() < signal_period) {
        return quiet_nan();
    }

    return ema_values(
        std::span<const float>(
            series.data(),
            series.size()
        ),
        signal_period
    );
}


// ============================================================================
// RSI SERIES
// ============================================================================

std::vector<float> FeatureEngine::rsi_series(
    const std::span<const Bar> bars,
    const std::size_t period
) {
    std::vector<float> output;

    if (
        period == 0 ||
        bars.size() < period + 1
    ) {
        return output;
    }

    output.reserve(
        bars.size() - period
    );

    double gain_sum = 0.0;
    double loss_sum = 0.0;

    for (std::size_t i = 1; i <= period; ++i) {
        const double delta =
            static_cast<double>(bars[i].close) -
            static_cast<double>(bars[i - 1].close);

        if (delta > 0.0) {
            gain_sum += delta;
        } else {
            loss_sum -= delta;
        }
    }

    double avg_gain =
        gain_sum / static_cast<double>(period);

    double avg_loss =
        loss_sum / static_cast<double>(period);

    auto append_rsi =
        [&output](const double gain,
                  const double loss) {

            if (loss == 0.0) {
                output.push_back(
                    gain == 0.0
                        ? 50.0F
                        : 100.0F
                );
                return;
            }

            if (gain == 0.0) {
                output.push_back(0.0F);
                return;
            }

            const double rs = gain / loss;

            output.push_back(
                static_cast<float>(
                    100.0 -
                    (100.0 / (1.0 + rs))
                )
            );
        };

    append_rsi(avg_gain, avg_loss);

    for (std::size_t i = period + 1;
         i < bars.size();
         ++i) {

        const double delta =
            static_cast<double>(bars[i].close) -
            static_cast<double>(bars[i - 1].close);

        const double gain =
            delta > 0.0 ? delta : 0.0;

        const double loss =
            delta < 0.0 ? -delta : 0.0;

        avg_gain =
            ((avg_gain * static_cast<double>(period - 1)) +
             gain) /
            static_cast<double>(period);

        avg_loss =
            ((avg_loss * static_cast<double>(period - 1)) +
             loss) /
            static_cast<double>(period);

        append_rsi(avg_gain, avg_loss);
    }

    return output;
}


// ============================================================================
// MACD SERIES
// ============================================================================

std::vector<float> FeatureEngine::macd_series(
    const std::span<const Bar> bars,
    const std::size_t fast_period,
    const std::size_t slow_period
) {
    std::vector<float> output;

    if (
        fast_period == 0 ||
        slow_period == 0 ||
        bars.empty()
    ) {
        return output;
    }

    const std::size_t warmup =
        std::max(fast_period, slow_period);

    if (bars.size() < warmup) {
        return output;
    }

    // Build an aligned MACD series beginning when both EMAs have enough
    // observations. Each point intentionally uses the same CPU reference EMA
    // implementation as macd_line(), making this suitable as a Metal oracle.
    output.reserve(
        bars.size() - warmup + 1
    );

    for (std::size_t end = warmup;
         end <= bars.size();
         ++end) {

        const auto window =
            bars.first(end);

        const float fast =
            ema(window, fast_period);

        const float slow =
            ema(window, slow_period);

        if (!finite(fast) || !finite(slow)) {
            output.push_back(quiet_nan());
        } else {
            output.push_back(fast - slow);
        }
    }

    return output;
}


// ============================================================================
// BAR VALIDATION
// ============================================================================

bool FeatureEngine::bars_valid(
    const std::span<const Bar> bars
) noexcept {
    if (bars.empty()) {
        return false;
    }

    std::uint64_t previous_timestamp = 0;

    for (const auto& bar : bars) {
        if (!bar.valid()) {
            return false;
        }

        if (
            previous_timestamp != 0 &&
            bar.timestamp_ns <= previous_timestamp
        ) {
            return false;
        }

        previous_timestamp = bar.timestamp_ns;
    }

    return true;
}


// ============================================================================
// WARMUP
// ============================================================================

std::size_t FeatureEngine::required_warmup() const noexcept {
    const std::size_t macd_warmup =
        std::max(
            config_.ema_fast_period,
            config_.ema_slow_period
        ) +
        config_.macd_signal_period -
        1;

    const std::size_t rsi_warmup =
        config_.rsi_period + 1;

    const std::size_t stoch_rsi_warmup =
        config_.rsi_period +
        config_.stoch_rsi_period;

    const std::size_t roc_warmup =
        config_.roc_period + 1;

    const std::size_t volatility_warmup =
        config_.volatility_period + 1;

    return std::max({
        config_.sma_fast_period,
        config_.sma_slow_period,
        config_.sma_50_period,
        config_.sma_200_period,
        config_.ema_fast_period,
        config_.ema_slow_period,
        macd_warmup,
        rsi_warmup,
        stoch_rsi_warmup,
        roc_warmup,
        config_.cci_period,
        volatility_warmup
    });
}


// ============================================================================
// FEATURE STATUS TEXT
// ============================================================================

std::string_view to_string(
    const FeatureStatus status
) noexcept {
    switch (status) {
        case FeatureStatus::Valid:
            return "valid";

        case FeatureStatus::Warmup:
            return "warmup";

        case FeatureStatus::MissingData:
            return "missing_data";

        case FeatureStatus::InvalidInput:
            return "invalid_input";
    }

    return "unknown";
}

} // namespace trading::analytics
