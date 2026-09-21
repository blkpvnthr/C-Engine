#pragma once

#include "orderbook.hpp"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <map>
#include <optional>
#include <shared_mutex>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace trading {

// ============================================================================
// MARKET DEPTH / TRADE-QUOTE OBSERVATION
// ============================================================================
//
// Observation only.
//
// This component never:
//   * creates, amends, cancels, or fills our orders
//   * mutates LimitOrderBook
//   * treats SIP NBBO as fabricated Level-II depth
//
// It accepts:
//   1. normalized consolidated MarketQuoteUpdate / MarketTradeUpdate
//   2. explicit provider-supplied MarketDepthUpdate messages when a provider
//      actually supplies depth.
//
// A SIP quote updates only the top-of-book observation. It does not create
// deeper levels.
// ============================================================================

enum class DepthUpdateAction : std::uint8_t {
    Upsert = 0,
    Delete = 1,
    ClearSide = 2,
    ClearBook = 3
};

enum class DepthSource : std::uint8_t {
    Unknown = 0,
    ConsolidatedQuote = 1,
    ProviderDepth = 2,
    Replay = 3
};

struct MarketDepthUpdate {
    std::string symbol;
    Side side{Side::Buy};
    DepthUpdateAction action{DepthUpdateAction::Upsert};

    std::int64_t price_ticks{0};
    std::uint64_t size{0};
    std::uint32_t order_count{0};

    std::string exchange;
    std::string provider;

    std::uint64_t timestamp_ns{0};
    std::uint64_t received_ns{0};
    std::uint64_t sequence{0};

    DepthSource source{DepthSource::ProviderDepth};

    [[nodiscard]]
    bool valid() const noexcept;
};

struct ObservedDepthLevel {
    std::int64_t price_ticks{0};
    std::uint64_t size{0};
    std::uint32_t order_count{0};

    std::string exchange;
    std::string provider;

    std::uint64_t timestamp_ns{0};
    std::uint64_t received_ns{0};
    std::uint64_t sequence{0};

    DepthSource source{DepthSource::Unknown};
};

struct TradeFlowSnapshot {
    std::uint64_t trade_count{0};
    std::uint64_t total_volume{0};

    std::uint64_t at_or_above_ask_volume{0};
    std::uint64_t at_or_below_bid_volume{0};
    std::uint64_t inside_spread_volume{0};
    std::uint64_t outside_quote_volume{0};

    std::optional<std::int64_t> last_trade_price_ticks;
    std::optional<std::uint64_t> last_trade_size;
    std::uint64_t last_trade_timestamp_ns{0};

    [[nodiscard]]
    std::int64_t signed_quote_volume() const noexcept;
};

struct MarketDepthSnapshot {
    std::string symbol;

    MarketDataStatus status{MarketDataStatus::Empty};
    DepthSource depth_source{DepthSource::Unknown};

    std::vector<ObservedDepthLevel> bids;
    std::vector<ObservedDepthLevel> asks;

    std::optional<MarketQuoteUpdate> last_quote;
    std::optional<MarketTradeUpdate> last_trade;

    TradeFlowSnapshot trade_flow;

    std::uint64_t version{0};
    std::uint64_t captured_ns{0};

    std::uint64_t quote_updates{0};
    std::uint64_t trade_updates{0};
    std::uint64_t depth_updates{0};

    std::uint64_t rejected_updates{0};
    std::uint64_t out_of_sequence_updates{0};

    [[nodiscard]]
    bool has_top_of_book() const noexcept;

    [[nodiscard]]
    std::optional<std::int64_t>
    best_bid_ticks() const noexcept;

    [[nodiscard]]
    std::optional<std::int64_t>
    best_ask_ticks() const noexcept;

    [[nodiscard]]
    std::optional<std::int64_t>
    midpoint_ticks() const noexcept;

    [[nodiscard]]
    std::optional<std::int64_t>
    spread_ticks() const noexcept;
};

// Contiguous fixed-depth export intended for CPU FeatureEngine / Metal upload.
// Missing levels are represented by zero price/size/count values.
struct MarketDepthSoA {
    std::string symbol;
    std::size_t depth{0};

    std::vector<std::int64_t> bid_price_ticks;
    std::vector<std::uint64_t> bid_sizes;
    std::vector<std::uint32_t> bid_order_counts;

    std::vector<std::int64_t> ask_price_ticks;
    std::vector<std::uint64_t> ask_sizes;
    std::vector<std::uint32_t> ask_order_counts;

    std::uint64_t version{0};
    std::uint64_t captured_ns{0};
};

class MarketDepthBook {
public:
    explicit MarketDepthBook(
        std::string symbol,
        std::size_t trade_window = 1024
    );

    MarketDepthBook(const MarketDepthBook&) = delete;
    MarketDepthBook& operator=(const MarketDepthBook&) = delete;

    // Consolidated quote observation. This updates only top-of-book state.
    [[nodiscard]]
    bool on_quote(const MarketQuoteUpdate& update);

    // Trade observation and quote-relative trade-flow classification.
    [[nodiscard]]
    bool on_trade(const MarketTradeUpdate& update);

    // Explicit provider depth. Use only when the source actually provides
    // depth-of-book data.
    [[nodiscard]]
    bool on_depth(const MarketDepthUpdate& update);

    [[nodiscard]]
    MarketDepthSnapshot snapshot(
        std::size_t max_depth,
        std::uint64_t captured_ns
    ) const;

    [[nodiscard]]
    MarketDepthSoA export_soa(
        std::size_t depth,
        std::uint64_t captured_ns
    ) const;

    [[nodiscard]]
    const std::string& symbol() const noexcept {
        return symbol_;
    }

    [[nodiscard]]
    std::uint64_t version() const noexcept {
        return version_.load(std::memory_order_acquire);
    }

    void clear();

private:
    struct TradeObservation {
        MarketTradeUpdate trade;
        std::optional<std::int64_t> bid_ticks;
        std::optional<std::int64_t> ask_ticks;
    };

    using BidMap = std::map<
        std::int64_t,
        ObservedDepthLevel,
        std::greater<std::int64_t>
    >;

    using AskMap = std::map<
        std::int64_t,
        ObservedDepthLevel,
        std::less<std::int64_t>
    >;

    [[nodiscard]]
    bool validate_symbol(std::string_view symbol) const noexcept;

    [[nodiscard]]
    bool sequence_ok(
        std::uint64_t incoming,
        std::uint64_t previous
    ) const noexcept;

    void trim_trade_window_locked();

    [[nodiscard]]
    TradeFlowSnapshot trade_flow_locked() const;

    std::string symbol_;
    std::size_t trade_window_;

    mutable std::shared_mutex mutex_;

    std::optional<MarketQuoteUpdate> last_quote_;
    std::optional<MarketTradeUpdate> last_trade_;

    BidMap depth_bids_;
    AskMap depth_asks_;

    std::deque<TradeObservation> trades_;

    std::uint64_t last_quote_sequence_{0};
    std::uint64_t last_trade_sequence_{0};
    std::uint64_t last_depth_sequence_{0};

    std::uint64_t last_quote_timestamp_ns_{0};
    std::uint64_t last_trade_timestamp_ns_{0};
    std::uint64_t last_depth_timestamp_ns_{0};

    std::uint64_t quote_updates_{0};
    std::uint64_t trade_updates_{0};
    std::uint64_t depth_updates_{0};
    std::uint64_t rejected_updates_{0};
    std::uint64_t out_of_sequence_updates_{0};

    DepthSource depth_source_{DepthSource::Unknown};

    std::atomic<std::uint64_t> version_{0};
};


class MarketDepthRegistry {
public:
    explicit MarketDepthRegistry(
        std::size_t trade_window = 1024
    );

    [[nodiscard]]
    bool add_symbol(std::string symbol);

    [[nodiscard]]
    bool contains(std::string_view symbol) const;

    [[nodiscard]]
    std::shared_ptr<MarketDepthBook>
    get(std::string_view symbol) const;

    [[nodiscard]]
    bool on_quote(const MarketQuoteUpdate& update);

    [[nodiscard]]
    bool on_trade(const MarketTradeUpdate& update);

    [[nodiscard]]
    bool on_depth(const MarketDepthUpdate& update);

    [[nodiscard]]
    std::vector<std::string> symbols() const;

private:
    std::size_t trade_window_;

    mutable std::shared_mutex mutex_;

    std::unordered_map<
        std::string,
        std::shared_ptr<MarketDepthBook>
    > books_;
};

}  // namespace trading
