#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace trading {

// ============================================================================
// ORDERBOOK / LIVE MARKET-DATA BOUNDARY
// ============================================================================
//
// This header deliberately separates:
//
//   1. OBSERVED LIVE MARKET STATE
//      Alpaca SIP quote/trade events normalized by Python and pushed into
//      ObservedMarketBook.
//
//   2. SIMULATED / INTERNAL ORDER STATE
//      LimitOrderBook owns our orders, queue priority, amendments,
//      cancellations, and simulated matching.
//
//   3. COMPOSITION
//      OrderBookEngine exposes both through one symbol-scoped interface.
//
// Alpaca credentials, WebSocket objects, JSON, TLS, reconnect logic, and API
// keys MUST NOT enter this C++ boundary.
//
// Python mapping:
//
//   QuoteEvent
//       -> MarketQuoteUpdate
//       -> OrderBookEngine::on_market_quote()
//
//   TradeEvent
//       -> MarketTradeUpdate
//       -> OrderBookEngine::on_market_trade()
//
//   BarEvent
//       -> FeatureEngine / bar pipeline
//
// IMPORTANT:
// Alpaca SIP provides consolidated quotes/trades. It is not a complete
// exchange-by-exchange Level-II depth feed. Therefore incoming SIP quotes
// update ObservedMarketBook and are NOT inserted as resting orders into
// LimitOrderBook.
//
// ============================================================================


// ============================================================================
// ENUMS
// ============================================================================

enum class Side : std::uint8_t {
    Buy,
    Sell
};

enum class OrderStatus : std::uint8_t {
    New,
    Accepted,
    PartiallyFilled,
    Filled,
    Cancelled,
    Rejected
};

enum class MarketDataStatus : std::uint8_t {
    Empty,
    Valid,
    Stale,
    Invalid,
    OutOfSequence
};


// ============================================================================
// LIVE MARKET-DATA UPDATE TYPES
// ============================================================================
//
// These are intentionally primitive / pybind11-friendly structures.
//
// They mirror the normalized Python dataclasses from alpaca_sip_stream.py.
// No provider-specific JSON structure is exposed below this layer.
// ============================================================================

struct MarketQuoteUpdate {
    std::string symbol;

    std::uint64_t timestamp_ns{0};
    std::uint64_t received_ns{0};

    std::int64_t bid_price_ticks{0};
    std::uint64_t bid_size{0};
    std::string bid_exchange;

    std::int64_t ask_price_ticks{0};
    std::uint64_t ask_size{0};
    std::string ask_exchange;

    // Optional monotonically increasing local/provider sequence.
    //
    // Alpaca normalized Python ingestion may populate this with a locally
    // assigned receive sequence if the source message does not expose a
    // directly useful sequence number.
    std::uint64_t sequence{0};

    // Provider label supplied by MarketDataNormalizer (e.g. "alpaca_sip").
    std::string provider;

    [[nodiscard]]
    bool valid() const noexcept {
        return !symbol.empty() &&
               timestamp_ns != 0 &&
               bid_price_ticks > 0 &&
               ask_price_ticks > 0 &&
               bid_price_ticks < ask_price_ticks;
    }

    [[nodiscard]]
    std::int64_t midpoint_ticks() const noexcept {
        return bid_price_ticks +
               ((ask_price_ticks - bid_price_ticks) / 2);
    }

    [[nodiscard]]
    std::int64_t spread_ticks() const noexcept {
        return ask_price_ticks - bid_price_ticks;
    }
};


struct MarketTradeUpdate {
    std::string symbol;

    std::uint64_t timestamp_ns{0};
    std::uint64_t received_ns{0};

    std::uint64_t trade_id{0};

    std::string exchange;

    std::int64_t price_ticks{0};
    std::uint64_t size{0};

    std::uint64_t sequence{0};

    std::string provider;

    [[nodiscard]]
    bool valid() const noexcept {
        return !symbol.empty() &&
               timestamp_ns != 0 &&
               price_ticks > 0 &&
               size > 0;
    }
};


// ============================================================================
// OBSERVED MARKET SNAPSHOT
// ============================================================================

struct ObservedMarketSnapshot {
    std::string symbol;

    MarketDataStatus status{MarketDataStatus::Empty};

    std::uint64_t version{0};

    std::uint64_t last_quote_timestamp_ns{0};
    std::uint64_t last_trade_timestamp_ns{0};

    std::uint64_t last_quote_received_ns{0};
    std::uint64_t last_trade_received_ns{0};

    std::optional<std::int64_t> bid_price_ticks;
    std::optional<std::uint64_t> bid_size;
    std::string bid_exchange;

    std::optional<std::int64_t> ask_price_ticks;
    std::optional<std::uint64_t> ask_size;
    std::string ask_exchange;

    std::optional<std::int64_t> last_trade_price_ticks;
    std::optional<std::uint64_t> last_trade_size;
    std::optional<std::uint64_t> last_trade_id;
    std::string last_trade_exchange;

    std::uint64_t quote_updates{0};
    std::uint64_t trade_updates{0};

    std::uint64_t rejected_updates{0};
    std::uint64_t out_of_sequence_updates{0};

    [[nodiscard]]
    bool has_quote() const noexcept {
        return bid_price_ticks.has_value() &&
               ask_price_ticks.has_value();
    }

    [[nodiscard]]
    bool has_trade() const noexcept {
        return last_trade_price_ticks.has_value();
    }

    [[nodiscard]]
    bool quote_valid() const noexcept {
        return has_quote() &&
               *bid_price_ticks > 0 &&
               *ask_price_ticks > 0 &&
               *bid_price_ticks < *ask_price_ticks;
    }

    [[nodiscard]]
    std::optional<std::int64_t>
    midpoint_ticks() const noexcept {
        if (!quote_valid()) {
            return std::nullopt;
        }

        return *bid_price_ticks +
               ((*ask_price_ticks - *bid_price_ticks) / 2);
    }

    [[nodiscard]]
    std::optional<std::int64_t>
    spread_ticks() const noexcept {
        if (!quote_valid()) {
            return std::nullopt;
        }

        return *ask_price_ticks - *bid_price_ticks;
    }
};


// ============================================================================
// OBSERVED MARKET BOOK
// ============================================================================
//
// Stores the latest consolidated SIP quote/trade state.
//
// This is NOT a matching engine.
//
// Threading:
//   - updates acquire exclusive lock
//   - snapshots/readers acquire shared lock
//
// Sequence behavior:
//   - timestamp regression is rejected by default
//   - nonzero sequence regression is rejected
//
// This protects downstream features from accidentally processing obviously
// older updates after reconnects or asynchronous queue delivery.
// ============================================================================

class ObservedMarketBook {
public:
    explicit ObservedMarketBook(std::string symbol);

    ObservedMarketBook(
        const ObservedMarketBook&
    ) = delete;

    ObservedMarketBook& operator=(
        const ObservedMarketBook&
    ) = delete;

    ObservedMarketBook(
        ObservedMarketBook&&
    ) = delete;

    ObservedMarketBook& operator=(
        ObservedMarketBook&&
    ) = delete;

    ~ObservedMarketBook() = default;

    // ------------------------------------------------------------------------
    // LIVE INGESTION
    // ------------------------------------------------------------------------

    [[nodiscard]]
    bool on_quote(
        const MarketQuoteUpdate& update
    );

    [[nodiscard]]
    bool on_trade(
        const MarketTradeUpdate& update
    );

    // ------------------------------------------------------------------------
    // READS
    // ------------------------------------------------------------------------

    [[nodiscard]]
    ObservedMarketSnapshot snapshot() const;

    [[nodiscard]]
    std::optional<MarketQuoteUpdate>
    last_quote() const;

    [[nodiscard]]
    std::optional<MarketTradeUpdate>
    last_trade() const;

    [[nodiscard]]
    std::optional<std::int64_t>
    best_bid_ticks() const;

    [[nodiscard]]
    std::optional<std::int64_t>
    best_ask_ticks() const;

    [[nodiscard]]
    std::optional<std::int64_t>
    midpoint_ticks() const;

    [[nodiscard]]
    std::optional<std::int64_t>
    spread_ticks() const;

    [[nodiscard]]
    bool has_valid_quote() const;

    [[nodiscard]]
    const std::string& symbol() const noexcept {
        return symbol_;
    }

    [[nodiscard]]
    std::uint64_t version() const noexcept {
        return version_.load(
            std::memory_order_acquire
        );
    }

    void clear();

private:
    std::string symbol_;

    mutable std::shared_mutex mutex_;

    std::optional<MarketQuoteUpdate> last_quote_;
    std::optional<MarketTradeUpdate> last_trade_;

    std::uint64_t last_quote_sequence_{0};
    std::uint64_t last_trade_sequence_{0};

    std::uint64_t quote_updates_{0};
    std::uint64_t trade_updates_{0};

    std::uint64_t rejected_updates_{0};
    std::uint64_t out_of_sequence_updates_{0};

    std::atomic<std::uint64_t> version_{0};

    [[nodiscard]]
    bool symbol_matches(
        std::string_view symbol
    ) const noexcept;

    [[nodiscard]]
    bool quote_in_sequence(
        const MarketQuoteUpdate& update
    ) const noexcept;

    [[nodiscard]]
    bool trade_in_sequence(
        const MarketTradeUpdate& update
    ) const noexcept;
};


// ============================================================================
// INTERNAL ORDER
// ============================================================================

struct Order {
    std::uint64_t id{0};

    std::string symbol;

    Side side{Side::Buy};

    // Current total requested quantity.
    std::uint64_t quantity{0};

    // Cumulative executed quantity.
    std::uint64_t filled_quantity{0};

    // quantity - filled_quantity.
    std::uint64_t remaining_quantity{0};

    // Integer ticks. Never floating point at the execution boundary.
    std::int64_t price_ticks{0};

    // Monotonic internal queue sequence.
    std::uint64_t sequence{0};

    OrderStatus status{OrderStatus::New};

    [[nodiscard]]
    bool is_terminal() const noexcept {
        return status == OrderStatus::Filled ||
               status == OrderStatus::Cancelled ||
               status == OrderStatus::Rejected;
    }

    [[nodiscard]]
    bool is_active() const noexcept {
        return !is_terminal() &&
               remaining_quantity > 0;
    }
};


// ============================================================================
// EXECUTION
// ============================================================================

struct Execution {
    std::uint64_t execution_id{0};

    // Resting order.
    std::uint64_t maker_id{0};

    // Incoming/aggressing order.
    std::uint64_t taker_id{0};

    std::string symbol;

    Side aggressor_side{Side::Buy};

    std::uint64_t quantity{0};

    // Execution occurs at resting/maker price.
    std::int64_t price_ticks{0};

    // Internal monotonic event sequence.
    std::uint64_t sequence{0};
};


// ============================================================================
// PRICE LEVEL SNAPSHOT
// ============================================================================

struct PriceLevelSnapshot {
    std::int64_t price_ticks{0};

    std::uint64_t total_quantity{0};

    std::size_t order_count{0};
};


// ============================================================================
// SIMULATED BOOK SNAPSHOT
// ============================================================================

struct BookSnapshot {
    std::string symbol;

    std::optional<std::int64_t> best_bid_ticks;
    std::optional<std::int64_t> best_ask_ticks;

    std::vector<PriceLevelSnapshot> bids;
    std::vector<PriceLevelSnapshot> asks;

    std::size_t active_order_count{0};

    std::uint64_t version{0};
};


// ============================================================================
// COMBINED SNAPSHOT
// ============================================================================
//
// Gives analytics one atomic-ish logical view of:
//
//   - current observed SIP/NBBO state
//   - current internal/simulated order state
//
// The two components have independent versions because they are separate
// authorities and may update concurrently.
// ============================================================================

struct OrderBookEngineSnapshot {
    std::string symbol;

    ObservedMarketSnapshot market;
    BookSnapshot simulated;

    std::uint64_t captured_ns{0};
};


// ============================================================================
// LIMIT ORDER BOOK
// ============================================================================
//
// Price-time-priority matching engine for OUR simulated/internal orders.
//
// It does not:
//   - authenticate to Alpaca
//   - parse WebSocket messages
//   - treat SIP quote sizes as executable local queue depth
//   - mutate itself from external quote/trade updates
//
// Ordering:
//   * Bids: highest price first.
//   * Asks: lowest price first.
//   * Orders at same price: FIFO.
//
// Amendments:
//   * same price + quantity reduction => retains priority
//   * same price + quantity increase  => loses priority
//   * price change                    => loses priority, may execute
//   * total quantity == filled qty    => cancels remainder
//
// IDs:
//   * order IDs are never reusable during the life of the book
//
// Threading:
//   * public mutating and read operations are internally synchronized
// ============================================================================

class LimitOrderBook {
public:
    explicit LimitOrderBook(std::string symbol);

    LimitOrderBook(
        const LimitOrderBook&
    ) = delete;

    LimitOrderBook& operator=(
        const LimitOrderBook&
    ) = delete;

    LimitOrderBook(
        LimitOrderBook&&
    ) = delete;

    LimitOrderBook& operator=(
        LimitOrderBook&&
    ) = delete;

    ~LimitOrderBook() = default;

    // ------------------------------------------------------------------------
    // ORDER ENTRY
    // ------------------------------------------------------------------------

    [[nodiscard]]
    std::vector<Execution> add_limit(
        std::uint64_t id,
        const std::string& symbol,
        Side side,
        std::uint64_t quantity,
        std::int64_t price_ticks
    );

    // ------------------------------------------------------------------------
    // AMEND
    // ------------------------------------------------------------------------

    [[nodiscard]]
    std::vector<Execution> amend(
        std::uint64_t order_id,
        std::int64_t new_price_ticks,
        std::uint64_t new_total_quantity
    );

    // ------------------------------------------------------------------------
    // CANCEL
    // ------------------------------------------------------------------------

    [[nodiscard]]
    Order cancel(
        std::uint64_t order_id
    );

    // ------------------------------------------------------------------------
    // LOOKUP
    // ------------------------------------------------------------------------

    [[nodiscard]]
    std::optional<Order> get(
        std::uint64_t order_id
    ) const;

    [[nodiscard]]
    bool contains(
        std::uint64_t order_id
    ) const;

    [[nodiscard]]
    bool order_id_seen(
        std::uint64_t order_id
    ) const;

    // ------------------------------------------------------------------------
    // TOP OF BOOK
    // ------------------------------------------------------------------------

    [[nodiscard]]
    std::optional<std::int64_t>
    best_bid_ticks() const;

    [[nodiscard]]
    std::optional<std::int64_t>
    best_ask_ticks() const;

    [[nodiscard]]
    std::optional<Order>
    best_bid_order() const;

    [[nodiscard]]
    std::optional<Order>
    best_ask_order() const;

    // ------------------------------------------------------------------------
    // ITERATION / SNAPSHOTS
    // ------------------------------------------------------------------------

    [[nodiscard]]
    std::vector<Order> bids() const;

    [[nodiscard]]
    std::vector<Order> asks() const;

    [[nodiscard]]
    std::vector<Order> orders() const;

    [[nodiscard]]
    BookSnapshot snapshot(
        std::size_t max_levels = 0
    ) const;

    // ------------------------------------------------------------------------
    // STATE
    // ------------------------------------------------------------------------

    [[nodiscard]]
    std::size_t size() const;

    [[nodiscard]]
    bool empty() const;

    [[nodiscard]]
    const std::string& symbol() const noexcept {
        return symbol_;
    }

    [[nodiscard]]
    std::uint64_t version() const noexcept {
        return version_.load(
            std::memory_order_acquire
        );
    }

    void clear();

private:
    using OrderPtr =
        std::shared_ptr<Order>;

    using PriceLevel =
        std::list<OrderPtr>;

    using BidBook =
        std::map<
            std::int64_t,
            PriceLevel,
            std::greater<std::int64_t>
        >;

    using AskBook =
        std::map<
            std::int64_t,
            PriceLevel,
            std::less<std::int64_t>
        >;

    struct Locator {
        Side side{Side::Buy};

        std::int64_t price_ticks{0};

        PriceLevel::iterator iterator;
    };

    std::string symbol_;

    BidBook bids_;
    AskBook asks_;

    std::unordered_map<
        std::uint64_t,
        Locator
    > locators_;

    // Prevent ID reuse after fill/cancel.
    std::unordered_set<
        std::uint64_t
    > seen_order_ids_;

    std::uint64_t next_order_sequence_{1};
    std::uint64_t next_execution_id_{1};
    std::uint64_t next_event_sequence_{1};

    std::atomic<std::uint64_t> version_{0};

    mutable std::mutex mutex_;

    // ------------------------------------------------------------------------
    // VALIDATION
    // ------------------------------------------------------------------------

    void validate_new_order(
        std::uint64_t id,
        const std::string& symbol,
        std::uint64_t quantity,
        std::int64_t price_ticks
    ) const;

    // ------------------------------------------------------------------------
    // RESTING ORDER MANAGEMENT
    // ------------------------------------------------------------------------

    void insert_resting(
        const OrderPtr& order
    );

    void remove_from_book(
        std::uint64_t order_id
    );

    // ------------------------------------------------------------------------
    // MATCHING
    // ------------------------------------------------------------------------

    [[nodiscard]]
    std::vector<Execution> match_and_rest(
        const OrderPtr& incoming
    );

    void match_buy(
        const OrderPtr& incoming,
        std::vector<Execution>& executions
    );

    void match_sell(
        const OrderPtr& incoming,
        std::vector<Execution>& executions
    );

    [[nodiscard]]
    Execution make_execution(
        const OrderPtr& maker,
        const OrderPtr& taker,
        std::uint64_t quantity,
        std::int64_t price_ticks
    );

    // ------------------------------------------------------------------------
    // SNAPSHOT HELPERS
    // ------------------------------------------------------------------------

    [[nodiscard]]
    static std::uint64_t
    aggregate_level_quantity(
        const PriceLevel& level
    );

    [[nodiscard]]
    static PriceLevelSnapshot
    make_level_snapshot(
        std::int64_t price_ticks,
        const PriceLevel& level
    );
};


// ============================================================================
// ORDER BOOK ENGINE
// ============================================================================
//
// Symbol-scoped composition root.
//
// This is the C++ object that the Python Alpaca stream should feed.
//
// LIVE DATA PATH:
//
//     AlpacaSIPStream
//           |
//           | QuoteEvent / TradeEvent
//           v
//     Python/C++ binding
//           |
//           | MarketQuoteUpdate / MarketTradeUpdate
//           v
//     OrderBookEngine
//           |
//           +----> ObservedMarketBook
//           |
//           +----> FeatureEngine / RiskEngine readers
//
// ORDER PATH:
//
//     TradingStrategy
//           |
//           v
//       RiskEngine
//           |
//        ACCEPT
//           |
//           v
//     LimitOrderBook
//
// Notice that on_market_quote()/on_market_trade() NEVER place, amend,
// cancel, or execute our own orders.
// ============================================================================

class OrderBookEngine {
public:
    explicit OrderBookEngine(
        std::string symbol
    );

    OrderBookEngine(
        const OrderBookEngine&
    ) = delete;

    OrderBookEngine& operator=(
        const OrderBookEngine&
    ) = delete;

    OrderBookEngine(
        OrderBookEngine&&
    ) = delete;

    OrderBookEngine& operator=(
        OrderBookEngine&&
    ) = delete;

    ~OrderBookEngine() = default;

    // ------------------------------------------------------------------------
    // LIVE SIP MARKET DATA
    // ------------------------------------------------------------------------

    [[nodiscard]]
    bool on_market_quote(
        const MarketQuoteUpdate& update
    );

    [[nodiscard]]
    bool on_market_trade(
        const MarketTradeUpdate& update
    );

    // ------------------------------------------------------------------------
    // INTERNAL ORDER ROUTING
    // ------------------------------------------------------------------------

    [[nodiscard]]
    std::vector<Execution> add_limit(
        std::uint64_t id,
        Side side,
        std::uint64_t quantity,
        std::int64_t price_ticks
    );

    [[nodiscard]]
    std::vector<Execution> amend(
        std::uint64_t order_id,
        std::int64_t new_price_ticks,
        std::uint64_t new_total_quantity
    );

    [[nodiscard]]
    Order cancel(
        std::uint64_t order_id
    );

    // ------------------------------------------------------------------------
    // SNAPSHOTS
    // ------------------------------------------------------------------------

    [[nodiscard]]
    ObservedMarketSnapshot
    market_snapshot() const;

    [[nodiscard]]
    BookSnapshot
    simulated_snapshot(
        std::size_t max_levels = 0
    ) const;

    [[nodiscard]]
    OrderBookEngineSnapshot snapshot(
        std::uint64_t captured_ns,
        std::size_t max_levels = 0
    ) const;

    // ------------------------------------------------------------------------
    // DIRECT COMPONENT ACCESS
    // ------------------------------------------------------------------------
    //
    // Intended for wiring FeatureEngine/RiskEngine, not for bypassing
    // authority checks in the Python orchestration layer.
    // ------------------------------------------------------------------------

    [[nodiscard]]
    const ObservedMarketBook&
    market() const noexcept {
        return market_;
    }

    [[nodiscard]]
    ObservedMarketBook&
    market() noexcept {
        return market_;
    }

    [[nodiscard]]
    const LimitOrderBook&
    simulated_book() const noexcept {
        return simulated_;
    }

    [[nodiscard]]
    LimitOrderBook&
    simulated_book() noexcept {
        return simulated_;
    }

    [[nodiscard]]
    const std::string&
    symbol() const noexcept {
        return symbol_;
    }

private:
    std::string symbol_;

    ObservedMarketBook market_;
    LimitOrderBook simulated_;
};


// ============================================================================
// MULTI-SYMBOL LIVE MARKET REGISTRY
// ============================================================================
//
// Alpaca should use one WebSocket connection for the entire universe.
//
// The Python stream can route each normalized event to a symbol-specific
// OrderBookEngine managed by this registry.
//
// This avoids:
//   - one WebSocket per symbol
//   - duplicate subscriptions
//   - provider code inside the book
// ============================================================================

class OrderBookRegistry {
public:
    OrderBookRegistry() = default;

    OrderBookRegistry(
        const OrderBookRegistry&
    ) = delete;

    OrderBookRegistry& operator=(
        const OrderBookRegistry&
    ) = delete;

    ~OrderBookRegistry() = default;

    void add_symbol(
        const std::string& symbol
    );

    [[nodiscard]]
    bool contains(
        std::string_view symbol
    ) const;

    [[nodiscard]]
    std::shared_ptr<OrderBookEngine>
    get(
        std::string_view symbol
    ) const;

    [[nodiscard]]
    bool on_market_quote(
        const MarketQuoteUpdate& update
    );

    [[nodiscard]]
    bool on_market_trade(
        const MarketTradeUpdate& update
    );

    [[nodiscard]]
    std::vector<std::string>
    symbols() const;

    [[nodiscard]]
    std::size_t size() const;

private:
    mutable std::shared_mutex mutex_;

    std::unordered_map<
        std::string,
        std::shared_ptr<OrderBookEngine>
    > books_;

};

} // namespace trading
