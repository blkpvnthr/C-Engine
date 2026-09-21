#pragma once

#include "orderbook/orderbook.hpp"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace trading {

// ============================================================================
// RISK DECISION
// ============================================================================

enum class RiskDecision : std::uint8_t {
    Accept,
    Reject
};


// ============================================================================
// RISK REJECT REASON
// ============================================================================

enum class RiskRejectReason : std::uint8_t {
    None,

    KillSwitchActive,

    InvalidOrderId,
    DuplicateOrderId,
    InvalidSymbol,
    InvalidQuantity,
    InvalidPrice,

    QuantityLimitExceeded,
    NotionalLimitExceeded,

    LongPositionLimitExceeded,
    ShortPositionLimitExceeded,
    GrossPositionLimitExceeded,
    GrossExposureLimitExceeded,

    OpenOrderLimitExceeded,
    OpenOrderQuantityLimitExceeded,

    MissingMarketData,
    StaleMarketData,
    InvalidMarketData,
    PriceCollarExceeded,

    InsufficientBuyingPower,

    Unknown
};


// ============================================================================
// MARKET STATE
// ============================================================================
//
// Market prices are represented as integer ticks, exactly as they are in the
// LimitOrderBook. This avoids floating-point price ambiguity at the execution
// and risk boundaries.
//
// timestamp_ns is expected to use a monotonic or consistently defined
// nanosecond clock chosen by the orchestration layer.
//

struct RiskMarketState {
    std::string symbol;

    std::int64_t bid_ticks{0};
    std::int64_t ask_ticks{0};

    std::optional<std::int64_t> last_ticks;

    std::uint64_t timestamp_ns{0};

    [[nodiscard]]
    bool valid() const noexcept {
        return !symbol.empty() &&
               bid_ticks > 0 &&
               ask_ticks > 0 &&
               bid_ticks < ask_ticks;
    }

    [[nodiscard]]
    std::int64_t midpoint_ticks() const noexcept {
        return bid_ticks + ((ask_ticks - bid_ticks) / 2);
    }
};


// ============================================================================
// ACCOUNT / PORTFOLIO RISK STATE
// ============================================================================
//
// Position is signed:
//
//      +100 = long 100
//      -100 = short 100
//
// cash_ticks and buying_power_ticks are monetary values expressed in the same
// smallest accounting unit selected by the application. If one price tick is
// one cent, these can also be cents.
//
// The risk engine does not own the portfolio. The authoritative portfolio /
// position manager should update this state after executions.
//

struct RiskAccountState {
    std::int64_t position{0};

    std::int64_t cash_ticks{0};
    std::int64_t buying_power_ticks{0};

    // Current absolute portfolio exposure before the proposed order.
    std::uint64_t gross_exposure_ticks{0};
};


// ============================================================================
// RISK LIMITS
// ============================================================================

struct RiskLimits {
    // ------------------------------------------------------------------------
    // Per-order limits
    // ------------------------------------------------------------------------

    std::uint64_t max_order_quantity{10'000};

    // quantity * price_ticks
    std::uint64_t max_order_notional_ticks{100'000'000};

    // ------------------------------------------------------------------------
    // Position limits
    // ------------------------------------------------------------------------

    std::int64_t max_long_position{100'000};

    // Absolute magnitude allowed on the short side.
    std::int64_t max_short_position{100'000};

    // Absolute projected position regardless of direction.
    std::uint64_t max_gross_position{100'000};

    // ------------------------------------------------------------------------
    // Portfolio exposure
    // ------------------------------------------------------------------------

    std::uint64_t max_gross_exposure_ticks{1'000'000'000};

    // ------------------------------------------------------------------------
    // Open-order limits
    // ------------------------------------------------------------------------

    std::size_t max_open_orders{10'000};

    std::uint64_t max_open_order_quantity{1'000'000};

    // ------------------------------------------------------------------------
    // Market-data freshness
    // ------------------------------------------------------------------------

    // Maximum permitted quote age in nanoseconds.
    //
    // Default: 1 second.
    std::uint64_t max_market_data_age_ns{1'000'000'000ULL};

    // ------------------------------------------------------------------------
    // Price collars
    // ------------------------------------------------------------------------

    // Maximum distance from the current reference price in basis points.
    //
    // 100 bps = 1%.
    //
    // A value of 0 disables the collar.
    std::uint32_t max_price_deviation_bps{500};

    // ------------------------------------------------------------------------
    // Buying power
    // ------------------------------------------------------------------------

    bool enforce_buying_power{true};
};


// ============================================================================
// OPEN ORDER RISK STATE
// ============================================================================

struct OpenOrderRiskState {
    std::size_t count{0};

    std::uint64_t total_remaining_quantity{0};

    std::uint64_t total_buy_notional_ticks{0};
    std::uint64_t total_sell_notional_ticks{0};
};


// ============================================================================
// RISK REQUEST
// ============================================================================
//
// The risk engine receives immutable request data. It does not need access to
// the mutable LimitOrderBook itself in order to make a pre-trade decision.
//

struct RiskRequest {
    std::uint64_t order_id{0};

    std::string symbol;

    Side side{Side::Buy};

    std::uint64_t quantity{0};

    std::int64_t price_ticks{0};

    RiskMarketState market;
    RiskAccountState account;
    OpenOrderRiskState open_orders;

    // Current monotonic/application timestamp used for quote-age validation.
    std::uint64_t now_ns{0};
};


// ============================================================================
// AMENDMENT RISK REQUEST
// ============================================================================
//
// An amendment must be evaluated using the proposed replacement state rather
// than merely checking the delta.
//
// filled_quantity is included so the engine can reject amendments that attempt
// to reduce total quantity below already-executed quantity.
//

struct AmendmentRiskRequest {
    std::uint64_t order_id{0};

    std::string symbol;

    Side side{Side::Buy};

    std::uint64_t current_total_quantity{0};
    std::uint64_t filled_quantity{0};
    std::int64_t current_price_ticks{0};

    std::uint64_t new_total_quantity{0};
    std::int64_t new_price_ticks{0};

    RiskMarketState market;
    RiskAccountState account;
    OpenOrderRiskState open_orders;

    std::uint64_t now_ns{0};
};


// ============================================================================
// RISK RESULT
// ============================================================================

struct RiskResult {
    RiskDecision decision{RiskDecision::Reject};

    RiskRejectReason reason{RiskRejectReason::Unknown};

    std::string message;

    // Useful diagnostics for audit / telemetry.
    std::uint64_t calculated_notional_ticks{0};

    std::int64_t projected_position{0};

    std::uint64_t projected_gross_exposure_ticks{0};

    [[nodiscard]]
    bool accepted() const noexcept {
        return decision == RiskDecision::Accept;
    }

    [[nodiscard]]
    bool rejected() const noexcept {
        return decision == RiskDecision::Reject;
    }

    [[nodiscard]]
    explicit operator bool() const noexcept {
        return accepted();
    }
};


// ============================================================================
// RISK ENGINE SNAPSHOT
// ============================================================================

struct RiskEngineSnapshot {
    bool kill_switch_active{false};

    RiskLimits limits;

    std::size_t tracked_order_ids{0};

    std::uint64_t accepted_orders{0};
    std::uint64_t rejected_orders{0};
    std::uint64_t accepted_amendments{0};
    std::uint64_t rejected_amendments{0};
};


// ============================================================================
// RISK ENGINE
// ============================================================================
//
// Pre-trade authority boundary.
//
// Intended execution flow:
//
//      Python TradingStrategy
//              |
//              v
//      Python/C++ OrderManager
//              |
//              v
//          RiskEngine
//          /       \
//       reject     accept
//                    |
//                    v
//             LimitOrderBook
//
// The RiskEngine does NOT:
//   * generate trading signals,
//   * match orders,
//   * mutate the order book,
//   * calculate strategy indicators,
//   * own portfolio positions,
//   * send broker orders.
//
// It evaluates whether a proposed order or amendment is permitted.
//
// Thread safety:
//   Public state-mutating and evaluation operations are synchronized.
//
// Duplicate-order policy:
//   The engine can track order IDs that have passed risk. The OrderManager
//   should call release_order_id() once the ID is permanently retired if ID
//   reuse is intentionally supported. Prefer never reusing order IDs.
//

class RiskEngine {
public:
    explicit RiskEngine(RiskLimits limits = {});

    RiskEngine(const RiskEngine&) = delete;
    RiskEngine& operator=(const RiskEngine&) = delete;

    RiskEngine(RiskEngine&&) = delete;
    RiskEngine& operator=(RiskEngine&&) = delete;

    ~RiskEngine() = default;

    // ------------------------------------------------------------------------
    // NEW ORDER EVALUATION
    // ------------------------------------------------------------------------

    [[nodiscard]]
    RiskResult evaluate(const RiskRequest& request);

    // ------------------------------------------------------------------------
    // AMENDMENT EVALUATION
    // ------------------------------------------------------------------------

    [[nodiscard]]
    RiskResult evaluate_amendment(
        const AmendmentRiskRequest& request
    );

    // ------------------------------------------------------------------------
    // KILL SWITCH
    // ------------------------------------------------------------------------

    void activate_kill_switch();

    void deactivate_kill_switch();

    [[nodiscard]]
    bool kill_switch_active() const;

    // ------------------------------------------------------------------------
    // LIMIT MANAGEMENT
    // ------------------------------------------------------------------------

    void set_limits(const RiskLimits& limits);

    [[nodiscard]]
    RiskLimits limits() const;

    // ------------------------------------------------------------------------
    // ORDER-ID LIFECYCLE
    // ------------------------------------------------------------------------

    [[nodiscard]]
    bool order_id_seen(std::uint64_t order_id) const;

    // Explicitly reserve an ID after an external component has accepted it.
    //
    // Returns false if already reserved.
    [[nodiscard]]
    bool reserve_order_id(std::uint64_t order_id);

    // Use only if the application intentionally permits order-ID reuse.
    void release_order_id(std::uint64_t order_id);

    // ------------------------------------------------------------------------
    // TELEMETRY
    // ------------------------------------------------------------------------

    [[nodiscard]]
    RiskEngineSnapshot snapshot() const;

    // ------------------------------------------------------------------------
    // HUMAN-READABLE ENUM TEXT
    // ------------------------------------------------------------------------

    [[nodiscard]]
    static std::string_view to_string(
        RiskDecision decision
    ) noexcept;

    [[nodiscard]]
    static std::string_view to_string(
        RiskRejectReason reason
    ) noexcept;

private:
    // ========================================================================
    // STATE
    // ========================================================================

    RiskLimits limits_;

    bool kill_switch_active_{false};

    // IDs accepted/reserved by the risk boundary.
    std::unordered_set<std::uint64_t> seen_order_ids_;

    std::uint64_t accepted_orders_{0};
    std::uint64_t rejected_orders_{0};

    std::uint64_t accepted_amendments_{0};
    std::uint64_t rejected_amendments_{0};

    mutable std::mutex mutex_;

    // ========================================================================
    // VALIDATION HELPERS
    // ========================================================================

    [[nodiscard]]
    static bool valid_limits(
        const RiskLimits& limits
    ) noexcept;

    [[nodiscard]]
    static bool valid_symbol(
        const std::string& symbol
    ) noexcept;

    [[nodiscard]]
    static bool safe_notional(
        std::uint64_t quantity,
        std::int64_t price_ticks,
        std::uint64_t& result
    ) noexcept;

    [[nodiscard]]
    static bool safe_add_u64(
        std::uint64_t lhs,
        std::uint64_t rhs,
        std::uint64_t& result
    ) noexcept;

    [[nodiscard]]
    static bool safe_project_position(
        std::int64_t current_position,
        Side side,
        std::uint64_t quantity,
        std::int64_t& projected_position
    ) noexcept;

    [[nodiscard]]
    static std::uint64_t absolute_position(
        std::int64_t position
    ) noexcept;

    // ------------------------------------------------------------------------
    // MARKET VALIDATION
    // ------------------------------------------------------------------------

    [[nodiscard]]
    RiskResult validate_market(
        const std::string& symbol,
        std::int64_t order_price_ticks,
        const RiskMarketState& market,
        std::uint64_t now_ns
    ) const;

    [[nodiscard]]
    RiskResult validate_price_collar(
        std::int64_t order_price_ticks,
        const RiskMarketState& market
    ) const;

    // ------------------------------------------------------------------------
    // ORDER VALIDATION
    // ------------------------------------------------------------------------

    [[nodiscard]]
    RiskResult validate_order_fields(
        const RiskRequest& request
    ) const;

    [[nodiscard]]
    RiskResult validate_order_limits(
        const RiskRequest& request,
        std::uint64_t notional_ticks,
        std::int64_t projected_position
    ) const;

    [[nodiscard]]
    RiskResult validate_open_order_limits(
        const OpenOrderRiskState& open_orders,
        std::uint64_t proposed_quantity
    ) const;

    [[nodiscard]]
    RiskResult validate_exposure(
        const RiskRequest& request,
        std::uint64_t notional_ticks,
        std::uint64_t& projected_gross_exposure_ticks
    ) const;

    [[nodiscard]]
    RiskResult validate_buying_power(
        const RiskRequest& request,
        std::uint64_t notional_ticks
    ) const;

    // ------------------------------------------------------------------------
    // AMENDMENT VALIDATION
    // ------------------------------------------------------------------------

    [[nodiscard]]
    RiskResult validate_amendment_fields(
        const AmendmentRiskRequest& request
    ) const;

    // ------------------------------------------------------------------------
    // RESULT HELPERS
    // ------------------------------------------------------------------------

    [[nodiscard]]
    static RiskResult accept_result(
        std::uint64_t notional_ticks,
        std::int64_t projected_position,
        std::uint64_t projected_gross_exposure_ticks
    );

    [[nodiscard]]
    static RiskResult reject_result(
        RiskRejectReason reason,
        std::string message,
        std::uint64_t notional_ticks = 0,
        std::int64_t projected_position = 0,
        std::uint64_t projected_gross_exposure_ticks = 0
    );
};

} // namespace trading
