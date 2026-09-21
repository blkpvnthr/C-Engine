#include "risk_engine.hpp"

#include <algorithm>
#include <cctype>
#include <limits>
#include <stdexcept>
#include <utility>

namespace trading {

namespace {

// ============================================================================
// INTERNAL CONSTANTS
// ============================================================================

constexpr std::uint64_t BPS_DENOMINATOR = 10'000ULL;


// ============================================================================
// INTERNAL HELPERS
// ============================================================================

[[nodiscard]]
std::uint64_t abs_i64_to_u64(
    const std::int64_t value
) noexcept {
    if (value >= 0) {
        return static_cast<std::uint64_t>(value);
    }

    // Avoid undefined behavior for INT64_MIN.
    return static_cast<std::uint64_t>(
        -(value + 1)
    ) + 1ULL;
}


[[nodiscard]]
bool safe_sub_u64(
    const std::uint64_t lhs,
    const std::uint64_t rhs,
    std::uint64_t& result
) noexcept {
    if (rhs > lhs) {
        return false;
    }

    result = lhs - rhs;
    return true;
}


[[nodiscard]]
bool safe_mul_u64(
    const std::uint64_t lhs,
    const std::uint64_t rhs,
    std::uint64_t& result
) noexcept {
    if (lhs == 0 || rhs == 0) {
        result = 0;
        return true;
    }

    if (
        lhs >
        std::numeric_limits<std::uint64_t>::max() / rhs
    ) {
        return false;
    }

    result = lhs * rhs;
    return true;
}


[[nodiscard]]
RiskResult ok_result() {
    RiskResult result;
    result.decision = RiskDecision::Accept;
    result.reason = RiskRejectReason::None;
    result.message = "accepted";
    return result;
}


[[nodiscard]]
bool is_ok(
    const RiskResult& result
) noexcept {
    return result.decision == RiskDecision::Accept;
}

} // namespace


// ============================================================================
// CONSTRUCTION
// ============================================================================

RiskEngine::RiskEngine(
    RiskLimits limits
)
    : limits_(std::move(limits)) {

    if (!valid_limits(limits_)) {
        throw std::invalid_argument(
            "RiskEngine: invalid risk limits"
        );
    }
}


// ============================================================================
// NEW ORDER EVALUATION
// ============================================================================

RiskResult RiskEngine::evaluate(
    const RiskRequest& request
) {
    std::scoped_lock lock(mutex_);

    if (kill_switch_active_) {
        ++rejected_orders_;

        return reject_result(
            RiskRejectReason::KillSwitchActive,
            "risk kill switch is active"
        );
    }

    if (
        request.order_id != 0 &&
        seen_order_ids_.find(request.order_id) !=
            seen_order_ids_.end()
    ) {
        ++rejected_orders_;

        return reject_result(
            RiskRejectReason::DuplicateOrderId,
            "order ID has already passed or been reserved at the risk boundary"
        );
    }

    auto result = validate_order_fields(request);

    if (!is_ok(result)) {
        ++rejected_orders_;
        return result;
    }

    result = validate_market(
        request.symbol,
        request.price_ticks,
        request.market,
        request.now_ns
    );

    if (!is_ok(result)) {
        ++rejected_orders_;
        return result;
    }

    std::uint64_t notional_ticks = 0;

    if (
        !safe_notional(
            request.quantity,
            request.price_ticks,
            notional_ticks
        )
    ) {
        ++rejected_orders_;

        return reject_result(
            RiskRejectReason::NotionalLimitExceeded,
            "order notional overflows uint64",
            0
        );
    }

    std::int64_t projected_position = 0;

    if (
        !safe_project_position(
            request.account.position,
            request.side,
            request.quantity,
            projected_position
        )
    ) {
        ++rejected_orders_;

        return reject_result(
            RiskRejectReason::GrossPositionLimitExceeded,
            "projected position exceeds int64 range",
            notional_ticks
        );
    }

    result = validate_order_limits(
        request,
        notional_ticks,
        projected_position
    );

    if (!is_ok(result)) {
        ++rejected_orders_;
        return result;
    }

    result = validate_open_order_limits(
        request.open_orders,
        request.quantity
    );

    if (!is_ok(result)) {
        result.calculated_notional_ticks = notional_ticks;
        result.projected_position = projected_position;

        ++rejected_orders_;
        return result;
    }

    std::uint64_t projected_gross_exposure_ticks = 0;

    result = validate_exposure(
        request,
        notional_ticks,
        projected_gross_exposure_ticks
    );

    if (!is_ok(result)) {
        result.projected_position = projected_position;

        ++rejected_orders_;
        return result;
    }

    result = validate_buying_power(
        request,
        notional_ticks
    );

    if (!is_ok(result)) {
        result.projected_position = projected_position;
        result.projected_gross_exposure_ticks =
            projected_gross_exposure_ticks;

        ++rejected_orders_;
        return result;
    }

    // Reserve only after every check has passed. Because evaluate() holds
    // mutex_, duplicate checking and reservation are atomic with respect to
    // other RiskEngine callers.
    const auto [it, inserted] =
        seen_order_ids_.insert(request.order_id);

    (void)it;

    if (!inserted) {
        ++rejected_orders_;

        return reject_result(
            RiskRejectReason::DuplicateOrderId,
            "order ID became reserved before acceptance",
            notional_ticks,
            projected_position,
            projected_gross_exposure_ticks
        );
    }

    ++accepted_orders_;

    return accept_result(
        notional_ticks,
        projected_position,
        projected_gross_exposure_ticks
    );
}


// ============================================================================
// AMENDMENT EVALUATION
// ============================================================================

RiskResult RiskEngine::evaluate_amendment(
    const AmendmentRiskRequest& request
) {
    std::scoped_lock lock(mutex_);

    if (kill_switch_active_) {
        ++rejected_amendments_;

        return reject_result(
            RiskRejectReason::KillSwitchActive,
            "risk kill switch is active"
        );
    }

    auto result =
        validate_amendment_fields(request);

    if (!is_ok(result)) {
        ++rejected_amendments_;
        return result;
    }

    result = validate_market(
        request.symbol,
        request.new_price_ticks,
        request.market,
        request.now_ns
    );

    if (!is_ok(result)) {
        ++rejected_amendments_;
        return result;
    }

    const std::uint64_t current_remaining =
        request.current_total_quantity -
        request.filled_quantity;

    const std::uint64_t new_remaining =
        request.new_total_quantity -
        request.filled_quantity;

    std::uint64_t new_notional_ticks = 0;

    if (
        !safe_notional(
            new_remaining,
            request.new_price_ticks,
            new_notional_ticks
        )
    ) {
        ++rejected_amendments_;

        return reject_result(
            RiskRejectReason::NotionalLimitExceeded,
            "amended remaining notional overflows uint64"
        );
    }

    // Apply per-order limits to the proposed total order quantity and price.
    std::uint64_t new_total_notional_ticks = 0;

    if (
        !safe_notional(
            request.new_total_quantity,
            request.new_price_ticks,
            new_total_notional_ticks
        )
    ) {
        ++rejected_amendments_;

        return reject_result(
            RiskRejectReason::NotionalLimitExceeded,
            "amended total order notional overflows uint64"
        );
    }

    if (
        request.new_total_quantity >
        limits_.max_order_quantity
    ) {
        ++rejected_amendments_;

        return reject_result(
            RiskRejectReason::QuantityLimitExceeded,
            "amended quantity exceeds max_order_quantity",
            new_total_notional_ticks
        );
    }

    if (
        new_total_notional_ticks >
        limits_.max_order_notional_ticks
    ) {
        ++rejected_amendments_;

        return reject_result(
            RiskRejectReason::NotionalLimitExceeded,
            "amended order notional exceeds max_order_notional_ticks",
            new_total_notional_ticks
        );
    }

    // Only the change in remaining exposure should affect projected position.
    // The existing resting remainder is already represented by the current
    // open-order state.
    std::int64_t projected_position =
        request.account.position;

    if (new_remaining > current_remaining) {
        const std::uint64_t increase =
            new_remaining - current_remaining;

        if (
            !safe_project_position(
                request.account.position,
                request.side,
                increase,
                projected_position
            )
        ) {
            ++rejected_amendments_;

            return reject_result(
                RiskRejectReason::GrossPositionLimitExceeded,
                "amendment causes projected position overflow",
                new_total_notional_ticks
            );
        }
    }

    if (
        projected_position >
        limits_.max_long_position
    ) {
        ++rejected_amendments_;

        return reject_result(
            RiskRejectReason::LongPositionLimitExceeded,
            "amendment exceeds maximum long position",
            new_total_notional_ticks,
            projected_position
        );
    }

    if (
        projected_position <
        -limits_.max_short_position
    ) {
        ++rejected_amendments_;

        return reject_result(
            RiskRejectReason::ShortPositionLimitExceeded,
            "amendment exceeds maximum short position",
            new_total_notional_ticks,
            projected_position
        );
    }

    if (
        absolute_position(projected_position) >
        limits_.max_gross_position
    ) {
        ++rejected_amendments_;

        return reject_result(
            RiskRejectReason::GrossPositionLimitExceeded,
            "amendment exceeds maximum gross position",
            new_total_notional_ticks,
            projected_position
        );
    }

    // The existing order is already included in open_orders. Replace its
    // remaining quantity rather than adding the amended order on top of it.
    std::uint64_t adjusted_open_quantity = 0;

    if (
        !safe_sub_u64(
            request.open_orders.total_remaining_quantity,
            current_remaining,
            adjusted_open_quantity
        )
    ) {
        ++rejected_amendments_;

        return reject_result(
            RiskRejectReason::Unknown,
            "open-order quantity state is inconsistent with amended order",
            new_total_notional_ticks,
            projected_position
        );
    }

    if (
        !safe_add_u64(
            adjusted_open_quantity,
            new_remaining,
            adjusted_open_quantity
        )
    ) {
        ++rejected_amendments_;

        return reject_result(
            RiskRejectReason::OpenOrderQuantityLimitExceeded,
            "amended open-order quantity overflows uint64",
            new_total_notional_ticks,
            projected_position
        );
    }

    if (
        adjusted_open_quantity >
        limits_.max_open_order_quantity
    ) {
        ++rejected_amendments_;

        return reject_result(
            RiskRejectReason::OpenOrderQuantityLimitExceeded,
            "amendment exceeds maximum aggregate open-order quantity",
            new_total_notional_ticks,
            projected_position
        );
    }

    // Amendments do not create another open order, so count is unchanged.
    if (
        request.open_orders.count >
        limits_.max_open_orders
    ) {
        ++rejected_amendments_;

        return reject_result(
            RiskRejectReason::OpenOrderLimitExceeded,
            "current open-order count exceeds configured limit",
            new_total_notional_ticks,
            projected_position
        );
    }

    // Replace the existing order's resting notional in gross exposure.
    std::uint64_t current_remaining_notional = 0;

    if (
        !safe_notional(
            current_remaining,
            request.current_price_ticks,
            current_remaining_notional
        )
    ) {
        ++rejected_amendments_;

        return reject_result(
            RiskRejectReason::Unknown,
            "current order notional overflows uint64",
            new_total_notional_ticks,
            projected_position
        );
    }

    std::uint64_t base_exposure = 0;

    if (
        !safe_sub_u64(
            request.account.gross_exposure_ticks,
            current_remaining_notional,
            base_exposure
        )
    ) {
        // Portfolio/account implementations may track gross exposure without
        // open-order exposure. In that case, fall back to treating the current
        // account exposure as the base and add only any incremental increase.
        base_exposure =
            request.account.gross_exposure_ticks;

        if (new_notional_ticks > current_remaining_notional) {
            const auto incremental =
                new_notional_ticks -
                current_remaining_notional;

            if (
                !safe_add_u64(
                    base_exposure,
                    incremental,
                    base_exposure
                )
            ) {
                ++rejected_amendments_;

                return reject_result(
                    RiskRejectReason::GrossExposureLimitExceeded,
                    "amended gross exposure overflows uint64",
                    new_total_notional_ticks,
                    projected_position
                );
            }
        }
    } else {
        if (
            !safe_add_u64(
                base_exposure,
                new_notional_ticks,
                base_exposure
            )
        ) {
            ++rejected_amendments_;

            return reject_result(
                RiskRejectReason::GrossExposureLimitExceeded,
                "amended gross exposure overflows uint64",
                new_total_notional_ticks,
                projected_position
            );
        }
    }

    const std::uint64_t projected_gross_exposure_ticks =
        base_exposure;

    if (
        projected_gross_exposure_ticks >
        limits_.max_gross_exposure_ticks
    ) {
        ++rejected_amendments_;

        return reject_result(
            RiskRejectReason::GrossExposureLimitExceeded,
            "amendment exceeds maximum gross exposure",
            new_total_notional_ticks,
            projected_position,
            projected_gross_exposure_ticks
        );
    }

    // For buys, only additional remaining notional requires additional buying
    // power. Reductions and sell amendments do not consume new buying power
    // under this simplified cash-risk model.
    if (
        limits_.enforce_buying_power &&
        request.side == Side::Buy &&
        new_notional_ticks > current_remaining_notional
    ) {
        const std::uint64_t additional_notional =
            new_notional_ticks -
            current_remaining_notional;

        if (request.account.buying_power_ticks < 0) {
            ++rejected_amendments_;

            return reject_result(
                RiskRejectReason::InsufficientBuyingPower,
                "buying power is negative",
                new_total_notional_ticks,
                projected_position,
                projected_gross_exposure_ticks
            );
        }

        if (
            additional_notional >
            static_cast<std::uint64_t>(
                request.account.buying_power_ticks
            )
        ) {
            ++rejected_amendments_;

            return reject_result(
                RiskRejectReason::InsufficientBuyingPower,
                "amendment requires more incremental buying power than available",
                new_total_notional_ticks,
                projected_position,
                projected_gross_exposure_ticks
            );
        }
    }

    ++accepted_amendments_;

    return accept_result(
        new_total_notional_ticks,
        projected_position,
        projected_gross_exposure_ticks
    );
}


// ============================================================================
// KILL SWITCH
// ============================================================================

void RiskEngine::activate_kill_switch() {
    std::scoped_lock lock(mutex_);
    kill_switch_active_ = true;
}


void RiskEngine::deactivate_kill_switch() {
    std::scoped_lock lock(mutex_);
    kill_switch_active_ = false;
}


bool RiskEngine::kill_switch_active() const {
    std::scoped_lock lock(mutex_);
    return kill_switch_active_;
}


// ============================================================================
// LIMIT MANAGEMENT
// ============================================================================

void RiskEngine::set_limits(
    const RiskLimits& limits
) {
    if (!valid_limits(limits)) {
        throw std::invalid_argument(
            "RiskEngine: invalid risk limits"
        );
    }

    std::scoped_lock lock(mutex_);
    limits_ = limits;
}


RiskLimits RiskEngine::limits() const {
    std::scoped_lock lock(mutex_);
    return limits_;
}


// ============================================================================
// ORDER-ID LIFECYCLE
// ============================================================================

bool RiskEngine::order_id_seen(
    const std::uint64_t order_id
) const {
    std::scoped_lock lock(mutex_);

    return seen_order_ids_.find(order_id) !=
           seen_order_ids_.end();
}


bool RiskEngine::reserve_order_id(
    const std::uint64_t order_id
) {
    if (order_id == 0) {
        return false;
    }

    std::scoped_lock lock(mutex_);

    const auto [it, inserted] =
        seen_order_ids_.insert(order_id);

    (void)it;

    return inserted;
}


void RiskEngine::release_order_id(
    const std::uint64_t order_id
) {
    std::scoped_lock lock(mutex_);
    seen_order_ids_.erase(order_id);
}


// ============================================================================
// TELEMETRY
// ============================================================================

RiskEngineSnapshot RiskEngine::snapshot() const {
    std::scoped_lock lock(mutex_);

    RiskEngineSnapshot result;

    result.kill_switch_active =
        kill_switch_active_;

    result.limits = limits_;

    result.tracked_order_ids =
        seen_order_ids_.size();

    result.accepted_orders =
        accepted_orders_;

    result.rejected_orders =
        rejected_orders_;

    result.accepted_amendments =
        accepted_amendments_;

    result.rejected_amendments =
        rejected_amendments_;

    return result;
}


// ============================================================================
// ENUM TEXT
// ============================================================================

std::string_view RiskEngine::to_string(
    const RiskDecision decision
) noexcept {
    switch (decision) {
        case RiskDecision::Accept:
            return "accept";

        case RiskDecision::Reject:
            return "reject";
    }

    return "unknown";
}


std::string_view RiskEngine::to_string(
    const RiskRejectReason reason
) noexcept {
    switch (reason) {
        case RiskRejectReason::None:
            return "none";

        case RiskRejectReason::KillSwitchActive:
            return "kill_switch_active";

        case RiskRejectReason::InvalidOrderId:
            return "invalid_order_id";

        case RiskRejectReason::DuplicateOrderId:
            return "duplicate_order_id";

        case RiskRejectReason::InvalidSymbol:
            return "invalid_symbol";

        case RiskRejectReason::InvalidQuantity:
            return "invalid_quantity";

        case RiskRejectReason::InvalidPrice:
            return "invalid_price";

        case RiskRejectReason::QuantityLimitExceeded:
            return "quantity_limit_exceeded";

        case RiskRejectReason::NotionalLimitExceeded:
            return "notional_limit_exceeded";

        case RiskRejectReason::LongPositionLimitExceeded:
            return "long_position_limit_exceeded";

        case RiskRejectReason::ShortPositionLimitExceeded:
            return "short_position_limit_exceeded";

        case RiskRejectReason::GrossPositionLimitExceeded:
            return "gross_position_limit_exceeded";

        case RiskRejectReason::GrossExposureLimitExceeded:
            return "gross_exposure_limit_exceeded";

        case RiskRejectReason::OpenOrderLimitExceeded:
            return "open_order_limit_exceeded";

        case RiskRejectReason::OpenOrderQuantityLimitExceeded:
            return "open_order_quantity_limit_exceeded";

        case RiskRejectReason::MissingMarketData:
            return "missing_market_data";

        case RiskRejectReason::StaleMarketData:
            return "stale_market_data";

        case RiskRejectReason::InvalidMarketData:
            return "invalid_market_data";

        case RiskRejectReason::PriceCollarExceeded:
            return "price_collar_exceeded";

        case RiskRejectReason::InsufficientBuyingPower:
            return "insufficient_buying_power";

        case RiskRejectReason::Unknown:
            return "unknown";
    }

    return "unknown";
}


// ============================================================================
// LIMIT VALIDATION
// ============================================================================

bool RiskEngine::valid_limits(
    const RiskLimits& limits
) noexcept {
    if (limits.max_order_quantity == 0) {
        return false;
    }

    if (limits.max_order_notional_ticks == 0) {
        return false;
    }

    if (limits.max_long_position < 0) {
        return false;
    }

    if (limits.max_short_position < 0) {
        return false;
    }

    if (limits.max_gross_position == 0) {
        return false;
    }

    if (limits.max_gross_exposure_ticks == 0) {
        return false;
    }

    if (limits.max_open_orders == 0) {
        return false;
    }

    if (limits.max_open_order_quantity == 0) {
        return false;
    }

    if (limits.max_market_data_age_ns == 0) {
        return false;
    }

    return true;
}


// ============================================================================
// SYMBOL VALIDATION
// ============================================================================

bool RiskEngine::valid_symbol(
    const std::string& symbol
) noexcept {
    if (symbol.empty() || symbol.size() > 32) {
        return false;
    }

    for (const unsigned char ch : symbol) {
        if (
            !(std::isalnum(ch) ||
              ch == '.' ||
              ch == '-' ||
              ch == '_' ||
              ch == '/')
        ) {
            return false;
        }
    }

    return true;
}


// ============================================================================
// SAFE ARITHMETIC
// ============================================================================

bool RiskEngine::safe_notional(
    const std::uint64_t quantity,
    const std::int64_t price_ticks,
    std::uint64_t& result
) noexcept {
    if (price_ticks <= 0) {
        result = 0;
        return false;
    }

    return safe_mul_u64(
        quantity,
        static_cast<std::uint64_t>(price_ticks),
        result
    );
}


bool RiskEngine::safe_add_u64(
    const std::uint64_t lhs,
    const std::uint64_t rhs,
    std::uint64_t& result
) noexcept {
    if (
        rhs >
        std::numeric_limits<std::uint64_t>::max() - lhs
    ) {
        result = 0;
        return false;
    }

    result = lhs + rhs;
    return true;
}


bool RiskEngine::safe_project_position(
    const std::int64_t current_position,
    const Side side,
    const std::uint64_t quantity,
    std::int64_t& projected_position
) noexcept {
    constexpr auto I64_MAX_U =
        static_cast<std::uint64_t>(
            std::numeric_limits<std::int64_t>::max()
        );

    if (side == Side::Buy) {
        if (current_position >= 0) {
            const auto current =
                static_cast<std::uint64_t>(current_position);

            if (quantity > I64_MAX_U - current) {
                return false;
            }

            projected_position =
                static_cast<std::int64_t>(
                    current + quantity
                );

            return true;
        }

        const std::uint64_t short_magnitude =
            abs_i64_to_u64(current_position);

        if (quantity <= short_magnitude) {
            const std::uint64_t remaining_short =
                short_magnitude - quantity;

            if (remaining_short == 0) {
                projected_position = 0;
                return true;
            }

            if (
                remaining_short ==
                (I64_MAX_U + 1ULL)
            ) {
                projected_position =
                    std::numeric_limits<std::int64_t>::min();

                return true;
            }

            projected_position =
                -static_cast<std::int64_t>(
                    remaining_short
                );

            return true;
        }

        const std::uint64_t resulting_long =
            quantity - short_magnitude;

        if (resulting_long > I64_MAX_U) {
            return false;
        }

        projected_position =
            static_cast<std::int64_t>(
                resulting_long
            );

        return true;
    }

    // Sell.
    if (current_position <= 0) {
        const std::uint64_t short_magnitude =
            abs_i64_to_u64(current_position);

        const std::uint64_t max_negative_magnitude =
            I64_MAX_U + 1ULL;

        if (
            quantity >
            max_negative_magnitude - short_magnitude
        ) {
            return false;
        }

        const std::uint64_t resulting_short =
            short_magnitude + quantity;

        if (resulting_short == 0) {
            projected_position = 0;
            return true;
        }

        if (
            resulting_short ==
            max_negative_magnitude
        ) {
            projected_position =
                std::numeric_limits<std::int64_t>::min();

            return true;
        }

        projected_position =
            -static_cast<std::int64_t>(
                resulting_short
            );

        return true;
    }

    const auto current_long =
        static_cast<std::uint64_t>(
            current_position
        );

    if (quantity <= current_long) {
        projected_position =
            static_cast<std::int64_t>(
                current_long - quantity
            );

        return true;
    }

    const std::uint64_t resulting_short =
        quantity - current_long;

    if (resulting_short > I64_MAX_U + 1ULL) {
        return false;
    }

    if (resulting_short == I64_MAX_U + 1ULL) {
        projected_position =
            std::numeric_limits<std::int64_t>::min();

        return true;
    }

    projected_position =
        -static_cast<std::int64_t>(
            resulting_short
        );

    return true;
}


std::uint64_t RiskEngine::absolute_position(
    const std::int64_t position
) noexcept {
    return abs_i64_to_u64(position);
}


// ============================================================================
// MARKET VALIDATION
// ============================================================================

RiskResult RiskEngine::validate_market(
    const std::string& symbol,
    const std::int64_t order_price_ticks,
    const RiskMarketState& market,
    const std::uint64_t now_ns
) const {
    if (market.symbol.empty()) {
        return reject_result(
            RiskRejectReason::MissingMarketData,
            "market data is missing a symbol"
        );
    }

    if (market.symbol != symbol) {
        return reject_result(
            RiskRejectReason::InvalidMarketData,
            "market-data symbol does not match order symbol"
        );
    }

    if (!market.valid()) {
        return reject_result(
            RiskRejectReason::InvalidMarketData,
            "market data has invalid bid/ask values"
        );
    }

    if (market.timestamp_ns == 0 || now_ns == 0) {
        return reject_result(
            RiskRejectReason::MissingMarketData,
            "market-data timestamp or evaluation timestamp is missing"
        );
    }

    if (now_ns < market.timestamp_ns) {
        return reject_result(
            RiskRejectReason::InvalidMarketData,
            "market-data timestamp is in the future"
        );
    }

    const std::uint64_t age =
        now_ns - market.timestamp_ns;

    if (age > limits_.max_market_data_age_ns) {
        return reject_result(
            RiskRejectReason::StaleMarketData,
            "market data exceeds configured maximum age"
        );
    }

    return validate_price_collar(
        order_price_ticks,
        market
    );
}


// ============================================================================
// PRICE COLLAR
// ============================================================================

RiskResult RiskEngine::validate_price_collar(
    const std::int64_t order_price_ticks,
    const RiskMarketState& market
) const {
    if (limits_.max_price_deviation_bps == 0) {
        return ok_result();
    }

    const std::int64_t reference =
        market.midpoint_ticks();

    if (reference <= 0) {
        return reject_result(
            RiskRejectReason::InvalidMarketData,
            "market midpoint is invalid"
        );
    }

    const std::uint64_t reference_u =
        static_cast<std::uint64_t>(reference);

    const std::uint64_t order_u =
        static_cast<std::uint64_t>(order_price_ticks);

    const std::uint64_t difference =
        order_u >= reference_u
            ? order_u - reference_u
            : reference_u - order_u;

    // Compare:
    //
    //     difference / reference <= bps / 10000
    //
    // without floating point. To avoid overflow, compute a conservative
    // integer threshold using quotient/remainder decomposition.
    const std::uint64_t whole =
        reference_u / BPS_DENOMINATOR;

    const std::uint64_t remainder =
        reference_u % BPS_DENOMINATOR;

    std::uint64_t allowed_whole = 0;

    if (
        !safe_mul_u64(
            whole,
            limits_.max_price_deviation_bps,
            allowed_whole
        )
    ) {
        // An overflow here means the permitted threshold is effectively
        // beyond representable order-price distance, so the collar cannot
        // be exceeded by a valid int64 price.
        return ok_result();
    }

    std::uint64_t allowed_remainder_product = 0;

    if (
        !safe_mul_u64(
            remainder,
            limits_.max_price_deviation_bps,
            allowed_remainder_product
        )
    ) {
        return ok_result();
    }

    const std::uint64_t allowed_remainder =
        allowed_remainder_product /
        BPS_DENOMINATOR;

    std::uint64_t allowed_difference = 0;

    if (
        !safe_add_u64(
            allowed_whole,
            allowed_remainder,
            allowed_difference
        )
    ) {
        return ok_result();
    }

    if (difference > allowed_difference) {
        return reject_result(
            RiskRejectReason::PriceCollarExceeded,
            "order price exceeds configured market-price collar"
        );
    }

    return ok_result();
}


// ============================================================================
// ORDER FIELD VALIDATION
// ============================================================================

RiskResult RiskEngine::validate_order_fields(
    const RiskRequest& request
) const {
    if (request.order_id == 0) {
        return reject_result(
            RiskRejectReason::InvalidOrderId,
            "order ID must be non-zero"
        );
    }

    if (!valid_symbol(request.symbol)) {
        return reject_result(
            RiskRejectReason::InvalidSymbol,
            "order symbol is invalid"
        );
    }

    if (request.quantity == 0) {
        return reject_result(
            RiskRejectReason::InvalidQuantity,
            "order quantity must be positive"
        );
    }

    if (request.price_ticks <= 0) {
        return reject_result(
            RiskRejectReason::InvalidPrice,
            "order price must be positive"
        );
    }

    return ok_result();
}


// ============================================================================
// ORDER LIMIT VALIDATION
// ============================================================================

RiskResult RiskEngine::validate_order_limits(
    const RiskRequest& request,
    const std::uint64_t notional_ticks,
    const std::int64_t projected_position
) const {
    if (
        request.quantity >
        limits_.max_order_quantity
    ) {
        return reject_result(
            RiskRejectReason::QuantityLimitExceeded,
            "order quantity exceeds max_order_quantity",
            notional_ticks,
            projected_position
        );
    }

    if (
        notional_ticks >
        limits_.max_order_notional_ticks
    ) {
        return reject_result(
            RiskRejectReason::NotionalLimitExceeded,
            "order notional exceeds max_order_notional_ticks",
            notional_ticks,
            projected_position
        );
    }

    if (
        projected_position >
        limits_.max_long_position
    ) {
        return reject_result(
            RiskRejectReason::LongPositionLimitExceeded,
            "projected position exceeds maximum long position",
            notional_ticks,
            projected_position
        );
    }

    if (
        projected_position <
        -limits_.max_short_position
    ) {
        return reject_result(
            RiskRejectReason::ShortPositionLimitExceeded,
            "projected position exceeds maximum short position",
            notional_ticks,
            projected_position
        );
    }

    if (
        absolute_position(projected_position) >
        limits_.max_gross_position
    ) {
        return reject_result(
            RiskRejectReason::GrossPositionLimitExceeded,
            "projected absolute position exceeds maximum gross position",
            notional_ticks,
            projected_position
        );
    }

    return ok_result();
}


// ============================================================================
// OPEN ORDER LIMITS
// ============================================================================

RiskResult RiskEngine::validate_open_order_limits(
    const OpenOrderRiskState& open_orders,
    const std::uint64_t proposed_quantity
) const {
    if (
        open_orders.count >=
        limits_.max_open_orders
    ) {
        return reject_result(
            RiskRejectReason::OpenOrderLimitExceeded,
            "maximum number of open orders reached"
        );
    }

    std::uint64_t projected_quantity = 0;

    if (
        !safe_add_u64(
            open_orders.total_remaining_quantity,
            proposed_quantity,
            projected_quantity
        )
    ) {
        return reject_result(
            RiskRejectReason::OpenOrderQuantityLimitExceeded,
            "aggregate open-order quantity overflows uint64"
        );
    }

    if (
        projected_quantity >
        limits_.max_open_order_quantity
    ) {
        return reject_result(
            RiskRejectReason::OpenOrderQuantityLimitExceeded,
            "aggregate open-order quantity exceeds configured limit"
        );
    }

    return ok_result();
}


// ============================================================================
// EXPOSURE
// ============================================================================

RiskResult RiskEngine::validate_exposure(
    const RiskRequest& request,
    const std::uint64_t notional_ticks,
    std::uint64_t& projected_gross_exposure_ticks
) const {
    if (
        !safe_add_u64(
            request.account.gross_exposure_ticks,
            notional_ticks,
            projected_gross_exposure_ticks
        )
    ) {
        projected_gross_exposure_ticks = 0;

        return reject_result(
            RiskRejectReason::GrossExposureLimitExceeded,
            "projected gross exposure overflows uint64",
            notional_ticks
        );
    }

    if (
        projected_gross_exposure_ticks >
        limits_.max_gross_exposure_ticks
    ) {
        return reject_result(
            RiskRejectReason::GrossExposureLimitExceeded,
            "projected gross exposure exceeds configured limit",
            notional_ticks,
            0,
            projected_gross_exposure_ticks
        );
    }

    return ok_result();
}


// ============================================================================
// BUYING POWER
// ============================================================================

RiskResult RiskEngine::validate_buying_power(
    const RiskRequest& request,
    const std::uint64_t notional_ticks
) const {
    if (!limits_.enforce_buying_power) {
        return ok_result();
    }

    // Under this conservative cash-risk model, buys consume buying power.
    // Short-sale margin rules belong in a richer account/margin model.
    if (request.side == Side::Sell) {
        return ok_result();
    }

    if (request.account.buying_power_ticks < 0) {
        return reject_result(
            RiskRejectReason::InsufficientBuyingPower,
            "buying power is negative",
            notional_ticks
        );
    }

    const auto buying_power =
        static_cast<std::uint64_t>(
            request.account.buying_power_ticks
        );

    if (notional_ticks > buying_power) {
        return reject_result(
            RiskRejectReason::InsufficientBuyingPower,
            "order notional exceeds available buying power",
            notional_ticks
        );
    }

    return ok_result();
}


// ============================================================================
// AMENDMENT FIELD VALIDATION
// ============================================================================

RiskResult RiskEngine::validate_amendment_fields(
    const AmendmentRiskRequest& request
) const {
    if (request.order_id == 0) {
        return reject_result(
            RiskRejectReason::InvalidOrderId,
            "amendment order ID must be non-zero"
        );
    }

    if (!valid_symbol(request.symbol)) {
        return reject_result(
            RiskRejectReason::InvalidSymbol,
            "amendment symbol is invalid"
        );
    }

    // An amendment is valid only for an ID already known to the risk boundary.
    if (
        seen_order_ids_.find(request.order_id) ==
        seen_order_ids_.end()
    ) {
        return reject_result(
            RiskRejectReason::InvalidOrderId,
            "amendment references an order ID not reserved by the risk engine"
        );
    }

    if (request.current_total_quantity == 0) {
        return reject_result(
            RiskRejectReason::InvalidQuantity,
            "current total quantity must be positive"
        );
    }

    if (
        request.filled_quantity >
        request.current_total_quantity
    ) {
        return reject_result(
            RiskRejectReason::InvalidQuantity,
            "filled quantity exceeds current total quantity"
        );
    }

    if (
        request.new_total_quantity <
        request.filled_quantity
    ) {
        return reject_result(
            RiskRejectReason::InvalidQuantity,
            "new total quantity cannot be below already-filled quantity"
        );
    }

    if (request.current_price_ticks <= 0) {
        return reject_result(
            RiskRejectReason::InvalidPrice,
            "current order price must be positive"
        );
    }

    if (request.new_price_ticks <= 0) {
        return reject_result(
            RiskRejectReason::InvalidPrice,
            "amended order price must be positive"
        );
    }

    return ok_result();
}


// ============================================================================
// RESULT HELPERS
// ============================================================================

RiskResult RiskEngine::accept_result(
    const std::uint64_t notional_ticks,
    const std::int64_t projected_position,
    const std::uint64_t projected_gross_exposure_ticks
) {
    RiskResult result;

    result.decision =
        RiskDecision::Accept;

    result.reason =
        RiskRejectReason::None;

    result.message =
        "accepted";

    result.calculated_notional_ticks =
        notional_ticks;

    result.projected_position =
        projected_position;

    result.projected_gross_exposure_ticks =
        projected_gross_exposure_ticks;

    return result;
}


RiskResult RiskEngine::reject_result(
    const RiskRejectReason reason,
    std::string message,
    const std::uint64_t notional_ticks,
    const std::int64_t projected_position,
    const std::uint64_t projected_gross_exposure_ticks
) {
    RiskResult result;

    result.decision =
        RiskDecision::Reject;

    result.reason =
        reason;

    result.message =
        std::move(message);

    result.calculated_notional_ticks =
        notional_ticks;

    result.projected_position =
        projected_position;

    result.projected_gross_exposure_ticks =
        projected_gross_exposure_ticks;

    return result;
}

} // namespace trading
