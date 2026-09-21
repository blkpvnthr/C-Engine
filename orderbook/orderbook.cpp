#include "orderbook.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <utility>

namespace trading {

namespace {

[[nodiscard]]
bool valid_symbol_text(
    const std::string_view symbol
) noexcept {
    if (symbol.empty() || symbol.size() > 32) {
        return false;
    }

    for (const char ch : symbol) {
        const bool alpha =
            (ch >= 'A' && ch <= 'Z') ||
            (ch >= 'a' && ch <= 'z');

        const bool digit =
            ch >= '0' && ch <= '9';

        const bool punctuation =
            ch == '.' ||
            ch == '-' ||
            ch == '_' ||
            ch == '/';

        if (!alpha && !digit && !punctuation) {
            return false;
        }
    }

    return true;
}


[[nodiscard]]
std::uint64_t next_counter(
    std::uint64_t& value,
    const char* name
) {
    if (
        value ==
        std::numeric_limits<std::uint64_t>::max()
    ) {
        throw std::overflow_error(
            std::string(name) + " exhausted"
        );
    }

    return value++;
}

} // namespace


// ============================================================================
// OBSERVED MARKET BOOK
// ============================================================================

ObservedMarketBook::ObservedMarketBook(
    std::string symbol
)
    : symbol_(std::move(symbol)) {

    if (!valid_symbol_text(symbol_)) {
        throw std::invalid_argument(
            "ObservedMarketBook: invalid symbol"
        );
    }
}


bool ObservedMarketBook::symbol_matches(
    const std::string_view symbol
) const noexcept {
    return symbol == symbol_;
}


bool ObservedMarketBook::quote_in_sequence(
    const MarketQuoteUpdate& update
) const noexcept {
    if (!last_quote_.has_value()) {
        return true;
    }

    if (
        update.timestamp_ns <
        last_quote_->timestamp_ns
    ) {
        return false;
    }

    if (
        update.sequence != 0 &&
        last_quote_sequence_ != 0 &&
        update.sequence <= last_quote_sequence_
    ) {
        return false;
    }

    // If timestamps are identical and neither side has a useful sequence,
    // accept the update. SIP/NBBO can legitimately update multiple fields
    // within the same timestamp granularity.
    return true;
}


bool ObservedMarketBook::trade_in_sequence(
    const MarketTradeUpdate& update
) const noexcept {
    if (!last_trade_.has_value()) {
        return true;
    }

    if (
        update.timestamp_ns <
        last_trade_->timestamp_ns
    ) {
        return false;
    }

    if (
        update.sequence != 0 &&
        last_trade_sequence_ != 0 &&
        update.sequence <= last_trade_sequence_
    ) {
        return false;
    }

    return true;
}


bool ObservedMarketBook::on_quote(
    const MarketQuoteUpdate& update
) {
    std::unique_lock lock(mutex_);

    if (
        !symbol_matches(update.symbol) ||
        !update.valid()
    ) {
        ++rejected_updates_;
        return false;
    }

    if (!quote_in_sequence(update)) {
        ++rejected_updates_;
        ++out_of_sequence_updates_;
        return false;
    }

    last_quote_ = update;

    if (update.sequence != 0) {
        last_quote_sequence_ = update.sequence;
    }

    ++quote_updates_;

    version_.fetch_add(
        1,
        std::memory_order_release
    );

    return true;
}


bool ObservedMarketBook::on_trade(
    const MarketTradeUpdate& update
) {
    std::unique_lock lock(mutex_);

    if (
        !symbol_matches(update.symbol) ||
        !update.valid()
    ) {
        ++rejected_updates_;
        return false;
    }

    if (!trade_in_sequence(update)) {
        ++rejected_updates_;
        ++out_of_sequence_updates_;
        return false;
    }

    last_trade_ = update;

    if (update.sequence != 0) {
        last_trade_sequence_ = update.sequence;
    }

    ++trade_updates_;

    version_.fetch_add(
        1,
        std::memory_order_release
    );

    return true;
}


ObservedMarketSnapshot
ObservedMarketBook::snapshot() const {
    std::shared_lock lock(mutex_);

    ObservedMarketSnapshot result;

    result.symbol = symbol_;

    result.version = version_.load(
        std::memory_order_acquire
    );

    result.quote_updates = quote_updates_;
    result.trade_updates = trade_updates_;

    result.rejected_updates = rejected_updates_;
    result.out_of_sequence_updates =
        out_of_sequence_updates_;

    if (last_quote_.has_value()) {
        const auto& quote = *last_quote_;

        result.last_quote_timestamp_ns =
            quote.timestamp_ns;

        result.last_quote_received_ns =
            quote.received_ns;

        result.bid_price_ticks =
            quote.bid_price_ticks;

        result.bid_size =
            quote.bid_size;

        result.bid_exchange =
            quote.bid_exchange;

        result.ask_price_ticks =
            quote.ask_price_ticks;

        result.ask_size =
            quote.ask_size;

        result.ask_exchange =
            quote.ask_exchange;
    }

    if (last_trade_.has_value()) {
        const auto& trade = *last_trade_;

        result.last_trade_timestamp_ns =
            trade.timestamp_ns;

        result.last_trade_received_ns =
            trade.received_ns;

        result.last_trade_price_ticks =
            trade.price_ticks;

        result.last_trade_size =
            trade.size;

        result.last_trade_id =
            trade.trade_id;

        result.last_trade_exchange =
            trade.exchange;
    }

    if (
        !last_quote_.has_value() &&
        !last_trade_.has_value()
    ) {
        result.status = MarketDataStatus::Empty;
    } else if (
        last_quote_.has_value() &&
        !last_quote_->valid()
    ) {
        result.status = MarketDataStatus::Invalid;
    } else {
        // Staleness is intentionally not inferred here because this method
        // has no caller-supplied "now". RiskEngine owns the freshness policy.
        result.status = MarketDataStatus::Valid;
    }

    return result;
}


std::optional<MarketQuoteUpdate>
ObservedMarketBook::last_quote() const {
    std::shared_lock lock(mutex_);
    return last_quote_;
}


std::optional<MarketTradeUpdate>
ObservedMarketBook::last_trade() const {
    std::shared_lock lock(mutex_);
    return last_trade_;
}


std::optional<std::int64_t>
ObservedMarketBook::best_bid_ticks() const {
    std::shared_lock lock(mutex_);

    if (!last_quote_.has_value()) {
        return std::nullopt;
    }

    return last_quote_->bid_price_ticks;
}


std::optional<std::int64_t>
ObservedMarketBook::best_ask_ticks() const {
    std::shared_lock lock(mutex_);

    if (!last_quote_.has_value()) {
        return std::nullopt;
    }

    return last_quote_->ask_price_ticks;
}


std::optional<std::int64_t>
ObservedMarketBook::midpoint_ticks() const {
    std::shared_lock lock(mutex_);

    if (
        !last_quote_.has_value() ||
        !last_quote_->valid()
    ) {
        return std::nullopt;
    }

    return last_quote_->midpoint_ticks();
}


std::optional<std::int64_t>
ObservedMarketBook::spread_ticks() const {
    std::shared_lock lock(mutex_);

    if (
        !last_quote_.has_value() ||
        !last_quote_->valid()
    ) {
        return std::nullopt;
    }

    return last_quote_->spread_ticks();
}


bool ObservedMarketBook::has_valid_quote() const {
    std::shared_lock lock(mutex_);

    return last_quote_.has_value() &&
           last_quote_->valid();
}


void ObservedMarketBook::clear() {
    std::unique_lock lock(mutex_);

    last_quote_.reset();
    last_trade_.reset();

    last_quote_sequence_ = 0;
    last_trade_sequence_ = 0;

    quote_updates_ = 0;
    trade_updates_ = 0;

    rejected_updates_ = 0;
    out_of_sequence_updates_ = 0;

    version_.fetch_add(
        1,
        std::memory_order_release
    );
}


// ============================================================================
// LIMIT ORDER BOOK
// ============================================================================

LimitOrderBook::LimitOrderBook(
    std::string symbol
)
    : symbol_(std::move(symbol)) {

    if (!valid_symbol_text(symbol_)) {
        throw std::invalid_argument(
            "LimitOrderBook: invalid symbol"
        );
    }
}


void LimitOrderBook::validate_new_order(
    const std::uint64_t id,
    const std::string& symbol,
    const std::uint64_t quantity,
    const std::int64_t price_ticks
) const {
    if (id == 0) {
        throw std::invalid_argument(
            "LimitOrderBook: order id must be nonzero"
        );
    }

    if (symbol != symbol_) {
        throw std::invalid_argument(
            "LimitOrderBook: symbol mismatch"
        );
    }

    if (quantity == 0) {
        throw std::invalid_argument(
            "LimitOrderBook: quantity must be positive"
        );
    }

    if (price_ticks <= 0) {
        throw std::invalid_argument(
            "LimitOrderBook: price_ticks must be positive"
        );
    }

    if (seen_order_ids_.contains(id)) {
        throw std::invalid_argument(
            "LimitOrderBook: order id has already been used"
        );
    }
}


std::vector<Execution>
LimitOrderBook::add_limit(
    const std::uint64_t id,
    const std::string& symbol,
    const Side side,
    const std::uint64_t quantity,
    const std::int64_t price_ticks
) {
    std::scoped_lock lock(mutex_);

    validate_new_order(
        id,
        symbol,
        quantity,
        price_ticks
    );

    auto order = std::make_shared<Order>();

    order->id = id;
    order->symbol = symbol_;
    order->side = side;
    order->quantity = quantity;
    order->filled_quantity = 0;
    order->remaining_quantity = quantity;
    order->price_ticks = price_ticks;

    order->sequence = next_counter(
        next_order_sequence_,
        "order sequence"
    );

    order->status = OrderStatus::Accepted;

    // Reserve the ID before matching so an exception after this point cannot
    // permit unsafe ID reuse.
    seen_order_ids_.insert(id);

    try {
        auto executions =
            match_and_rest(order);

        version_.fetch_add(
            1,
            std::memory_order_release
        );

        return executions;
    } catch (...) {
        // The order ID remains permanently seen by design. Once an order has
        // crossed the authority boundary, reuse is prohibited even if a later
        // internal failure occurs.
        throw;
    }
}


std::vector<Execution>
LimitOrderBook::amend(
    const std::uint64_t order_id,
    const std::int64_t new_price_ticks,
    const std::uint64_t new_total_quantity
) {
    std::scoped_lock lock(mutex_);

    if (new_price_ticks <= 0) {
        throw std::invalid_argument(
            "LimitOrderBook::amend: price must be positive"
        );
    }

    const auto locator_it =
        locators_.find(order_id);

    if (locator_it == locators_.end()) {
        throw std::out_of_range(
            "LimitOrderBook::amend: active order not found"
        );
    }

    auto order =
        *(locator_it->second.iterator);

    if (!order || order->is_terminal()) {
        throw std::logic_error(
            "LimitOrderBook::amend: order is terminal"
        );
    }

    if (
        new_total_quantity <
        order->filled_quantity
    ) {
        throw std::invalid_argument(
            "LimitOrderBook::amend: total quantity "
            "cannot be below filled quantity"
        );
    }

    const auto old_price =
        order->price_ticks;

    const auto old_total =
        order->quantity;

    const auto old_remaining =
        order->remaining_quantity;

    const auto new_remaining =
        new_total_quantity -
        order->filled_quantity;

    // ------------------------------------------------------------------------
    // CANCEL REMAINDER
    // ------------------------------------------------------------------------

    if (new_remaining == 0) {
        remove_from_book(order_id);

        order->quantity =
            new_total_quantity;

        order->remaining_quantity = 0;
        order->status = OrderStatus::Cancelled;

        version_.fetch_add(
            1,
            std::memory_order_release
        );

        return {};
    }

    const bool same_price =
        new_price_ticks == old_price;

    const bool quantity_reduction =
        new_total_quantity <= old_total;

    // ------------------------------------------------------------------------
    // SAME PRICE + REDUCTION: KEEP PRIORITY
    // ------------------------------------------------------------------------

    if (same_price && quantity_reduction) {
        order->quantity =
            new_total_quantity;

        order->remaining_quantity =
            new_remaining;

        order->status =
            order->filled_quantity > 0
                ? OrderStatus::PartiallyFilled
                : OrderStatus::Accepted;

        version_.fetch_add(
            1,
            std::memory_order_release
        );

        return {};
    }

    // ------------------------------------------------------------------------
    // PRICE CHANGE OR QUANTITY INCREASE: LOSE PRIORITY
    // ------------------------------------------------------------------------

    remove_from_book(order_id);

    order->price_ticks =
        new_price_ticks;

    order->quantity =
        new_total_quantity;

    order->remaining_quantity =
        new_remaining;

    order->sequence = next_counter(
        next_order_sequence_,
        "order sequence"
    );

    order->status =
        order->filled_quantity > 0
            ? OrderStatus::PartiallyFilled
            : OrderStatus::Accepted;

    try {
        auto executions =
            match_and_rest(order);

        version_.fetch_add(
            1,
            std::memory_order_release
        );

        return executions;
    } catch (...) {
        // Attempt to restore a coherent resting order if matching failed
        // before the order became terminal.
        if (
            order->remaining_quantity > 0 &&
            !locators_.contains(order_id)
        ) {
            order->price_ticks = old_price;
            order->quantity = old_total;
            order->remaining_quantity =
                old_remaining;

            insert_resting(order);
        }

        throw;
    }
}


Order LimitOrderBook::cancel(
    const std::uint64_t order_id
) {
    std::scoped_lock lock(mutex_);

    const auto locator_it =
        locators_.find(order_id);

    if (locator_it == locators_.end()) {
        throw std::out_of_range(
            "LimitOrderBook::cancel: active order not found"
        );
    }

    auto order =
        *(locator_it->second.iterator);

    remove_from_book(order_id);

    order->status = OrderStatus::Cancelled;

    version_.fetch_add(
        1,
        std::memory_order_release
    );

    return *order;
}


std::optional<Order>
LimitOrderBook::get(
    const std::uint64_t order_id
) const {
    std::scoped_lock lock(mutex_);

    const auto it =
        locators_.find(order_id);

    if (it == locators_.end()) {
        return std::nullopt;
    }

    const auto& order =
        *(it->second.iterator);

    if (!order) {
        return std::nullopt;
    }

    return *order;
}


bool LimitOrderBook::contains(
    const std::uint64_t order_id
) const {
    std::scoped_lock lock(mutex_);
    return locators_.contains(order_id);
}


bool LimitOrderBook::order_id_seen(
    const std::uint64_t order_id
) const {
    std::scoped_lock lock(mutex_);
    return seen_order_ids_.contains(order_id);
}


std::optional<std::int64_t>
LimitOrderBook::best_bid_ticks() const {
    std::scoped_lock lock(mutex_);

    if (bids_.empty()) {
        return std::nullopt;
    }

    return bids_.begin()->first;
}


std::optional<std::int64_t>
LimitOrderBook::best_ask_ticks() const {
    std::scoped_lock lock(mutex_);

    if (asks_.empty()) {
        return std::nullopt;
    }

    return asks_.begin()->first;
}


std::optional<Order>
LimitOrderBook::best_bid_order() const {
    std::scoped_lock lock(mutex_);

    if (bids_.empty()) {
        return std::nullopt;
    }

    const auto& level =
        bids_.begin()->second;

    if (level.empty() || !level.front()) {
        return std::nullopt;
    }

    return *level.front();
}


std::optional<Order>
LimitOrderBook::best_ask_order() const {
    std::scoped_lock lock(mutex_);

    if (asks_.empty()) {
        return std::nullopt;
    }

    const auto& level =
        asks_.begin()->second;

    if (level.empty() || !level.front()) {
        return std::nullopt;
    }

    return *level.front();
}


std::vector<Order>
LimitOrderBook::bids() const {
    std::scoped_lock lock(mutex_);

    std::vector<Order> result;
    result.reserve(locators_.size());

    for (const auto& [price, level] : bids_) {
        static_cast<void>(price);

        for (const auto& order : level) {
            if (order) {
                result.push_back(*order);
            }
        }
    }

    return result;
}


std::vector<Order>
LimitOrderBook::asks() const {
    std::scoped_lock lock(mutex_);

    std::vector<Order> result;
    result.reserve(locators_.size());

    for (const auto& [price, level] : asks_) {
        static_cast<void>(price);

        for (const auto& order : level) {
            if (order) {
                result.push_back(*order);
            }
        }
    }

    return result;
}


std::vector<Order>
LimitOrderBook::orders() const {
    std::scoped_lock lock(mutex_);

    std::vector<Order> result;
    result.reserve(locators_.size());

    for (const auto& [price, level] : bids_) {
        static_cast<void>(price);

        for (const auto& order : level) {
            if (order) {
                result.push_back(*order);
            }
        }
    }

    for (const auto& [price, level] : asks_) {
        static_cast<void>(price);

        for (const auto& order : level) {
            if (order) {
                result.push_back(*order);
            }
        }
    }

    return result;
}


BookSnapshot LimitOrderBook::snapshot(
    const std::size_t max_levels
) const {
    std::scoped_lock lock(mutex_);

    BookSnapshot result;

    result.symbol = symbol_;

    result.active_order_count =
        locators_.size();

    result.version = version_.load(
        std::memory_order_acquire
    );

    if (!bids_.empty()) {
        result.best_bid_ticks =
            bids_.begin()->first;
    }

    if (!asks_.empty()) {
        result.best_ask_ticks =
            asks_.begin()->first;
    }

    const std::size_t bid_limit =
        max_levels == 0
            ? bids_.size()
            : std::min(max_levels, bids_.size());

    const std::size_t ask_limit =
        max_levels == 0
            ? asks_.size()
            : std::min(max_levels, asks_.size());

    result.bids.reserve(bid_limit);
    result.asks.reserve(ask_limit);

    std::size_t count = 0;

    for (const auto& [price, level] : bids_) {
        if (
            max_levels != 0 &&
            count >= max_levels
        ) {
            break;
        }

        result.bids.push_back(
            make_level_snapshot(
                price,
                level
            )
        );

        ++count;
    }

    count = 0;

    for (const auto& [price, level] : asks_) {
        if (
            max_levels != 0 &&
            count >= max_levels
        ) {
            break;
        }

        result.asks.push_back(
            make_level_snapshot(
                price,
                level
            )
        );

        ++count;
    }

    return result;
}


std::size_t LimitOrderBook::size() const {
    std::scoped_lock lock(mutex_);
    return locators_.size();
}


bool LimitOrderBook::empty() const {
    std::scoped_lock lock(mutex_);
    return locators_.empty();
}


void LimitOrderBook::clear() {
    std::scoped_lock lock(mutex_);

    bids_.clear();
    asks_.clear();
    locators_.clear();

    // seen_order_ids_ intentionally remains populated. clear() resets active
    // book state but does not make historical IDs reusable.

    version_.fetch_add(
        1,
        std::memory_order_release
    );
}


// ============================================================================
// RESTING ORDER MANAGEMENT
// ============================================================================

void LimitOrderBook::insert_resting(
    const OrderPtr& order
) {
    if (!order) {
        throw std::invalid_argument(
            "LimitOrderBook::insert_resting: null order"
        );
    }

    if (
        order->remaining_quantity == 0 ||
        order->price_ticks <= 0
    ) {
        throw std::logic_error(
            "LimitOrderBook::insert_resting: invalid active order"
        );
    }

    if (locators_.contains(order->id)) {
        throw std::logic_error(
            "LimitOrderBook::insert_resting: duplicate active id"
        );
    }

    if (order->side == Side::Buy) {
        auto [level_it, inserted] =
            bids_.try_emplace(
                order->price_ticks
            );

        static_cast<void>(inserted);

        auto& level = level_it->second;

        level.push_back(order);

        auto order_it =
            std::prev(level.end());

        try {
            locators_.emplace(
                order->id,
                Locator{
                    Side::Buy,
                    order->price_ticks,
                    order_it
                }
            );
        } catch (...) {
            level.erase(order_it);

            if (level.empty()) {
                bids_.erase(level_it);
            }

            throw;
        }

        return;
    }

    auto [level_it, inserted] =
        asks_.try_emplace(
            order->price_ticks
        );

    static_cast<void>(inserted);

    auto& level = level_it->second;

    level.push_back(order);

    auto order_it =
        std::prev(level.end());

    try {
        locators_.emplace(
            order->id,
            Locator{
                Side::Sell,
                order->price_ticks,
                order_it
            }
        );
    } catch (...) {
        level.erase(order_it);

        if (level.empty()) {
            asks_.erase(level_it);
        }

        throw;
    }
}


void LimitOrderBook::remove_from_book(
    const std::uint64_t order_id
) {
    const auto locator_it =
        locators_.find(order_id);

    if (locator_it == locators_.end()) {
        throw std::out_of_range(
            "LimitOrderBook::remove_from_book: "
            "active order not found"
        );
    }

    const Locator locator =
        locator_it->second;

    if (locator.side == Side::Buy) {
        const auto level_it =
            bids_.find(locator.price_ticks);

        if (level_it == bids_.end()) {
            throw std::logic_error(
                "LimitOrderBook: bid locator references "
                "missing price level"
            );
        }

        level_it->second.erase(
            locator.iterator
        );

        if (level_it->second.empty()) {
            bids_.erase(level_it);
        }
    } else {
        const auto level_it =
            asks_.find(locator.price_ticks);

        if (level_it == asks_.end()) {
            throw std::logic_error(
                "LimitOrderBook: ask locator references "
                "missing price level"
            );
        }

        level_it->second.erase(
            locator.iterator
        );

        if (level_it->second.empty()) {
            asks_.erase(level_it);
        }
    }

    locators_.erase(locator_it);
}


// ============================================================================
// MATCHING
// ============================================================================

std::vector<Execution>
LimitOrderBook::match_and_rest(
    const OrderPtr& incoming
) {
    if (!incoming) {
        throw std::invalid_argument(
            "LimitOrderBook::match_and_rest: null order"
        );
    }

    std::vector<Execution> executions;

    if (incoming->side == Side::Buy) {
        match_buy(
            incoming,
            executions
        );
    } else {
        match_sell(
            incoming,
            executions
        );
    }

    if (incoming->remaining_quantity > 0) {
        incoming->status =
            incoming->filled_quantity > 0
                ? OrderStatus::PartiallyFilled
                : OrderStatus::Accepted;

        insert_resting(incoming);
    } else {
        incoming->status =
            OrderStatus::Filled;
    }

    return executions;
}


void LimitOrderBook::match_buy(
    const OrderPtr& incoming,
    std::vector<Execution>& executions
) {
    while (
        incoming->remaining_quantity > 0 &&
        !asks_.empty()
    ) {
        auto level_it = asks_.begin();

        const auto maker_price =
            level_it->first;

        if (maker_price > incoming->price_ticks) {
            break;
        }

        auto& level = level_it->second;

        while (
            incoming->remaining_quantity > 0 &&
            !level.empty()
        ) {
            auto maker_it = level.begin();
            auto maker = *maker_it;

            if (!maker) {
                throw std::logic_error(
                    "LimitOrderBook: null ask order"
                );
            }

            const std::uint64_t fill_quantity =
                std::min(
                    incoming->remaining_quantity,
                    maker->remaining_quantity
                );

            if (fill_quantity == 0) {
                throw std::logic_error(
                    "LimitOrderBook: zero-sized match"
                );
            }

            maker->filled_quantity +=
                fill_quantity;

            maker->remaining_quantity -=
                fill_quantity;

            incoming->filled_quantity +=
                fill_quantity;

            incoming->remaining_quantity -=
                fill_quantity;

            executions.push_back(
                make_execution(
                    maker,
                    incoming,
                    fill_quantity,
                    maker_price
                )
            );

            if (maker->remaining_quantity == 0) {
                maker->status =
                    OrderStatus::Filled;

                locators_.erase(maker->id);
                level.erase(maker_it);
            } else {
                maker->status =
                    OrderStatus::PartiallyFilled;
            }
        }

        if (level.empty()) {
            asks_.erase(level_it);
        }
    }
}


void LimitOrderBook::match_sell(
    const OrderPtr& incoming,
    std::vector<Execution>& executions
) {
    while (
        incoming->remaining_quantity > 0 &&
        !bids_.empty()
    ) {
        auto level_it = bids_.begin();

        const auto maker_price =
            level_it->first;

        if (maker_price < incoming->price_ticks) {
            break;
        }

        auto& level = level_it->second;

        while (
            incoming->remaining_quantity > 0 &&
            !level.empty()
        ) {
            auto maker_it = level.begin();
            auto maker = *maker_it;

            if (!maker) {
                throw std::logic_error(
                    "LimitOrderBook: null bid order"
                );
            }

            const std::uint64_t fill_quantity =
                std::min(
                    incoming->remaining_quantity,
                    maker->remaining_quantity
                );

            if (fill_quantity == 0) {
                throw std::logic_error(
                    "LimitOrderBook: zero-sized match"
                );
            }

            maker->filled_quantity +=
                fill_quantity;

            maker->remaining_quantity -=
                fill_quantity;

            incoming->filled_quantity +=
                fill_quantity;

            incoming->remaining_quantity -=
                fill_quantity;

            executions.push_back(
                make_execution(
                    maker,
                    incoming,
                    fill_quantity,
                    maker_price
                )
            );

            if (maker->remaining_quantity == 0) {
                maker->status =
                    OrderStatus::Filled;

                locators_.erase(maker->id);
                level.erase(maker_it);
            } else {
                maker->status =
                    OrderStatus::PartiallyFilled;
            }
        }

        if (level.empty()) {
            bids_.erase(level_it);
        }
    }
}


Execution LimitOrderBook::make_execution(
    const OrderPtr& maker,
    const OrderPtr& taker,
    const std::uint64_t quantity,
    const std::int64_t price_ticks
) {
    if (!maker || !taker) {
        throw std::invalid_argument(
            "LimitOrderBook::make_execution: null order"
        );
    }

    Execution execution;

    execution.execution_id = next_counter(
        next_execution_id_,
        "execution id"
    );

    execution.maker_id = maker->id;
    execution.taker_id = taker->id;

    execution.symbol = symbol_;

    execution.aggressor_side =
        taker->side;

    execution.quantity = quantity;
    execution.price_ticks = price_ticks;

    execution.sequence = next_counter(
        next_event_sequence_,
        "event sequence"
    );

    return execution;
}


// ============================================================================
// SNAPSHOT HELPERS
// ============================================================================

std::uint64_t
LimitOrderBook::aggregate_level_quantity(
    const PriceLevel& level
) {
    std::uint64_t total = 0;

    for (const auto& order : level) {
        if (!order) {
            continue;
        }

        if (
            order->remaining_quantity >
            std::numeric_limits<std::uint64_t>::max() -
            total
        ) {
            throw std::overflow_error(
                "LimitOrderBook: price-level quantity overflow"
            );
        }

        total += order->remaining_quantity;
    }

    return total;
}


PriceLevelSnapshot
LimitOrderBook::make_level_snapshot(
    const std::int64_t price_ticks,
    const PriceLevel& level
) {
    PriceLevelSnapshot result;

    result.price_ticks = price_ticks;

    result.total_quantity =
        aggregate_level_quantity(level);

    result.order_count =
        level.size();

    return result;
}


// ============================================================================
// ORDER BOOK ENGINE
// ============================================================================

OrderBookEngine::OrderBookEngine(
    std::string symbol
)
    : symbol_(std::move(symbol)),
      market_(symbol_),
      simulated_(symbol_) {

    if (!valid_symbol_text(symbol_)) {
        throw std::invalid_argument(
            "OrderBookEngine: invalid symbol"
        );
    }
}


bool OrderBookEngine::on_market_quote(
    const MarketQuoteUpdate& update
) {
    return market_.on_quote(update);
}


bool OrderBookEngine::on_market_trade(
    const MarketTradeUpdate& update
) {
    return market_.on_trade(update);
}


std::vector<Execution>
OrderBookEngine::add_limit(
    const std::uint64_t id,
    const Side side,
    const std::uint64_t quantity,
    const std::int64_t price_ticks
) {
    return simulated_.add_limit(
        id,
        symbol_,
        side,
        quantity,
        price_ticks
    );
}


std::vector<Execution>
OrderBookEngine::amend(
    const std::uint64_t order_id,
    const std::int64_t new_price_ticks,
    const std::uint64_t new_total_quantity
) {
    return simulated_.amend(
        order_id,
        new_price_ticks,
        new_total_quantity
    );
}


Order OrderBookEngine::cancel(
    const std::uint64_t order_id
) {
    return simulated_.cancel(order_id);
}


ObservedMarketSnapshot
OrderBookEngine::market_snapshot() const {
    return market_.snapshot();
}


BookSnapshot
OrderBookEngine::simulated_snapshot(
    const std::size_t max_levels
) const {
    return simulated_.snapshot(max_levels);
}


OrderBookEngineSnapshot
OrderBookEngine::snapshot(
    const std::uint64_t captured_ns,
    const std::size_t max_levels
) const {
    OrderBookEngineSnapshot result;

    result.symbol = symbol_;

    // The components intentionally own independent locks. This is a coherent
    // pair of versioned snapshots, not a globally atomic market+order state.
    result.market =
        market_.snapshot();

    result.simulated =
        simulated_.snapshot(max_levels);

    result.captured_ns =
        captured_ns;

    return result;
}


// ============================================================================
// MULTI-SYMBOL REGISTRY
// ============================================================================

void OrderBookRegistry::add_symbol(
    const std::string& symbol
) {
    if (!valid_symbol_text(symbol)) {
        throw std::invalid_argument(
            "OrderBookRegistry::add_symbol: invalid symbol"
        );
    }

    if (symbol == "VIX" ||
        symbol == "VXN") {
        throw std::invalid_argument(
            "OrderBookRegistry::add_symbol: "
            "VIX/VXN are read-only statistical factors"
        );
    }

    std::unique_lock lock(mutex_);

    if (books_.contains(symbol)) {
        return;
    }

    books_.emplace(
        symbol,
        std::make_shared<OrderBookEngine>(
            symbol
        )
    );
}


bool OrderBookRegistry::contains(
    const std::string_view symbol
) const {
    std::shared_lock lock(mutex_);

    return books_.find(
        std::string(symbol)
    ) != books_.end();
}


std::shared_ptr<OrderBookEngine>
OrderBookRegistry::get(
    const std::string_view symbol
) const {
    std::shared_lock lock(mutex_);

    const auto it =
        books_.find(
            std::string(symbol)
        );

    if (it == books_.end()) {
        return {};
    }

    return it->second;
}


bool OrderBookRegistry::on_market_quote(
    const MarketQuoteUpdate& update
) {
    std::shared_ptr<OrderBookEngine> book;

    {
        std::shared_lock lock(mutex_);

        const auto it =
            books_.find(update.symbol);

        if (it == books_.end()) {
            return false;
        }

        book = it->second;
    }

    return book->on_market_quote(update);
}


bool OrderBookRegistry::on_market_trade(
    const MarketTradeUpdate& update
) {
    std::shared_ptr<OrderBookEngine> book;

    {
        std::shared_lock lock(mutex_);

        const auto it =
            books_.find(update.symbol);

        if (it == books_.end()) {
            return false;
        }

        book = it->second;
    }

    return book->on_market_trade(update);
}


std::vector<std::string>
OrderBookRegistry::symbols() const {
    std::shared_lock lock(mutex_);

    std::vector<std::string> result;
    result.reserve(books_.size());

    for (const auto& [symbol, book] : books_) {
        static_cast<void>(book);
        result.push_back(symbol);
    }

    std::sort(
        result.begin(),
        result.end()
    );

    return result;
}


std::size_t OrderBookRegistry::size() const {
    std::shared_lock lock(mutex_);
    return books_.size();
}

} // namespace trading
