#include "market-book.hpp"

#include <algorithm>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <utility>

namespace trading {

namespace {

bool valid_symbol_text(const std::string_view symbol) noexcept {
    return !symbol.empty() && symbol.size() <= 32;
}

template <typename Map>
void copy_levels(
    const Map& source,
    std::vector<ObservedDepthLevel>& destination,
    const std::size_t max_depth
) {
    destination.clear();
    destination.reserve(
        std::min(max_depth, source.size())
    );

    std::size_t count = 0;
    for (const auto& [price, level] : source) {
        (void)price;

        if (count >= max_depth) {
            break;
        }

        destination.push_back(level);
        ++count;
    }
}

}  // namespace


bool MarketDepthUpdate::valid() const noexcept {
    if (!valid_symbol_text(symbol) ||
        timestamp_ns == 0 ||
        received_ns == 0) {
        return false;
    }

    switch (action) {
        case DepthUpdateAction::Upsert:
            return price_ticks > 0 && size > 0;

        case DepthUpdateAction::Delete:
            return price_ticks > 0;

        case DepthUpdateAction::ClearSide:
        case DepthUpdateAction::ClearBook:
            return true;
    }

    return false;
}


std::int64_t
TradeFlowSnapshot::signed_quote_volume() const noexcept {
    const auto max_i64 =
        static_cast<std::uint64_t>(
            std::numeric_limits<std::int64_t>::max()
        );

    const auto buy =
        std::min(at_or_above_ask_volume, max_i64);
    const auto sell =
        std::min(at_or_below_bid_volume, max_i64);

    return static_cast<std::int64_t>(buy) -
           static_cast<std::int64_t>(sell);
}


bool MarketDepthSnapshot::has_top_of_book() const noexcept {
    return best_bid_ticks().has_value() &&
           best_ask_ticks().has_value();
}


std::optional<std::int64_t>
MarketDepthSnapshot::best_bid_ticks() const noexcept {
    if (!bids.empty()) {
        return bids.front().price_ticks;
    }

    if (last_quote.has_value()) {
        return last_quote->bid_price_ticks;
    }

    return std::nullopt;
}


std::optional<std::int64_t>
MarketDepthSnapshot::best_ask_ticks() const noexcept {
    if (!asks.empty()) {
        return asks.front().price_ticks;
    }

    if (last_quote.has_value()) {
        return last_quote->ask_price_ticks;
    }

    return std::nullopt;
}


std::optional<std::int64_t>
MarketDepthSnapshot::midpoint_ticks() const noexcept {
    const auto bid = best_bid_ticks();
    const auto ask = best_ask_ticks();

    if (!bid.has_value() ||
        !ask.has_value() ||
        *bid >= *ask) {
        return std::nullopt;
    }

    return *bid + ((*ask - *bid) / 2);
}


std::optional<std::int64_t>
MarketDepthSnapshot::spread_ticks() const noexcept {
    const auto bid = best_bid_ticks();
    const auto ask = best_ask_ticks();

    if (!bid.has_value() ||
        !ask.has_value() ||
        *bid >= *ask) {
        return std::nullopt;
    }

    return *ask - *bid;
}


MarketDepthBook::MarketDepthBook(
    std::string symbol,
    const std::size_t trade_window
)
    : symbol_(std::move(symbol)),
      trade_window_(trade_window) {
    if (!valid_symbol_text(symbol_)) {
        throw std::invalid_argument(
            "MarketDepthBook: invalid symbol"
        );
    }

    if (symbol_ == "VIX" ||
        symbol_ == "VXN") {
        throw std::invalid_argument(
            "MarketDepthBook: VIX/VXN are factor observations, "
            "not executable equity depth books"
        );
    }

    if (trade_window_ == 0) {
        throw std::invalid_argument(
            "MarketDepthBook: trade_window must be positive"
        );
    }
}


bool MarketDepthBook::validate_symbol(
    const std::string_view symbol
) const noexcept {
    return symbol == symbol_;
}


bool MarketDepthBook::sequence_ok(
    const std::uint64_t incoming,
    const std::uint64_t previous
) const noexcept {
    return incoming == 0 ||
           previous == 0 ||
           incoming > previous;
}


bool MarketDepthBook::on_quote(
    const MarketQuoteUpdate& update
) {
    if (!validate_symbol(update.symbol) ||
        !update.valid()) {
        std::unique_lock lock(mutex_);
        ++rejected_updates_;
        return false;
    }

    std::unique_lock lock(mutex_);

    if (update.timestamp_ns <
        last_quote_timestamp_ns_) {
        ++rejected_updates_;
        ++out_of_sequence_updates_;
        return false;
    }

    if (!sequence_ok(
            update.sequence,
            last_quote_sequence_
        )) {
        ++rejected_updates_;
        ++out_of_sequence_updates_;
        return false;
    }

    last_quote_ = update;
    last_quote_timestamp_ns_ =
        update.timestamp_ns;

    if (update.sequence != 0) {
        last_quote_sequence_ =
            update.sequence;
    }

    ++quote_updates_;

    // IMPORTANT:
    // A consolidated quote is not Level-II. We intentionally do not insert
    // bid/ask quote sizes into depth_bids_/depth_asks_. Snapshot readers can
    // still use last_quote as the observed NBBO fallback.
    if (depth_source_ == DepthSource::Unknown) {
        depth_source_ =
            DepthSource::ConsolidatedQuote;
    }

    version_.fetch_add(
        1,
        std::memory_order_acq_rel
    );

    return true;
}


bool MarketDepthBook::on_trade(
    const MarketTradeUpdate& update
) {
    if (!validate_symbol(update.symbol) ||
        !update.valid()) {
        std::unique_lock lock(mutex_);
        ++rejected_updates_;
        return false;
    }

    std::unique_lock lock(mutex_);

    if (update.timestamp_ns <
        last_trade_timestamp_ns_) {
        ++rejected_updates_;
        ++out_of_sequence_updates_;
        return false;
    }

    if (!sequence_ok(
            update.sequence,
            last_trade_sequence_
        )) {
        ++rejected_updates_;
        ++out_of_sequence_updates_;
        return false;
    }

    TradeObservation observation;
    observation.trade = update;

    if (last_quote_.has_value()) {
        observation.bid_ticks =
            last_quote_->bid_price_ticks;
        observation.ask_ticks =
            last_quote_->ask_price_ticks;
    }

    trades_.push_back(
        std::move(observation)
    );

    trim_trade_window_locked();

    last_trade_ = update;
    last_trade_timestamp_ns_ =
        update.timestamp_ns;

    if (update.sequence != 0) {
        last_trade_sequence_ =
            update.sequence;
    }

    ++trade_updates_;

    version_.fetch_add(
        1,
        std::memory_order_acq_rel
    );

    return true;
}


bool MarketDepthBook::on_depth(
    const MarketDepthUpdate& update
) {
    if (!validate_symbol(update.symbol) ||
        !update.valid()) {
        std::unique_lock lock(mutex_);
        ++rejected_updates_;
        return false;
    }

    if (update.source ==
        DepthSource::ConsolidatedQuote) {
        // Consolidated NBBO belongs in on_quote(), not the L2 maps.
        std::unique_lock lock(mutex_);
        ++rejected_updates_;
        return false;
    }

    std::unique_lock lock(mutex_);

    if (update.timestamp_ns <
        last_depth_timestamp_ns_) {
        ++rejected_updates_;
        ++out_of_sequence_updates_;
        return false;
    }

    if (!sequence_ok(
            update.sequence,
            last_depth_sequence_
        )) {
        ++rejected_updates_;
        ++out_of_sequence_updates_;
        return false;
    }

    if (update.action ==
        DepthUpdateAction::ClearBook) {
        depth_bids_.clear();
        depth_asks_.clear();
    } else if (
        update.action ==
        DepthUpdateAction::ClearSide
    ) {
        if (update.side == Side::Buy) {
            depth_bids_.clear();
        } else {
            depth_asks_.clear();
        }
    } else if (
        update.action ==
        DepthUpdateAction::Delete
    ) {
        if (update.side == Side::Buy) {
            depth_bids_.erase(
                update.price_ticks
            );
        } else {
            depth_asks_.erase(
                update.price_ticks
            );
        }
    } else {
        ObservedDepthLevel level;
        level.price_ticks =
            update.price_ticks;
        level.size =
            update.size;
        level.order_count =
            update.order_count;
        level.exchange =
            update.exchange;
        level.provider =
            update.provider;
        level.timestamp_ns =
            update.timestamp_ns;
        level.received_ns =
            update.received_ns;
        level.sequence =
            update.sequence;
        level.source =
            update.source;

        if (update.side == Side::Buy) {
            depth_bids_[
                update.price_ticks
            ] = std::move(level);
        } else {
            depth_asks_[
                update.price_ticks
            ] = std::move(level);
        }
    }

    last_depth_timestamp_ns_ =
        update.timestamp_ns;

    if (update.sequence != 0) {
        last_depth_sequence_ =
            update.sequence;
    }

    depth_source_ =
        update.source;

    ++depth_updates_;

    version_.fetch_add(
        1,
        std::memory_order_acq_rel
    );

    return true;
}


void MarketDepthBook::trim_trade_window_locked() {
    while (trades_.size() > trade_window_) {
        trades_.pop_front();
    }
}


TradeFlowSnapshot
MarketDepthBook::trade_flow_locked() const {
    TradeFlowSnapshot result;

    for (const auto& observation : trades_) {
        const auto& trade =
            observation.trade;

        ++result.trade_count;

        if (
            std::numeric_limits<std::uint64_t>::max() -
                result.total_volume <
            trade.size
        ) {
            result.total_volume =
                std::numeric_limits<std::uint64_t>::max();
        } else {
            result.total_volume +=
                trade.size;
        }

        if (observation.bid_ticks.has_value() &&
            observation.ask_ticks.has_value()) {
            if (trade.price_ticks >=
                *observation.ask_ticks) {
                result.at_or_above_ask_volume +=
                    trade.size;
            } else if (
                trade.price_ticks <=
                *observation.bid_ticks
            ) {
                result.at_or_below_bid_volume +=
                    trade.size;
            } else {
                result.inside_spread_volume +=
                    trade.size;
            }
        } else {
            result.outside_quote_volume +=
                trade.size;
        }
    }

    if (last_trade_.has_value()) {
        result.last_trade_price_ticks =
            last_trade_->price_ticks;
        result.last_trade_size =
            last_trade_->size;
        result.last_trade_timestamp_ns =
            last_trade_->timestamp_ns;
    }

    return result;
}


MarketDepthSnapshot
MarketDepthBook::snapshot(
    const std::size_t max_depth,
    const std::uint64_t captured_ns
) const {
    MarketDepthSnapshot result;
    result.symbol = symbol_;
    result.captured_ns = captured_ns;

    std::shared_lock lock(mutex_);

    result.version =
        version_.load(
            std::memory_order_acquire
        );

    result.last_quote =
        last_quote_;
    result.last_trade =
        last_trade_;

    result.depth_source =
        depth_source_;

    copy_levels(
        depth_bids_,
        result.bids,
        max_depth
    );

    copy_levels(
        depth_asks_,
        result.asks,
        max_depth
    );

    result.trade_flow =
        trade_flow_locked();

    result.quote_updates =
        quote_updates_;
    result.trade_updates =
        trade_updates_;
    result.depth_updates =
        depth_updates_;
    result.rejected_updates =
        rejected_updates_;
    result.out_of_sequence_updates =
        out_of_sequence_updates_;

    if (last_quote_.has_value() ||
        !depth_bids_.empty() ||
        !depth_asks_.empty() ||
        last_trade_.has_value()) {
        result.status =
            MarketDataStatus::Valid;
    }

    const auto bid =
        result.best_bid_ticks();
    const auto ask =
        result.best_ask_ticks();

    if (bid.has_value() &&
        ask.has_value() &&
        *bid >= *ask) {
        result.status =
            MarketDataStatus::Invalid;
    }

    return result;
}


MarketDepthSoA
MarketDepthBook::export_soa(
    const std::size_t depth,
    const std::uint64_t captured_ns
) const {
    MarketDepthSoA result;
    result.symbol = symbol_;
    result.depth = depth;
    result.captured_ns =
        captured_ns;

    result.bid_price_ticks.assign(
        depth,
        0
    );
    result.bid_sizes.assign(
        depth,
        0
    );
    result.bid_order_counts.assign(
        depth,
        0
    );

    result.ask_price_ticks.assign(
        depth,
        0
    );
    result.ask_sizes.assign(
        depth,
        0
    );
    result.ask_order_counts.assign(
        depth,
        0
    );

    std::shared_lock lock(mutex_);

    result.version =
        version_.load(
            std::memory_order_acquire
        );

    std::size_t index = 0;
    for (const auto& [price, level] :
         depth_bids_) {
        if (index >= depth) {
            break;
        }

        result.bid_price_ticks[index] =
            price;
        result.bid_sizes[index] =
            level.size;
        result.bid_order_counts[index] =
            level.order_count;

        ++index;
    }

    index = 0;
    for (const auto& [price, level] :
         depth_asks_) {
        if (index >= depth) {
            break;
        }

        result.ask_price_ticks[index] =
            price;
        result.ask_sizes[index] =
            level.size;
        result.ask_order_counts[index] =
            level.order_count;

        ++index;
    }

    // When no genuine provider depth exists, export only observed NBBO as
    // level 0. This is explicitly one level, never fabricated deeper depth.
    if (depth > 0 &&
        depth_bids_.empty() &&
        depth_asks_.empty() &&
        last_quote_.has_value()) {
        result.bid_price_ticks[0] =
            last_quote_->bid_price_ticks;
        result.bid_sizes[0] =
            last_quote_->bid_size;

        result.ask_price_ticks[0] =
            last_quote_->ask_price_ticks;
        result.ask_sizes[0] =
            last_quote_->ask_size;
    }

    return result;
}


void MarketDepthBook::clear() {
    std::unique_lock lock(mutex_);

    last_quote_.reset();
    last_trade_.reset();

    depth_bids_.clear();
    depth_asks_.clear();
    trades_.clear();

    last_quote_sequence_ = 0;
    last_trade_sequence_ = 0;
    last_depth_sequence_ = 0;

    last_quote_timestamp_ns_ = 0;
    last_trade_timestamp_ns_ = 0;
    last_depth_timestamp_ns_ = 0;

    quote_updates_ = 0;
    trade_updates_ = 0;
    depth_updates_ = 0;
    rejected_updates_ = 0;
    out_of_sequence_updates_ = 0;

    depth_source_ =
        DepthSource::Unknown;

    version_.fetch_add(
        1,
        std::memory_order_acq_rel
    );
}


MarketDepthRegistry::MarketDepthRegistry(
    const std::size_t trade_window
)
    : trade_window_(trade_window) {
    if (trade_window_ == 0) {
        throw std::invalid_argument(
            "MarketDepthRegistry: trade_window must be positive"
        );
    }
}


bool MarketDepthRegistry::add_symbol(
    std::string symbol
) {
    if (!valid_symbol_text(symbol) ||
        symbol == "VIX" ||
        symbol == "VXN") {
        return false;
    }

    std::unique_lock lock(mutex_);

    if (books_.contains(symbol)) {
        return false;
    }

    auto book =
        std::make_shared<MarketDepthBook>(
            symbol,
            trade_window_
        );

    books_.emplace(
        std::move(symbol),
        std::move(book)
    );

    return true;
}


bool MarketDepthRegistry::contains(
    const std::string_view symbol
) const {
    std::shared_lock lock(mutex_);

    return books_.find(
        std::string(symbol)
    ) != books_.end();
}


std::shared_ptr<MarketDepthBook>
MarketDepthRegistry::get(
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


bool MarketDepthRegistry::on_quote(
    const MarketQuoteUpdate& update
) {
    const auto book =
        get(update.symbol);

    return book &&
           book->on_quote(update);
}


bool MarketDepthRegistry::on_trade(
    const MarketTradeUpdate& update
) {
    const auto book =
        get(update.symbol);

    return book &&
           book->on_trade(update);
}


bool MarketDepthRegistry::on_depth(
    const MarketDepthUpdate& update
) {
    const auto book =
        get(update.symbol);

    return book &&
           book->on_depth(update);
}


std::vector<std::string>
MarketDepthRegistry::symbols() const {
    std::vector<std::string> result;

    std::shared_lock lock(mutex_);

    result.reserve(
        books_.size()
    );

    for (const auto& [symbol, book] :
         books_) {
        (void)book;
        result.push_back(symbol);
    }

    std::sort(
        result.begin(),
        result.end()
    );

    return result;
}

}  // namespace trading
