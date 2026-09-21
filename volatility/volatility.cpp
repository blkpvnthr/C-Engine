#include "volatility_factor_book.hpp"

#include <algorithm>
#include <limits>
#include <mutex>

namespace trading {

namespace {

std::uint64_t abs_diff(
    const std::uint64_t lhs,
    const std::uint64_t rhs
) noexcept {
    return lhs >= rhs
        ? lhs - rhs
        : rhs - lhs;
}

std::uint64_t market_timestamp(
    const ObservedMarketSnapshot& snapshot
) noexcept {
    std::uint64_t result = 0;

    result = std::max(
        snapshot.last_quote_timestamp_ns,
        snapshot.last_trade_timestamp_ns
    );

    return result;
}

bool market_fresh(
    const ObservedMarketSnapshot& snapshot,
    const std::uint64_t now_ns,
    const std::uint64_t max_age_ns
) noexcept {
    if (snapshot.status != MarketDataStatus::Valid) {
        return false;
    }

    const auto timestamp =
        market_timestamp(snapshot);

    if (timestamp == 0 ||
        now_ns < timestamp) {
        return false;
    }

    return (now_ns - timestamp) <=
           max_age_ns;
}

}  // namespace


bool VolatilityFactorUpdate::valid() const noexcept {
    return VolatilityFactorBook::supported(symbol) &&
           value_ticks > 0 &&
           timestamp_ns != 0 &&
           received_ns != 0;
}


bool VolatilityFactorState::usable(
    const std::uint64_t now_ns,
    const std::uint64_t max_age_ns
) const noexcept {
    return status == FactorStatus::Valid &&
           value_ticks.has_value() &&
           age_ns(now_ns) <= max_age_ns;
}


std::uint64_t VolatilityFactorState::age_ns(
    const std::uint64_t now_ns
) const noexcept {
    if (timestamp_ns == 0 ||
        now_ns < timestamp_ns) {
        return std::numeric_limits<
            std::uint64_t
        >::max();
    }

    return now_ns - timestamp_ns;
}


bool VolatilityFactorSnapshot::complete() const noexcept {
    return vix.has_value() &&
           vxn.has_value();
}


bool VolatilityFactorSnapshot::fresh(
    const std::uint64_t max_age_ns
) const noexcept {
    return complete() &&
           vix->usable(captured_ns, max_age_ns) &&
           vxn->usable(captured_ns, max_age_ns);
}


std::optional<std::int64_t>
VolatilityFactorSnapshot::vxn_minus_vix_ticks() const noexcept {
    if (!vix.has_value() ||
        !vxn.has_value() ||
        !vix->value_ticks.has_value() ||
        !vxn->value_ticks.has_value()) {
        return std::nullopt;
    }

    return *vxn->value_ticks -
           *vix->value_ticks;
}


std::uint64_t
VolatilityFactorSnapshot::source_skew_ns() const noexcept {
    if (!complete()) {
        return std::numeric_limits<
            std::uint64_t
        >::max();
    }

    return abs_diff(
        vix->timestamp_ns,
        vxn->timestamp_ns
    );
}


bool VolatilityFactorBook::supported(
    const std::string_view symbol
) noexcept {
    return symbol == "VIX" ||
           symbol == "VXN";
}


VolatilityFactorState
VolatilityFactorBook::to_public_state(
    const InternalState& state
) {
    VolatilityFactorState result;

    result.symbol =
        state.update.symbol;

    result.status =
        state.update.valid()
            ? FactorStatus::Valid
            : FactorStatus::Invalid;

    if (state.update.value_ticks > 0) {
        result.value_ticks =
            state.update.value_ticks;
    }

    result.timestamp_ns =
        state.update.timestamp_ns;
    result.received_ns =
        state.update.received_ns;
    result.sequence =
        state.update.sequence;
    result.version =
        state.version;
    result.provider =
        state.update.provider;

    result.accepted_updates =
        state.accepted_updates;
    result.rejected_updates =
        state.rejected_updates;
    result.out_of_sequence_updates =
        state.out_of_sequence_updates;

    return result;
}


bool VolatilityFactorBook::on_update(
    const VolatilityFactorUpdate& update
) {
    if (!supported(update.symbol)) {
        return false;
    }

    std::unique_lock lock(mutex_);

    auto& state =
        states_[update.symbol];

    if (!update.valid()) {
        ++state.rejected_updates;
        return false;
    }

    if (state.update.timestamp_ns != 0) {
        if (update.timestamp_ns <
            state.update.timestamp_ns) {
            ++state.rejected_updates;
            ++state.out_of_sequence_updates;
            return false;
        }

        if (update.sequence != 0 &&
            state.update.sequence != 0 &&
            update.sequence <=
                state.update.sequence) {
            ++state.rejected_updates;
            ++state.out_of_sequence_updates;
            return false;
        }
    }

    state.update = update;
    ++state.accepted_updates;

    state.version =
        version_.fetch_add(
            1,
            std::memory_order_acq_rel
        ) + 1;

    return true;
}


std::optional<VolatilityFactorState>
VolatilityFactorBook::get(
    const std::string_view symbol
) const {
    if (!supported(symbol)) {
        return std::nullopt;
    }

    std::shared_lock lock(mutex_);

    const auto it =
        states_.find(std::string(symbol));

    if (it == states_.end() ||
        it->second.update.timestamp_ns == 0) {
        return std::nullopt;
    }

    return to_public_state(
        it->second
    );
}


VolatilityFactorSnapshot
VolatilityFactorBook::snapshot(
    const std::uint64_t captured_ns
) const {
    VolatilityFactorSnapshot result;
    result.captured_ns =
        captured_ns;

    std::shared_lock lock(mutex_);

    const auto vix =
        states_.find("VIX");
    if (vix != states_.end() &&
        vix->second.update.timestamp_ns != 0) {
        result.vix =
            to_public_state(vix->second);
    }

    const auto vxn =
        states_.find("VXN");
    if (vxn != states_.end() &&
        vxn->second.update.timestamp_ns != 0) {
        result.vxn =
            to_public_state(vxn->second);
    }

    result.version =
        version_.load(
            std::memory_order_acquire
        );

    return result;
}


void VolatilityFactorBook::clear() {
    std::unique_lock lock(mutex_);

    states_.clear();

    version_.fetch_add(
        1,
        std::memory_order_acq_rel
    );
}


bool StatArbMarketSnapshot::has_core_equities() const noexcept {
    return qqq.has_value() &&
           sqqq.has_value();
}


bool StatArbMarketSnapshot::has_volatility_factors() const noexcept {
    return volatility.complete();
}


bool StatArbMarketSnapshot::equity_markets_valid() const noexcept {
    return has_core_equities() &&
           qqq->status == MarketDataStatus::Valid &&
           sqqq->status == MarketDataStatus::Valid &&
           market_timestamp(*qqq) != 0 &&
           market_timestamp(*sqqq) != 0;
}


bool StatArbMarketSnapshot::factors_valid() const noexcept {
    return has_volatility_factors() &&
           volatility.vix->status ==
               FactorStatus::Valid &&
           volatility.vxn->status ==
               FactorStatus::Valid &&
           volatility.vix->value_ticks.has_value() &&
           volatility.vxn->value_ticks.has_value();
}


std::uint64_t
StatArbMarketSnapshot::equity_skew_ns() const noexcept {
    if (!has_core_equities()) {
        return std::numeric_limits<
            std::uint64_t
        >::max();
    }

    const auto qqq_ts =
        market_timestamp(*qqq);
    const auto sqqq_ts =
        market_timestamp(*sqqq);

    if (qqq_ts == 0 ||
        sqqq_ts == 0) {
        return std::numeric_limits<
            std::uint64_t
        >::max();
    }

    return abs_diff(
        qqq_ts,
        sqqq_ts
    );
}


std::uint64_t
StatArbMarketSnapshot::cross_asset_skew_ns() const noexcept {
    if (!has_core_equities() ||
        !has_volatility_factors()) {
        return std::numeric_limits<
            std::uint64_t
        >::max();
    }

    const auto qqq_ts =
        market_timestamp(*qqq);
    const auto sqqq_ts =
        market_timestamp(*sqqq);

    if (qqq_ts == 0 ||
        sqqq_ts == 0) {
        return std::numeric_limits<
            std::uint64_t
        >::max();
    }

    const auto equity_latest =
        std::max(qqq_ts, sqqq_ts);

    const auto factor_latest =
        std::max(
            volatility.vix->timestamp_ns,
            volatility.vxn->timestamp_ns
        );

    return abs_diff(
        equity_latest,
        factor_latest
    );
}


bool StatArbMarketSnapshot::ready(
    const StatArbSnapshotPolicy& policy
) const noexcept {
    if (captured_ns == 0 ||
        !equity_markets_valid() ||
        !factors_valid()) {
        return false;
    }

    if (!market_fresh(
            *qqq,
            captured_ns,
            policy.max_equity_age_ns
        ) ||
        !market_fresh(
            *sqqq,
            captured_ns,
            policy.max_equity_age_ns
        )) {
        return false;
    }

    if (!volatility.fresh(
            policy.max_factor_age_ns
        )) {
        return false;
    }

    if (equity_skew_ns() >
        policy.max_equity_skew_ns) {
        return false;
    }

    if (volatility.source_skew_ns() >
        policy.max_factor_skew_ns) {
        return false;
    }

    if (cross_asset_skew_ns() >
        policy.max_cross_asset_skew_ns) {
        return false;
    }

    return true;
}


StatArbMarketSnapshotBuilder::
StatArbMarketSnapshotBuilder(
    const OrderBookRegistry& equities,
    const VolatilityFactorBook& volatility
) noexcept
    : equities_(&equities),
      volatility_(&volatility) {}


StatArbMarketSnapshot
StatArbMarketSnapshotBuilder::capture(
    const std::uint64_t captured_ns
) const {
    StatArbMarketSnapshot result;
    result.captured_ns =
        captured_ns;

    const auto qqq_engine =
        equities_->get("QQQ");
    if (qqq_engine) {
        result.qqq =
            qqq_engine->market_snapshot();
    }

    const auto sqqq_engine =
        equities_->get("SQQQ");
    if (sqqq_engine) {
        result.sqqq =
            sqqq_engine->market_snapshot();
    }

    result.volatility =
        volatility_->snapshot(
            captured_ns
        );

    return result;
}

}  // namespace trading
