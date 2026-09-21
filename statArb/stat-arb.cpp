#include "stat-arb.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace trading::statarb {

namespace {

double safe_abs(const double value) noexcept {
    return std::abs(value);
}

std::uint64_t abs_u64_diff(
    const std::uint64_t lhs,
    const std::uint64_t rhs
) noexcept {
    return lhs >= rhs
        ? lhs - rhs
        : rhs - lhs;
}

std::uint64_t quote_timestamp(
    const ObservedMarketSnapshot& market
) noexcept {
    return market.last_quote_timestamp_ns;
}

std::optional<std::int64_t>
touch_price(
    const ObservedMarketSnapshot& market,
    const Side side
) noexcept {
    if (!market.quote_valid()) {
        return std::nullopt;
    }

    return side == Side::Buy
        ? market.ask_price_ticks
        : market.bid_price_ticks;
}

std::uint64_t saturating_notional(
    const std::uint64_t quantity,
    const std::int64_t price_ticks
) noexcept {
    if (price_ticks <= 0) {
        return 0;
    }

    const auto price =
        static_cast<std::uint64_t>(
            price_ticks
        );

    if (quantity != 0 &&
        price >
            std::numeric_limits<std::uint64_t>::max() /
            quantity) {
        return std::numeric_limits<
            std::uint64_t
        >::max();
    }

    return quantity * price;
}

}  // namespace


void StatArbConfig::validate() const {
    if (dependent_symbol.empty() ||
        hedge_symbol.empty() ||
        dependent_symbol == hedge_symbol) {
        throw std::invalid_argument(
            "StatArbConfig: invalid pair symbols"
        );
    }

    if (lookback < 3 ||
        min_samples < 3 ||
        min_samples > lookback) {
        throw std::invalid_argument(
            "StatArbConfig: invalid sample window"
        );
    }

    if (!(exit_z >= 0.0) ||
        !(entry_z > exit_z) ||
        !(stop_z > entry_z)) {
        throw std::invalid_argument(
            "StatArbConfig: require "
            "0 <= exit_z < entry_z < stop_z"
        );
    }

    if (min_abs_correlation < 0.0 ||
        min_abs_correlation > 1.0) {
        throw std::invalid_argument(
            "StatArbConfig: correlation threshold "
            "must be in [0,1]"
        );
    }

    if (!(min_residual_std > 0.0)) {
        throw std::invalid_argument(
            "StatArbConfig: min_residual_std "
            "must be positive"
        );
    }

    if (target_gross_notional_ticks == 0 ||
        max_pair_notional_ticks == 0 ||
        target_gross_notional_ticks >
            max_pair_notional_ticks) {
        throw std::invalid_argument(
            "StatArbConfig: invalid notional limits"
        );
    }
}


SimulatedOrderBookVenue::SimulatedOrderBookVenue(
    OrderBookRegistry& registry
) noexcept
    : registry_(&registry) {}


VenueSubmitResult
SimulatedOrderBookVenue::submit_limit(
    const LegIntent& intent
) {
    VenueSubmitResult result;
    result.order_id =
        intent.order_id;

    const auto engine =
        registry_->get(intent.symbol);

    if (!engine) {
        result.message =
            "symbol has no simulated order-book engine";
        return result;
    }

    try {
        result.executions =
            engine->add_limit(
                intent.order_id,
                intent.side,
                intent.quantity,
                intent.limit_price_ticks
            );

        result.accepted = true;

        std::uint64_t filled = 0;
        for (const auto& execution :
             result.executions) {
            if (execution.maker_id ==
                    intent.order_id ||
                execution.taker_id ==
                    intent.order_id) {
                filled += execution.quantity;
            }
        }

        result.filled_quantity =
            std::min(
                intent.quantity,
                filled
            );

        result.remaining_quantity =
            intent.quantity -
            result.filled_quantity;

        result.message =
            "accepted by simulated C++ order book";
    } catch (const std::exception& exc) {
        result.message = exc.what();
    }

    return result;
}


bool SimulatedOrderBookVenue::cancel(
    const std::uint64_t order_id
) {
    for (const auto& symbol :
         registry_->symbols()) {
        const auto engine =
            registry_->get(symbol);

        if (!engine) {
            continue;
        }

        try {
            (void)engine->cancel(order_id);
            return true;
        } catch (const std::exception&) {
            // Continue searching. A production venue should use an
            // order-id -> venue-symbol locator instead of scanning.
        }
    }

    return false;
}


RollingPairModel::RollingPairModel(
    StatArbConfig config
)
    : config_(std::move(config)) {
    config_.validate();
}


std::optional<PairStatistics>
RollingPairModel::update(
    const StatArbMarketSnapshot& snapshot
) {
    if (!snapshot.qqq.has_value() ||
        !snapshot.sqqq.has_value()) {
        return std::nullopt;
    }

    const auto& qqq =
        *snapshot.qqq;
    const auto& sqqq =
        *snapshot.sqqq;

    if (!qqq.quote_valid() ||
        !sqqq.quote_valid()) {
        return std::nullopt;
    }

    // Do not manufacture repeated returns from the same observed versions.
    if (qqq.version ==
            previous_qqq_version_ &&
        sqqq.version ==
            previous_sqqq_version_) {
        return std::nullopt;
    }

    const auto qqq_mid =
        qqq.midpoint_ticks();
    const auto sqqq_mid =
        sqqq.midpoint_ticks();

    if (!qqq_mid.has_value() ||
        !sqqq_mid.has_value() ||
        *qqq_mid <= 0 ||
        *sqqq_mid <= 0) {
        return std::nullopt;
    }

    const auto qqq_ts =
        quote_timestamp(qqq);
    const auto sqqq_ts =
        quote_timestamp(sqqq);

    if (qqq_ts == 0 ||
        sqqq_ts == 0 ||
        abs_u64_diff(
            qqq_ts,
            sqqq_ts
        ) >
            config_.snapshot_policy
                .max_equity_skew_ns) {
        return std::nullopt;
    }

    previous_qqq_version_ =
        qqq.version;
    previous_sqqq_version_ =
        sqqq.version;

    if (!previous_qqq_mid_.has_value() ||
        !previous_sqqq_mid_.has_value()) {
        previous_qqq_mid_ =
            *qqq_mid;
        previous_sqqq_mid_ =
            *sqqq_mid;
        return std::nullopt;
    }

    const double qqq_return =
        std::log(
            static_cast<double>(*qqq_mid) /
            static_cast<double>(
                *previous_qqq_mid_
            )
        );

    const double sqqq_return =
        std::log(
            static_cast<double>(*sqqq_mid) /
            static_cast<double>(
                *previous_sqqq_mid_
            )
        );

    previous_qqq_mid_ =
        *qqq_mid;
    previous_sqqq_mid_ =
        *sqqq_mid;

    if (!std::isfinite(qqq_return) ||
        !std::isfinite(sqqq_return)) {
        return std::nullopt;
    }

    samples_.push_back(
        ReturnSample{
            .y = qqq_return,
            .x = sqqq_return
        }
    );

    while (samples_.size() >
           config_.lookback) {
        samples_.pop_front();
    }

    if (samples_.size() <
        config_.min_samples) {
        PairStatistics warmup;
        warmup.samples =
            samples_.size();
        warmup.qqq_return =
            qqq_return;
        warmup.sqqq_return =
            sqqq_return;
        return warmup;
    }

    return compute_statistics();
}


PairStatistics
RollingPairModel::compute_statistics() const {
    PairStatistics result;
    result.samples =
        samples_.size();

    if (samples_.size() < 3) {
        return result;
    }

    const double n =
        static_cast<double>(
            samples_.size()
        );

    double sum_x = 0.0;
    double sum_y = 0.0;

    for (const auto& sample :
         samples_) {
        sum_x += sample.x;
        sum_y += sample.y;
    }

    const double mean_x =
        sum_x / n;
    const double mean_y =
        sum_y / n;

    double sxx = 0.0;
    double syy = 0.0;
    double sxy = 0.0;

    for (const auto& sample :
         samples_) {
        const double dx =
            sample.x - mean_x;
        const double dy =
            sample.y - mean_y;

        sxx += dx * dx;
        syy += dy * dy;
        sxy += dx * dy;
    }

    if (!(sxx >
          std::numeric_limits<double>::epsilon()) ||
        !(syy >
          std::numeric_limits<double>::epsilon())) {
        return result;
    }

    result.beta =
        sxy / sxx;

    result.alpha =
        mean_y -
        result.beta * mean_x;

    result.correlation =
        sxy /
        std::sqrt(sxx * syy);

    double residual_sum = 0.0;

    std::vector<double> residuals;
    residuals.reserve(
        samples_.size()
    );

    for (const auto& sample :
         samples_) {
        const double residual =
            sample.y -
            (result.alpha +
             result.beta * sample.x);

        residuals.push_back(
            residual
        );

        residual_sum +=
            residual;
    }

    result.residual_mean =
        residual_sum / n;

    double residual_ss = 0.0;

    for (const double residual :
         residuals) {
        const double delta =
            residual -
            result.residual_mean;

        residual_ss +=
            delta * delta;
    }

    result.residual_std =
        std::sqrt(
            residual_ss /
            std::max(1.0, n - 1.0)
        );

    const auto& latest =
        samples_.back();

    result.qqq_return =
        latest.y;
    result.sqqq_return =
        latest.x;

    result.residual =
        latest.y -
        (result.alpha +
         result.beta * latest.x);

    if (!(result.residual_std >
          config_.min_residual_std) ||
        !std::isfinite(
            result.correlation
        ) ||
        !std::isfinite(
            result.beta
        )) {
        return result;
    }

    result.zscore =
        (result.residual -
         result.residual_mean) /
        result.residual_std;

    result.valid =
        std::isfinite(result.zscore) &&
        safe_abs(result.correlation) >=
            config_.min_abs_correlation;

    return result;
}


void RollingPairModel::clear() {
    previous_qqq_mid_.reset();
    previous_sqqq_mid_.reset();

    previous_qqq_version_ = 0;
    previous_sqqq_version_ = 0;

    samples_.clear();
}


StatArbSignalEngine::StatArbSignalEngine(
    StatArbConfig config
)
    : config_(std::move(config)),
      model_(config_) {
    config_.validate();
}


bool StatArbSignalEngine::volatility_gate(
    const StatArbMarketSnapshot& snapshot
) const noexcept {
    if (!snapshot.factors_valid()) {
        return false;
    }

    const auto vix =
        *snapshot.volatility.vix
             ->value_ticks;
    const auto vxn =
        *snapshot.volatility.vxn
             ->value_ticks;

    if (config_.max_vix_ticks > 0 &&
        vix > config_.max_vix_ticks) {
        return false;
    }

    if (config_.max_vxn_ticks > 0 &&
        vxn > config_.max_vxn_ticks) {
        return false;
    }

    return true;
}


PairSignal StatArbSignalEngine::evaluate(
    const StatArbMarketSnapshot& snapshot,
    const PairPosition current_position
) {
    PairSignal signal;
    signal.current_position =
        current_position;

    signal.snapshot_ready =
        snapshot.ready(
            config_.snapshot_policy
        );

    if (!signal.snapshot_ready) {
        signal.reason =
            "composite market snapshot is not ready";
        return signal;
    }

    signal.volatility_gate_open =
        volatility_gate(snapshot);

    if (!signal.volatility_gate_open) {
        signal.reason =
            "VIX/VXN volatility gate is closed";
        return signal;
    }

    const auto statistics =
        model_.update(snapshot);

    if (!statistics.has_value()) {
        signal.reason =
            "waiting for a new synchronized pair observation";
        return signal;
    }

    signal.statistics =
        *statistics;

    if (!statistics->valid) {
        signal.reason =
            "pair model warming up or relationship unstable";
        return signal;
    }

    const double z =
        statistics->zscore;
    const double abs_z =
        safe_abs(z);

    if (current_position ==
        PairPosition::Flat) {
        if (abs_z >= config_.stop_z) {
            signal.reason =
                "residual exceeds stop threshold; "
                "new entry suppressed";
            return signal;
        }

        if (z <= -config_.entry_z) {
            signal.action =
                SignalAction::EnterLongResidual;
            signal.reason =
                "negative residual entry threshold crossed";
            return signal;
        }

        if (z >= config_.entry_z) {
            signal.action =
                SignalAction::EnterShortResidual;
            signal.reason =
                "positive residual entry threshold crossed";
            return signal;
        }

        signal.reason =
            "flat: residual inside entry band";
        return signal;
    }

    if (current_position ==
            PairPosition::LongResidual ||
        current_position ==
            PairPosition::ShortResidual) {
        if (abs_z >= config_.stop_z) {
            signal.action =
                SignalAction::Stop;
            signal.reason =
                "residual stop threshold crossed";
            return signal;
        }

        if (abs_z <= config_.exit_z) {
            signal.action =
                SignalAction::Exit;
            signal.reason =
                "residual mean-reversion exit threshold reached";
            return signal;
        }

        signal.reason =
            "position held: residual has not reached exit/stop";
        return signal;
    }

    signal.reason =
        "engine position does not permit new signal";
    return signal;
}


void StatArbSignalEngine::reset() {
    model_.clear();
}


StatArbExecutionEngine::StatArbExecutionEngine(
    StatArbConfig config,
    PairRiskGate& risk_gate,
    ExecutionVenue& venue,
    const std::uint64_t first_order_id,
    const std::uint64_t first_pair_id
)
    : config_(std::move(config)),
      signals_(config_),
      risk_gate_(&risk_gate),
      venue_(&venue),
      next_order_id_(first_order_id),
      next_pair_id_(first_pair_id) {
    config_.validate();

    if (first_order_id == 0 ||
        first_pair_id == 0) {
        throw std::invalid_argument(
            "StatArbExecutionEngine: IDs must be positive"
        );
    }
}


PairSignal StatArbExecutionEngine::observe(
    const StatArbMarketSnapshot& snapshot
) {
    return signals_.evaluate(
        snapshot,
        position_
    );
}


PairExecutionResult
StatArbExecutionEngine::process(
    const StatArbMarketSnapshot& snapshot,
    const std::uint64_t now_ns
) {
    PairExecutionResult result;

    if (position_ ==
        PairPosition::Halted) {
        result.status =
            PairExecutionStatus::Rejected;
        result.message =
            "stat-arb engine is halted";
        return result;
    }

    if (last_action_ns_ != 0 &&
        now_ns >= last_action_ns_ &&
        now_ns - last_action_ns_ <
            config_.cooldown_ns) {
        result.status =
            PairExecutionStatus::NoAction;
        result.message =
            "cooldown active";
        return result;
    }

    const auto signal =
        observe(snapshot);

    if (signal.action ==
        SignalAction::None) {
        result.status =
            PairExecutionStatus::NoAction;
        result.message =
            signal.reason;
        return result;
    }

    // Exit/stop requires live position/fill inventory, which is intentionally
    // not inferred from signal state. Until the execution journal/position
    // adapter is wired, fail closed rather than invent flatten quantities.
    if (signal.action ==
            SignalAction::Exit ||
        signal.action ==
            SignalAction::Stop) {
        position_ =
            PairPosition::Degraded;

        result.status =
            PairExecutionStatus::Degraded;
        result.message =
            "flatten requested but authoritative pair inventory "
            "is not yet connected";
        return result;
    }

    const auto intent =
        build_intent(
            signal,
            snapshot,
            now_ns
        );

    if (!intent.has_value()) {
        result.status =
            PairExecutionStatus::Rejected;
        result.message =
            "could not construct a valid pair intent";
        return result;
    }

    result =
        execute_pair(
            *intent,
            snapshot
        );

    if (result.status ==
        PairExecutionStatus::Established) {
        position_ =
            signal.action ==
                SignalAction::EnterLongResidual
            ? PairPosition::LongResidual
            : PairPosition::ShortResidual;

        last_action_ns_ =
            now_ns;
    } else if (
        result.status ==
            PairExecutionStatus::PartiallyEstablished ||
        result.status ==
            PairExecutionStatus::Degraded
    ) {
        position_ =
            PairPosition::Degraded;
        last_action_ns_ =
            now_ns;
    }

    return result;
}


std::optional<PairOrderIntent>
StatArbExecutionEngine::build_intent(
    const PairSignal& signal,
    const StatArbMarketSnapshot& snapshot,
    const std::uint64_t now_ns
) {
    if (!snapshot.qqq.has_value() ||
        !snapshot.sqqq.has_value() ||
        !signal.statistics.valid) {
        return std::nullopt;
    }

    double dependent_weight = 0.0;
    double hedge_weight = 0.0;

    if (signal.action ==
        SignalAction::EnterLongResidual) {
        dependent_weight = 1.0;
        hedge_weight =
            -signal.statistics.beta;
    } else if (
        signal.action ==
        SignalAction::EnterShortResidual
    ) {
        dependent_weight = -1.0;
        hedge_weight =
            signal.statistics.beta;
    } else {
        return std::nullopt;
    }

    const double gross_weight =
        safe_abs(dependent_weight) +
        safe_abs(hedge_weight);

    if (!(gross_weight > 0.0) ||
        !std::isfinite(gross_weight)) {
        return std::nullopt;
    }

    const double gross_target =
        static_cast<double>(
            config_.target_gross_notional_ticks
        );

    const double dependent_notional =
        gross_target *
        dependent_weight /
        gross_weight;

    const double hedge_notional =
        gross_target *
        hedge_weight /
        gross_weight;

    PairOrderIntent intent;
    intent.pair_id =
        next_pair_id();
    intent.action =
        signal.action;
    intent.dependent_weight =
        dependent_weight;
    intent.hedge_weight =
        hedge_weight;
    intent.created_ns =
        now_ns;

    intent.dependent =
        build_leg(
            config_.dependent_symbol,
            dependent_notional,
            *snapshot.qqq
        );

    intent.hedge =
        build_leg(
            config_.hedge_symbol,
            hedge_notional,
            *snapshot.sqqq
        );

    if (intent.dependent.quantity == 0 ||
        intent.hedge.quantity == 0) {
        return std::nullopt;
    }

    const auto dep_notional =
        saturating_notional(
            intent.dependent.quantity,
            intent.dependent.limit_price_ticks
        );

    const auto hedge_actual_notional =
        saturating_notional(
            intent.hedge.quantity,
            intent.hedge.limit_price_ticks
        );

    if (dep_notional >
            config_.max_pair_notional_ticks ||
        hedge_actual_notional >
            config_.max_pair_notional_ticks ||
        dep_notional >
            config_.max_pair_notional_ticks -
                std::min(
                    hedge_actual_notional,
                    config_.max_pair_notional_ticks
                )) {
        return std::nullopt;
    }

    return intent;
}


LegIntent StatArbExecutionEngine::build_leg(
    std::string symbol,
    const double signed_notional_ticks,
    const ObservedMarketSnapshot& market
) {
    LegIntent leg;
    leg.order_id =
        next_order_id();
    leg.symbol =
        std::move(symbol);
    leg.signed_target_notional_ticks =
        signed_notional_ticks;

    leg.side =
        signed_notional_ticks >= 0.0
            ? Side::Buy
            : Side::Sell;

    const auto price =
        touch_price(
            market,
            leg.side
        );

    if (!price.has_value() ||
        *price <= 0) {
        return leg;
    }

    leg.limit_price_ticks =
        *price;

    const double raw_quantity =
        safe_abs(signed_notional_ticks) /
        static_cast<double>(*price);

    if (!std::isfinite(raw_quantity) ||
        raw_quantity < 1.0) {
        return leg;
    }

    const double capped =
        std::min(
            raw_quantity,
            static_cast<double>(
                std::numeric_limits<
                    std::uint64_t
                >::max()
            )
        );

    leg.quantity =
        static_cast<std::uint64_t>(
            std::floor(capped)
        );

    return leg;
}


PairExecutionResult
StatArbExecutionEngine::execute_pair(
    const PairOrderIntent& intent,
    const StatArbMarketSnapshot& snapshot
) {
    PairExecutionResult result;
    result.pair_id =
        intent.pair_id;

    const auto risk =
        risk_gate_->evaluate_pair(
            intent,
            snapshot
        );

    if (!risk.accepted) {
        result.status =
            PairExecutionStatus::Rejected;
        result.message =
            risk.reason.empty()
                ? "joint pair risk rejected intent"
                : risk.reason;
        return result;
    }

    // Submit the dependent leg first for now. A production venue should make
    // lead-leg choice configurable from observed liquidity/slippage.
    result.dependent =
        venue_->submit_limit(
            intent.dependent
        );

    if (!result.dependent->accepted) {
        result.status =
            PairExecutionStatus::Rejected;
        result.message =
            "dependent leg rejected: " +
            result.dependent->message;
        return result;
    }

    result.hedge =
        venue_->submit_limit(
            intent.hedge
        );

    if (!result.hedge->accepted) {
        // Best-effort cancellation. If the first leg filled, cancellation
        // cannot undo that exposure; mark degraded and require inventory-based
        // emergency hedging/flattening by the next execution layer.
        (void)venue_->cancel(
            intent.dependent.order_id
        );

        if (result.dependent->
                filled_quantity > 0) {
            result.status =
                PairExecutionStatus::Degraded;
            result.message =
                "hedge leg rejected after dependent fill; "
                "manual/automatic emergency hedge required";
        } else {
            result.status =
                PairExecutionStatus::Rejected;
            result.message =
                "hedge leg rejected; dependent leg cancelled";
        }

        return result;
    }

    const bool dep_filled =
        result.dependent->
            remaining_quantity == 0;

    const bool hedge_filled =
        result.hedge->
            remaining_quantity == 0;

    if (dep_filled &&
        hedge_filled) {
        result.status =
            PairExecutionStatus::Established;
        result.message =
            "both pair legs fully established";
        return result;
    }

    // Accepted/resting is not equivalent to established exposure.
    result.status =
        PairExecutionStatus::PartiallyEstablished;
    result.message =
        "both legs accepted but one or both remain unfilled";

    return result;
}


std::uint64_t
StatArbExecutionEngine::next_order_id() noexcept {
    return next_order_id_++;
}


std::uint64_t
StatArbExecutionEngine::next_pair_id() noexcept {
    return next_pair_id_++;
}


void StatArbExecutionEngine::reset() {
    signals_.reset();
    position_ =
        PairPosition::Flat;
    last_action_ns_ = 0;
}

}  // namespace trading::statarb
