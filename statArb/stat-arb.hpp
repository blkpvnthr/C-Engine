#pragma once

#include "orderbook.hpp"
#include "volatility_factor_book.hpp"

#include <cstddef>
#include <cstdint>
#include <deque>
#include <optional>
#include <string>
#include <vector>

namespace trading::statarb {

// ============================================================================
// QQQ / SQQQ STATISTICAL-ARBITRAGE ENGINE
// ============================================================================
//
// Signal model:
//     y = QQQ short-horizon log return
//     x = SQQQ short-horizon log return
//
//     y = alpha + beta*x + residual
//
// The engine does NOT assume beta == -1/3 or -3. It estimates beta from the
// synchronized rolling return window.
//
// VIX/VXN are read-only regime factors. They never become execution legs.
//
// "Long residual" means:
//     +1 * QQQ  + (-beta) * SQQQ
//
// With a negative beta this can produce two BUY legs. That is intentional:
// residual exposure is represented by signed weights, not a hard-coded
// long-one/short-the-other assumption.
// ============================================================================

enum class PairPosition : std::uint8_t {
    Flat = 0,
    LongResidual = 1,
    ShortResidual = 2,
    Degraded = 3,
    Halted = 4
};

enum class SignalAction : std::uint8_t {
    None = 0,
    EnterLongResidual = 1,
    EnterShortResidual = 2,
    Exit = 3,
    Stop = 4
};

enum class PairExecutionStatus : std::uint8_t {
    NoAction = 0,
    Rejected = 1,
    Submitted = 2,
    PartiallyEstablished = 3,
    Established = 4,
    Flattened = 5,
    Degraded = 6
};

struct StatArbConfig {
    std::string dependent_symbol{"QQQ"};
    std::string hedge_symbol{"SQQQ"};

    std::size_t lookback{120};
    std::size_t min_samples{60};

    // Engineering defaults only; calibrate through research/backtests.
    double entry_z{2.0};
    double exit_z{0.5};
    double stop_z{4.0};

    double min_abs_correlation{0.80};
    double min_residual_std{1.0e-8};

    // Optional volatility regime gate. Zero disables the limit.
    std::int64_t max_vxn_ticks{0};
    std::int64_t max_vix_ticks{0};

    // Gross target expressed in price-tick * share units.
    std::uint64_t target_gross_notional_ticks{10'000'000};
    std::uint64_t max_pair_notional_ticks{25'000'000};

    // Aggressive simulated limit: buy at observed ask, sell at observed bid.
    // This is a price policy only. SIP NBBO is NOT inserted into the local
    // LimitOrderBook and is not assumed executable externally.
    bool use_touch_prices{true};

    std::uint64_t cooldown_ns{1'000'000'000ULL};
    std::uint64_t max_legging_ns{500'000'000ULL};

    StatArbSnapshotPolicy snapshot_policy{};

    void validate() const;
};

struct PairStatistics {
    std::size_t samples{0};

    double alpha{0.0};
    double beta{0.0};
    double correlation{0.0};

    double residual{0.0};
    double residual_mean{0.0};
    double residual_std{0.0};
    double zscore{0.0};

    double qqq_return{0.0};
    double sqqq_return{0.0};

    bool valid{false};
};

struct PairSignal {
    SignalAction action{SignalAction::None};
    PairPosition current_position{PairPosition::Flat};

    PairStatistics statistics;

    bool snapshot_ready{false};
    bool volatility_gate_open{false};

    std::string reason;
};

struct LegIntent {
    std::uint64_t order_id{0};
    std::string symbol;
    Side side{Side::Buy};
    std::uint64_t quantity{0};
    std::int64_t limit_price_ticks{0};

    // Signed desired notional before conversion to quantity.
    double signed_target_notional_ticks{0.0};
};

struct PairOrderIntent {
    std::uint64_t pair_id{0};
    SignalAction action{SignalAction::None};

    LegIntent dependent;
    LegIntent hedge;

    double dependent_weight{0.0};
    double hedge_weight{0.0};

    std::uint64_t created_ns{0};
};

struct PairRiskDecision {
    bool accepted{false};
    std::string reason;
};

// Joint pair-risk authority.
//
// This is intentionally NOT implemented by calling RiskEngine::evaluate()
// twice. The current RiskEngine tracks accepted order IDs, so two independent
// calls are not an atomic/non-mutating pair preflight.
class PairRiskGate {
public:
    virtual ~PairRiskGate() = default;

    [[nodiscard]]
    virtual PairRiskDecision evaluate_pair(
        const PairOrderIntent& intent,
        const StatArbMarketSnapshot& market
    ) = 0;
};

struct VenueSubmitResult {
    bool accepted{false};
    std::uint64_t order_id{0};
    std::uint64_t filled_quantity{0};
    std::uint64_t remaining_quantity{0};
    std::vector<Execution> executions;
    std::string message;
};

class ExecutionVenue {
public:
    virtual ~ExecutionVenue() = default;

    [[nodiscard]]
    virtual VenueSubmitResult submit_limit(
        const LegIntent& intent
    ) = 0;

    [[nodiscard]]
    virtual bool cancel(
        std::uint64_t order_id
    ) = 0;
};

// Simulation venue only. It wraps OrderBookRegistry/OrderBookEngine.
// A future AlpacaBrokerVenue should implement the same interface separately.
class SimulatedOrderBookVenue final : public ExecutionVenue {
public:
    explicit SimulatedOrderBookVenue(
        OrderBookRegistry& registry
    ) noexcept;

    [[nodiscard]]
    VenueSubmitResult submit_limit(
        const LegIntent& intent
    ) override;

    [[nodiscard]]
    bool cancel(
        std::uint64_t order_id
    ) override;

private:
    OrderBookRegistry* registry_;
};

struct PairExecutionResult {
    PairExecutionStatus status{
        PairExecutionStatus::NoAction
    };

    std::uint64_t pair_id{0};

    std::optional<VenueSubmitResult> dependent;
    std::optional<VenueSubmitResult> hedge;

    std::string message;
};

class RollingPairModel {
public:
    explicit RollingPairModel(
        StatArbConfig config
    );

    // Returns a new statistical observation only when both QQQ and SQQQ
    // midpoints advanced to a usable synchronized snapshot.
    [[nodiscard]]
    std::optional<PairStatistics> update(
        const StatArbMarketSnapshot& snapshot
    );

    [[nodiscard]]
    std::size_t size() const noexcept {
        return samples_.size();
    }

    void clear();

private:
    struct ReturnSample {
        double y{0.0};
        double x{0.0};
    };

    [[nodiscard]]
    PairStatistics compute_statistics() const;

    StatArbConfig config_;

    std::optional<std::int64_t> previous_qqq_mid_;
    std::optional<std::int64_t> previous_sqqq_mid_;

    std::uint64_t previous_qqq_version_{0};
    std::uint64_t previous_sqqq_version_{0};

    std::deque<ReturnSample> samples_;
};

class StatArbSignalEngine {
public:
    explicit StatArbSignalEngine(
        StatArbConfig config
    );

    [[nodiscard]]
    PairSignal evaluate(
        const StatArbMarketSnapshot& snapshot,
        PairPosition current_position
    );

    void reset();

private:
    [[nodiscard]]
    bool volatility_gate(
        const StatArbMarketSnapshot& snapshot
    ) const noexcept;

    StatArbConfig config_;
    RollingPairModel model_;
};

class StatArbExecutionEngine {
public:
    StatArbExecutionEngine(
        StatArbConfig config,
        PairRiskGate& risk_gate,
        ExecutionVenue& venue,
        std::uint64_t first_order_id = 4'000'000'000ULL,
        std::uint64_t first_pair_id = 1
    );

    [[nodiscard]]
    PairSignal observe(
        const StatArbMarketSnapshot& snapshot
    );

    [[nodiscard]]
    PairExecutionResult process(
        const StatArbMarketSnapshot& snapshot,
        std::uint64_t now_ns
    );

    [[nodiscard]]
    PairPosition position() const noexcept {
        return position_;
    }

    void halt() noexcept {
        position_ = PairPosition::Halted;
    }

    void reset();

private:
    [[nodiscard]]
    std::optional<PairOrderIntent>
    build_intent(
        const PairSignal& signal,
        const StatArbMarketSnapshot& snapshot,
        std::uint64_t now_ns
    );

    [[nodiscard]]
    LegIntent build_leg(
        std::string symbol,
        double signed_notional_ticks,
        const ObservedMarketSnapshot& market
    );

    [[nodiscard]]
    PairExecutionResult execute_pair(
        const PairOrderIntent& intent,
        const StatArbMarketSnapshot& snapshot
    );

    [[nodiscard]]
    std::uint64_t next_order_id() noexcept;

    [[nodiscard]]
    std::uint64_t next_pair_id() noexcept;

    StatArbConfig config_;
    StatArbSignalEngine signals_;

    PairRiskGate* risk_gate_;
    ExecutionVenue* venue_;

    PairPosition position_{PairPosition::Flat};

    std::uint64_t next_order_id_;
    std::uint64_t next_pair_id_;
    std::uint64_t last_action_ns_{0};
};

}  // namespace trading::statarb