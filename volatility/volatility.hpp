#pragma once

#include "orderbook.hpp"

#include <atomic>
#include <cstdint>
#include <optional>
#include <shared_mutex>
#include <string>
#include <string_view>
#include <unordered_map>

namespace trading {

// ============================================================================
// VOLATILITY FACTORS
// ============================================================================
//
// Observation only. VIX/VXN are statistical inputs, never executable books.
// Values use the same integer tick convention as the rest of the engine.
// ============================================================================

enum class FactorStatus : std::uint8_t {
    Empty = 0,
    Valid = 1,
    Stale = 2,
    Invalid = 3,
    OutOfSequence = 4
};

struct VolatilityFactorUpdate {
    std::string symbol;
    std::int64_t value_ticks{0};

    std::uint64_t timestamp_ns{0};
    std::uint64_t received_ns{0};
    std::uint64_t sequence{0};

    std::string provider;

    [[nodiscard]]
    bool valid() const noexcept;
};

struct VolatilityFactorState {
    std::string symbol;
    FactorStatus status{FactorStatus::Empty};

    std::optional<std::int64_t> value_ticks;

    std::uint64_t timestamp_ns{0};
    std::uint64_t received_ns{0};
    std::uint64_t sequence{0};
    std::uint64_t version{0};

    std::string provider;

    std::uint64_t accepted_updates{0};
    std::uint64_t rejected_updates{0};
    std::uint64_t out_of_sequence_updates{0};

    [[nodiscard]]
    bool usable(
        std::uint64_t now_ns,
        std::uint64_t max_age_ns
    ) const noexcept;

    [[nodiscard]]
    std::uint64_t age_ns(
        std::uint64_t now_ns
    ) const noexcept;
};

struct VolatilityFactorSnapshot {
    std::optional<VolatilityFactorState> vix;
    std::optional<VolatilityFactorState> vxn;

    std::uint64_t version{0};
    std::uint64_t captured_ns{0};

    [[nodiscard]]
    bool complete() const noexcept;

    [[nodiscard]]
    bool fresh(
        std::uint64_t max_age_ns
    ) const noexcept;

    [[nodiscard]]
    std::optional<std::int64_t>
    vxn_minus_vix_ticks() const noexcept;

    [[nodiscard]]
    std::uint64_t source_skew_ns() const noexcept;
};

class VolatilityFactorBook {
public:
    VolatilityFactorBook() = default;

    VolatilityFactorBook(
        const VolatilityFactorBook&
    ) = delete;

    VolatilityFactorBook& operator=(
        const VolatilityFactorBook&
    ) = delete;

    [[nodiscard]]
    bool on_update(
        const VolatilityFactorUpdate& update
    );

    [[nodiscard]]
    std::optional<VolatilityFactorState>
    get(std::string_view symbol) const;

    [[nodiscard]]
    VolatilityFactorSnapshot snapshot(
        std::uint64_t captured_ns
    ) const;

    [[nodiscard]]
    std::uint64_t version() const noexcept {
        return version_.load(
            std::memory_order_acquire
        );
    }

    void clear();

    [[nodiscard]]
    static bool supported(
        std::string_view symbol
    ) noexcept;

private:
    struct InternalState {
        VolatilityFactorUpdate update;
        std::uint64_t version{0};

        std::uint64_t accepted_updates{0};
        std::uint64_t rejected_updates{0};
        std::uint64_t out_of_sequence_updates{0};
    };

    [[nodiscard]]
    static VolatilityFactorState
    to_public_state(
        const InternalState& state
    );

    mutable std::shared_mutex mutex_;

    std::unordered_map<
        std::string,
        InternalState
    > states_;

    std::atomic<std::uint64_t> version_{0};
};


// ============================================================================
// STAT-ARB COMPOSITE SNAPSHOT
// ============================================================================
//
// A composite observation assembled from independently versioned market
// sources. captured_ns means "when assembled", NOT "all legs arrived
// atomically".
// ============================================================================

struct StatArbSnapshotPolicy {
    std::uint64_t max_equity_age_ns{
        1'000'000'000ULL
    };

    std::uint64_t max_factor_age_ns{
        2'000'000'000ULL
    };

    std::uint64_t max_equity_skew_ns{
        250'000'000ULL
    };

    std::uint64_t max_factor_skew_ns{
        500'000'000ULL
    };

    std::uint64_t max_cross_asset_skew_ns{
        1'000'000'000ULL
    };
};

struct StatArbMarketSnapshot {
    std::optional<ObservedMarketSnapshot> qqq;
    std::optional<ObservedMarketSnapshot> sqqq;

    VolatilityFactorSnapshot volatility;

    std::uint64_t captured_ns{0};

    [[nodiscard]]
    bool has_core_equities() const noexcept;

    [[nodiscard]]
    bool has_volatility_factors() const noexcept;

    [[nodiscard]]
    bool equity_markets_valid() const noexcept;

    [[nodiscard]]
    bool factors_valid() const noexcept;

    [[nodiscard]]
    std::uint64_t equity_skew_ns() const noexcept;

    [[nodiscard]]
    std::uint64_t cross_asset_skew_ns() const noexcept;

    [[nodiscard]]
    bool ready(
        const StatArbSnapshotPolicy& policy
    ) const noexcept;
};

class StatArbMarketSnapshotBuilder {
public:
    StatArbMarketSnapshotBuilder(
        const OrderBookRegistry& equities,
        const VolatilityFactorBook& volatility
    ) noexcept;

    [[nodiscard]]
    StatArbMarketSnapshot capture(
        std::uint64_t captured_ns
    ) const;

private:
    const OrderBookRegistry* equities_;
    const VolatilityFactorBook* volatility_;
};

}  // namespace trading
