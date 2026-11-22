// ============================================================================
// discovery_engine.hpp
// ----------------------------------------------------------------------------
// Phase B.2 — authoritative native market-state ingestion for opportunity
// discovery. The DiscoveryEngine is the single native owner of per-symbol
// discovery state. It COMPOSES the existing authoritative structures rather
// than duplicating them (directive §19):
//
//   * L1 observed quote/trade state  -> trading::OrderBookRegistry
//     (ObservedMarketBook per symbol — the authoritative L1 book).
//   * Per-symbol rolling bar history  -> native ring buffers added HERE
//     (no native equivalent existed; the caller used to own bar history).
//   * Stable integer symbol identity + generation + membership version -> HERE
//     (the existing registries key by std::string only; no id/generation).
//   * Dirty-symbol tracking           -> HERE (for incremental recompute).
//
// Everything is framework-agnostic C++ (no Metal / Objective-C) so it compiles
// and is testable on every platform; Phase B.3+ layers Metal technical-feature
// computation and screen_market on top of the state this class maintains.
//
// Prices crossing the boundary are int64 "ticks" (x10,000, 1 tick = $0.0001),
// matching the Alpaca IEX normalized events and the native *_ticks structs. Bar
// history is stored in analytics::Bar (float dollars) for the feature engine, so
// bar ingestion converts ticks -> dollars via the configured price scale.
// ============================================================================
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "analytics/features.hpp"   // trading::analytics::Bar
#include "orderbook/orderbook.hpp"  // trading::OrderBookRegistry

namespace trading::discovery {

// Default price scale: 1 tick == $0.0001 (matches market_data/alpaca_stream.py
// DEFAULT_PRICE_SCALE and the native *_ticks convention).
inline constexpr std::int64_t kDefaultPriceScale = 10'000;

enum class BarTimeframe : std::uint8_t { OneMinute = 0, FiveMinute = 1 };

// --- batched ingestion inputs (keyed by stable symbol id) -------------------
// Keeping these SoA-friendly and id-keyed lets the Python supervisor accumulate
// IEX events and flush whole batches, avoiding per-symbol boundary calls (§21).
struct QuoteObs {
    std::uint32_t symbol_id{0};
    std::uint64_t timestamp_ns{0};
    std::int64_t bid_ticks{0};
    std::int64_t ask_ticks{0};
    std::uint64_t bid_size{0};
    std::uint64_t ask_size{0};
    std::uint64_t sequence{0};
};

struct TradeObs {
    std::uint32_t symbol_id{0};
    std::uint64_t timestamp_ns{0};
    std::int64_t price_ticks{0};
    std::uint64_t size{0};
    std::uint64_t sequence{0};
};

struct BarObs {
    std::uint32_t symbol_id{0};
    std::uint64_t timestamp_ns{0};
    std::int64_t open_ticks{0};
    std::int64_t high_ticks{0};
    std::int64_t low_ticks{0};
    std::int64_t close_ticks{0};
    std::int64_t volume{0};
};

// --- per-symbol state snapshot (host-readable) ------------------------------
struct SymbolSnapshot {
    std::uint32_t symbol_id{0};
    std::uint32_t generation{0};
    std::string symbol;
    bool active{false};
    bool has_quote{false};
    bool has_trade{false};
    std::int64_t bid_ticks{0};
    std::int64_t ask_ticks{0};
    std::int64_t last_trade_ticks{0};
    std::uint64_t last_obs_ns{0};
    std::uint64_t last_sequence{0};
    std::uint64_t quote_updates{0};
    std::uint64_t trade_updates{0};
    std::size_t bars_1m{0};
    std::size_t bars_5m{0};
    bool dirty{false};
};

// --- batched bar-window export (feeds the B.3 technical_features kernel) -----
// Row-major (symbol_count * max_window) float matrices of the last `window` bars
// for the requested symbol ids, ready to pass straight to
// MetalRuntime.technical_features. counts[row] is the valid bar depth.
struct BarWindow {
    std::vector<float> closes;
    std::vector<float> highs;
    std::vector<float> lows;
    std::vector<float> volumes;
    std::vector<std::uint32_t> counts;
    std::uint32_t symbol_count{0};
    std::uint32_t max_window{0};
};

// --- coverage summary (telemetry) -------------------------------------------
struct CoverageSummary {
    std::size_t symbols{0};          // active symbols
    std::size_t observed{0};         // has a valid quote or trade
    std::size_t with_quote{0};
    std::size_t with_trade{0};
    std::size_t with_bars_1m{0};
    std::size_t with_bars_5m{0};
    std::size_t dirty{0};
    std::uint64_t membership_version{0};
    std::uint64_t total_quote_updates{0};
    std::uint64_t total_trade_updates{0};
    std::uint64_t total_bar_updates{0};
    std::uint64_t rejected{0};
};

class DiscoveryEngine {
public:
    // bar_capacity bounds each per-symbol rolling bar window (oldest dropped).
    explicit DiscoveryEngine(std::size_t bar_capacity = 512,
                             std::int64_t price_scale = kDefaultPriceScale);
    ~DiscoveryEngine();
    DiscoveryEngine(const DiscoveryEngine&) = delete;
    DiscoveryEngine& operator=(const DiscoveryEngine&) = delete;

    // --- membership ---------------------------------------------------------
    // Set the ACTIVE discovery universe. New symbols get a fresh stable id
    // (generation 1). A symbol dropped from the set becomes inactive but keeps
    // its id. A previously-known symbol re-entering bumps its generation and
    // clears its (now stale) observed state + bar buffers. Returns the new
    // membership_version (incremented only when the active set changes).
    std::uint64_t update_membership(const std::vector<std::string>& symbols);

    [[nodiscard]] std::optional<std::uint32_t> symbol_id(
        const std::string& symbol) const;
    [[nodiscard]] std::string symbol_name(std::uint32_t id) const;
    [[nodiscard]] std::uint32_t generation(std::uint32_t id) const;
    [[nodiscard]] std::uint64_t membership_version() const noexcept;
    [[nodiscard]] std::size_t symbol_count() const noexcept;  // active symbols
    [[nodiscard]] std::vector<std::string> symbols() const;   // active symbols

    // --- batched ingestion --------------------------------------------------
    // Each returns the number of rows actually applied. Invalid rows (unknown /
    // inactive id, non-finite or non-positive prices, crossed quotes) are
    // rejected and counted in coverage().rejected; applied rows mark the symbol
    // dirty and update observation sequence / freshness.
    std::size_t ingest_quotes(const std::vector<QuoteObs>& quotes);
    std::size_t ingest_trades(const std::vector<TradeObs>& trades);
    std::size_t ingest_bars(const std::vector<BarObs>& bars, BarTimeframe tf);

    // --- dirty-symbol tracking ---------------------------------------------
    [[nodiscard]] std::vector<std::uint32_t> dirty_symbols() const;  // sorted
    void clear_dirty();

    // --- introspection ------------------------------------------------------
    [[nodiscard]] SymbolSnapshot snapshot(std::uint32_t id) const;
    [[nodiscard]] std::vector<analytics::Bar> bars(std::uint32_t id,
                                                   BarTimeframe tf) const;
    [[nodiscard]] std::optional<float> last_bar_close(std::uint32_t id,
                                                      BarTimeframe tf) const;

    // Export the last `window` bars for `ids` as a row-major matrix (one row per
    // requested id, in order). Rows for unknown ids are left empty (count 0).
    [[nodiscard]] BarWindow export_bar_window(
        const std::vector<std::uint32_t>& ids, BarTimeframe tf,
        std::uint32_t window) const;

    [[nodiscard]] CoverageSummary coverage() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace trading::discovery
