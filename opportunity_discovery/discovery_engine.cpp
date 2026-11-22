// ============================================================================
// discovery_engine.cpp — Phase B.2 native market-state ingestion.
// See discovery_engine.hpp. Composes OrderBookRegistry (authoritative L1) and
// adds the symbol-id/generation registry, per-symbol bar ring buffers, and
// dirty-symbol tracking that did not previously exist natively.
// ============================================================================
#include "opportunity_discovery/discovery_engine.hpp"

#include <algorithm>
#include <deque>
#include <mutex>
#include <unordered_map>
#include <unordered_set>

namespace trading::discovery {

namespace {
// Largest finite tick magnitude we accept, to reject obviously corrupt inputs
// without imposing a real-world price ceiling.
constexpr std::int64_t kMaxTicks = 1'000'000'000'000LL;  // $100,000,000
}  // namespace

struct DiscoveryEngine::Impl {
    struct SymbolRecord {
        std::string name;
        std::uint32_t generation{1};
        bool active{true};
        std::uint64_t last_obs_ns{0};
        std::uint64_t last_sequence{0};
        std::deque<analytics::Bar> bars_1m;
        std::deque<analytics::Bar> bars_5m;
    };

    std::size_t bar_capacity;
    std::int64_t price_scale;
    mutable std::mutex mtx;

    OrderBookRegistry registry;                      // authoritative L1 books
    std::unordered_map<std::string, std::uint32_t> name_to_id;
    std::vector<SymbolRecord> records;               // indexed by symbol id
    std::unordered_set<std::uint32_t> dirty;

    std::uint64_t membership_version{0};
    std::uint64_t total_quote_updates{0};
    std::uint64_t total_trade_updates{0};
    std::uint64_t total_bar_updates{0};
    std::uint64_t rejected{0};

    [[nodiscard]] bool valid_active(std::uint32_t id) const {
        return id < records.size() && records[id].active;
    }

    [[nodiscard]] float to_dollars(std::int64_t ticks) const {
        return static_cast<float>(ticks) / static_cast<float>(price_scale);
    }

    void push_bar(std::deque<analytics::Bar>& buf, const analytics::Bar& bar) {
        buf.push_back(bar);
        while (buf.size() > bar_capacity) buf.pop_front();
    }
};

DiscoveryEngine::DiscoveryEngine(std::size_t bar_capacity,
                                 std::int64_t price_scale)
    : impl_(std::make_unique<Impl>()) {
    impl_->bar_capacity = bar_capacity == 0 ? 1 : bar_capacity;
    impl_->price_scale = price_scale <= 0 ? kDefaultPriceScale : price_scale;
}

DiscoveryEngine::~DiscoveryEngine() = default;

std::uint64_t DiscoveryEngine::update_membership(
    const std::vector<std::string>& symbols) {
    std::lock_guard<std::mutex> lk(impl_->mtx);
    std::unordered_set<std::string> wanted(symbols.begin(), symbols.end());
    bool changed = false;

    for (const std::string& name : symbols) {
        auto it = impl_->name_to_id.find(name);
        if (it == impl_->name_to_id.end()) {
            const auto id = static_cast<std::uint32_t>(impl_->records.size());
            Impl::SymbolRecord rec;
            rec.name = name;
            rec.generation = 1;
            rec.active = true;
            impl_->records.push_back(std::move(rec));
            impl_->name_to_id.emplace(name, id);
            impl_->registry.add_symbol(name);
            changed = true;
        } else {
            Impl::SymbolRecord& rec = impl_->records[it->second];
            if (!rec.active) {
                // Re-entry: bump generation and drop now-stale state so a
                // consumer can reject results tagged with the old generation.
                ++rec.generation;
                rec.active = true;
                rec.last_obs_ns = 0;
                rec.last_sequence = 0;
                rec.bars_1m.clear();
                rec.bars_5m.clear();
                if (auto eng = impl_->registry.get(rec.name)) eng->market().clear();
                impl_->dirty.erase(it->second);
                changed = true;
            }
        }
    }

    for (std::uint32_t id = 0; id < impl_->records.size(); ++id) {
        Impl::SymbolRecord& rec = impl_->records[id];
        if (rec.active && wanted.find(rec.name) == wanted.end()) {
            rec.active = false;
            impl_->dirty.erase(id);
            changed = true;
        }
    }

    if (changed) ++impl_->membership_version;
    return impl_->membership_version;
}

std::optional<std::uint32_t> DiscoveryEngine::symbol_id(
    const std::string& symbol) const {
    std::lock_guard<std::mutex> lk(impl_->mtx);
    auto it = impl_->name_to_id.find(symbol);
    if (it == impl_->name_to_id.end()) return std::nullopt;
    return it->second;
}

std::string DiscoveryEngine::symbol_name(std::uint32_t id) const {
    std::lock_guard<std::mutex> lk(impl_->mtx);
    return id < impl_->records.size() ? impl_->records[id].name : std::string{};
}

std::uint32_t DiscoveryEngine::generation(std::uint32_t id) const {
    std::lock_guard<std::mutex> lk(impl_->mtx);
    return id < impl_->records.size() ? impl_->records[id].generation : 0u;
}

std::uint64_t DiscoveryEngine::membership_version() const noexcept {
    std::lock_guard<std::mutex> lk(impl_->mtx);
    return impl_->membership_version;
}

std::size_t DiscoveryEngine::symbol_count() const noexcept {
    std::lock_guard<std::mutex> lk(impl_->mtx);
    std::size_t n = 0;
    for (const auto& rec : impl_->records) {
        if (rec.active) ++n;
    }
    return n;
}

std::vector<std::string> DiscoveryEngine::symbols() const {
    std::lock_guard<std::mutex> lk(impl_->mtx);
    std::vector<std::string> out;
    for (const auto& rec : impl_->records) {
        if (rec.active) out.push_back(rec.name);
    }
    return out;
}

std::size_t DiscoveryEngine::ingest_quotes(const std::vector<QuoteObs>& quotes) {
    std::lock_guard<std::mutex> lk(impl_->mtx);
    std::size_t applied = 0;
    for (const QuoteObs& q : quotes) {
        if (!impl_->valid_active(q.symbol_id) ||
            q.bid_ticks <= 0 || q.ask_ticks <= 0 ||
            q.bid_ticks >= q.ask_ticks || q.ask_ticks > kMaxTicks) {
            ++impl_->rejected;
            continue;
        }
        Impl::SymbolRecord& rec = impl_->records[q.symbol_id];
        MarketQuoteUpdate upd;
        upd.symbol = rec.name;
        upd.timestamp_ns = q.timestamp_ns;
        upd.bid_price_ticks = q.bid_ticks;
        upd.bid_size = q.bid_size;
        upd.ask_price_ticks = q.ask_ticks;
        upd.ask_size = q.ask_size;
        upd.sequence = q.sequence;
        upd.provider = "discovery";
        if (!upd.valid() || !impl_->registry.on_market_quote(upd)) {
            ++impl_->rejected;
            continue;
        }
        ++applied;
        ++impl_->total_quote_updates;
        rec.last_obs_ns = std::max(rec.last_obs_ns, q.timestamp_ns);
        rec.last_sequence = std::max(rec.last_sequence, q.sequence);
        impl_->dirty.insert(q.symbol_id);
    }
    return applied;
}

std::size_t DiscoveryEngine::ingest_trades(const std::vector<TradeObs>& trades) {
    std::lock_guard<std::mutex> lk(impl_->mtx);
    std::size_t applied = 0;
    for (const TradeObs& t : trades) {
        if (!impl_->valid_active(t.symbol_id) || t.price_ticks <= 0 ||
            t.price_ticks > kMaxTicks || t.size == 0) {
            ++impl_->rejected;
            continue;
        }
        Impl::SymbolRecord& rec = impl_->records[t.symbol_id];
        MarketTradeUpdate upd;
        upd.symbol = rec.name;
        upd.timestamp_ns = t.timestamp_ns;
        upd.price_ticks = t.price_ticks;
        upd.size = t.size;
        upd.sequence = t.sequence;
        upd.provider = "discovery";
        if (!upd.valid() || !impl_->registry.on_market_trade(upd)) {
            ++impl_->rejected;
            continue;
        }
        ++applied;
        ++impl_->total_trade_updates;
        rec.last_obs_ns = std::max(rec.last_obs_ns, t.timestamp_ns);
        rec.last_sequence = std::max(rec.last_sequence, t.sequence);
        impl_->dirty.insert(t.symbol_id);
    }
    return applied;
}

std::size_t DiscoveryEngine::ingest_bars(const std::vector<BarObs>& bars,
                                         BarTimeframe tf) {
    std::lock_guard<std::mutex> lk(impl_->mtx);
    std::size_t applied = 0;
    for (const BarObs& b : bars) {
        if (!impl_->valid_active(b.symbol_id) || b.open_ticks <= 0 ||
            b.high_ticks <= 0 || b.low_ticks <= 0 || b.close_ticks <= 0 ||
            b.high_ticks < b.low_ticks || b.close_ticks > kMaxTicks ||
            b.volume < 0) {
            ++impl_->rejected;
            continue;
        }
        Impl::SymbolRecord& rec = impl_->records[b.symbol_id];
        analytics::Bar bar;
        bar.timestamp_ns = b.timestamp_ns;
        bar.open = impl_->to_dollars(b.open_ticks);
        bar.high = impl_->to_dollars(b.high_ticks);
        bar.low = impl_->to_dollars(b.low_ticks);
        bar.close = impl_->to_dollars(b.close_ticks);
        bar.volume = static_cast<float>(b.volume);
        auto& buf = tf == BarTimeframe::OneMinute ? rec.bars_1m : rec.bars_5m;
        impl_->push_bar(buf, bar);
        ++applied;
        ++impl_->total_bar_updates;
        rec.last_obs_ns = std::max(rec.last_obs_ns, b.timestamp_ns);
        impl_->dirty.insert(b.symbol_id);
    }
    return applied;
}

std::vector<std::uint32_t> DiscoveryEngine::dirty_symbols() const {
    std::lock_guard<std::mutex> lk(impl_->mtx);
    std::vector<std::uint32_t> out(impl_->dirty.begin(), impl_->dirty.end());
    std::sort(out.begin(), out.end());
    return out;
}

void DiscoveryEngine::clear_dirty() {
    std::lock_guard<std::mutex> lk(impl_->mtx);
    impl_->dirty.clear();
}

SymbolSnapshot DiscoveryEngine::snapshot(std::uint32_t id) const {
    std::lock_guard<std::mutex> lk(impl_->mtx);
    SymbolSnapshot s;
    if (id >= impl_->records.size()) return s;
    const Impl::SymbolRecord& rec = impl_->records[id];
    s.symbol_id = id;
    s.generation = rec.generation;
    s.symbol = rec.name;
    s.active = rec.active;
    s.last_obs_ns = rec.last_obs_ns;
    s.last_sequence = rec.last_sequence;
    s.bars_1m = rec.bars_1m.size();
    s.bars_5m = rec.bars_5m.size();
    s.dirty = impl_->dirty.find(id) != impl_->dirty.end();
    if (auto eng = impl_->registry.get(rec.name)) {
        const ObservedMarketSnapshot ms = eng->market_snapshot();
        s.has_quote = ms.has_quote();
        s.has_trade = ms.has_trade();
        if (ms.bid_price_ticks) s.bid_ticks = *ms.bid_price_ticks;
        if (ms.ask_price_ticks) s.ask_ticks = *ms.ask_price_ticks;
        if (ms.last_trade_price_ticks) s.last_trade_ticks = *ms.last_trade_price_ticks;
        s.quote_updates = ms.quote_updates;
        s.trade_updates = ms.trade_updates;
    }
    return s;
}

std::vector<analytics::Bar> DiscoveryEngine::bars(std::uint32_t id,
                                                  BarTimeframe tf) const {
    std::lock_guard<std::mutex> lk(impl_->mtx);
    if (id >= impl_->records.size()) return {};
    const auto& buf = tf == BarTimeframe::OneMinute ? impl_->records[id].bars_1m
                                                    : impl_->records[id].bars_5m;
    return {buf.begin(), buf.end()};
}

std::optional<float> DiscoveryEngine::last_bar_close(std::uint32_t id,
                                                     BarTimeframe tf) const {
    std::lock_guard<std::mutex> lk(impl_->mtx);
    if (id >= impl_->records.size()) return std::nullopt;
    const auto& buf = tf == BarTimeframe::OneMinute ? impl_->records[id].bars_1m
                                                    : impl_->records[id].bars_5m;
    if (buf.empty()) return std::nullopt;
    return buf.back().close;
}

BarWindow DiscoveryEngine::export_bar_window(
    const std::vector<std::uint32_t>& ids, BarTimeframe tf,
    std::uint32_t window) const {
    std::lock_guard<std::mutex> lk(impl_->mtx);
    BarWindow w;
    w.max_window = window == 0 ? 1 : window;
    w.symbol_count = static_cast<std::uint32_t>(ids.size());
    const std::size_t cells =
        static_cast<std::size_t>(w.symbol_count) * w.max_window;
    w.closes.assign(cells, 0.0f);
    w.highs.assign(cells, 0.0f);
    w.lows.assign(cells, 0.0f);
    w.volumes.assign(cells, 0.0f);
    w.counts.assign(w.symbol_count, 0u);

    for (std::uint32_t row = 0; row < ids.size(); ++row) {
        const std::uint32_t id = ids[row];
        if (id >= impl_->records.size()) continue;
        const auto& buf = tf == BarTimeframe::OneMinute
                              ? impl_->records[id].bars_1m
                              : impl_->records[id].bars_5m;
        const std::size_t avail = buf.size();
        const std::size_t take =
            avail < w.max_window ? avail : static_cast<std::size_t>(w.max_window);
        const std::size_t start = avail - take;  // last `take` bars, chronological
        const std::size_t base = static_cast<std::size_t>(row) * w.max_window;
        for (std::size_t i = 0; i < take; ++i) {
            const analytics::Bar& b = buf[start + i];
            w.closes[base + i] = b.close;
            w.highs[base + i] = b.high;
            w.lows[base + i] = b.low;
            w.volumes[base + i] = b.volume;
        }
        w.counts[row] = static_cast<std::uint32_t>(take);
    }
    return w;
}

CoverageSummary DiscoveryEngine::coverage() const {
    std::lock_guard<std::mutex> lk(impl_->mtx);
    CoverageSummary c;
    c.membership_version = impl_->membership_version;
    c.total_quote_updates = impl_->total_quote_updates;
    c.total_trade_updates = impl_->total_trade_updates;
    c.total_bar_updates = impl_->total_bar_updates;
    c.rejected = impl_->rejected;
    c.dirty = impl_->dirty.size();
    for (std::uint32_t id = 0; id < impl_->records.size(); ++id) {
        const Impl::SymbolRecord& rec = impl_->records[id];
        if (!rec.active) continue;
        ++c.symbols;
        if (!rec.bars_1m.empty()) ++c.with_bars_1m;
        if (!rec.bars_5m.empty()) ++c.with_bars_5m;
        bool q = false;
        bool t = false;
        if (auto eng = impl_->registry.get(rec.name)) {
            const ObservedMarketSnapshot ms = eng->market_snapshot();
            q = ms.has_quote();
            t = ms.has_trade();
        }
        if (q) ++c.with_quote;
        if (t) ++c.with_trade;
        if (q || t) ++c.observed;
    }
    return c;
}

}  // namespace trading::discovery
