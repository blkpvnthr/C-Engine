#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "opportunity_discovery/discovery_engine.hpp"
#include "orderbook/orderbook.hpp"
#include "risk/risk_engine.hpp"

namespace py = pybind11;
using namespace trading;

// Implemented in bindings/discovery_bindings.cpp. Registers the native
// Objective-C++ Metal runtime + kernel registry on the same module. A no-op
// stub is provided there for non-Apple builds so this call always links.
void register_discovery(py::module_& m);

// Implemented in bindings/portfolio_bindings.cpp. Registers the native
// deterministic portfolio allocator + portfolio-risk math (Milestone 3).
void register_portfolio(py::module_& m);

PYBIND11_MODULE(_cengine_native, m) {
    m.doc() = "C-Engine native risk, order-book boundary, and Metal runtime";
    py::enum_<Side>(m, "Side").value("BUY", Side::Buy).value("SELL", Side::Sell);
    py::enum_<RiskDecision>(m, "RiskDecision")
        .value("ACCEPT", RiskDecision::Accept).value("REJECT", RiskDecision::Reject);
    py::enum_<RiskRejectReason>(m, "RiskRejectReason")
        .value("NONE", RiskRejectReason::None).value("KILL_SWITCH_ACTIVE", RiskRejectReason::KillSwitchActive)
        .value("INVALID_ORDER_ID", RiskRejectReason::InvalidOrderId).value("DUPLICATE_ORDER_ID", RiskRejectReason::DuplicateOrderId)
        .value("INVALID_SYMBOL", RiskRejectReason::InvalidSymbol).value("INVALID_QUANTITY", RiskRejectReason::InvalidQuantity)
        .value("INVALID_PRICE", RiskRejectReason::InvalidPrice).value("QUANTITY_LIMIT_EXCEEDED", RiskRejectReason::QuantityLimitExceeded)
        .value("NOTIONAL_LIMIT_EXCEEDED", RiskRejectReason::NotionalLimitExceeded).value("MISSING_MARKET_DATA", RiskRejectReason::MissingMarketData)
        .value("STALE_MARKET_DATA", RiskRejectReason::StaleMarketData).value("INVALID_MARKET_DATA", RiskRejectReason::InvalidMarketData)
        .value("PRICE_COLLAR_EXCEEDED", RiskRejectReason::PriceCollarExceeded).value("INSUFFICIENT_BUYING_POWER", RiskRejectReason::InsufficientBuyingPower)
        .value("UNKNOWN", RiskRejectReason::Unknown);
    py::class_<RiskMarketState>(m, "RiskMarketState").def(py::init<>())
        .def_readwrite("symbol", &RiskMarketState::symbol).def_readwrite("bid_ticks", &RiskMarketState::bid_ticks)
        .def_readwrite("ask_ticks", &RiskMarketState::ask_ticks).def_readwrite("last_ticks", &RiskMarketState::last_ticks)
        .def_readwrite("timestamp_ns", &RiskMarketState::timestamp_ns);
    py::class_<RiskAccountState>(m, "RiskAccountState").def(py::init<>())
        .def_readwrite("position", &RiskAccountState::position).def_readwrite("cash_ticks", &RiskAccountState::cash_ticks)
        .def_readwrite("buying_power_ticks", &RiskAccountState::buying_power_ticks)
        .def_readwrite("gross_exposure_ticks", &RiskAccountState::gross_exposure_ticks);
    py::class_<OpenOrderRiskState>(m, "OpenOrderRiskState").def(py::init<>())
        .def_readwrite("count", &OpenOrderRiskState::count)
        .def_readwrite("total_remaining_quantity", &OpenOrderRiskState::total_remaining_quantity)
        .def_readwrite("total_buy_notional_ticks", &OpenOrderRiskState::total_buy_notional_ticks)
        .def_readwrite("total_sell_notional_ticks", &OpenOrderRiskState::total_sell_notional_ticks);
    py::class_<RiskRequest>(m, "RiskRequest").def(py::init<>())
        .def_readwrite("order_id", &RiskRequest::order_id).def_readwrite("symbol", &RiskRequest::symbol)
        .def_readwrite("side", &RiskRequest::side).def_readwrite("quantity", &RiskRequest::quantity)
        .def_readwrite("price_ticks", &RiskRequest::price_ticks).def_readwrite("market", &RiskRequest::market)
        .def_readwrite("account", &RiskRequest::account).def_readwrite("open_orders", &RiskRequest::open_orders)
        .def_readwrite("now_ns", &RiskRequest::now_ns);
    py::class_<RiskResult>(m, "RiskResult").def_property_readonly("accepted", &RiskResult::accepted)
        .def_readonly("decision", &RiskResult::decision).def_readonly("reason", &RiskResult::reason)
        .def_readonly("message", &RiskResult::message).def_readonly("calculated_notional_ticks", &RiskResult::calculated_notional_ticks)
        .def_readonly("projected_position", &RiskResult::projected_position)
        .def_readonly("projected_gross_exposure_ticks", &RiskResult::projected_gross_exposure_ticks);
    py::class_<RiskEngine>(m, "RiskEngine").def(py::init<>()).def("evaluate", &RiskEngine::evaluate)
        .def("activate_kill_switch", &RiskEngine::activate_kill_switch)
        .def("deactivate_kill_switch", &RiskEngine::deactivate_kill_switch)
        .def_property_readonly("kill_switch_active", &RiskEngine::kill_switch_active);
    py::class_<Execution>(m, "Execution").def_readonly("execution_id", &Execution::execution_id)
        .def_readonly("maker_id", &Execution::maker_id).def_readonly("taker_id", &Execution::taker_id)
        .def_readonly("symbol", &Execution::symbol).def_readonly("quantity", &Execution::quantity)
        .def_readonly("price_ticks", &Execution::price_ticks);
    py::class_<LimitOrderBook>(m, "LimitOrderBook").def(py::init<std::string>())
        .def("add_limit", &LimitOrderBook::add_limit).def("amend", &LimitOrderBook::amend)
        .def("cancel", &LimitOrderBook::cancel).def("contains", &LimitOrderBook::contains)
        .def_property_readonly("size", &LimitOrderBook::size).def_property_readonly("symbol", &LimitOrderBook::symbol);

    // ------------------------------------------------------------------ Phase B.2
    // Native market-state ingestion (discovery_engine.hpp). Pure C++ (no Metal),
    // always available. The supervisor batches IEX observations into these.
    {
        using namespace trading::discovery;
        py::enum_<BarTimeframe>(m, "BarTimeframe")
            .value("ONE_MINUTE", BarTimeframe::OneMinute)
            .value("FIVE_MINUTE", BarTimeframe::FiveMinute);

        py::class_<QuoteObs>(m, "QuoteObs")
            .def(py::init([](std::uint32_t sid, std::uint64_t ts, std::int64_t bid,
                             std::int64_t ask, std::uint64_t bsz, std::uint64_t asz,
                             std::uint64_t seq) {
                     return QuoteObs{sid, ts, bid, ask, bsz, asz, seq};
                 }),
                 py::arg("symbol_id"), py::arg("timestamp_ns"), py::arg("bid_ticks"),
                 py::arg("ask_ticks"), py::arg("bid_size") = 0,
                 py::arg("ask_size") = 0, py::arg("sequence") = 0)
            .def_readwrite("symbol_id", &QuoteObs::symbol_id)
            .def_readwrite("timestamp_ns", &QuoteObs::timestamp_ns)
            .def_readwrite("bid_ticks", &QuoteObs::bid_ticks)
            .def_readwrite("ask_ticks", &QuoteObs::ask_ticks)
            .def_readwrite("bid_size", &QuoteObs::bid_size)
            .def_readwrite("ask_size", &QuoteObs::ask_size)
            .def_readwrite("sequence", &QuoteObs::sequence);

        py::class_<TradeObs>(m, "TradeObs")
            .def(py::init([](std::uint32_t sid, std::uint64_t ts, std::int64_t px,
                             std::uint64_t sz, std::uint64_t seq) {
                     return TradeObs{sid, ts, px, sz, seq};
                 }),
                 py::arg("symbol_id"), py::arg("timestamp_ns"),
                 py::arg("price_ticks"), py::arg("size") = 0, py::arg("sequence") = 0)
            .def_readwrite("symbol_id", &TradeObs::symbol_id)
            .def_readwrite("timestamp_ns", &TradeObs::timestamp_ns)
            .def_readwrite("price_ticks", &TradeObs::price_ticks)
            .def_readwrite("size", &TradeObs::size)
            .def_readwrite("sequence", &TradeObs::sequence);

        py::class_<BarObs>(m, "BarObs")
            .def(py::init([](std::uint32_t sid, std::uint64_t ts, std::int64_t o,
                             std::int64_t h, std::int64_t l, std::int64_t c,
                             std::int64_t v) {
                     return BarObs{sid, ts, o, h, l, c, v};
                 }),
                 py::arg("symbol_id"), py::arg("timestamp_ns"), py::arg("open_ticks"),
                 py::arg("high_ticks"), py::arg("low_ticks"), py::arg("close_ticks"),
                 py::arg("volume") = 0)
            .def_readwrite("symbol_id", &BarObs::symbol_id)
            .def_readwrite("timestamp_ns", &BarObs::timestamp_ns)
            .def_readwrite("open_ticks", &BarObs::open_ticks)
            .def_readwrite("high_ticks", &BarObs::high_ticks)
            .def_readwrite("low_ticks", &BarObs::low_ticks)
            .def_readwrite("close_ticks", &BarObs::close_ticks)
            .def_readwrite("volume", &BarObs::volume);

        py::class_<SymbolSnapshot>(m, "SymbolSnapshot")
            .def_readonly("symbol_id", &SymbolSnapshot::symbol_id)
            .def_readonly("generation", &SymbolSnapshot::generation)
            .def_readonly("symbol", &SymbolSnapshot::symbol)
            .def_readonly("active", &SymbolSnapshot::active)
            .def_readonly("has_quote", &SymbolSnapshot::has_quote)
            .def_readonly("has_trade", &SymbolSnapshot::has_trade)
            .def_readonly("bid_ticks", &SymbolSnapshot::bid_ticks)
            .def_readonly("ask_ticks", &SymbolSnapshot::ask_ticks)
            .def_readonly("last_trade_ticks", &SymbolSnapshot::last_trade_ticks)
            .def_readonly("last_obs_ns", &SymbolSnapshot::last_obs_ns)
            .def_readonly("last_sequence", &SymbolSnapshot::last_sequence)
            .def_readonly("quote_updates", &SymbolSnapshot::quote_updates)
            .def_readonly("trade_updates", &SymbolSnapshot::trade_updates)
            .def_readonly("bars_1m", &SymbolSnapshot::bars_1m)
            .def_readonly("bars_5m", &SymbolSnapshot::bars_5m)
            .def_readonly("dirty", &SymbolSnapshot::dirty);

        py::class_<BarWindow>(m, "BarWindow")
            .def_readonly("closes", &BarWindow::closes)
            .def_readonly("highs", &BarWindow::highs)
            .def_readonly("lows", &BarWindow::lows)
            .def_readonly("volumes", &BarWindow::volumes)
            .def_readonly("counts", &BarWindow::counts)
            .def_readonly("symbol_count", &BarWindow::symbol_count)
            .def_readonly("max_window", &BarWindow::max_window);

        py::class_<CoverageSummary>(m, "DiscoveryCoverage")
            .def_readonly("symbols", &CoverageSummary::symbols)
            .def_readonly("observed", &CoverageSummary::observed)
            .def_readonly("with_quote", &CoverageSummary::with_quote)
            .def_readonly("with_trade", &CoverageSummary::with_trade)
            .def_readonly("with_bars_1m", &CoverageSummary::with_bars_1m)
            .def_readonly("with_bars_5m", &CoverageSummary::with_bars_5m)
            .def_readonly("dirty", &CoverageSummary::dirty)
            .def_readonly("membership_version", &CoverageSummary::membership_version)
            .def_readonly("total_quote_updates", &CoverageSummary::total_quote_updates)
            .def_readonly("total_trade_updates", &CoverageSummary::total_trade_updates)
            .def_readonly("total_bar_updates", &CoverageSummary::total_bar_updates)
            .def_readonly("rejected", &CoverageSummary::rejected);

        py::class_<DiscoveryEngine>(m, "DiscoveryEngine")
            .def(py::init<std::size_t, std::int64_t>(),
                 py::arg("bar_capacity") = 512,
                 py::arg("price_scale") = kDefaultPriceScale)
            .def("update_membership", &DiscoveryEngine::update_membership,
                 py::arg("symbols"))
            .def("symbol_id", &DiscoveryEngine::symbol_id, py::arg("symbol"))
            .def("symbol_name", &DiscoveryEngine::symbol_name, py::arg("symbol_id"))
            .def("generation", &DiscoveryEngine::generation, py::arg("symbol_id"))
            .def_property_readonly("membership_version",
                                   &DiscoveryEngine::membership_version)
            .def_property_readonly("symbol_count", &DiscoveryEngine::symbol_count)
            .def("symbols", &DiscoveryEngine::symbols)
            .def("ingest_quotes", &DiscoveryEngine::ingest_quotes, py::arg("quotes"),
                 py::call_guard<py::gil_scoped_release>())
            .def("ingest_trades", &DiscoveryEngine::ingest_trades, py::arg("trades"),
                 py::call_guard<py::gil_scoped_release>())
            .def("ingest_bars", &DiscoveryEngine::ingest_bars, py::arg("bars"),
                 py::arg("timeframe"), py::call_guard<py::gil_scoped_release>())
            .def("dirty_symbols", &DiscoveryEngine::dirty_symbols)
            .def("clear_dirty", &DiscoveryEngine::clear_dirty)
            .def("snapshot", &DiscoveryEngine::snapshot, py::arg("symbol_id"))
            .def("last_bar_close", &DiscoveryEngine::last_bar_close,
                 py::arg("symbol_id"), py::arg("timeframe"))
            .def("export_bar_window", &DiscoveryEngine::export_bar_window,
                 py::arg("ids"), py::arg("timeframe"), py::arg("window"),
                 py::call_guard<py::gil_scoped_release>())
            .def("coverage", &DiscoveryEngine::coverage);
    }

    register_discovery(m);
    register_portfolio(m);
}
