#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "orderbook/orderbook.hpp"
#include "risk/risk_engine.hpp"

namespace py = pybind11;
using namespace trading;

PYBIND11_MODULE(_cengine_native, m) {
    m.doc() = "C-Engine native risk and order-book boundary";
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
}
