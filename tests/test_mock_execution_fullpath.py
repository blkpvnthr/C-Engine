"""Full-path mock-execution acceptance (directive §9).

Exercises all five strategy families through the REAL production components —
MarketRegimeEngine, the real family evaluators, arbitration + economic ranking, the
execution bridge, the EXISTING CandidateRouter + exposure gate, the REAL native C++
RiskEngine, the OrderManager, an isolated mock broker, and fill reconciliation —
emitting traceable decision evidence. No fabricated final recommendations: every
recommendation is produced by the real evaluators from controlled feature rows; only
the market observations and the (isolated, test-only) authorization are supplied.

No live orders. The native risk engine runs against a fresh, per-test RuntimeState
(never the process-global one).
"""

from __future__ import annotations

import time

import pytest

from cengine.decision_events import DecisionEventSink, DecisionEventType
from cengine.execution_policy import ConfiguredExecutionPolicy
from cengine.exposure_registry import ExposureRegistry
from cengine.mock_broker import FillPlan, SimulatedBroker
from cengine.portfolio import AccountSnapshot, AccountState
from cengine.regime_strategy_pipeline import RegimeStrategyPipeline
from cengine.strategy_execution_bridge import (
    StrategyExecutionBridge,
    make_decision_audit_handler,
)
from cengine.strategy_families import StrategyFamily, StrategyFamilyRegistry
from cengine.strategy_maturity import ExecutionAuthorization
from cengine.strategy_state import StrategyStateBook
from market_data.alpaca_stream import QuoteEvent
from native_adapters import NativeRiskAdapter, RuntimeState
from order_manager import OrderManager, OrderStatus, OrderType, TimeInForce
from trade import CandidateRouter

PRICE_TICKS = 1_000_000  # $100.00 at scale 10_000


# --------------------------------------------------------------------------- rows
def _row(**over) -> dict:
    base = 100.0
    row = {
        "pdi": 32.0, "mdi": 9.0, "di_plus_5m": 32.0, "di_minus_5m": 9.0, "adx": 30.0,
        "price": 101.0, "vwap": base, "sma20": base, "sma20_5m": base, "bb_mid": base,
        "bb_middle_5m": base, "bb_width": 0.05, "vwap_dist": 0.01, "bb_dist": 0.012,
        "short_mom": 0.005, "momentum": 0.005, "momentum_accel": 0.002,
        "developing_1m_price": 101.1, "previous_completed_1m_close": base,
        "atr_pct": 0.003, "rel_volume": 1.6, "spread_bps": 3.0,
        "atr_expansion_ratio": 1.0, "rsi": 27.0, "cci": -130.0, "htf_direction": 1.0,
        "trend_15m": 1.0, "trend_1h": 1.0, "trend_1d": 1.0,
        "sector_rel_strength": 0.3, "realized_vol": 0.003, "observation_ns": 0,
    }
    row.update(over)
    return row


# Regime that makes each family's conditions pass (verified empirically):
FAMILY_ROW = {
    # bullish / expansion
    StrategyFamily.MOMENTUM_CONTINUATION: _row(atr_expansion_ratio=1.3, realized_vol=0.01),
    StrategyFamily.VOLATILITY_BREAKOUT: _row(atr_expansion_ratio=1.3, realized_vol=0.01,
                                             donchian_high=100.2),
    # bullish / recovery (momentum<0 turning up; short_mom>0 for reversal confirmation)
    StrategyFamily.TREND_PULLBACK: _row(momentum=-0.002, momentum_accel=0.002,
                                        short_mom=0.004, adx=22.0),
    StrategyFamily.MEAN_REVERSION: _row(momentum=-0.002, momentum_accel=0.002,
                                        short_mom=0.004, adx=22.0),
    # multi-day swing ignores instrument trend compat; needs aligned timeframes
    StrategyFamily.MULTI_DAY_SWING: _row(),
}


def _quote(sym: str, ns: int) -> QuoteEvent:
    return QuoteEvent(symbol=sym, timestamp_ns=ns, bid_price_ticks=PRICE_TICKS - 100,
                      bid_size=500, bid_exchange="V", ask_price_ticks=PRICE_TICKS + 100,
                      ask_size=500, ask_exchange="V", conditions=(), tape="C",
                      sequence=1, received_ns=ns)


class _Qty:
    """Fixed-share sizing (native risk remains authoritative)."""

    def __init__(self, qty: int = 5) -> None:
        self._qty = qty

    def quantity_for(self, candidate):
        return self._qty

    def quantity_to_close(self, candidate):
        return self._qty


class Stack:
    """A full real execution stack with native risk + isolated mock broker."""

    def __init__(self, *, buying_power: int = 10_000_000_000,
                 fill_plan: FillPlan = FillPlan.FULL, qty: int = 5,
                 authorization: ExecutionAuthorization = ExecutionAuthorization.MOCK,
                 exposure: ExposureRegistry | None = None,
                 symbols: tuple[str, ...] = ()) -> None:
        self.ns = time.time_ns()
        self.state = RuntimeState()
        self.state.account = AccountState()
        self.state.account.replace_from_broker(AccountSnapshot(
            cash_ticks=buying_power, buying_power_ticks=buying_power,
            equity_ticks=buying_power, realized_pnl_ticks=0, updated_ns=1))
        for sym in symbols:
            self.state.quotes[sym] = _quote(sym, self.ns)
        self.risk = NativeRiskAdapter(state=self.state)
        self.broker = SimulatedBroker(default_plan=fill_plan,
                                      default_fill_price_ticks=PRICE_TICKS)
        self.sink = DecisionEventSink()
        self.registry = StrategyFamilyRegistry()
        self.pipeline = RegimeStrategyPipeline(registry=self.registry, sink=self.sink)
        self.state_book = StrategyStateBook()
        self.corr_index: dict = {}
        audit = make_decision_audit_handler(self.sink, self.corr_index)

        def on_audit(event, order):
            self.state_book.on_order(order)
            audit(event, order)

        self.mgr = OrderManager(risk_engine=self.risk, venue=self.broker,
                                on_audit=on_audit)
        self.exposure = exposure if exposure is not None else ExposureRegistry()
        self.router = CandidateRouter(
            order_manager=self.mgr, entry_quantity_policy=_Qty(qty),
            exit_quantity_provider=_Qty(qty),
            execution_policy=ConfiguredExecutionPolicy(OrderType.MARKET, TimeInForce.DAY),
            strategy_state=self.state_book,
            exposure_authorizer=self.exposure.authorize_entry)
        self.bridge = StrategyExecutionBridge(
            router=self.router, registry=self.registry, sink=self.sink,
            price_ticks_fn=self.state.price_ticks, authorization=authorization,
            correlation_index=self.corr_index)

    def recommendations_for(self, symbol: str, row: dict) -> list:
        row = {**row, "observation_ns": self.ns}
        refs = {s: _row(observation_ns=self.ns) for s in ("SPY", "QQQ", "DIA", "IWM")}
        out = self.pipeline.run({symbol: row}, now_ns=self.ns, ref_features=refs)
        return out["recommendations"]

    async def drain_fills(self) -> None:
        for order in self.mgr.all_orders():
            if order.venue_order_id:
                for upd in self.broker.drain_updates(order.venue_order_id):
                    await self.mgr.reconcile(upd)


# ----------------------------------------------------------- per-family full path
@pytest.mark.parametrize("family", list(StrategyFamily))
@pytest.mark.asyncio
async def test_each_family_routes_through_real_path_to_fill(family):
    symbol = f"EQ{family.value[:3].upper()}"
    stack = Stack(symbols=(symbol,))
    stack.registry.mark_mock_execution_eligible(family, test_authorization=True)
    recs = [r for r in stack.recommendations_for(symbol, FAMILY_ROW[family])
            if r.strategy_family is family]
    assert recs, f"{family.value}: real evaluator produced no recommendation"
    assert recs[0].conditions_passed, f"{family.value}: conditions did not pass"

    outcome = await stack.bridge.route_recommendations(recs, now_ns=stack.ns)
    assert outcome.submitted == 1, outcome.dispositions
    await stack.drain_fills()

    order = stack.mgr.all_orders()[0]
    assert order.status is OrderStatus.FILLED
    assert order.cumulative_filled_quantity == 5
    # traceability: recommendation decision_id -> order intent -> fill
    types = {r.event_type for r in stack.sink.peek()}
    assert {DecisionEventType.ORDER_INTENT, DecisionEventType.RISK_APPROVAL,
            DecisionEventType.BROKER_ACK, DecisionEventType.COMPLETE_FILL} <= types


# ------------------------------------------------------- instrument-class coverage
@pytest.mark.parametrize("symbol,authorize,expect_route", [
    ("AAPL", False, True),    # conventional equity: not in registry -> auto-authorized
    ("SPY", False, True),     # broad-market ETF (1x) -> authorized
    ("XLK", False, True),     # sector ETF (1x) -> authorized
    ("TQQQ", False, False),   # leveraged long -> blocked unless authorized
    ("TQQQ", True, True),
    ("SH", False, False),     # inverse -> blocked unless authorized
    ("SH", True, True),
    ("SQQQ", False, False),   # leveraged inverse -> blocked unless authorized
    ("SQQQ", True, True),
])
@pytest.mark.asyncio
async def test_instrument_classes_respect_exposure_gate(symbol, authorize, expect_route):
    exposure = ExposureRegistry()
    if authorize:
        exposure.authorize(symbol, True)
    stack = Stack(symbols=(symbol,), exposure=exposure)
    fam = StrategyFamily.MOMENTUM_CONTINUATION
    stack.registry.mark_mock_execution_eligible(fam, test_authorization=True)
    recs = [r for r in stack.recommendations_for(symbol, FAMILY_ROW[fam])
            if r.strategy_family is fam]
    assert recs and recs[0].conditions_passed
    outcome = await stack.bridge.route_recommendations(recs, now_ns=stack.ns)
    if expect_route:
        assert outcome.submitted == 1, outcome.dispositions
    else:
        assert outcome.submitted == 0
        assert outcome.by_outcome("exposure_blocked")


# -------------------------------------------------------------- negative scenarios
@pytest.mark.asyncio
async def test_stale_observation_yields_no_recommendation():
    symbol = "STALE"
    stack = Stack(symbols=(symbol,))
    stack.registry.mark_mock_execution_eligible(
        StrategyFamily.MOMENTUM_CONTINUATION, test_authorization=True)
    # observation stamped now, but queried 10 minutes later -> regime stale -> invalid
    row = {**FAMILY_ROW[StrategyFamily.MOMENTUM_CONTINUATION], "observation_ns": stack.ns}
    refs = {s: _row(observation_ns=stack.ns) for s in ("SPY", "QQQ", "DIA", "IWM")}
    future = stack.ns + 10 * 60 * 1_000_000_000  # 10 minutes later (> 5m staleness)
    out = stack.pipeline.run({symbol: row}, now_ns=future, ref_features=refs)
    assert out["assessments"][symbol].valid is False
    # a stale/invalid instrument regime produces no per-family recommendations
    assert [r for r in out["recommendations"] if r.symbol_id == symbol] == []


@pytest.mark.asyncio
async def test_insufficient_capital_rejected_by_native_risk():
    symbol = "POORCAP"
    stack = Stack(symbols=(symbol,), buying_power=1_000)  # far below order notional
    fam = StrategyFamily.MOMENTUM_CONTINUATION
    stack.registry.mark_mock_execution_eligible(fam, test_authorization=True)
    recs = [r for r in stack.recommendations_for(symbol, FAMILY_ROW[fam])
            if r.strategy_family is fam]
    outcome = await stack.bridge.route_recommendations(recs, now_ns=stack.ns)
    assert outcome.submitted == 0
    assert outcome.by_outcome("risk_rejected")
    # risk rejection is captured as decision evidence
    types = {r.event_type for r in stack.sink.peek()}
    assert DecisionEventType.RISK_REJECTION in types


@pytest.mark.asyncio
async def test_partial_then_full_fill_evidence():
    symbol = "PARTIAL"
    stack = Stack(symbols=(symbol,), fill_plan=FillPlan.PARTIAL_THEN_FULL, qty=10)
    fam = StrategyFamily.MOMENTUM_CONTINUATION
    stack.registry.mark_mock_execution_eligible(fam, test_authorization=True)
    recs = [r for r in stack.recommendations_for(symbol, FAMILY_ROW[fam])
            if r.strategy_family is fam]
    await stack.bridge.route_recommendations(recs, now_ns=stack.ns)
    await stack.drain_fills()
    order = stack.mgr.all_orders()[0]
    assert order.status is OrderStatus.FILLED
    types = [r.event_type for r in stack.sink.peek()]
    assert DecisionEventType.PARTIAL_FILL in types
    assert DecisionEventType.COMPLETE_FILL in types


@pytest.mark.asyncio
async def test_broker_rejection_is_terminal():
    symbol = "REJ"
    stack = Stack(symbols=(symbol,), fill_plan=FillPlan.REJECT)
    fam = StrategyFamily.MOMENTUM_CONTINUATION
    stack.registry.mark_mock_execution_eligible(fam, test_authorization=True)
    recs = [r for r in stack.recommendations_for(symbol, FAMILY_ROW[fam])
            if r.strategy_family is fam]
    await stack.bridge.route_recommendations(recs, now_ns=stack.ns)
    await stack.drain_fills()
    order = stack.mgr.all_orders()[0]
    assert order.status is OrderStatus.REJECTED


@pytest.mark.asyncio
async def test_duplicate_order_prevented_by_pending_guard():
    symbol = "DUP"
    stack = Stack(symbols=(symbol,), fill_plan=FillPlan.NOFILL)
    fam = StrategyFamily.MOMENTUM_CONTINUATION
    stack.registry.mark_mock_execution_eligible(fam, test_authorization=True)
    recs = [r for r in stack.recommendations_for(symbol, FAMILY_ROW[fam])
            if r.strategy_family is fam]
    first = await stack.bridge.route_recommendations(recs, now_ns=stack.ns)
    assert first.submitted == 1
    # second pass before the first order terminalizes -> pending guard withholds it
    second = await stack.bridge.route_recommendations(recs, now_ns=stack.ns + 1)
    assert second.submitted == 0
    assert second.by_outcome("exposure_blocked")  # router returned submitted==0
    assert stack.broker.submitted == 1
