"""Full-path intelligent-execution mock certification (Milestone 4, §24).

portfolio recommendation -> ExecutionPlanner (fresh-quote limit price from the L1
microstructure engine) -> CapitalReservation -> EXISTING CandidateRouter -> exposure
gate -> REAL native C++ RiskEngine -> OrderManager -> SimulatedBroker -> fills ->
TransactionCostAnalysis, with execution decision evidence.

No live orders; the native risk engine runs against a fresh per-test RuntimeState.
Timestamps use real time.time_ns() so the native freshness check passes.
"""

from __future__ import annotations

import time
from pathlib import Path

import _cengine_native as native
import numpy as np
import pytest

from cengine.capital_reservation import CapitalReservationLedger
from cengine.decision_events import DecisionEventSink, DecisionEventType
from cengine.execution_planner import ExecutionPlanner, PlannedExecutionPolicy
from cengine.exposure_registry import ExposureRegistry
from cengine.microstructure import MicrostructureEngine
from cengine.mock_broker import FillPlan, SimulatedBroker
from cengine.portfolio import AccountSnapshot, AccountState
from cengine.portfolio_analytics import PortfolioAnalytics
from cengine.portfolio_constraints import PortfolioConstraints
from cengine.portfolio_optimizer import OptimizerConfig, PortfolioOptimizer
from cengine.portfolio_pipeline import PortfolioPipeline, PortfolioQuantityPolicy
from cengine.portfolio_state import (
    PORTFOLIO_STATE_VERSION,
    PortfolioState,
    ReconciliationStatus,
)
from cengine.strategy_execution_bridge import make_decision_audit_handler
from cengine.strategy_maturity import ExecutionAuthorization
from cengine.strategy_state import StrategyStateBook
from cengine.transaction_cost import analyze_execution
from market_data.alpaca_stream import QuoteEvent, TradeEvent
from native_adapters import NativeRiskAdapter, RuntimeState
from order_manager import OrderManager, OrderStatus, OrderType, Side
from trade import CandidateRouter

PRICE = 1_000_000  # $100.00 at scale 10_000


def _metallib() -> str | None:
    for c in (Path(native.__file__).parent / "cengine.metallib",
              Path("build-metal/cengine.metallib")):
        if c.exists():
            return str(c)
    return None


def _quote(sym, ns, px=PRICE, half=100):
    return QuoteEvent(symbol=sym, timestamp_ns=ns, bid_price_ticks=px - half, bid_size=500,
                      bid_exchange="V", ask_price_ticks=px + half, ask_size=400,
                      ask_exchange="V", conditions=(), tape="C", sequence=1, received_ns=ns)


def _trade(sym, ns, px, sz):
    return TradeEvent(symbol=sym, timestamp_ns=ns, trade_id=1, exchange="V",
                      price_ticks=px, size=sz, conditions=(), tape="C", sequence=1,
                      received_ns=ns)


def _returns(symbols, obs=40, seed=5):
    rng = np.random.default_rng(seed)
    ts = list(range(1, obs + 1))
    return {s: {t: float(rng.normal(0.0004, 0.01 * (1 + 0.3 * k))) for t in ts}
            for k, s in enumerate(symbols)}


class ExecStack:
    def __init__(self, *, buying_power=100_000_000, fill_plan=FillPlan.FULL,
                 authorization=ExecutionAuthorization.MOCK, exposure=None, symbols=(),
                 half_spread=100, per_asset_cap=0.5, micro_age_s=0.0, gpu=True):
        self.ns = time.time_ns()
        self.symbols = symbols
        self.state_rt = RuntimeState()
        self.state_rt.account = AccountState()
        self.state_rt.account.replace_from_broker(AccountSnapshot(
            cash_ticks=buying_power, buying_power_ticks=buying_power,
            equity_ticks=buying_power, realized_pnl_ticks=0, updated_ns=1))
        self.micro = MicrostructureEngine()
        for s in symbols:
            self._feed(s, half_spread, micro_age_s)
            # fresh quote for the native risk freshness check (always at ~now)
            self.state_rt.quotes[s] = _quote(s, self.ns, half=half_spread)
        self.risk = NativeRiskAdapter(state=self.state_rt)
        self.broker = SimulatedBroker(default_plan=fill_plan, default_fill_price_ticks=PRICE)
        self.sink = DecisionEventSink()
        self.state_book = StrategyStateBook()
        self.corr: dict = {}
        audit = make_decision_audit_handler(self.sink, self.corr)
        self.mgr = OrderManager(risk_engine=self.risk, venue=self.broker,
                                on_audit=lambda e, o: (self.state_book.on_order(o), audit(e, o)))
        self.exposure = exposure if exposure is not None else ExposureRegistry()
        self.qpol = PortfolioQuantityPolicy()
        self.planned = PlannedExecutionPolicy()
        self.router = CandidateRouter(
            order_manager=self.mgr, entry_quantity_policy=self.qpol,
            exit_quantity_provider=self.qpol, execution_policy=self.planned,
            strategy_state=self.state_book, exposure_authorizer=self.exposure.authorize_entry)
        lib = _metallib()
        runtime = (native.MetalRuntime(lib) if (lib and gpu) else None)
        self.analytics = PortfolioAnalytics(runtime=runtime, ridge=1e-8)
        self.planner = ExecutionPlanner(execution_authorization=authorization.value)
        self.pipeline = PortfolioPipeline(
            router=self.router, quantity_policy=self.qpol,
            optimizer=PortfolioOptimizer(OptimizerConfig(
                constraints=PortfolioConstraints(max_position_weight=per_asset_cap,
                    max_portfolio_volatility=1.0),
                per_asset_cap=per_asset_cap, gross_cap=0.9), analytics=self.analytics),
            analytics=self.analytics, reservation_ledger=CapitalReservationLedger(
                risk_reservations=self.state_rt.reservations),
            sink=self.sink, authorization=authorization, correlation_index=self.corr,
            execution_planner=self.planner, microstructure=self.micro,
            planned_policy=self.planned)

    def _feed(self, sym, half, micro_age_s):
        # Feed ~40 quote+trade events ending ``micro_age_s`` before now (0 => fresh).
        end = self.ns - int(micro_age_s * 1e9)
        px = PRICE
        for i in range(40):
            t = end - (40 - i) * 100_000_000
            self.micro.on_quote_event(_quote(sym, t, px, half))
            self.micro.on_trade_event(_trade(sym, t + 1, px + half, 300))
            px += 20

    def state(self, *, equity=100_000_000, cash=100_000_000,
              status=ReconciliationStatus.RECONCILED):
        return PortfolioState(PORTFOLIO_STATE_VERSION, self.ns, equity, cash, cash, cash,
                              0, 0, 0, 0, 0, 0, 0, 0, reconciliation_status=status)

    async def run(self):
        return await self.pipeline.run(
            symbols=self.symbols, returns_by_symbol=_returns(self.symbols),
            state=self.state(), now_ns=self.ns, price_ticks=self.state_rt.price_ticks)

    async def drain(self):
        for o in self.mgr.all_orders():
            if o.venue_order_id:
                for u in self.broker.drain_updates(o.venue_order_id):
                    await self.mgr.reconcile(u)

    def event_types(self):
        return {e.event_type for e in self.sink.peek()}


# --------------------------------------------------------- narrow-spread happy path
@pytest.mark.asyncio
async def test_narrow_spread_marketable_limit_fills_with_evidence():
    st = ExecStack(symbols=("AAA", "BBB"), half_spread=100, per_asset_cap=0.4)
    res = await st.run()
    assert res.routed >= 1, res.dispositions
    await st.drain()
    orders = st.mgr.all_orders()
    assert any(o.intent.order_type is OrderType.LIMIT and o.intent.limit_price_ticks
               for o in orders)
    assert any(o.status is OrderStatus.FILLED for o in orders)
    t = st.event_types()
    assert DecisionEventType.EXECUTION_PLAN in t
    assert DecisionEventType.MICROSTRUCTURE_SNAPSHOT in t
    assert DecisionEventType.COMPLETE_FILL in t


# ----------------------------------------------------------- instrument classes
@pytest.mark.parametrize("symbol,authorize,expect", [
    ("AAA", False, True), ("SPY", False, True), ("XLK", False, True),
    ("TQQQ", False, False), ("TQQQ", True, True),
    ("SH", False, False), ("SH", True, True),
    ("SQQQ", False, False), ("SQQQ", True, True),
])
@pytest.mark.asyncio
async def test_instrument_class_exposure_gate(symbol, authorize, expect):
    exposure = ExposureRegistry()
    if authorize:
        exposure.authorize(symbol, True)
    st = ExecStack(symbols=(symbol,), exposure=exposure, per_asset_cap=0.4)
    res = await st.run()
    await st.drain()
    routed = {o.intent.symbol for o in st.mgr.all_orders()}
    if expect:
        assert symbol in routed, res.dispositions
    else:
        assert symbol not in routed


# ------------------------------------------------------------- wide spread -> passive
@pytest.mark.asyncio
async def test_wide_spread_passive_limit():
    st = ExecStack(symbols=("AAA",), half_spread=6_000, per_asset_cap=0.4)  # ~120 bps
    await st.run()
    await st.drain()
    orders = st.mgr.all_orders()
    assert orders and orders[0].intent.order_type is OrderType.LIMIT


# ------------------------------------------------------------- stale microstructure
@pytest.mark.asyncio
async def test_stale_microstructure_uses_immediate_market():
    # microstructure quotes are old (60s) -> plan IMMEDIATE_EXECUTION (market); the
    # RuntimeState quote stays fresh so native risk still approves.
    st = ExecStack(symbols=("AAA",), micro_age_s=60.0, per_asset_cap=0.4)
    await st.run()
    await st.drain()
    orders = st.mgr.all_orders()
    assert orders and orders[0].intent.order_type is OrderType.MARKET


# ------------------------------------------------------------- partial fills
@pytest.mark.asyncio
async def test_partial_then_full_fill():
    st = ExecStack(symbols=("AAA",), fill_plan=FillPlan.PARTIAL_THEN_FULL, per_asset_cap=0.4)
    await st.run()
    await st.drain()
    assert any(o.status is OrderStatus.FILLED for o in st.mgr.all_orders())
    t = st.event_types()
    assert DecisionEventType.PARTIAL_FILL in t and DecisionEventType.COMPLETE_FILL in t


# ------------------------------------------------------------- broker rejection
@pytest.mark.asyncio
async def test_broker_rejection_terminal():
    st = ExecStack(symbols=("AAA",), fill_plan=FillPlan.REJECT, per_asset_cap=0.4)
    await st.run()
    await st.drain()
    assert any(o.status is OrderStatus.REJECTED for o in st.mgr.all_orders())


# ------------------------------------------------------------- insufficient capital
@pytest.mark.asyncio
async def test_insufficient_capital_native_risk_rejects():
    st = ExecStack(symbols=("AAA",), buying_power=1_000, per_asset_cap=0.4)
    res = await st.run()
    assert res.routed == 0
    assert any("risk_rejected" in str(d[1]) for d in res.dispositions)


# ------------------------------------------------------------- advisory default
@pytest.mark.asyncio
async def test_advisory_default_routes_nothing_but_emits_recommendations():
    # Advisory (authorization NONE): recommendations + reservations + SHADOW execution
    # plans + microstructure are produced and emitted as evidence (§21/§24), but
    # nothing routes to the broker.
    st = ExecStack(symbols=("AAA",), authorization=ExecutionAuthorization.NONE,
                   per_asset_cap=0.4)
    res = await st.run()
    assert res.routed == 0
    assert st.broker.submitted == 0
    t = st.event_types()
    assert DecisionEventType.PORTFOLIO_RECOMMENDATION in t
    assert DecisionEventType.EXECUTION_PLAN in t  # shadow plan emitted, not routed


# ------------------------------------------------------------- GPU fallback
@pytest.mark.asyncio
async def test_gpu_unavailable_cpu_fallback_still_routes():
    st = ExecStack(symbols=("AAA", "BBB"), gpu=False, per_asset_cap=0.4)
    res = await st.run()
    assert res.covariance_source == "cpu"
    assert res.routed >= 1


# ------------------------------------------------------------- duplicate / restart
@pytest.mark.asyncio
async def test_duplicate_run_idempotent_reservation():
    st = ExecStack(symbols=("AAA",), fill_plan=FillPlan.NOFILL, per_asset_cap=0.4)
    await st.run()
    first = st.broker.submitted
    await st.run()  # same now_ns + pending guard -> no duplicate order
    assert st.broker.submitted == first


# ------------------------------------------------------------- TCA on a fill
@pytest.mark.asyncio
async def test_transaction_cost_analysis_on_fill():
    st = ExecStack(symbols=("AAA",), per_asset_cap=0.4)
    await st.run()
    await st.drain()
    order = next((o for o in st.mgr.all_orders() if o.status is OrderStatus.FILLED), None)
    assert order is not None
    outcome = analyze_execution(
        order, side=Side.BUY, decision_timestamp_ns=st.ns,
        decision_price_ticks=PRICE, arrival_price_ticks=PRICE,
        submission_timestamp_ns=st.ns + 1, acknowledgement_timestamp_ns=st.ns + 2,
        final_fill_timestamp_ns=st.ns + 3, quoted_spread_bps=2.0)
    assert outcome.implementation_shortfall_bps is not None
    assert outcome.realized_slippage_bps is not None
    assert outcome.filled_quantity == order.cumulative_filled_quantity
    st.sink.record_transaction_cost_analysis(outcome, st.ns + 3,
                                             decision_id="x")
    assert DecisionEventType.TRANSACTION_COST_ANALYSIS in st.event_types()


# ------------------------------------------------------------- exit independence
@pytest.mark.asyncio
async def test_mandatory_exit_independent_of_execution_planner():
    # A protective-exit close fires through the same OrderManager regardless of the
    # intelligent execution planner (which is for elective entries only).
    from protective_exit import ProtectiveExitCoordinator

    class _Pos:
        def __init__(s, sym, q, c):
            s.symbol, s.quantity, s.cost_ticks = sym, q, c

    class _Book:
        def __init__(s, p):
            s._p = p
        def positions(s):
            return list(s._p)
        def quantity(s, sym, strategy_id=None):
            return next((p.quantity for p in s._p if p.symbol == sym), 0)

    class _Prices:
        def price_ticks(s, sym):
            return 990_000  # -1% vs 1_000_000 avg -> breaches -0.5%

    class _Res:
        def projected_position(s, sym, held):
            return held

    st = ExecStack(symbols=("AAA",), per_asset_cap=0.4)
    coord = ProtectiveExitCoordinator(
        order_manager=st.mgr, positions=_Book([_Pos("NVDA", 10, 10_000_000)]),
        price_source=_Prices(), reservations=_Res(), strategy_state=st.state_book,
        threshold=-0.005, interval_s=0.0)
    # fresh quote so native risk approves the protective close
    st.state_rt.quotes["NVDA"] = _quote("NVDA", st.ns, 990_000)
    submitted = await coord.scan_and_exit()
    assert submitted == ["NVDA"]
    order = next(o for o in st.mgr.all_orders() if o.intent.symbol == "NVDA")
    assert order.intent.strategy_id == "protective_exit"
    assert order.intent.side is Side.SELL
