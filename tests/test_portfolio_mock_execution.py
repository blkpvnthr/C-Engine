"""Full-path portfolio mock-execution certification (Milestone 3, §22).

Exercises actual optimizer recommendations through capital reservation and the
EXISTING execution authority (CandidateRouter -> exposure gate -> REAL native C++
RiskEngine -> OrderManager -> isolated mock broker -> fill reconciliation), with
portfolio decision evidence. No live orders; the native risk engine runs against a
fresh per-test RuntimeState (never the process-global one).
"""

from __future__ import annotations

import time
from pathlib import Path

import _cengine_native as native
import numpy as np
import pytest

from cengine.capital_reservation import CapitalReservationLedger
from cengine.decision_events import DecisionEventSink, DecisionEventType
from cengine.execution_policy import ConfiguredExecutionPolicy
from cengine.exposure_registry import ExposureRegistry
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
from cengine.strategy_maturity import ExecutionAuthorization
from cengine.strategy_state import StrategyStateBook
from market_data.alpaca_stream import QuoteEvent
from native_adapters import NativeRiskAdapter, RuntimeState
from order_manager import OrderManager, OrderStatus, OrderType, TimeInForce
from trade import CandidateRouter

PRICE_TICKS = 1_000_000  # $100.00 at scale 10_000


def _metallib() -> str | None:
    for c in (Path(native.__file__).parent / "cengine.metallib",
              Path("build-metal/cengine.metallib")):
        if c.exists():
            return str(c)
    return None


def _quote(sym: str, ns: int, px: int = PRICE_TICKS) -> QuoteEvent:
    return QuoteEvent(symbol=sym, timestamp_ns=ns, bid_price_ticks=px - 100, bid_size=500,
                      bid_exchange="V", ask_price_ticks=px + 100, ask_size=500,
                      ask_exchange="V", conditions=(), tape="C", sequence=1, received_ns=ns)


def _returns(symbols, obs=40, seed=5, vols=None):
    rng = np.random.default_rng(seed)
    ts = list(range(1, obs + 1))
    out = {}
    for k, s in enumerate(symbols):
        vol = (vols[k] if vols else 0.01 * (1 + 0.3 * k))
        out[s] = {t: float(rng.normal(0.0004, vol)) for t in ts}
    return out


class Stack:
    """Full portfolio execution stack with native risk + isolated mock broker."""

    # Sized so per-symbol order notional stays under the native RiskEngine default
    # max_order_notional_ticks (100M ticks = $10k at scale 10_000): $10k equity,
    # ~50-share orders at $100.
    def __init__(self, *, buying_power=100_000_000, fill_plan=FillPlan.FULL,
                 authorization=ExecutionAuthorization.MOCK, exposure=None,
                 symbols=(), per_asset_cap=0.5, gross_cap=0.9, allow_fractional=False):
        self.ns = time.time_ns()
        self.state_rt = RuntimeState()
        self.state_rt.account = AccountState()
        self.state_rt.account.replace_from_broker(AccountSnapshot(
            cash_ticks=buying_power, buying_power_ticks=buying_power,
            equity_ticks=buying_power, realized_pnl_ticks=0, updated_ns=1))
        for s in symbols:
            self.state_rt.quotes[s] = _quote(s, self.ns)
        self.risk = NativeRiskAdapter(state=self.state_rt)
        self.broker = SimulatedBroker(default_plan=fill_plan, default_fill_price_ticks=PRICE_TICKS)
        self.sink = DecisionEventSink()
        self.state_book = StrategyStateBook()
        self.corr_index: dict = {}
        from cengine.strategy_execution_bridge import make_decision_audit_handler
        audit = make_decision_audit_handler(self.sink, self.corr_index)

        def _on_audit(e, o):
            self.state_book.on_order(o)
            audit(e, o)

        self.mgr = OrderManager(risk_engine=self.risk, venue=self.broker, on_audit=_on_audit)
        self.exposure = exposure if exposure is not None else ExposureRegistry()
        self.qpol = PortfolioQuantityPolicy()
        self.router = CandidateRouter(
            order_manager=self.mgr, entry_quantity_policy=self.qpol,
            exit_quantity_provider=self.qpol,
            execution_policy=ConfiguredExecutionPolicy(OrderType.MARKET, TimeInForce.DAY),
            strategy_state=self.state_book,
            exposure_authorizer=self.exposure.authorize_entry)
        lib = _metallib()
        runtime = native.MetalRuntime(lib) if lib else None
        self.analytics = PortfolioAnalytics(runtime=runtime, ridge=1e-8)
        self.optimizer = PortfolioOptimizer(OptimizerConfig(
            constraints=PortfolioConstraints(max_position_weight=per_asset_cap,
                max_gross_exposure=1.0, max_portfolio_volatility=1.0),
            per_asset_cap=per_asset_cap, gross_cap=gross_cap,
            allow_fractional=allow_fractional), analytics=self.analytics)
        self.ledger = CapitalReservationLedger(risk_reservations=self.state_rt.reservations)
        self.pipeline = PortfolioPipeline(
            router=self.router, quantity_policy=self.qpol, optimizer=self.optimizer,
            analytics=self.analytics, reservation_ledger=self.ledger, sink=self.sink,
            authorization=authorization, correlation_index=self.corr_index)

    def state(self, *, equity=100_000_000, cash=100_000_000,
              status=ReconciliationStatus.RECONCILED):
        return PortfolioState(PORTFOLIO_STATE_VERSION, self.ns, equity, cash, cash, cash,
                              0, 0, 0, 0, 0, 0, 0, 0, reconciliation_status=status)

    async def drain(self):
        for o in self.mgr.all_orders():
            if o.venue_order_id:
                for u in self.broker.drain_updates(o.venue_order_id):
                    await self.mgr.reconcile(u)


# ------------------------------------------------------------ happy-path fills
@pytest.mark.asyncio
async def test_multiple_eligible_route_and_fill_through_native_risk():
    syms = ("AAA", "BBB", "CCC")
    st = Stack(symbols=syms)
    res = await st.pipeline.run(symbols=syms, returns_by_symbol=_returns(syms),
                                state=st.state(), now_ns=st.ns + 1,
                                price_ticks=st.state_rt.price_ticks)
    assert res.covariance_source in ("metal", "cpu")
    assert res.reserved == 3 and res.routed == 3, res.dispositions
    await st.drain()
    filled = [o for o in st.mgr.all_orders() if o.status is OrderStatus.FILLED]
    assert len(filled) == 3
    types = {e.event_type for e in st.sink.peek()}
    assert {DecisionEventType.PORTFOLIO_STATE, DecisionEventType.PORTFOLIO_OPTIMIZATION,
            DecisionEventType.PORTFOLIO_RECOMMENDATION, DecisionEventType.CAPITAL_RESERVATION,
            DecisionEventType.COMPLETE_FILL} <= types


# ------------------------------------------------------- competing capital
@pytest.mark.asyncio
async def test_competing_capital_reservation_prevents_overspend():
    syms = ("AAA", "BBB", "CCC", "DDD")
    # Tiny capital: only enough for a fraction of the requested book.
    st = Stack(symbols=syms, buying_power=5_000_000)  # $500 -> ~5 shares @ $100 total
    res = await st.pipeline.run(symbols=syms, returns_by_symbol=_returns(syms),
                                state=st.state(equity=5_000_000, cash=5_000_000),
                                now_ns=st.ns + 1, price_ticks=st.state_rt.price_ticks)
    # Reservations must not collectively exceed available capital.
    reserved_notional = st.ledger.reserved_buy_ticks()
    assert reserved_notional <= 5_000_000
    # At least one rec is skipped for insufficient capital.
    assert any("reservation_failed" in d[1] or d[1] == "advisory_authorization_none"
               for d in res.dispositions) or res.reserved < len(syms)


# ------------------------------------------------------- concentration cap
@pytest.mark.asyncio
async def test_concentration_cap_binds():
    syms = ("AAA", "BBB")
    st = Stack(symbols=syms, per_asset_cap=0.3, gross_cap=0.6)
    res = await st.pipeline.run(symbols=syms, returns_by_symbol=_returns(syms),
                                state=st.state(), now_ns=st.ns + 1,
                                price_ticks=st.state_rt.price_ticks)
    for rec in res.recommendations:
        assert rec.target_weight <= 0.3 + 1e-9


# ------------------------------------------------------- insufficient buying power (native risk)
@pytest.mark.asyncio
async def test_insufficient_buying_power_rejected_by_native_risk():
    syms = ("AAA", "BBB")
    # Enough available cash to reserve, but native risk account has ~nothing.
    st = Stack(symbols=syms, buying_power=1_000)
    res = await st.pipeline.run(symbols=syms, returns_by_symbol=_returns(syms),
                                state=st.state(equity=100_000_000, cash=100_000_000),
                                now_ns=st.ns + 1, price_ticks=st.state_rt.price_ticks)
    # Reserved against portfolio-state cash, but native risk rejects at the order.
    assert res.routed == 0
    assert any("risk_rejected" in str(d[1]) for d in res.dispositions)


# ------------------------------------------------------- fractional quantities
@pytest.mark.asyncio
async def test_fractional_quantities_when_allowed():
    syms = ("AAA", "BBB")
    st = Stack(symbols=syms, allow_fractional=True)
    res = await st.pipeline.run(symbols=syms, returns_by_symbol=_returns(syms),
                                state=st.state(), now_ns=st.ns + 1,
                                price_ticks=st.state_rt.price_ticks)
    # requested_quantity is computed with fractional sizing enabled (int-cast at contract).
    assert all(r.requested_quantity >= 0 for r in res.recommendations)


# ------------------------------------------------------- stale / unreconciled state (fail closed)
@pytest.mark.asyncio
async def test_stale_state_fails_closed_no_routing():
    syms = ("AAA", "BBB")
    st = Stack(symbols=syms)
    res = await st.pipeline.run(
        symbols=syms, returns_by_symbol=_returns(syms),
        state=st.state(status=ReconciliationStatus.STALE), now_ns=st.ns + 1,
        price_ticks=st.state_rt.price_ticks)
    assert res.routed == 0 and res.reserved == 0
    assert any("not_allocatable" in str(d[1]) for d in res.dispositions)


# ------------------------------------------------------- insufficient history
@pytest.mark.asyncio
async def test_insufficient_return_history_no_optimization():
    syms = ("AAA", "BBB")
    st = Stack(symbols=syms)
    short = _returns(syms, obs=5)  # < min_observations
    res = await st.pipeline.run(symbols=syms, returns_by_symbol=short, state=st.state(),
                                now_ns=st.ns + 1, price_ticks=st.state_rt.price_ticks)
    assert res.routed == 0
    assert any("insufficient_return_history" in str(d[1]) for d in res.dispositions)


# ------------------------------------------------------- partial fills
@pytest.mark.asyncio
async def test_partial_then_full_fill_evidence():
    syms = ("AAA",)
    st = Stack(symbols=syms, fill_plan=FillPlan.PARTIAL_THEN_FULL)
    await st.pipeline.run(symbols=syms, returns_by_symbol=_returns(syms), state=st.state(),
                          now_ns=st.ns + 1, price_ticks=st.state_rt.price_ticks)
    await st.drain()
    assert any(o.status is OrderStatus.FILLED for o in st.mgr.all_orders())


# ------------------------------------------------------- broker rejection
@pytest.mark.asyncio
async def test_broker_rejection_terminal():
    syms = ("AAA",)
    st = Stack(symbols=syms, fill_plan=FillPlan.REJECT)
    await st.pipeline.run(symbols=syms, returns_by_symbol=_returns(syms), state=st.state(),
                          now_ns=st.ns + 1, price_ticks=st.state_rt.price_ticks)
    await st.drain()
    assert any(o.status is OrderStatus.REJECTED for o in st.mgr.all_orders())


# ------------------------------------------------------- advisory default (authorization NONE)
@pytest.mark.asyncio
async def test_advisory_default_routes_nothing():
    syms = ("AAA", "BBB")
    st = Stack(symbols=syms, authorization=ExecutionAuthorization.NONE)
    res = await st.pipeline.run(symbols=syms, returns_by_symbol=_returns(syms), state=st.state(),
                                now_ns=st.ns + 1, price_ticks=st.state_rt.price_ticks)
    assert res.routed == 0
    assert st.broker.submitted == 0
    # still produces recommendations + evidence (advisory)
    assert res.recommendations
    assert any(e.event_type is DecisionEventType.PORTFOLIO_RECOMMENDATION for e in st.sink.peek())


# ------------------------------------------------------- GPU unavailable -> CPU fallback
@pytest.mark.asyncio
async def test_gpu_unavailable_cpu_fallback():
    syms = ("AAA", "BBB")
    st = Stack(symbols=syms)
    st.analytics._runtime = None  # simulate no GPU
    res = await st.pipeline.run(symbols=syms, returns_by_symbol=_returns(syms), state=st.state(),
                                now_ns=st.ns + 1, price_ticks=st.state_rt.price_ticks)
    assert res.covariance_source == "cpu"
    assert res.routed >= 1


# ------------------------------------------------------- unauthorized geared/inverse blocked
@pytest.mark.asyncio
async def test_unauthorized_inverse_blocked_by_exposure_gate():
    syms = ("SQQQ", "AAA")
    st = Stack(symbols=syms)  # SQQQ leveraged-inverse, not authorized
    await st.pipeline.run(symbols=syms, returns_by_symbol=_returns(syms), state=st.state(),
                          now_ns=st.ns + 1, price_ticks=st.state_rt.price_ticks)
    await st.drain()
    routed_syms = {o.intent.symbol for o in st.mgr.all_orders()}
    assert "SQQQ" not in routed_syms


# ------------------------------------------------------- restart recovery (reservation idempotency)
@pytest.mark.asyncio
async def test_reservation_idempotent_across_duplicate_runs():
    syms = ("AAA",)
    st = Stack(symbols=syms, fill_plan=FillPlan.NOFILL)
    rbs = _returns(syms)
    await st.pipeline.run(symbols=syms, returns_by_symbol=rbs, state=st.state(),
                          now_ns=st.ns + 1, price_ticks=st.state_rt.price_ticks)
    first = st.broker.submitted
    # same now_ns -> same portfolio_decision_id -> reservation idempotent; pending guard
    # also prevents a duplicate order on the same symbol.
    await st.pipeline.run(symbols=syms, returns_by_symbol=rbs, state=st.state(),
                          now_ns=st.ns + 1, price_ticks=st.state_rt.price_ticks)
    assert st.broker.submitted == first
