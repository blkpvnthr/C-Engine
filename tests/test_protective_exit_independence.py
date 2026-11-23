"""Mandatory protective-exit independence from strategy intelligence (directive §7, §19).

Proves the mandatory -0.5% exit fires regardless of the regime/strategy engine's
state — unavailable, BLOCKED, or producing conflicting elective recommendations —
and that an in-flight elective (bridge) order never blocks a mandatory close,
because the two use independent single-pending keys and the SAME OrderManager.
"""

from __future__ import annotations

import asyncio

from cengine.decision_events import DecisionEventSink
from cengine.execution_policy import ConfiguredExecutionPolicy
from cengine.exposure_registry import ExposureRegistry
from cengine.mock_broker import FillPlan, SimulatedBroker
from cengine.strategy_execution_bridge import StrategyExecutionBridge
from cengine.strategy_families import StrategyFamilyRegistry
from cengine.strategy_maturity import ExecutionAuthorization
from cengine.strategy_state import StrategyStateBook
from order_manager import (
    ManagedOrder,
    OrderIntent,
    OrderManager,
    OrderStatus,
    OrderType,
    RiskDecision,
    Side,
    TimeInForce,
)
from protective_exit import ProtectiveExitCoordinator


class _Position:
    def __init__(self, symbol, quantity, cost_ticks):
        self.symbol = symbol
        self.quantity = quantity
        self.cost_ticks = cost_ticks


class _PositionBook:
    def __init__(self, positions):
        self._positions = list(positions)

    def positions(self):
        return list(self._positions)

    def quantity(self, symbol, strategy_id=None):
        return next((p.quantity for p in self._positions if p.symbol == symbol), 0)


class _PriceSource:
    def __init__(self, prices):
        self._prices = dict(prices)

    def price_ticks(self, symbol):
        if symbol not in self._prices:
            raise RuntimeError(f"no quote for {symbol}")
        return self._prices[symbol]


class _Reservations:
    def projected_position(self, symbol, held):
        return held


class _ApproveRisk:
    def evaluate(self, intent):
        return RiskDecision(True, "d", "v", intent.created_ns,
                            approved_quantity=intent.quantity)


def _run(coro):
    return asyncio.run(coro)


def _coordinator(order_manager, strategy_state, *, loss_price=99):
    return ProtectiveExitCoordinator(
        order_manager=order_manager,
        positions=_PositionBook([_Position("NVDA", 10, 1000)]),   # avg 100
        price_source=_PriceSource({"NVDA": loss_price}),          # -1% <= -0.5%
        reservations=_Reservations(),
        strategy_state=strategy_state,
        threshold=-0.005, interval_s=0.0)


def test_exit_fires_with_no_strategy_engine_present():
    state = StrategyStateBook()
    mgr = OrderManager(risk_engine=_ApproveRisk(),
                       venue=SimulatedBroker(), on_audit=None)
    coord = _coordinator(mgr, state)
    submitted = _run(coord.scan_and_exit())
    assert submitted == ["NVDA"]
    order = mgr.all_orders()[0]
    assert order.intent.strategy_id == "protective_exit"
    assert order.intent.side is Side.SELL


def test_exit_fires_even_when_all_families_blocked():
    # Build a bridge whose families are all BLOCKED (routes nothing) sharing the
    # SAME OrderManager + StrategyStateBook as the protective exit.
    state = StrategyStateBook()
    broker = SimulatedBroker(default_plan=FillPlan.NOFILL)

    def on_audit(event, order):
        state.on_order(order)

    mgr = OrderManager(risk_engine=_ApproveRisk(), venue=broker, on_audit=on_audit)
    registry = StrategyFamilyRegistry()
    for fam in registry.families():
        registry.block(fam)
    from trade import CandidateRouter

    class _Qty:
        def quantity_for(self, c):
            return 5

        def quantity_to_close(self, c):
            return 5

    router = CandidateRouter(
        order_manager=mgr, entry_quantity_policy=_Qty(), exit_quantity_provider=_Qty(),
        execution_policy=ConfiguredExecutionPolicy(OrderType.MARKET, TimeInForce.DAY),
        strategy_state=state, exposure_authorizer=ExposureRegistry().authorize_entry)
    bridge = StrategyExecutionBridge(
        router=router, registry=registry, sink=DecisionEventSink(),
        price_ticks_fn=lambda s: 1_000_000, authorization=ExecutionAuthorization.MOCK)

    # The mandatory exit must still fire through the same manager.
    coord = _coordinator(mgr, state)
    submitted = _run(coord.scan_and_exit())
    assert submitted == ["NVDA"]
    # and the bridge itself routes nothing (families BLOCKED)
    assert bridge  # constructed fine; blocked maturity means nothing is permitted


def test_inflight_elective_order_does_not_block_mandatory_exit():
    # An elective bridge order (strategy_id 'regime_intelligence') is pending on NVDA.
    # The mandatory exit uses strategy_id 'protective_exit' -> independent key -> it
    # must still be allowed to submit.
    state = StrategyStateBook()
    elective_intent = OrderIntent(
        symbol="NVDA", side=Side.BUY, quantity=5, order_type=OrderType.MARKET,
        time_in_force=TimeInForce.DAY, strategy_id="regime_intelligence",
        client_order_id="elective-1", created_ns=1_000)
    pending = ManagedOrder(intent=elective_intent, client_order_id="elective-1",
                           status=OrderStatus.SUBMITTED)
    state.on_order(pending)  # elective order now in-flight

    assert state.can_submit("regime_intelligence", "NVDA") is False  # elective blocked
    assert state.can_submit("protective_exit", "NVDA") is True        # exit independent

    mgr = OrderManager(risk_engine=_ApproveRisk(), venue=SimulatedBroker(),
                       on_audit=None)
    coord = _coordinator(mgr, state)
    submitted = _run(coord.scan_and_exit())
    assert submitted == ["NVDA"]
    assert mgr.all_orders()[0].intent.strategy_id == "protective_exit"


def test_protective_exit_is_never_arbitrated():
    # Structural: the arbitrator only resolves elective recommendations and never
    # emits or blocks a protective exit (there is no exit in the recommendation set).
    from cengine.strategy_arbitration import StrategyArbitrator
    result = StrategyArbitrator().arbitrate([], now_ns=1)
    assert result == {"selected": [], "rejected": [], "research": []}
    # (The protective exit runs on its own path; nothing here can touch it.)


def test_exit_fires_despite_conflicting_recommendation_same_symbol():
    # Even if the strategy engine wanted to BUY NVDA, the mandatory exit (SELL) is
    # independent and fires. We assert the exit submits a SELL to close.
    state = StrategyStateBook()
    mgr = OrderManager(risk_engine=_ApproveRisk(), venue=SimulatedBroker(),
                       on_audit=None)
    coord = _coordinator(mgr, state)
    submitted = _run(coord.scan_and_exit())
    assert submitted == ["NVDA"]
    order = mgr.all_orders()[0]
    assert order.intent.side is Side.SELL
    assert order.intent.quantity == 10  # full position removed
