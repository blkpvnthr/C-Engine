"""Mandatory protective-exit independence from the execution planner (Milestone 4, §15).

The mandatory -0.5% exit fires through the OrderManager as a broker-compatible MARKET
order regardless of the execution planner / microstructure engine, and no
discretionary profit-protection action can precede it in the exit arbiter.
"""

from __future__ import annotations

import asyncio

from cengine.execution_planner import ExecutionPlanner, PlannedExecutionPolicy, Urgency
from cengine.microstructure import MicrostructureEngine, MicrostructureFeatures
from cengine.mock_broker import SimulatedBroker
from cengine.position_management import (
    AdaptivePositionManager,
    ExitArbiter,
    ExitPriority,
    PositionAction,
    PositionActionType,
    PositionView,
)
from cengine.strategy_maturity import ExecutionAuthorization
from cengine.strategy_state import StrategyStateBook
from order_manager import (
    OrderManager,
    OrderType,
    RiskDecision,
    Side,
)
from protective_exit import ProtectiveExitCoordinator


class _Position:
    def __init__(self, symbol, quantity, cost_ticks):
        self.symbol, self.quantity, self.cost_ticks = symbol, quantity, cost_ticks


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
        positions=_PositionBook([_Position("NVDA", 10, 1000)]),
        price_source=_PriceSource({"NVDA": loss_price}),
        reservations=_Reservations(), strategy_state=strategy_state,
        threshold=-0.005, interval_s=0.0)


def test_exit_fires_with_no_execution_planner():
    mgr = OrderManager(risk_engine=_ApproveRisk(), venue=SimulatedBroker(), on_audit=None)
    submitted = _run(_coordinator(mgr, StrategyStateBook()).scan_and_exit())
    assert submitted == ["NVDA"]
    order = mgr.all_orders()[0]
    assert order.intent.strategy_id == "protective_exit"
    assert order.intent.side is Side.SELL
    # mandatory exits are broker-compatible MARKET orders, NOT planner limit orders
    assert order.intent.order_type is OrderType.MARKET
    assert order.intent.limit_price_ticks is None


def test_exit_fires_despite_planner_and_microstructure_existing():
    # The planner + microstructure engine exist but are NOT part of the exit path.
    planner = ExecutionPlanner()
    micro = MicrostructureEngine()
    _ = PlannedExecutionPolicy()
    # even a (hypothetical) stale/crossed microstructure snapshot is irrelevant:
    stale = MicrostructureFeatures(symbol="NVDA", is_stale=True, crossed_or_locked=True,
                                   quality="CROSSED")
    _ = planner.plan(symbol="NVDA", side=Side.SELL, quantity=10, now_ns=1,
                     features=stale, urgency=Urgency.NORMAL)
    mgr = OrderManager(risk_engine=_ApproveRisk(), venue=SimulatedBroker(), on_audit=None)
    submitted = _run(_coordinator(mgr, StrategyStateBook()).scan_and_exit())
    assert submitted == ["NVDA"]
    assert mgr.all_orders()[0].intent.order_type is OrderType.MARKET
    # sanity: the engines were constructed and did not interfere
    assert micro.tracked_symbols() == []


def test_mandatory_exit_outranks_discretionary_profit_protection():
    # A discretionary profit-protection action is shadow-only by default and must not
    # precede the mandatory exit in the arbiter.
    view = PositionView(symbol="NVDA", quantity=10, unrealized_return=0.004,
                        max_favorable_excursion=0.03, atr_pct=0.005,
                        momentum_accel=-0.001, buyer_exhaustion=0.7)
    discretionary = AdaptivePositionManager(
        authorization=ExecutionAuthorization.NONE).evaluate(view)
    assert discretionary and all(a.shadow_only for a in discretionary)

    mandatory = PositionAction(
        symbol="NVDA", priority=ExitPriority.MANDATORY_PROTECTIVE,
        action_type=PositionActionType.EXIT_FULL, reason="mandatory_-0.5pct",
        shadow_only=False)
    chosen = ExitArbiter().arbitrate(discretionary + [mandatory])
    assert chosen.priority is ExitPriority.MANDATORY_PROTECTIVE
    assert chosen.reason == "mandatory_-0.5pct"


def test_discretionary_shadow_does_not_outrank_mandatory_even_if_listed_first():
    mandatory = PositionAction("NVDA", ExitPriority.MANDATORY_PROTECTIVE,
                               PositionActionType.EXIT_FULL, "mandatory", shadow_only=False)
    discretionary = PositionAction("NVDA", ExitPriority.PROFIT_PROTECTION,
                                   PositionActionType.EXIT_FULL, "trail", shadow_only=True)
    assert ExitArbiter().arbitrate([discretionary, mandatory]).reason == "mandatory"
