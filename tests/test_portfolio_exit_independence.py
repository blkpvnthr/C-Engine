"""Mandatory protective-exit independence from the portfolio optimizer (§12).

The mandatory -0.5% exit fires through the OrderManager regardless of optimizer /
covariance / GPU availability, and an in-flight portfolio order never blocks it
(separate single-pending key).
"""

from __future__ import annotations

import asyncio

from cengine.mock_broker import SimulatedBroker
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
from strategies import StrategyName


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


def test_exit_fires_with_no_optimizer_present():
    state = StrategyStateBook()
    mgr = OrderManager(risk_engine=_ApproveRisk(), venue=SimulatedBroker(), on_audit=None)
    coord = _coordinator(mgr, state)
    submitted = _run(coord.scan_and_exit())
    assert submitted == ["NVDA"]
    order = mgr.all_orders()[0]
    assert order.intent.strategy_id == "protective_exit"
    assert order.intent.side is Side.SELL


def test_exit_fires_when_optimizer_pipeline_would_raise():
    # A portfolio pipeline that raises on run() simulates optimizer/GPU failure. The
    # mandatory exit path never calls the optimizer, so it must still fire.
    class _ExplodingPipeline:
        async def run(self, **kwargs):
            raise RuntimeError("optimizer/covariance unavailable")

    pipeline = _ExplodingPipeline()
    # Prove the pipeline truly fails, independently.
    import pytest
    with pytest.raises(RuntimeError):
        _run(pipeline.run())

    state = StrategyStateBook()
    mgr = OrderManager(risk_engine=_ApproveRisk(), venue=SimulatedBroker(), on_audit=None)
    coord = _coordinator(mgr, state)
    submitted = _run(coord.scan_and_exit())
    assert submitted == ["NVDA"]
    assert mgr.all_orders()[0].intent.side is Side.SELL


def test_inflight_portfolio_order_does_not_block_mandatory_exit():
    state = StrategyStateBook()
    portfolio_intent = OrderIntent(
        symbol="NVDA", side=Side.BUY, quantity=5, order_type=OrderType.MARKET,
        time_in_force=TimeInForce.DAY, strategy_id=StrategyName.PORTFOLIO.value,
        client_order_id="pf-1", created_ns=1_000)
    pending = ManagedOrder(intent=portfolio_intent, client_order_id="pf-1",
                           status=OrderStatus.SUBMITTED)
    state.on_order(pending)  # portfolio order in-flight on NVDA

    assert state.can_submit(StrategyName.PORTFOLIO.value, "NVDA") is False  # portfolio blocked
    assert state.can_submit("protective_exit", "NVDA") is True              # exit independent

    mgr = OrderManager(risk_engine=_ApproveRisk(), venue=SimulatedBroker(), on_audit=None)
    coord = _coordinator(mgr, state)
    submitted = _run(coord.scan_and_exit())
    assert submitted == ["NVDA"]
    assert mgr.all_orders()[0].intent.strategy_id == "protective_exit"


def test_exit_removes_full_position():
    state = StrategyStateBook()
    mgr = OrderManager(risk_engine=_ApproveRisk(), venue=SimulatedBroker(), on_audit=None)
    coord = _coordinator(mgr, state)
    submitted = _run(coord.scan_and_exit())
    assert submitted == ["NVDA"]
    order = mgr.all_orders()[0]
    assert order.intent.side is Side.SELL
    assert order.intent.quantity == 10
