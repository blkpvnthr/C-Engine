"""Tests for the engine-side protective-exit coordinator.

The coordinator detects positions whose unrealised return is at/below the
mandatory loss-removal threshold (−0.5%) and submits a close through the
engine's single OrderManager path. It is independent of the scanner and
strategies, guards against duplicate in-flight closes via the single-pending
machinery, and never raises into the engine loop.
"""

from __future__ import annotations

import asyncio

import pytest

from order_manager import OrderType, Side, TimeInForce
from protective_exit import ProtectiveExitCoordinator


# --------------------------------------------------------------------- fakes
class FakePosition:
    def __init__(self, symbol, quantity, cost_ticks):
        self.symbol = symbol
        self.quantity = quantity
        self.cost_ticks = cost_ticks


class FakePositionBook:
    def __init__(self, positions):
        self._positions = list(positions)

    def positions(self):
        return list(self._positions)

    def quantity(self, symbol, strategy_id=None):
        for p in self._positions:
            if p.symbol == symbol:
                return p.quantity
        return 0


class FakePriceSource:
    def __init__(self, prices):
        self._prices = dict(prices)

    def price_ticks(self, symbol):
        if symbol not in self._prices:
            raise RuntimeError(f"no quote for {symbol}")
        return self._prices[symbol]


class FakeReservations:
    """projected_position returns net qty incl. pending; default = held qty."""
    def __init__(self, overrides=None):
        self._overrides = overrides or {}

    def projected_position(self, symbol, held):
        return self._overrides.get(symbol, held)


class FakeStrategyState:
    def __init__(self, blocked=()):
        self._blocked = set(blocked)

    def can_submit(self, strategy_id, symbol):
        return symbol not in self._blocked


class FakeOrderManager:
    def __init__(self, raise_for=()):
        self.submitted = []
        self._raise_for = set(raise_for)

    async def submit(self, intent):
        if intent.symbol in self._raise_for:
            raise RuntimeError("risk rejected")
        self.submitted.append(intent)
        return object()


def _coord(*, positions, prices, reservations=None, strategy_state=None,
           order_manager=None, threshold=-0.005):
    return ProtectiveExitCoordinator(
        order_manager=order_manager or FakeOrderManager(),
        positions=FakePositionBook(positions),
        price_source=FakePriceSource(prices),
        reservations=reservations or FakeReservations(),
        strategy_state=strategy_state or FakeStrategyState(),
        threshold=threshold,
        interval_s=0.0,
    )


def _run(coro):
    return asyncio.run(coro)


# ----------------------------------------------------------------- unit: math
def test_unrealized_return_long():
    # entry 100 ticks, price 99 ticks -> -1%
    assert ProtectiveExitCoordinator.unrealized_return(
        FakePosition("X", 10, 1000), 99) == pytest.approx(-0.01)


def test_unrealized_return_short():
    # short 10 @ avg 100 (cost_ticks -1000), price 101 -> short loses 1%
    r = ProtectiveExitCoordinator.unrealized_return(FakePosition("X", -10, -1000), 101)
    assert r == pytest.approx(-0.01)


# ----------------------------------------------------------- behaviour: breach
def test_long_breach_submits_market_sell():
    om = FakeOrderManager()
    c = _coord(positions=[FakePosition("NVDA", 10, 1000)],  # avg 100
               prices={"NVDA": 99},                         # -1% <= -0.5%
               order_manager=om)
    submitted = _run(c.scan_and_exit())
    assert submitted == ["NVDA"]
    (intent,) = om.submitted
    assert intent.symbol == "NVDA"
    assert intent.side == Side.SELL
    assert intent.quantity == 10
    assert intent.order_type == OrderType.MARKET
    assert intent.time_in_force == TimeInForce.DAY
    assert intent.strategy_id == "protective_exit"
    assert intent.created_ns > 0
    assert intent.correlation_id.startswith("protective:NVDA:")


def test_exactly_at_threshold_triggers():
    om = FakeOrderManager()
    c = _coord(positions=[FakePosition("AMD", 10, 1000)],  # avg 100
               prices={"AMD": 995},  # price_ticks=995? no -> use fractional below
               order_manager=om)
    # Use avg 1000 ticks, price 995 ticks => -0.5% exactly
    c._positions = FakePositionBook([FakePosition("AMD", 10, 10000)])  # avg 1000
    c._prices = FakePriceSource({"AMD": 995})
    submitted = _run(c.scan_and_exit())
    assert submitted == ["AMD"]


def test_no_breach_no_submit():
    om = FakeOrderManager()
    c = _coord(positions=[FakePosition("AMD", 10, 1000)],  # avg 100
               prices={"AMD": 998},  # -0.2% > -0.5%
               order_manager=om)
    assert _run(c.scan_and_exit()) == []
    assert om.submitted == []


def test_short_breach_submits_market_buy():
    om = FakeOrderManager()
    c = _coord(positions=[FakePosition("SQQQ", -10, -1000)],  # short avg 100
               prices={"SQQQ": 101},  # short down 1%
               order_manager=om)
    assert _run(c.scan_and_exit()) == ["SQQQ"]
    assert om.submitted[0].side == Side.BUY
    assert om.submitted[0].quantity == 10


# ----------------------------------------------------------- behaviour: guards
def test_zero_quantity_skipped():
    om = FakeOrderManager()
    c = _coord(positions=[FakePosition("X", 0, 0)], prices={"X": 50}, order_manager=om)
    assert _run(c.scan_and_exit()) == []


def test_missing_price_skips_symbol_but_continues():
    om = FakeOrderManager()
    c = _coord(
        positions=[FakePosition("NOPX", 10, 1000), FakePosition("NVDA", 10, 1000)],
        prices={"NVDA": 99},  # NOPX has no quote
        order_manager=om,
    )
    submitted = _run(c.scan_and_exit())
    assert submitted == ["NVDA"]  # NOPX skipped, NVDA still handled


def test_pending_close_not_duplicated():
    om = FakeOrderManager()
    c = _coord(positions=[FakePosition("NVDA", 10, 1000)], prices={"NVDA": 99},
               strategy_state=FakeStrategyState(blocked={"NVDA"}), order_manager=om)
    assert _run(c.scan_and_exit()) == []
    assert om.submitted == []


def test_fully_reserved_exit_skipped():
    om = FakeOrderManager()
    # projected position already 0 (an exit is fully in flight) -> nothing to close
    c = _coord(positions=[FakePosition("NVDA", 10, 1000)], prices={"NVDA": 99},
               reservations=FakeReservations({"NVDA": 0}), order_manager=om)
    assert _run(c.scan_and_exit()) == []


def test_close_qty_uses_projected_position():
    om = FakeOrderManager()
    # hold 10 but only 6 remain after a partial pending exit
    c = _coord(positions=[FakePosition("NVDA", 10, 1000)], prices={"NVDA": 99},
               reservations=FakeReservations({"NVDA": 6}), order_manager=om)
    _run(c.scan_and_exit())
    assert om.submitted[0].quantity == 6


def test_submit_error_isolated_per_symbol():
    om = FakeOrderManager(raise_for={"BAD"})
    c = _coord(
        positions=[FakePosition("BAD", 10, 1000), FakePosition("NVDA", 10, 1000)],
        prices={"BAD": 99, "NVDA": 99},
        order_manager=om,
    )
    # BAD raises on submit; NVDA must still be submitted; scan must not raise.
    submitted = _run(c.scan_and_exit())
    assert submitted == ["NVDA"]
    assert [i.symbol for i in om.submitted] == ["NVDA"]


def test_market_closed_predicate_blocks_submit():
    om = FakeOrderManager()
    c = _coord(positions=[FakePosition("NVDA", 10, 1000)], prices={"NVDA": 99},
               order_manager=om)
    c.market_open_fn = lambda: False
    assert _run(c.scan_and_exit()) == []
    assert om.submitted == []


# --------------------------------------------- real OrderManager path (contract)
def test_coordinator_intent_flows_through_real_order_manager():
    """Prove a coordinator-built OrderIntent survives the REAL OrderManager.submit
    (intent validation + risk gate + venue ack) -- not just a fake."""
    from order_manager import ExecutionAck, OrderManager, RiskDecision

    class _RiskEngine:
        def evaluate(self, intent):
            return RiskDecision(approved=True, decision_id="d1", policy_version="v1",
                                decided_ns=1)

    class _Venue:
        def __init__(self):
            self.submitted = []

        def submit(self, intent):
            self.submitted.append(intent)
            return ExecutionAck(client_order_id=intent.client_order_id,
                                venue_order_id="venue-1", accepted_ns=1)

        def cancel(self, *, client_order_id, venue_order_id):
            return None

    venue = _Venue()
    om = OrderManager(risk_engine=_RiskEngine(), venue=venue)
    c = ProtectiveExitCoordinator(
        order_manager=om,
        positions=FakePositionBook([FakePosition("NVDA", 10, 1000)]),  # avg 100
        price_source=FakePriceSource({"NVDA": 99}),                    # -1%
        reservations=FakeReservations(),
        strategy_state=FakeStrategyState(),
    )
    submitted = _run(c.scan_and_exit())
    assert submitted == ["NVDA"]
    # The real venue received exactly one SELL for the full held quantity.
    assert len(venue.submitted) == 1
    assert venue.submitted[0].symbol == "NVDA"
    assert venue.submitted[0].side == Side.SELL
    assert venue.submitted[0].quantity == 10
    open_orders = om.open_orders()
    assert len(open_orders) == 1
