"""CandidateRouter exposure-authorization gate (§8/§10): geared/inverse products
are blocked from elective ENTRY unless authorized; exits are never gated."""

import pytest

from cengine.execution_policy import ConfiguredExecutionPolicy
from cengine.exposure_registry import ExposureRegistry
from cengine.strategy_state import StrategyStateBook
from order_manager import OrderType, TimeInForce
from strategies import SignalSide, StrategyName, TradeCandidate
from trade import CandidateRouter


class _Manager:
    def __init__(self):
        self.intents = []

    async def submit(self, intent):
        self.intents.append(intent)


class _EntryQty:
    def quantity_for(self, candidate):
        return 5


class _ExitQty:
    def quantity_to_close(self, candidate):
        return 3


def _cand(symbol, side):
    return TradeCandidate(StrategyName.TURTLE, symbol, side, 1, 100, "test")


def _router(manager, registry):
    return CandidateRouter(
        order_manager=manager,
        entry_quantity_policy=_EntryQty(),
        exit_quantity_provider=_ExitQty(),
        execution_policy=ConfiguredExecutionPolicy(OrderType.MARKET, TimeInForce.DAY),
        strategy_state=StrategyStateBook(),
        exposure_authorizer=registry.authorize_entry,
    )


@pytest.mark.asyncio
async def test_geared_inverse_entry_blocked_by_default():
    reg = ExposureRegistry()
    manager = _Manager()
    router = _router(manager, reg)
    # SQQQ is a classified leveraged-inverse ETF, not execution-authorized -> block.
    submitted, conflicted = await router.route((_cand("SQQQ", SignalSide.LONG),))
    assert submitted == 0 and not conflicted
    assert manager.intents == []


@pytest.mark.asyncio
async def test_conventional_equity_entry_allowed():
    reg = ExposureRegistry()
    manager = _Manager()
    router = _router(manager, reg)
    # AAPL is not in the exposure registry -> treated as conventional equity -> allowed.
    submitted, _ = await router.route((_cand("AAPL", SignalSide.LONG),))
    assert submitted == 1
    assert manager.intents[0].symbol == "AAPL"


@pytest.mark.asyncio
async def test_exit_of_geared_product_never_gated():
    reg = ExposureRegistry()
    manager = _Manager()
    router = _router(manager, reg)
    # An EXIT of SQQQ must proceed even though SQQQ is not execution-authorized.
    submitted, _ = await router.route((_cand("SQQQ", SignalSide.EXIT_LONG),))
    assert submitted == 1
    assert manager.intents[0].side.value == "sell"


@pytest.mark.asyncio
async def test_explicit_authorization_allows_geared_entry():
    reg = ExposureRegistry()
    assert reg.authorize("SQQQ", True) is True
    manager = _Manager()
    router = _router(manager, reg)
    submitted, _ = await router.route((_cand("SQQQ", SignalSide.LONG),))
    assert submitted == 1
    assert manager.intents[0].symbol == "SQQQ"


@pytest.mark.asyncio
async def test_unlevered_etf_entry_allowed():
    reg = ExposureRegistry()
    manager = _Manager()
    router = _router(manager, reg)
    # SPY is a 1x broad-market ETF -> auto-authorized.
    submitted, _ = await router.route((_cand("SPY", SignalSide.LONG),))
    assert submitted == 1
