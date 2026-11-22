"""End-to-end universe-to-execution acceptance (§37-40, §46).

Traces a representative instrument of EACH universe category through the genuinely
wired path:

    master membership -> exposure metadata/classification -> exposure
    authorization gate -> CandidateRouter -> real OrderManager -> mock venue -> fill

and verifies the negative paths (unauthorized geared/inverse blocked; exits never
gated). This proves a traceable functional path, not mere object construction.

Scope honesty: this exercises the wired authorization/execution path with a mock
broker venue (no live orders). It does NOT assert that the full live SIP discovery
loop subscribes/warms/ranks every master-universe symbol — that broader loop wiring
is tracked separately. Membership/metadata stages use the real Alpaca membership
cache when present (else those assertions skip).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from cengine.execution_policy import ConfiguredExecutionPolicy
from cengine.exposure_registry import ExposureRegistry, InstrumentType
from cengine.membership import MembershipRegistry
from cengine.strategy_state import StrategyStateBook
from order_manager import (
    ExecutionAck,
    ExecutionUpdate,
    OrderManager,
    OrderStatus,
    OrderType,
    RiskDecision,
    TimeInForce,
)
from strategies import SignalSide, StrategyName, TradeCandidate
from trade import CandidateRouter

_CACHE = Path(__file__).resolve().parents[1] / "data" / "membership_snapshot.json"

# category -> (symbol, expected instrument_type, expected signed leverage, auto_auth)
CATEGORIES = {
    "conventional_equity": ("AAPL", None, None, True),            # not in exposure registry
    "broad_market_etf": ("SPY", InstrumentType.BROAD_MARKET_ETF, 1.0, True),
    "sector_etf": ("XLK", InstrumentType.SECTOR_ETF, 1.0, True),
    "leveraged_bull_etf": ("TQQQ", InstrumentType.LEVERAGED_LONG_ETF, 3.0, False),
    "inverse_etf": ("SH", InstrumentType.INVERSE_ETF, -1.0, False),
    "leveraged_inverse_etf": ("SQQQ", InstrumentType.LEVERAGED_INVERSE_ETF, -3.0, False),
    "leveraged_inverse_sector_etf": ("SOXS", InstrumentType.LEVERAGED_INVERSE_ETF, -3.0, False),
}


class _Risk:
    def evaluate(self, intent):
        return RiskDecision(True, "d1", "ok", intent.created_ns,
                            approved_quantity=intent.quantity)


class _Venue:
    def __init__(self):
        self.submitted = []

    def submit(self, intent):
        self.submitted.append(intent.symbol)
        return ExecutionAck(intent.client_order_id, f"v-{len(self.submitted)}",
                            intent.created_ns + 1)

    def cancel(self, **kwargs):
        return None


class _EntryQty:
    def quantity_for(self, candidate):
        return 5


class _ExitQty:
    def quantity_to_close(self, candidate):
        return 3


def _router(registry: ExposureRegistry):
    venue = _Venue()
    manager = OrderManager(risk_engine=_Risk(), venue=venue)
    router = CandidateRouter(
        order_manager=manager, entry_quantity_policy=_EntryQty(),
        exit_quantity_provider=_ExitQty(),
        execution_policy=ConfiguredExecutionPolicy(OrderType.MARKET, TimeInForce.DAY),
        strategy_state=StrategyStateBook(),
        exposure_authorizer=registry.authorize_entry)
    return router, manager, venue


def _cand(symbol, side, ts=1):
    return TradeCandidate(StrategyName.TURTLE, symbol, side, ts, 100, "e2e")


@pytest.fixture(scope="module")
def membership():
    reg = MembershipRegistry()
    if not reg.load_cache(_CACHE):
        pytest.skip("no membership cache (run tools/build_membership.py)")
    return reg


# -------------------------------------------------- membership + metadata stage
@pytest.mark.parametrize("category", list(CATEGORIES))
def test_master_membership_and_metadata(category, membership):
    symbol, itype, lev, _auto = CATEGORIES[category]
    info = membership.get(symbol)
    assert info is not None, f"{symbol} absent from master universe"
    assert info.tradable, f"{symbol} not tradable on Alpaca"
    exp = ExposureRegistry().get(symbol)
    if itype is None:
        assert exp is None  # conventional equity: not in exposure registry
    else:
        assert exp is not None and exp.instrument_type == itype
        assert exp.target_daily_leverage == lev
        assert (exp.exposure_direction.value == "INVERSE") == (lev < 0)


# -------------------------------------------------- authorization + execution stage
@pytest.mark.parametrize("category", list(CATEGORIES))
@pytest.mark.asyncio
async def test_entry_authorization_path(category):
    symbol, _itype, _lev, auto = CATEGORIES[category]
    reg = ExposureRegistry()
    router, manager, venue = _router(reg)
    submitted, _ = await router.route((_cand(symbol, SignalSide.LONG),))
    if auto:
        assert submitted == 1 and venue.submitted == [symbol]
    else:
        # geared/inverse blocked by default (§10)
        assert submitted == 0 and venue.submitted == []
        # explicit authorization opens the full path to the mock venue
        assert reg.authorize(symbol, True)
        router2, manager2, venue2 = _router(reg)
        submitted2, _ = await router2.route((_cand(symbol, SignalSide.LONG),))
        assert submitted2 == 1 and venue2.submitted == [symbol]


@pytest.mark.asyncio
async def test_authorized_geared_reaches_fill_reconciliation():
    reg = ExposureRegistry()
    reg.authorize("SQQQ", True)
    router, manager, venue = _router(reg)
    submitted, _ = await router.route((_cand("SQQQ", SignalSide.LONG),))
    assert submitted == 1
    order = next(iter(manager._orders.values()))
    assert order.status is OrderStatus.SUBMITTED
    filled = await manager.reconcile(ExecutionUpdate(
        order.client_order_id, order.venue_order_id, OrderStatus.FILLED,
        20, 5, 5, 100, venue_sequence=1))
    assert filled.status is OrderStatus.FILLED  # full traceable path to a fill


@pytest.mark.asyncio
async def test_exit_of_unauthorized_inverse_never_gated():
    reg = ExposureRegistry()  # SOXS NOT authorized
    router, manager, venue = _router(reg)
    submitted, _ = await router.route((_cand("SOXS", SignalSide.EXIT_LONG),))
    assert submitted == 1 and venue.submitted == ["SOXS"]  # risk reduction proceeds


@pytest.mark.asyncio
async def test_opposing_exposure_detectable_for_portfolio_policy():
    # TQQQ + SQQQ held => economically opposing on NASDAQ100 (portfolio must see it).
    from cengine.exposure_registry import detect_opposing, seed_exposure_table
    t = seed_exposure_table()
    opp = detect_opposing({s: t[s] for s in ("TQQQ", "SQQQ")})
    assert opp and opp[0][2] == "NASDAQ100"
