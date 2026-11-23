"""Strategy execution bridge (directive §3, §4, §14-17).

Verifies the bridge routes ONLY maturity-authorized, condition-passing, non-expired
LONG recommendations through the EXISTING CandidateRouter, emits traceable decision
evidence, and defaults to routing nothing.
"""

from __future__ import annotations

import pytest

from cengine.decision_events import DecisionEventSink, DecisionEventType
from cengine.execution_policy import ConfiguredExecutionPolicy
from cengine.exposure_registry import ExposureRegistry
from cengine.mock_broker import SimulatedBroker
from cengine.strategy_arbitration import StrategyRecommendation
from cengine.strategy_execution_bridge import (
    StrategyExecutionBridge,
    make_decision_audit_handler,
)
from cengine.strategy_families import (
    ExposureDirection,
    StrategyFamily,
    StrategyFamilyRegistry,
)
from cengine.strategy_maturity import ExecutionAuthorization
from cengine.strategy_state import StrategyStateBook
from order_manager import (
    OrderManager,
    OrderType,
    RiskDecision,
    TimeInForce,
)
from trade import CandidateRouter

PRICE = 1_000_000


class _ApproveRisk:
    def evaluate(self, intent):
        return RiskDecision(True, "d", "v", intent.created_ns,
                            approved_quantity=intent.quantity)


class _RejectRisk:
    def evaluate(self, intent):
        return RiskDecision(False, "d", "v", intent.created_ns, "insufficient_buying_power")


class _Qty:
    def quantity_for(self, candidate):
        return 5

    def quantity_to_close(self, candidate):
        return 5


def _rec(symbol="AAPL", family=StrategyFamily.MOMENTUM_CONTINUATION, *,
         conditions_passed=True, now_ns=1_000, ttl=10**12,
         direction=ExposureDirection.LONG, rank=0.5):
    rec = StrategyRecommendation(
        decision_id=f"{symbol}:{family.value}:{now_ns}", symbol_id=symbol,
        strategy_id="mapped", strategy_family=family, strategy_version="v1",
        regime_version="regime-1.0.0", feature_version="features-1.0.0",
        observation_timestamp=now_ns, expiration_timestamp=now_ns + ttl,
        exposure_direction=direction, rank_score=rank)
    rec.conditions_passed = conditions_passed
    return rec


def _build(*, risk=None, authorization=ExecutionAuthorization.MOCK,
          registry=None, exposure=None, broker=None, sink=None,
          prices=None, promote=True):
    registry = registry or StrategyFamilyRegistry()
    if promote:
        for fam in registry.families():
            registry.mark_mock_execution_eligible(fam, test_authorization=True)
    broker = broker or SimulatedBroker(default_fill_price_ticks=PRICE)
    sink = sink or DecisionEventSink()
    state = StrategyStateBook()
    mgr = OrderManager(risk_engine=risk or _ApproveRisk(), venue=broker,
                       on_audit=None)
    exposure = exposure if exposure is not None else ExposureRegistry()
    router = CandidateRouter(
        order_manager=mgr, entry_quantity_policy=_Qty(),
        exit_quantity_provider=_Qty(),
        execution_policy=ConfiguredExecutionPolicy(OrderType.MARKET, TimeInForce.DAY),
        strategy_state=state,
        exposure_authorizer=exposure.authorize_entry)
    prices = prices or {}
    bridge = StrategyExecutionBridge(
        router=router, registry=registry, sink=sink,
        price_ticks_fn=lambda s: prices.get(s, PRICE),
        authorization=authorization)
    return bridge, mgr, broker, sink, state


@pytest.mark.asyncio
async def test_default_authorization_routes_nothing():
    bridge, mgr, broker, sink, _ = _build(authorization=ExecutionAuthorization.NONE)
    out = await bridge.route_recommendations([_rec()], now_ns=2_000)
    assert out.submitted == 0
    assert broker.submitted == 0
    assert out.by_outcome("skipped")[0].reason == "authorization_none"


@pytest.mark.asyncio
async def test_research_maturity_not_routed_even_under_mock():
    # promote=False -> families stay RESEARCH maturity
    bridge, mgr, broker, sink, _ = _build(promote=False)
    out = await bridge.route_recommendations([_rec()], now_ns=2_000)
    assert out.submitted == 0
    assert "maturity_research_not_permitted" in out.by_outcome("skipped")[0].reason


@pytest.mark.asyncio
async def test_mock_authorized_family_routes_and_fills():
    bridge, mgr, broker, sink, _ = _build()
    out = await bridge.route_recommendations([_rec()], now_ns=2_000)
    assert out.submitted == 1
    assert broker.submitted == 1
    # ORDER_INTENT evidence with recommendation linkage emitted
    rows = sink.peek()
    intents = [r for r in rows if r.event_type is DecisionEventType.ORDER_INTENT]
    assert len(intents) == 1
    assert intents[0].decision_id == "AAPL:momentum_continuation:1000"


@pytest.mark.asyncio
async def test_conditions_not_passed_is_skipped():
    bridge, *_ = _build()
    out = await bridge.route_recommendations([_rec(conditions_passed=False)], now_ns=2_000)
    assert out.submitted == 0
    assert out.by_outcome("skipped")[0].reason == "conditions_not_passed"


@pytest.mark.asyncio
async def test_expired_recommendation_skipped():
    bridge, *_ = _build()
    out = await bridge.route_recommendations([_rec(now_ns=1_000, ttl=10)],
                                             now_ns=10_000)
    assert out.submitted == 0
    assert out.by_outcome("skipped")[0].reason == "expired"


@pytest.mark.asyncio
async def test_unauthorized_inverse_etf_blocked_by_exposure_gate():
    # SQQQ is leveraged-inverse; not authorized by default -> exposure gate blocks.
    bridge, mgr, broker, sink, _ = _build()
    out = await bridge.route_recommendations([_rec(symbol="SQQQ")], now_ns=2_000)
    assert out.submitted == 0
    assert broker.submitted == 0
    assert out.by_outcome("exposure_blocked")


@pytest.mark.asyncio
async def test_authorized_inverse_etf_routes():
    exposure = ExposureRegistry()
    exposure.authorize("SQQQ", True)
    bridge, mgr, broker, sink, _ = _build(exposure=exposure)
    out = await bridge.route_recommendations([_rec(symbol="SQQQ")], now_ns=2_000)
    assert out.submitted == 1


@pytest.mark.asyncio
async def test_risk_rejection_surfaced_as_disposition():
    bridge, *_ = _build(risk=_RejectRisk())
    out = await bridge.route_recommendations([_rec()], now_ns=2_000)
    assert out.submitted == 0
    assert out.by_outcome("risk_rejected")


@pytest.mark.asyncio
async def test_non_long_direction_skipped():
    bridge, *_ = _build()
    out = await bridge.route_recommendations(
        [_rec(direction=ExposureDirection.SHORT)], now_ns=2_000)
    assert out.submitted == 0
    assert "non_long_direction" in out.by_outcome("skipped")[0].reason


@pytest.mark.asyncio
async def test_audit_handler_emits_ack_and_fill_evidence():
    registry = StrategyFamilyRegistry()
    for fam in registry.families():
        registry.mark_mock_execution_eligible(fam, test_authorization=True)
    broker = SimulatedBroker(default_fill_price_ticks=PRICE)
    sink = DecisionEventSink()
    corr_index: dict = {}
    state = StrategyStateBook()
    audit = make_decision_audit_handler(sink, corr_index)

    def on_audit(event, order):
        state.on_order(order)
        audit(event, order)

    mgr = OrderManager(risk_engine=_ApproveRisk(), venue=broker, on_audit=on_audit)
    router = CandidateRouter(
        order_manager=mgr, entry_quantity_policy=_Qty(), exit_quantity_provider=_Qty(),
        execution_policy=ConfiguredExecutionPolicy(OrderType.MARKET, TimeInForce.DAY),
        strategy_state=state, exposure_authorizer=ExposureRegistry().authorize_entry)
    bridge = StrategyExecutionBridge(
        router=router, registry=registry, sink=sink,
        price_ticks_fn=lambda s: PRICE,
        authorization=ExecutionAuthorization.MOCK, correlation_index=corr_index)

    out = await bridge.route_recommendations([_rec()], now_ns=2_000)
    assert out.submitted == 1
    order = mgr.all_orders()[0]
    for upd in broker.drain_updates(order.venue_order_id):
        await mgr.reconcile(upd)

    types = {r.event_type for r in sink.peek()}
    assert DecisionEventType.ORDER_INTENT in types
    assert DecisionEventType.RISK_APPROVAL in types
    assert DecisionEventType.BROKER_ACK in types
    assert DecisionEventType.COMPLETE_FILL in types
    # traceability: every emitted execution event shares the decision_id
    fills = [r for r in sink.peek() if r.event_type is DecisionEventType.COMPLETE_FILL]
    assert fills[0].decision_id == "AAPL:momentum_continuation:1000"
