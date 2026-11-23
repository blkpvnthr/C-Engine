"""Portfolio decision-event contract (Milestone 3, §18)."""

from __future__ import annotations

from cengine.decision_events import (
    DecisionEventSink,
    DecisionEventType,
    events_from_rows,
)
from cengine.portfolio_optimizer import (
    OPTIMIZER_VERSION,
    PortfolioRecommendation,
    RecommendationEligibility,
)
from cengine.portfolio_replacement import ReplacementDecision
from cengine.portfolio_state import (
    PORTFOLIO_STATE_VERSION,
    PortfolioState,
    ReconciliationStatus,
)


def _state() -> PortfolioState:
    return PortfolioState(
        portfolio_version=PORTFOLIO_STATE_VERSION, observation_timestamp=100,
        account_equity_ticks=1_000_000, settled_cash_ticks=800_000,
        available_cash_ticks=800_000, buying_power_ticks=1_600_000,
        reserved_capital_ticks=0, pending_buy_exposure_ticks=0,
        pending_sell_exposure_ticks=0, realized_pl_ticks=0, unrealized_pl_ticks=0,
        gross_exposure_ticks=200_000, net_exposure_ticks=200_000,
        effective_exposure_ticks=200_000,
        reconciliation_status=ReconciliationStatus.RECONCILED)


def _rec(decision_id="pf:AAA:100", source="strat:AAA:50") -> PortfolioRecommendation:
    return PortfolioRecommendation(
        portfolio_decision_id=decision_id, source_strategy_decision_id=source,
        symbol_id="AAA", instrument_generation=1, portfolio_version=PORTFOLIO_STATE_VERSION,
        strategy_id="portfolio", strategy_family="momentum_continuation",
        regime_version="regime-1.0.0", feature_version="features-1.0.0",
        optimizer_version=OPTIMIZER_VERSION, configuration_version="portfolio-cfg-1",
        observation_timestamp=100, decision_timestamp=100, expiration_timestamp=200,
        current_weight=0.0, target_weight=0.3, requested_quantity=10,
        requested_notional_ticks=300_000, capital_required_ticks=300_000,
        expected_risk_contribution=0.002, eligibility=RecommendationEligibility.PROPOSED)


def test_portfolio_state_event():
    sink = DecisionEventSink()
    sink.record_portfolio_state(_state(), 100)
    (ev,) = sink.peek()
    assert ev.event_type is DecisionEventType.PORTFOLIO_STATE
    assert ev.payload["equity_ticks"] == 1_000_000
    assert ev.payload["gross_exposure_ticks"] == 200_000
    assert ev.payload["reconciliation_status"] == "reconciled"


def test_portfolio_optimization_event():
    sink = DecisionEventSink()
    sink.record_portfolio_optimization(
        now_ns=100, policy="risk_based", covariance_source="metal",
        covariance_version="portfolio-cov-1.0.0", return_basis="deterministic_risk_based",
        binding=["max_position_weight:AAA"], candidate_count=3)
    (ev,) = sink.peek()
    assert ev.event_type is DecisionEventType.PORTFOLIO_OPTIMIZATION
    assert ev.payload["allocation_policy"] == "risk_based"
    assert ev.payload["covariance_source"] == "metal"
    assert ev.payload["binding_constraints"] == ["max_position_weight:AAA"]


def test_portfolio_recommendation_event_links_source_and_keeps_unknowns_none():
    sink = DecisionEventSink()
    rec = _rec()
    sink.record_portfolio_recommendation(rec, 100)
    (ev,) = sink.peek()
    assert ev.event_type is DecisionEventType.PORTFOLIO_RECOMMENDATION
    assert ev.decision_id == rec.portfolio_decision_id
    assert ev.payload["source_strategy_decision_id"] == "strat:AAA:50"
    assert ev.payload["target_weight"] == 0.3
    assert ev.payload["eligibility"] == "proposed"
    # unknown economics are never fabricated
    assert ev.payload["expected_incremental_net_return"] is None


def test_capital_reservation_event():
    sink = DecisionEventSink()
    sink.record_capital_reservation(decision_id="pf:AAA:100", symbol_id="AAA",
                                    notional_ticks=300_000, accepted=True, reason="reserved",
                                    now_ns=100)
    (ev,) = sink.peek()
    assert ev.event_type is DecisionEventType.CAPITAL_RESERVATION
    assert ev.payload["accepted"] is True
    assert ev.payload["reason"] == "reserved"


def test_replacement_decision_event():
    sink = DecisionEventSink()
    sink.record_replacement_decision(
        ReplacementDecision("NVDA", "AAPL", True, "selected", considered=["MSFT", "AAPL"]), 100)
    (ev,) = sink.peek()
    assert ev.event_type is DecisionEventType.REPLACEMENT_DECISION
    assert ev.symbol_id == "NVDA"
    assert ev.payload["replacement_symbol"] == "AAPL"
    assert ev.payload["authorized"] is True


def test_traceability_recommendation_to_fill():
    # A portfolio_decision_id flows from the recommendation through to a fill event.
    sink = DecisionEventSink()
    rec = _rec(decision_id="pf:AAA:100")
    sink.record_portfolio_recommendation(rec, 100)

    class _Intent:
        symbol = "AAA"
        correlation_id = "AAA:100:portfolio"

    class _Order:
        intent = _Intent()
        average_fill_price_ticks = 1_000_000.0

    class _Update:
        from order_manager import OrderStatus as _OS
        status = _OS.FILLED
        venue_order_id = "MOCK-1"
        last_fill_quantity = 10
        cumulative_filled_quantity = 10
        last_fill_price_ticks = 1_000_000

    sink.record_fill(_Order(), _Update(), 101, decision_id=rec.portfolio_decision_id)
    fills = [e for e in sink.peek() if e.event_type is DecisionEventType.COMPLETE_FILL]
    assert len(fills) == 1
    assert fills[0].decision_id == "pf:AAA:100"


def test_events_from_rows_roundtrips_portfolio_event_types():
    sink = DecisionEventSink()
    sink.record_portfolio_state(_state(), 100)
    sink.record_portfolio_recommendation(_rec(), 100)
    sink.record_capital_reservation(decision_id="pf:AAA:100", symbol_id="AAA",
                                    notional_ticks=300_000, accepted=True, reason="reserved",
                                    now_ns=100)
    rows = sink.drain()
    recovered = events_from_rows(rows)
    types = {e.event_type for e in recovered}
    assert DecisionEventType.PORTFOLIO_STATE in types
    assert DecisionEventType.PORTFOLIO_RECOMMENDATION in types
    assert DecisionEventType.CAPITAL_RESERVATION in types
    assert len(recovered) == 3
