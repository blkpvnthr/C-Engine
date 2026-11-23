"""Decision-evidence contract (§17): nonblocking sink, attribution, drain, and
best-effort journal mirroring. No ML platform. Deterministic; no network."""

from __future__ import annotations

from cengine.decision_events import (
    DECISION_EVENT_SCHEMA_VERSION,
    DecisionEvent,
    DecisionEventSink,
    DecisionEventType,
)
from cengine.regime_engine import MarketRegimeEngine
from cengine.strategy_arbitration import StrategyRecommendation
from cengine.strategy_families import EligibilityState, ExposureDirection, StrategyFamily

NS = 1_000_000_000


def test_emit_drain_and_attribution_preserved():
    sink = DecisionEventSink()
    sink.emit(DecisionEvent(event_type=DecisionEventType.ORDER_INTENT, timestamp_ns=NS,
                            decision_id="d1", symbol_id="AAA",
                            strategy_family="momentum_continuation",
                            strategy_version="v1", regime_version="rv1",
                            feature_version="fv1", payload={"side": "long"}))
    assert sink.size == 1 and sink.emitted == 1
    rows = sink.drain()
    assert sink.size == 0
    r = rows[0]
    assert r["event_type"] == "order_intent" and r["decision_id"] == "d1"
    assert r["symbol_id"] == "AAA" and r["strategy_version"] == "v1"
    assert r["regime_version"] == "rv1" and r["feature_version"] == "fv1"
    assert r["schema_version"] == DECISION_EVENT_SCHEMA_VERSION


def test_sink_is_bounded_and_drops_oldest_without_raising():
    sink = DecisionEventSink(capacity=3)
    for i in range(10):
        sink.emit(DecisionEvent(event_type=DecisionEventType.RANKING_COMPONENTS,
                                timestamp_ns=NS + i, decision_id=f"d{i}"))
    assert sink.size == 3 and sink.emitted == 10 and sink.dropped == 7
    ids = [e.decision_id for e in sink.peek()]
    assert ids == ["d7", "d8", "d9"]  # oldest evicted


def test_emit_never_raises_on_bad_journal():
    class _BadJournal:
        def append(self, *a, **k):
            raise RuntimeError("journal down")

    sink = DecisionEventSink(journal=_BadJournal())
    # must not raise even though the journal throws
    sink.emit(DecisionEvent(event_type=DecisionEventType.RISK_REJECTION, timestamp_ns=NS,
                            decision_id="d"))
    assert sink.emitted == 1 and sink.size == 1


def test_journal_mirroring_best_effort():
    captured = []

    class _Journal:
        def append(self, kind, ts, obj):
            captured.append((kind, ts, obj["event_type"]))

    sink = DecisionEventSink(journal=_Journal())
    sink.emit(DecisionEvent(event_type=DecisionEventType.COMPLETE_FILL, timestamp_ns=NS,
                            decision_id="d"))
    assert captured == [("decision_event", NS, "complete_fill")]


def test_record_regime_transition_and_recommendation():
    sink = DecisionEventSink()
    eng = MarketRegimeEngine()
    row = {"pdi": 40.0, "mdi": 10.0, "adx": 35.0, "vwap_dist": 0.01,
           "bb_dist": 0.01, "short_mom": 0.004}
    assessment = eng.assess_instrument("AAA", row, now_ns=NS)
    sink.record_regime_transition(assessment, NS)
    rec = StrategyRecommendation(
        decision_id="d", symbol_id="AAA",
        strategy_family=StrategyFamily.MOMENTUM_CONTINUATION, strategy_id="s",
        strategy_version="v", regime_version=eng.version, feature_version="fv",
        observation_timestamp=NS, expiration_timestamp=2 * NS,
        exposure_direction=ExposureDirection.LONG, confidence=0.7,
        eligibility=EligibilityState.RESEARCH, rank_score=0.7)
    sink.record_recommendation(rec, NS)
    rows = sink.drain()
    assert rows[0]["event_type"] == "regime_transition"
    assert rows[0]["payload"]["direction"] == "bullish"
    assert rows[1]["event_type"] == "strategy_recommendation"
    assert rows[1]["payload"]["eligibility"] == "research"
    assert rows[1]["strategy_family"] == "momentum_continuation"
