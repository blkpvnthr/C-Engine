"""P4: universe-selection decision persistence (§9).

Verifies the decision-event contract extension, the recorder adapter, and that
the events route to the correct bronze datasets — without fabricating outcomes.
"""

from __future__ import annotations

from cengine.decision_events import DecisionEventSink, DecisionEventType
from cengine.dynamic_universe.events import UniverseDecisionRecorder
from cengine.dynamic_universe.layers import UniverseLayers
from cengine.dynamic_universe.rotation import RotationResult
from cengine.lakehouse.bronze import _EVENT_DATASET, BRONZE_DATASETS


def _result():
    return RotationResult(
        active=("SPY", "NVDA", "AAA"),
        added=("NVDA",),
        removed=("OLD",),
        protected=("SPY",),
        active_baseline=("SPY", "AAA"),
        active_opportunity=("NVDA",),
        reasons={"NVDA": "opportunity", "SPY": "protected", "AAA": "baseline",
                 "OLD": "rotated_out"},
        degraded=False,
        dropped_mandatory=(),
        stale_baseline=("BBB",),
    )


def _layers():
    return UniverseLayers.build(
        baseline=["SPY", "AAA", "BBB"], opportunity=["NVDA"], held=["SPY"],
        pending=[], active=["SPY", "NVDA", "AAA"], version=7,
    )


def test_record_universe_rotation_emits_summary_and_per_symbol():
    sink = DecisionEventSink()
    sink.record_universe_rotation(
        now_ns=1_000, generation=7,
        active=["SPY", "NVDA", "AAA"], added=["NVDA"], removed=["OLD"],
        active_baseline=["SPY", "AAA"], active_opportunity=["NVDA"],
        protected=["SPY"], stale_baseline=["BBB"], degraded=False,
        reasons={"NVDA": "opportunity", "OLD": "rotated_out"},
        scores={"NVDA": 0.91}, baseline_set={"SPY", "AAA", "BBB"},
        regime_version="r1", model_version="deterministic-v1",
    )
    rows = sink.peek()
    kinds = [r.event_type for r in rows]
    assert DecisionEventType.UNIVERSE_ROTATION in kinds
    assert DecisionEventType.UNIVERSE_PROMOTION in kinds
    assert DecisionEventType.UNIVERSE_DEMOTION in kinds
    assert DecisionEventType.BASELINE_REVISIT in kinds

    promo = next(r for r in rows if r.event_type is DecisionEventType.UNIVERSE_PROMOTION)
    assert promo.symbol_id == "NVDA"
    assert promo.payload["opportunity_score"] == 0.91
    assert promo.payload["is_baseline"] is False
    assert promo.payload["selection_reason"] == "opportunity"

    summary = next(r for r in rows if r.event_type is DecisionEventType.UNIVERSE_ROTATION)
    assert summary.payload["generation"] == 7
    assert summary.payload["active_size"] == 3
    assert summary.payload["model_version"] == "deterministic-v1"


def test_recorder_adapter_translates_rotation_result():
    sink = DecisionEventSink()
    rec = UniverseDecisionRecorder(sink, model_version="deterministic-v1")
    rec(_result(), _layers(), {"NVDA": 0.91, "SPY": 0.0}, now_s=1.0)
    rows = sink.peek()
    promo = next(r for r in rows if r.event_type is DecisionEventType.UNIVERSE_PROMOTION)
    assert promo.symbol_id == "NVDA" and promo.payload["opportunity_score"] == 0.91
    demo = next(r for r in rows if r.event_type is DecisionEventType.UNIVERSE_DEMOTION)
    assert demo.symbol_id == "OLD"
    revisit = next(r for r in rows if r.event_type is DecisionEventType.BASELINE_REVISIT)
    assert revisit.symbol_id == "BBB"


def test_events_route_to_bronze_datasets():
    assert _EVENT_DATASET["universe_rotation"] == "universe_membership"
    for et in ("universe_selection", "universe_promotion", "universe_demotion",
               "baseline_revisit"):
        assert _EVENT_DATASET[et] == "universe_selection"
    assert "universe_selection" in BRONZE_DATASETS


def test_recorder_is_nonblocking_on_none_sink():
    rec = UniverseDecisionRecorder(None)
    rec(_result(), _layers(), {}, now_s=1.0)   # must not raise
