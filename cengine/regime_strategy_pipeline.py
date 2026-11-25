"""Regime → strategy → arbitration → decision-evidence composition (milestone).

Wires the regime engine, five-family evaluators, regime-compatibility policy,
economic ranker, arbitrator and decision-event sink into a single ADVISORY pipeline
that consumes the native GPU screen outputs (``gpu_discovery`` ScreenRows) plus
benchmark reference rows and the GPU breadth snapshot.

It is deliberately advisory: every family defaults to RESEARCH eligibility, so the
pipeline produces recommendations + decision evidence but routes NOTHING to
execution unless an operator has explicitly validated and enabled a family. The
existing order path (strategies.py → CandidateRouter → RiskEngine → OrderManager)
and the independent protective exit are untouched.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

from cengine.decision_events import DecisionEventSink
from cengine.regime_engine import MarketRegimeEngine, RegimeLevel
from cengine.strategy_arbitration import (
    FAMILY_EVALUATORS,
    EconomicRanker,
    RegimeCompatibilityPolicy,
    StrategyArbitrator,
)
from cengine.strategy_families import StrategyFamilyRegistry

REFERENCES = ("SPY", "QQQ", "DIA", "IWM")


def screen_row_to_features(row: Any, daily: Optional[dict] = None,
                           now_ns: int = 0) -> dict:
    """Adapt a ``gpu_discovery.ScreenRow`` (+ optional daily stats) to the feature
    dict consumed by the regime engine and family evaluators. Only fields the native
    screen actually produces are populated; absent inputs stay absent (never faked)."""
    daily = daily or {}
    price = getattr(row, "price", None)
    vwap = getattr(row, "vwap", None)
    bb_mid = getattr(row, "bb_middle", None)
    sma = getattr(row, "sma", None)
    prev_1m = getattr(row, "previous_close_1m", None)
    bb_upper = getattr(row, "bb_upper", None)
    bb_lower = getattr(row, "bb_lower", None)
    bb_width = None
    if bb_mid and bb_upper is not None and bb_lower is not None and bb_mid > 0:
        bb_width = (bb_upper - bb_lower) / bb_mid
    vwap_dist = ((price - vwap) / vwap) if (price and vwap) else None
    bb_dist = ((price - bb_mid) / bb_mid) if (price and bb_mid) else None
    return {
        # directional / contract inputs
        "pdi": getattr(row, "di_plus", None), "mdi": getattr(row, "di_minus", None),
        "di_plus_5m": getattr(row, "di_plus", None),
        "di_minus_5m": getattr(row, "di_minus", None),
        "adx": getattr(row, "adx", None),
        "price": price, "vwap": vwap, "sma20": sma, "sma20_5m": sma,
        "bb_mid": bb_mid, "bb_middle_5m": bb_mid, "bb_width": bb_width,
        "vwap_dist": vwap_dist, "bb_dist": bb_dist,
        "short_mom": getattr(row, "momentum_1m_pct", None),
        "momentum": getattr(row, "momentum_1m_pct", None),
        "developing_1m_price": price,
        "previous_completed_1m_close": prev_1m,
        "atr_pct": (getattr(row, "atr", None) / price
                    if (getattr(row, "atr", None) and price) else None),
        # merged daily/market + confirmation stats where available (never fabricated;
        # these are genuine indicators computed from the native bar window and passed
        # in via daily_stats -- the family evaluators reference them but the feature
        # pipeline previously never supplied them, so every confirmation gate failed).
        "rel_volume": daily.get("rel_volume"),
        "spread_bps": daily.get("spread_bps"),
        "momentum_accel": daily.get("momentum_accel"),
        "htf_direction": daily.get("htf_direction"),
        "donchian_high": daily.get("donchian_high"),
        "atr_expansion_ratio": daily.get("atr_expansion_ratio"),
        "observation_ns": now_ns,
    }


class RegimeStrategyPipeline:
    def __init__(self, *, registry: Optional[StrategyFamilyRegistry] = None,
                 regime_engine: Optional[MarketRegimeEngine] = None,
                 sink: Optional[DecisionEventSink] = None,
                 opposing_fn: Optional[Callable] = None,
                 max_concurrent: int = 10) -> None:
        self.registry = registry or StrategyFamilyRegistry()
        self.regime = regime_engine or MarketRegimeEngine()
        self.policy = RegimeCompatibilityPolicy(self.registry)
        self.ranker = EconomicRanker()
        self.arbitrator = StrategyArbitrator(ranker=self.ranker, opposing_fn=opposing_fn)
        self.sink = sink or DecisionEventSink()
        self.max_concurrent = int(max_concurrent)
        self.runs = 0

    def run(self, screen_rows: dict, *, now_ns: int,
            ref_features: Optional[dict] = None, breadth: Optional[dict] = None,
            daily_stats: Optional[dict] = None) -> dict:
        """One advisory pass. ``screen_rows`` maps symbol -> ScreenRow (or feature
        dict). Returns {broad, assessments, arbitration} and emits decision events.
        Never raises into the caller's hot path semantics beyond normal exceptions —
        the engine schedules it off the critical path."""
        daily_stats = daily_stats or {}
        self.runs += 1
        # --- broad regime from benchmark references + breadth
        ref_features = ref_features or {}
        broad = self.regime.assess_broad(ref_features, breadth, now_ns=now_ns)
        if broad.transitioned:
            self.sink.record_regime_transition(broad, now_ns)

        assessments: dict = {}
        recommendations: list = []
        for symbol, row in screen_rows.items():
            feats = (row if isinstance(row, dict)
                     else screen_row_to_features(row, daily_stats.get(symbol), now_ns))
            assessment = self.regime.assess_instrument(
                symbol, feats, now_ns=now_ns, level=RegimeLevel.INSTRUMENT)
            assessments[symbol] = assessment
            if assessment.transitioned:
                self.sink.record_regime_transition(assessment, now_ns)
            if not assessment.valid:
                continue
            for family, evaluate in FAMILY_EVALUATORS.items():
                spec = self.registry.spec(family)
                rec = evaluate(symbol, feats, assessment, spec, self.policy, now_ns)
                self.ranker.rank(rec)
                self.sink.record_recommendation(rec, now_ns)
                recommendations.append(rec)

        result = self.arbitrator.arbitrate(
            recommendations, now_ns=now_ns, max_concurrent=self.max_concurrent)
        self.sink.record_arbitration(result, now_ns)
        return {"broad": broad, "assessments": assessments, "arbitration": result,
                "recommendations": recommendations}

    def telemetry(self) -> dict:
        return {"runs": self.runs, "events_emitted": self.sink.emitted,
                "events_buffered": self.sink.size,
                "execution_eligible_families": [
                    f.value for f in self.registry.families()
                    if self.registry.is_execution_eligible(f)]}
