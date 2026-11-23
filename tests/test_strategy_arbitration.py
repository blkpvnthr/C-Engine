"""Regime-aware arbitration + economic ranking + family evaluators (§14-16).
Deterministic; reuses the real regime engine. No device/network."""

from __future__ import annotations

from cengine.regime_engine import MarketRegimeEngine
from cengine.strategy_arbitration import (
    FAMILY_EVALUATORS,
    EconomicRanker,
    RegimeCompatibilityPolicy,
    StrategyArbitrator,
    StrategyRecommendation,
    evaluate_mean_reversion,
    evaluate_momentum_continuation,
    evaluate_multi_day_swing,
    evaluate_trend_pullback,
)
from cengine.strategy_families import (
    EligibilityState,
    ExposureDirection,
    StrategyFamily,
    StrategyFamilyRegistry,
)

NS = 1_000_000_000


def _bull_regime(eng, symbol="AAA"):
    row = {"pdi": 40.0, "mdi": 10.0, "adx": 35.0, "vwap_dist": 0.01,
           "bb_dist": 0.01, "short_mom": 0.004, "momentum": 0.004}
    return eng.assess_instrument(symbol, row, now_ns=NS)


def _mom_row(**over):
    row = {"di_plus_5m": 30.0, "di_minus_5m": 10.0, "price": 101.0, "vwap": 100.0,
           "sma20_5m": 99.0, "bb_middle_5m": 100.5, "developing_1m_price": 101.2,
           "previous_completed_1m_close": 100.9, "adx": 35.0, "rel_volume": 1.2,
           "momentum_accel": 0.001, "htf_direction": 1.0}
    row.update(over)
    return row


def test_momentum_research_by_default_not_executable():
    reg = StrategyFamilyRegistry()
    eng = MarketRegimeEngine()
    pol = RegimeCompatibilityPolicy(reg)
    spec = reg.spec(StrategyFamily.MOMENTUM_CONTINUATION)
    rec = evaluate_momentum_continuation("AAA", _mom_row(), _bull_regime(eng), spec, pol, NS)
    # conditions pass, but family is RESEARCH -> not executable
    assert rec.eligibility == EligibilityState.RESEARCH
    assert rec.rejection_reason.startswith("family_not_execution_eligible")
    assert not rec.is_executable


def test_momentum_executable_after_validation():
    reg = StrategyFamilyRegistry()
    reg.mark_execution_eligible(StrategyFamily.MOMENTUM_CONTINUATION, validated=True)
    eng = MarketRegimeEngine()
    pol = RegimeCompatibilityPolicy(reg)
    spec = reg.spec(StrategyFamily.MOMENTUM_CONTINUATION)
    rec = evaluate_momentum_continuation("AAA", _mom_row(), _bull_regime(eng), spec, pol, NS)
    assert rec.eligibility == EligibilityState.EXECUTION and rec.is_executable
    assert rec.rejection_reason is None


def test_momentum_requires_independent_confirmation():
    reg = StrategyFamilyRegistry()
    reg.mark_execution_eligible(StrategyFamily.MOMENTUM_CONTINUATION, validated=True)
    eng = MarketRegimeEngine()
    pol = RegimeCompatibilityPolicy(reg)
    spec = reg.spec(StrategyFamily.MOMENTUM_CONTINUATION)
    row = _mom_row(rel_volume=0.1, momentum_accel=-0.01, htf_direction=0.0)
    rec = evaluate_momentum_continuation("AAA", row, _bull_regime(eng), spec, pol, NS)
    assert rec.rejection_reason == "no_independent_momentum_confirmation"


def test_mean_reversion_requires_positive_reversal_not_just_oversold():
    reg = StrategyFamilyRegistry()
    reg.mark_execution_eligible(StrategyFamily.MEAN_REVERSION, validated=True)
    eng = MarketRegimeEngine()
    pol = RegimeCompatibilityPolicy(reg)
    spec = reg.spec(StrategyFamily.MEAN_REVERSION)
    # oversold but NO reversal confirmation -> rejected (never buy solely oversold)
    row = _mom_row(bb_dist=-0.02, rsi=25.0, momentum_accel=-0.01, short_mom=-0.002)
    rec = evaluate_mean_reversion("AAA", row, _bull_regime(eng), spec, pol, NS)
    assert rec.rejection_reason == "no_positive_reversal_confirmation"


def test_trend_pullback_requires_recovery():
    reg = StrategyFamilyRegistry()
    reg.mark_execution_eligible(StrategyFamily.TREND_PULLBACK, validated=True)
    eng = MarketRegimeEngine()
    pol = RegimeCompatibilityPolicy(reg)
    spec = reg.spec(StrategyFamily.TREND_PULLBACK)
    row = _mom_row(htf_direction=1.0, momentum_accel=-0.01, short_mom=-0.01)
    rec = evaluate_trend_pullback("AAA", row, _bull_regime(eng), spec, pol, NS)
    assert rec.rejection_reason == "recovery_not_confirmed"


def test_multi_day_swing_research_only_even_when_conditions_pass():
    reg = StrategyFamilyRegistry()
    # even if someone tries to promote it, conditions passing must not auto-execute
    eng = MarketRegimeEngine()
    pol = RegimeCompatibilityPolicy(reg)
    spec = reg.spec(StrategyFamily.MULTI_DAY_SWING)
    row = _mom_row(trend_15m=1, trend_1h=1, trend_1d=1, sector_rel_strength=1.0)
    rec = evaluate_multi_day_swing("AAA", row, _bull_regime(eng), spec, pol, NS)
    assert rec.eligibility == EligibilityState.RESEARCH
    assert any("-0.5%" in n for n in rec.notes)


def test_regime_compatibility_inverse_not_auto_prohibited_and_uncertain_fails_closed():
    reg = StrategyFamilyRegistry()
    pol = RegimeCompatibilityPolicy(reg)
    eng = MarketRegimeEngine()
    # An inverse ETF rising (its own direction bullish) is compatible with a
    # momentum family regardless of a bearish UNDERLYING — not auto-prohibited.
    bull = _bull_regime(eng, "SH")
    ok, _ = pol.compatible(StrategyFamily.MOMENTUM_CONTINUATION, bull)
    assert ok
    # Uncertain regime fails closed for elective entry.
    unc = eng.assess_instrument("ZZZ", {"pdi": None, "mdi": None}, now_ns=NS)
    ok2, why2 = pol.compatible(StrategyFamily.MOMENTUM_CONTINUATION, unc)
    assert not ok2 and why2 == "regime_uncertain"


def test_economic_ranker_net_subtracts_cost_once_and_unknown_stays_none():
    r = EconomicRanker()
    calibrated = StrategyRecommendation(
        decision_id="d", symbol_id="AAA", strategy_id="s",
        strategy_family=StrategyFamily.MOMENTUM_CONTINUATION, strategy_version="v",
        regime_version="rv", feature_version="fv", observation_timestamp=NS,
        expiration_timestamp=2 * NS, exposure_direction=ExposureDirection.LONG,
        expected_gross_return=0.01, estimated_transaction_cost=0.002)
    r.rank(calibrated)
    assert abs(calibrated.expected_net_return - 0.008) < 1e-12  # cost subtracted ONCE
    assert calibrated.rank_basis == "calibrated_economic"
    # unknown estimates -> deterministic fallback, net stays None (not fabricated)
    unknown = StrategyRecommendation(
        decision_id="d2", symbol_id="BBB", strategy_id="s",
        strategy_family=StrategyFamily.MOMENTUM_CONTINUATION, strategy_version="v",
        regime_version="rv", feature_version="fv", observation_timestamp=NS,
        expiration_timestamp=2 * NS, exposure_direction=ExposureDirection.LONG,
        confidence=0.7)
    r.rank(unknown)
    assert unknown.expected_net_return is None
    assert unknown.rank_basis == "deterministic_research" and unknown.rank_score == 0.7


def _exec_rec(symbol, now=NS, direction=ExposureDirection.LONG, conf=0.5, dec=None):
    return StrategyRecommendation(
        decision_id=dec or f"{symbol}:{direction.value}", symbol_id=symbol,
        strategy_id="s", strategy_family=StrategyFamily.MOMENTUM_CONTINUATION,
        strategy_version="v", regime_version="rv", feature_version="fv",
        observation_timestamp=now, expiration_timestamp=now + 10 * NS,
        exposure_direction=direction, confidence=conf,
        eligibility=EligibilityState.EXECUTION)


def test_arbitrator_resolves_stale_dupe_conflict_opposing_and_cap():
    arb = StrategyArbitrator(opposing_fn=lambda a, b: {a, b} == {"TQQQ", "SQQQ"})
    recs = [
        _exec_rec("AAA", conf=0.9, dec="a1"),
        _exec_rec("AAA", conf=0.5, dec="a2"),                         # dup symbol+dir
        _exec_rec("BBB", conf=0.8, direction=ExposureDirection.SHORT, dec="b-short"),
        _exec_rec("BBB", conf=0.3, direction=ExposureDirection.LONG, dec="b-long"),  # conflict
        _exec_rec("TQQQ", conf=0.7, dec="t"),
        _exec_rec("SQQQ", conf=0.6, dec="sq"),                        # opposing TQQQ
    ]
    stale = _exec_rec("CCC", now=0, dec="stale")
    stale.expiration_timestamp = NS // 2
    recs.append(stale)
    # a research rec must never be selected
    research = _exec_rec("DDD", dec="research")
    research.eligibility = EligibilityState.RESEARCH
    recs.append(research)

    out = arb.arbitrate(recs, now_ns=NS, max_concurrent=10)
    selected_ids = {r.decision_id for r in out["selected"]}
    reasons = {r.decision_id: why for r, why in out["rejected"]}
    assert "a1" in selected_ids and reasons.get("a2") == "duplicate_symbol_direction"
    assert "b-short" in selected_ids
    assert reasons.get("b-long") == "conflicting_direction_same_symbol"
    assert "t" in selected_ids and reasons.get("sq") == "opposing_exposure_conflict"
    assert reasons.get("stale") == "expired"
    assert "research" not in selected_ids  # research never selected
    assert any(r.symbol_id == "DDD" for r in out["research"])


def test_arbitrator_capital_contention_cap():
    arb = StrategyArbitrator()
    recs = [_exec_rec(f"S{i}", conf=i / 10.0, dec=f"s{i}") for i in range(5)]
    out = arb.arbitrate(recs, now_ns=NS, max_concurrent=3)
    assert len(out["selected"]) == 3
    assert sum(1 for _, why in out["rejected"] if why == "capital_contention_cap") == 2


def test_family_evaluators_registry_complete():
    assert set(FAMILY_EVALUATORS) == set(StrategyFamily)
