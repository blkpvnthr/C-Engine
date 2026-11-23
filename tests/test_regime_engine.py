"""MarketRegimeEngine (milestone §4-5): four levels, four independent dimensions,
hysteresis/persistence, staleness, transitions, inverse-ETF underlying separation.
Pure/deterministic; reuses amps_v4_regime scoring. No device/network."""

from __future__ import annotations

from cengine.regime_engine import (
    LiquidityRegime,
    MarketRegimeEngine,
    RegimeDirection,
    RegimeEngineConfig,
    RegimeLevel,
    TrendStructure,
    VolatilityRegime,
)

NS = 1_000_000_000


def _bull(**over):
    row = {"pdi": 40.0, "mdi": 10.0, "adx": 35.0, "vwap_dist": 0.01,
           "bb_dist": 0.01, "short_mom": 0.004, "momentum": 0.004}
    row.update(over)
    return row


def _bear(**over):
    row = {"pdi": 10.0, "mdi": 40.0, "adx": 35.0, "vwap_dist": -0.01,
           "bb_dist": -0.01, "short_mom": -0.004, "momentum": -0.004}
    row.update(over)
    return row


def test_direction_dimensions_independent():
    eng = MarketRegimeEngine()
    a = eng.assess_instrument("AAA", _bull(), now_ns=NS)
    b = eng.assess_instrument("BBB", _bear(), now_ns=NS)
    assert a.dimensions.direction == RegimeDirection.BULLISH
    assert b.dimensions.direction == RegimeDirection.BEARISH
    assert a.dimensions.directional_score > 0 and b.dimensions.directional_score < 0


def test_confidence_is_not_calibrated_probability():
    eng = MarketRegimeEngine()
    a = eng.assess_instrument("AAA", _bull(), now_ns=NS)
    assert 0.0 <= a.dimensions.confidence <= 1.0
    assert a.dimensions.confidence_is_calibrated is False


def test_volatility_dimension_thresholds():
    eng = MarketRegimeEngine()
    low = eng.assess_instrument("L", _bull(realized_vol=0.0005), now_ns=NS)
    elev = eng.assess_instrument("E", _bull(realized_vol=0.005), now_ns=NS)
    extreme = eng.assess_instrument("X", _bull(realized_vol=0.02), now_ns=NS)
    assert low.dimensions.volatility == VolatilityRegime.LOW
    assert elev.dimensions.volatility == VolatilityRegime.ELEVATED
    assert extreme.dimensions.volatility == VolatilityRegime.EXTREME


def test_liquidity_dimension():
    eng = MarketRegimeEngine()
    ok = eng.assess_instrument("A", _bull(spread_bps=5.0, rel_volume=1.2), now_ns=NS)
    deg = eng.assess_instrument("B", _bull(spread_bps=40.0, rel_volume=1.0), now_ns=NS)
    ins = eng.assess_instrument("C", _bull(spread_bps=120.0), now_ns=NS)
    assert ok.dimensions.liquidity == LiquidityRegime.NORMAL
    assert deg.dimensions.liquidity == LiquidityRegime.DEGRADED
    assert ins.dimensions.liquidity == LiquidityRegime.INSUFFICIENT


def test_trend_structure_variants():
    eng = MarketRegimeEngine()
    comp = eng.assess_instrument("C", _bull(adx=15.0, bb_width=0.01), now_ns=NS)
    exp = eng.assess_instrument("E", _bull(atr_expansion_ratio=1.5), now_ns=NS)
    pers = eng.assess_instrument("P", _bull(adx=35.0), now_ns=NS)
    rng = eng.assess_instrument("R", _bull(adx=12.0), now_ns=NS)
    exh = eng.assess_instrument("X", _bull(momentum=0.01, momentum_accel=-0.01), now_ns=NS)
    rec = eng.assess_instrument("V", _bull(momentum=-0.01, momentum_accel=0.01), now_ns=NS)
    assert comp.dimensions.trend == TrendStructure.COMPRESSION
    assert exp.dimensions.trend == TrendStructure.EXPANSION
    assert pers.dimensions.trend == TrendStructure.PERSISTENT
    assert rng.dimensions.trend == TrendStructure.RANGE_BOUND
    assert exh.dimensions.trend == TrendStructure.EXHAUSTION
    assert rec.dimensions.trend == TrendStructure.RECOVERY


def test_hysteresis_requires_persistence_to_flip():
    eng = MarketRegimeEngine(RegimeEngineConfig(min_persistence_updates=2))
    # establish bullish
    eng.assess_instrument("S", _bull(), now_ns=NS)
    # a single bearish reading must NOT flip direction (hysteresis)
    a = eng.assess_instrument("S", _bear(), now_ns=2 * NS)
    assert a.dimensions.direction == RegimeDirection.BULLISH
    assert not a.transitioned
    # a second persistent bearish reading flips it
    b = eng.assess_instrument("S", _bear(), now_ns=3 * NS)
    assert b.dimensions.direction == RegimeDirection.BEARISH
    assert b.transitioned and b.previous_direction == RegimeDirection.BULLISH


def test_staleness_marks_invalid():
    eng = MarketRegimeEngine(RegimeEngineConfig(max_age_ns=60 * NS))
    row = _bull(observation_ns=NS)
    a = eng.assess_instrument("S", row, now_ns=NS + 10 * NS)   # 10s old -> fresh
    assert a.valid and not a.stale
    b = eng.assess_instrument("S", _bull(observation_ns=NS), now_ns=NS + 120 * NS)  # 120s
    assert b.stale and not b.valid


def test_inverse_etf_underlying_direction_separate():
    eng = MarketRegimeEngine()
    # SH (inverse SPY) rising while SPY underlying is bearish: the ETF's own
    # direction is bullish, but the underlying exposure is reported bearish.
    a = eng.assess_instrument("SH", _bull(), now_ns=NS, underlying_score=-60.0)
    assert a.dimensions.direction == RegimeDirection.BULLISH
    assert a.dimensions.underlying_direction == RegimeDirection.BEARISH


def test_broad_assessment_breadth_tilt_and_transition():
    eng = MarketRegimeEngine()
    refs = {"SPY": _bull(), "QQQ": _bull(), "DIA": _bull(), "IWM": _bull()}
    breadth = {"available": True, "valid": 100, "bullish": 80, "bearish": 10}
    a = eng.assess_broad(refs, breadth, now_ns=NS)
    assert a.level == RegimeLevel.BROAD and a.scope == "MARKET"
    assert a.dimensions.direction == RegimeDirection.BULLISH
    assert "breadth" in a.inputs_used


def test_version_is_stamped():
    eng = MarketRegimeEngine()
    a = eng.assess_instrument("A", _bull(), now_ns=NS)
    assert a.version == eng.version and eng.version.startswith("regime-")
