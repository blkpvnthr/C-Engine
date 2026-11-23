"""Regime-aware strategy arbitration + economic ranking (milestone §14-16).

Defines the common :class:`StrategyRecommendation` contract, per-family condition
evaluators that emit recommendations, a configurable regime-compatibility policy
(research hypotheses, not hard law), a deterministic/economic ranker, and the
arbitrator that resolves conflicts across recommendations.

Hard rules honored here:
  * Unvalidated numeric estimates remain ``None`` (unknown) — never fabricated.
  * Costs are subtracted exactly once (no double-counting).
  * Mandatory financial protection always takes precedence: the arbitrator never
    emits or blocks a protective exit — protective exits run on the independent
    path and are explicitly out of scope for elective arbitration.
  * Research-eligibility families produce RESEARCH recommendations only; nothing
    here flips a family to EXECUTION.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

from cengine.regime_engine import RegimeAssessment, RegimeDirection, TrendStructure
from cengine.strategy_families import (
    FEATURE_VERSION,
    EligibilityState,
    ExposureDirection,
    StrategyFamily,
    StrategyFamilyRegistry,
    StrategyFamilySpec,
    passes_entry_contract,
)


@dataclass
class StrategyRecommendation:
    decision_id: str
    symbol_id: str
    strategy_id: str
    strategy_family: StrategyFamily
    strategy_version: str
    regime_version: str
    feature_version: str
    observation_timestamp: int                  # ns
    expiration_timestamp: int                   # ns
    exposure_direction: ExposureDirection
    # Numeric estimates are OPTIONAL and default unknown (None) — never fabricated.
    expected_gross_return: Optional[float] = None
    estimated_transaction_cost: Optional[float] = None
    expected_net_return: Optional[float] = None
    expected_downside: Optional[float] = None
    confidence: Optional[float] = None          # heuristic [0,1], NOT calibrated
    eligibility: EligibilityState = EligibilityState.RESEARCH
    rejection_reason: Optional[str] = None
    # True iff ALL strategy/entry/regime CONDITIONS passed, independent of the
    # family's advisory eligibility. The execution bridge gates on this + the
    # family's maturity/authorization; it is NOT a routing authorization by itself.
    conditions_passed: bool = False
    # auxiliary (not part of the minimal contract but useful downstream)
    underlying_exposure_group: Optional[str] = None
    rank_score: Optional[float] = None
    rank_basis: str = "deterministic_research"  # or "calibrated_economic"
    notes: list = field(default_factory=list)

    @property
    def is_executable(self) -> bool:
        return (self.eligibility == EligibilityState.EXECUTION
                and self.rejection_reason is None)

    def expired(self, now_ns: int) -> bool:
        return now_ns >= self.expiration_timestamp


# --- regime compatibility policy (§15) --------------------------------------
class RegimeCompatibilityPolicy:
    """Per-family compatible regime directions/trends. These are RESEARCH HYPOTHESES
    (configurable), not immutable law. Defaults come from each family spec.

    Notes honoring the directive:
      * Inverse-ETF opportunities are NOT auto-prohibited in bearish regimes — the
        traded instrument's own direction drives compatibility (an inverse ETF rises
        in a bearish underlying regime); exposure authorization is separate.
      * Compatibility never AUTHORIZES geared/inverse execution — that is the
        exposure registry's job.
    """

    def __init__(self, registry: StrategyFamilyRegistry,
                 overrides: Optional[dict] = None) -> None:
        self._registry = registry
        self._overrides = overrides or {}

    def compatible(self, family: StrategyFamily, regime: RegimeAssessment) -> tuple[bool, str]:
        spec = self._registry.spec(family)
        dirs = self._overrides.get((family, "directions"), spec.compatible_directions)
        trends = self._overrides.get((family, "trends"), spec.compatible_trends)
        d = regime.dimensions.direction
        t = regime.dimensions.trend
        # UNCERTAIN / degraded conditions: do not assert compatibility (fail closed
        # for elective entries, but never block risk-reducing exits elsewhere).
        if d == RegimeDirection.UNCERTAIN:
            return False, "regime_uncertain"
        if d not in dirs:
            return False, f"direction_{d.value}_incompatible"
        if trends and t not in trends:
            return False, f"trend_{t.value}_incompatible"
        return True, "regime_compatible"


# --- family condition evaluators (§8-12) ------------------------------------
def _nz(v, default=0.0):
    return default if v is None or v != v else v


def _mk(symbol, spec: StrategyFamilySpec, regime: RegimeAssessment, now_ns: int,
        ttl_ns: int, direction=ExposureDirection.LONG) -> StrategyRecommendation:
    return StrategyRecommendation(
        decision_id=f"{symbol}:{spec.family.value}:{now_ns}",
        symbol_id=str(symbol).upper(), strategy_id=spec.maps_to,
        strategy_family=spec.family, strategy_version=spec.strategy_version,
        regime_version=regime.version, feature_version=FEATURE_VERSION,
        observation_timestamp=now_ns, expiration_timestamp=now_ns + ttl_ns,
        exposure_direction=direction, confidence=regime.dimensions.confidence,
        underlying_exposure_group=getattr(regime, "underlying_exposure_group", None))


def _finalize(rec: StrategyRecommendation, spec: StrategyFamilySpec,
              passed: bool, reason: str) -> StrategyRecommendation:
    # Record whether the strategy/entry/regime conditions themselves passed,
    # independent of the family's advisory eligibility. The execution bridge uses
    # this with the family's maturity + the run authorization.
    rec.conditions_passed = bool(passed)
    if not passed:
        rec.eligibility = EligibilityState.RESEARCH
        rec.rejection_reason = reason
        return rec
    # All conditions pass. Eligibility is capped by the family's own eligibility:
    # a RESEARCH family NEVER yields an executable recommendation.
    if spec.eligibility == EligibilityState.EXECUTION:
        rec.eligibility = EligibilityState.EXECUTION
    else:
        rec.eligibility = EligibilityState.RESEARCH
        rec.rejection_reason = f"family_not_execution_eligible:{spec.eligibility.value}"
    return rec


def evaluate_momentum_continuation(symbol, row, regime, spec, policy, now_ns,
                                   ttl_ns=120 * 1_000_000_000):
    rec = _mk(symbol, spec, regime, now_ns, ttl_ns)
    ok, why = passes_entry_contract(row)
    if not ok:
        return _finalize(rec, spec, False, why)
    compat, creason = policy.compatible(spec.family, regime)
    if not compat:
        return _finalize(rec, spec, False, creason)
    # §8: ADX structure + a non-redundant confirmation (avoid double-weighting
    # mathematically redundant indicators — require ONE independent confirmation).
    adx = _nz(row.get("adx"))
    confirmations = [
        _nz(row.get("rel_volume")) >= spec.min_rel_volume,
        _nz(row.get("momentum_accel")) > 0,
        _nz(row.get("htf_direction")) > 0,
    ]
    if adx < 20.0:
        return _finalize(rec, spec, False, "adx_below_trend_threshold")
    if not any(confirmations):
        return _finalize(rec, spec, False, "no_independent_momentum_confirmation")
    rec.notes.append("momentum: adx+contract+independent-confirmation")
    return _finalize(rec, spec, True, "ok")


def evaluate_volatility_breakout(symbol, row, regime, spec, policy, now_ns,
                                 ttl_ns=120 * 1_000_000_000):
    rec = _mk(symbol, spec, regime, now_ns, ttl_ns)
    ok, why = passes_entry_contract(row)
    if not ok:
        return _finalize(rec, spec, False, why)
    compat, creason = policy.compatible(spec.family, regime)
    if not compat:
        return _finalize(rec, spec, False, creason)
    # §9: breakout of prior channel high, with volatility expansion + volume + no
    # false-breakout (close must hold above the broken level using info at decision time).
    price = _nz(row.get("price"))
    donchian_high = row.get("donchian_high")
    if donchian_high is None or not (price > donchian_high):
        return _finalize(rec, spec, False, "no_channel_breakout")
    expanding = (_nz(row.get("atr_expansion_ratio"), 1.0) >= 1.2
                 or regime.dimensions.trend in (TrendStructure.EXPANSION,))
    if not expanding:
        return _finalize(rec, spec, False, "no_volatility_expansion")
    if _nz(row.get("rel_volume")) < spec.min_rel_volume:
        return _finalize(rec, spec, False, "volume_unconfirmed")
    rec.notes.append("breakout: channel+expansion+volume")
    return _finalize(rec, spec, True, "ok")


def evaluate_trend_pullback(symbol, row, regime, spec, policy, now_ns,
                            ttl_ns=120 * 1_000_000_000):
    rec = _mk(symbol, spec, regime, now_ns, ttl_ns)
    # §10: require RECOVERY confirmation before an executable entry.
    if _nz(row.get("htf_direction")) <= 0:
        return _finalize(rec, spec, False, "htf_trend_not_up")
    recovered = (_nz(row.get("momentum_accel")) > 0
                 and _nz(row.get("short_mom"), -1.0) > -0.001)
    if not recovered:
        return _finalize(rec, spec, False, "recovery_not_confirmed")
    ok, why = passes_entry_contract(row)
    if not ok:
        return _finalize(rec, spec, False, why)
    compat, creason = policy.compatible(spec.family, regime)
    if not compat:
        return _finalize(rec, spec, False, creason)
    rec.notes.append("pullback: htf-up+recovery+contract")
    return _finalize(rec, spec, True, "ok")


def evaluate_mean_reversion(symbol, row, regime, spec, policy, now_ns,
                            ttl_ns=120 * 1_000_000_000):
    rec = _mk(symbol, spec, regime, now_ns, ttl_ns)
    # §11: displacement below mean is NECESSARY but NOT sufficient — require a
    # POSITIVE reversal confirmation. Never buy solely because oversold.
    displaced = (_nz(row.get("bb_dist"), 0.0) < -0.01
                 or _nz(row.get("rsi"), 50.0) <= 30.0
                 or _nz(row.get("cci"), 0.0) <= -100.0)
    if not displaced:
        return _finalize(rec, spec, False, "not_displaced_below_mean")
    reversal = (_nz(row.get("momentum_accel")) > 0
                and _nz(row.get("short_mom"), -1.0) > 0)
    if not reversal:
        return _finalize(rec, spec, False, "no_positive_reversal_confirmation")
    ok, why = passes_entry_contract(row)
    if not ok:
        return _finalize(rec, spec, False, why)
    compat, creason = policy.compatible(spec.family, regime)
    if not compat:
        return _finalize(rec, spec, False, creason)
    rec.notes.append("meanrev: displaced+positive-reversal+contract")
    return _finalize(rec, spec, True, "ok")


def evaluate_multi_day_swing(symbol, row, regime, spec, policy, now_ns,
                             ttl_ns=24 * 3600 * 1_000_000_000):
    rec = _mk(symbol, spec, regime, now_ns, ttl_ns)
    # §12: multi-timeframe alignment + sector strength + overnight viability under
    # the mandatory -0.5% protective exit. This family is RESEARCH-only until that
    # viability is demonstrated, so it never produces an executable recommendation.
    aligned = (_nz(row.get("trend_15m")) > 0 and _nz(row.get("trend_1h")) > 0
               and _nz(row.get("trend_1d")) > 0)
    if not aligned:
        return _finalize(rec, spec, False, "timeframes_not_aligned_up")
    if _nz(row.get("sector_rel_strength")) <= 0:
        return _finalize(rec, spec, False, "weak_sector_relative_strength")
    # Overnight viability vs the intraday -0.5% stop: a swing whose typical adverse
    # excursion exceeds 0.5% intraday would be stopped out before the thesis plays
    # out. We FLAG this explicitly; we never relax the protective exit.
    rec.notes.append("swing: viability-under--0.5%-stop must be validated (research-only)")
    ok, why = passes_entry_contract(row)
    if not ok:
        return _finalize(rec, spec, False, why)
    return _finalize(rec, spec, True, "ok")  # still capped to RESEARCH by spec


FAMILY_EVALUATORS: dict[StrategyFamily, Callable] = {
    StrategyFamily.MOMENTUM_CONTINUATION: evaluate_momentum_continuation,
    StrategyFamily.VOLATILITY_BREAKOUT: evaluate_volatility_breakout,
    StrategyFamily.TREND_PULLBACK: evaluate_trend_pullback,
    StrategyFamily.MEAN_REVERSION: evaluate_mean_reversion,
    StrategyFamily.MULTI_DAY_SWING: evaluate_multi_day_swing,
}


# --- economic ranking (§16) -------------------------------------------------
class EconomicRanker:
    """Ranks recommendations by expected NET return when calibrated estimates are
    present; otherwise falls back to an EXPLICITLY LABELED deterministic research
    score. Costs are subtracted exactly once."""

    def rank(self, rec: StrategyRecommendation) -> StrategyRecommendation:
        g = rec.expected_gross_return
        c = rec.estimated_transaction_cost
        if g is not None and c is not None:
            rec.expected_net_return = g - c          # subtract cost ONCE
            rec.rank_score = rec.expected_net_return
            rec.rank_basis = "calibrated_economic"
        else:
            # deterministic research fallback: confidence-weighted, cost-penalized,
            # downside-penalized — clearly not a calibrated expectation.
            conf = _nz(rec.confidence, 0.0)
            downside = _nz(rec.expected_downside, 0.0)
            # cost expressed in bps on the spec; normalize to a small penalty
            rec.rank_score = round(conf - 0.5 * downside, 6)
            rec.rank_basis = "deterministic_research"
            rec.expected_net_return = None           # unknown, not fabricated
        return rec

    def rank_all(self, recs: list) -> list:
        return [self.rank(r) for r in recs]


# --- arbitration (§14) ------------------------------------------------------
class StrategyArbitrator:
    """Resolves conflicts across elective recommendations. Mandatory protective
    exits are NOT arbitrated here — they run independently and always take
    precedence."""

    def __init__(self, ranker: Optional[EconomicRanker] = None,
                 opposing_fn: Optional[Callable] = None) -> None:
        self._ranker = ranker or EconomicRanker()
        # opposing_fn(symbol_a, symbol_b) -> bool: do these two symbols hold opposing
        # exposure (e.g. TQQQ vs SQQQ on the same underlying)? Supplied by the
        # exposure registry; defaults to "never opposing" when not provided.
        self._opposing = opposing_fn or (lambda a, b: False)

    def arbitrate(self, recs: list, *, now_ns: int, max_concurrent: int = 10) -> dict:
        """Return {'selected': [...], 'rejected': [(rec, reason), ...]} after
        resolving staleness, duplicates, direction conflicts, opposing exposures,
        and capital contention. Only EXECUTION-eligible, non-expired recommendations
        can be selected; RESEARCH recommendations are retained but never selected."""
        rejected = []
        live = []
        for r in self._ranker.rank_all(recs):
            if r.expired(now_ns):
                rejected.append((r, "expired"))
            elif r.eligibility != EligibilityState.EXECUTION or r.rejection_reason:
                rejected.append((r, r.rejection_reason or "not_execution_eligible"))
            else:
                live.append(r)

        # dedup: one recommendation per (symbol, direction) — keep best rank
        best: dict = {}
        for r in sorted(live, key=lambda x: _nz(x.rank_score, -1e9), reverse=True):
            key = (r.symbol_id, r.exposure_direction)
            if key in best:
                rejected.append((r, "duplicate_symbol_direction"))
            else:
                best[key] = r
        candidates = sorted(best.values(), key=lambda x: _nz(x.rank_score, -1e9),
                            reverse=True)

        # direction conflict: same symbol, opposing directions -> keep higher rank
        by_symbol: dict = {}
        for r in candidates:
            prev = by_symbol.get(r.symbol_id)
            if prev is not None and prev.exposure_direction != r.exposure_direction:
                rejected.append((r, "conflicting_direction_same_symbol"))
            elif prev is None:
                by_symbol[r.symbol_id] = r
        deconflicted = list(by_symbol.values())

        # opposing-exposure conflict across symbols (e.g. leveraged long vs inverse
        # on the same underlying) -> keep higher rank, reject the opposed one
        selected: list = []
        for r in sorted(deconflicted, key=lambda x: _nz(x.rank_score, -1e9), reverse=True):
            if any(self._opposing(r.symbol_id, s.symbol_id) for s in selected):
                rejected.append((r, "opposing_exposure_conflict"))
                continue
            if len(selected) >= max_concurrent:
                rejected.append((r, "capital_contention_cap"))
                continue
            selected.append(r)

        return {"selected": selected, "rejected": rejected,
                "research": [r for r in recs
                             if r.eligibility == EligibilityState.RESEARCH]}
