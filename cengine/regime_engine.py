"""Native-assisted Market Regime Intelligence (milestone §4-5).

A versioned ``MarketRegimeEngine`` that classifies market conditions at four levels
(broad market, benchmark, sector/industry, individual instrument) along FOUR
INDEPENDENT dimensions:

    Direction   : BULLISH | BEARISH | NEUTRAL | UNCERTAIN
    Trend       : PERSISTENT | RANGE_BOUND | COMPRESSION | EXPANSION |
                  EXHAUSTION | RECOVERY
    Volatility  : LOW | NORMAL | ELEVATED | EXTREME
    Liquidity   : NORMAL | DEGRADED | INSUFFICIENT

It REUSES the existing analytics rather than duplicating them:
  * ``amps_v4_regime.directional_score`` for the directional score (already the
    scanner's regime scorer), fed by the native ``technical_features`` outputs;
  * the GPU ``market_breadth`` snapshot (participation / advance-decline proxy);
  * ``indicators`` for any CPU-side helper math.

Design properties:
  * Pure/deterministic: inputs are plain feature dicts and an injected clock
    (``now_ns``), so it is fully unit-testable offline with no device or network.
  * Hysteresis + minimum persistence per scope (a candidate label must persist
    before it becomes active), timestamp validity, and explicit transition
    detection.
  * ``confidence`` is a HEURISTIC in [0, 1] — explicitly NOT a calibrated
    probability (see ``RegimeDimensions.confidence_is_calibrated == False``).
  * Inverse/geared ETFs: the instrument's own price *direction* is reported from
    its own momentum, but the *underlying benchmark* directional exposure is
    reported separately (``underlying_direction``) so a rising inverse ETF is not
    mistaken for bullish underlying exposure.

This module authorizes nothing: it is advisory context consumed by strategy
families, arbitration and ranking. Execution authorization remains with the
exposure registry and the native RiskEngine.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import amps_v4_regime as amps

REGIME_ENGINE_VERSION = "regime-1.0.0"


class RegimeLevel(str, Enum):
    BROAD = "broad"
    BENCHMARK = "benchmark"
    SECTOR = "sector"
    INSTRUMENT = "instrument"


class RegimeDirection(str, Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"
    UNCERTAIN = "uncertain"


class TrendStructure(str, Enum):
    PERSISTENT = "persistent"
    RANGE_BOUND = "range_bound"
    COMPRESSION = "compression"
    EXPANSION = "expansion"
    EXHAUSTION = "exhaustion"
    RECOVERY = "recovery"


class VolatilityRegime(str, Enum):
    LOW = "low"
    NORMAL = "normal"
    ELEVATED = "elevated"
    EXTREME = "extreme"


class LiquidityRegime(str, Enum):
    NORMAL = "normal"
    DEGRADED = "degraded"
    INSUFFICIENT = "insufficient"


@dataclass
class RegimeEngineConfig:
    # direction (on amps directional_score in [-100, 100])
    bullish_thresh: float = 25.0
    bearish_thresh: float = -25.0
    neutral_band: float = 12.0            # |score| below this and low ADX -> NEUTRAL
    # trend structure
    adx_persistent: float = 25.0          # ADX above -> trend has structure
    adx_rangebound: float = 18.0          # ADX below -> range-bound
    bb_compression: float = 0.02          # (bb_upper-bb_lower)/bb_mid below -> compressed
    atr_expansion_ratio: float = 1.25     # atr_pct / atr_pct_baseline above -> expanding
    exhaustion_decel: float = -0.0        # momentum_accel < this while |momentum| high
    # volatility (per-bar realized vol or atr_pct)
    vol_low: float = 0.0015
    vol_elevated: float = 0.004
    vol_extreme: float = 0.008
    # liquidity
    spread_degraded_bps: float = 25.0
    spread_insufficient_bps: float = 75.0
    min_rel_volume: float = 0.3           # below -> liquidity degraded
    # hysteresis / persistence
    min_persistence_updates: int = 2      # a candidate must persist N updates to flip
    max_age_ns: int = 5 * 60 * 1_000_000_000  # observations older than 5m are stale


@dataclass
class RegimeDimensions:
    direction: RegimeDirection
    trend: TrendStructure
    volatility: VolatilityRegime
    liquidity: LiquidityRegime
    confidence: float                     # heuristic [0,1] — NOT a probability
    confidence_is_calibrated: bool = False
    directional_score: float = float("nan")
    underlying_direction: Optional[RegimeDirection] = None  # for inverse/geared ETFs


@dataclass
class RegimeAssessment:
    level: RegimeLevel
    scope: str                            # "MARKET" | benchmark/sector/symbol
    dimensions: RegimeDimensions
    as_of_ns: int
    version: str = REGIME_ENGINE_VERSION
    valid: bool = True
    stale: bool = False
    transitioned: bool = False            # active direction changed vs the prior active
    previous_direction: Optional[RegimeDirection] = None
    inputs_used: list = field(default_factory=list)


@dataclass
class _ScopeState:
    """Per-scope hysteresis state for the DIRECTION dimension (the one that gates
    elective participation); other dimensions are reported instantaneously."""
    active: RegimeDirection = RegimeDirection.UNCERTAIN
    candidate: RegimeDirection = RegimeDirection.UNCERTAIN
    streak: int = 0


class MarketRegimeEngine:
    def __init__(self, config: Optional[RegimeEngineConfig] = None) -> None:
        self.config = config or RegimeEngineConfig()
        self.version = REGIME_ENGINE_VERSION
        self._states: dict[str, _ScopeState] = {}

    # -- dimension classifiers (instantaneous) -------------------------------
    def _direction_from_score(self, score: float, adx: float) -> RegimeDirection:
        c = self.config
        if score != score:  # NaN
            return RegimeDirection.UNCERTAIN
        if score >= c.bullish_thresh:
            return RegimeDirection.BULLISH
        if score <= c.bearish_thresh:
            return RegimeDirection.BEARISH
        if abs(score) <= c.neutral_band and (adx != adx or adx < c.adx_persistent):
            return RegimeDirection.NEUTRAL
        return RegimeDirection.UNCERTAIN

    def _trend(self, row: dict) -> TrendStructure:
        c = self.config
        adx = row.get("adx", float("nan"))
        bb_width = row.get("bb_width")
        atr_ratio = row.get("atr_expansion_ratio")
        mom = row.get("momentum", 0.0) or 0.0
        accel = row.get("momentum_accel")
        # exhaustion: strong move decelerating; recovery: negative move turning up
        if accel is not None:
            if mom > 0 and accel < c.exhaustion_decel and abs(mom) >= 0.003:
                return TrendStructure.EXHAUSTION
            if mom < 0 and accel > 0:
                return TrendStructure.RECOVERY
        if atr_ratio is not None and atr_ratio >= c.atr_expansion_ratio:
            return TrendStructure.EXPANSION
        if bb_width is not None and bb_width <= c.bb_compression:
            return TrendStructure.COMPRESSION
        if adx == adx and adx >= c.adx_persistent:
            return TrendStructure.PERSISTENT
        if adx == adx and adx <= c.adx_rangebound:
            return TrendStructure.RANGE_BOUND
        return TrendStructure.RANGE_BOUND

    def _volatility(self, row: dict) -> VolatilityRegime:
        c = self.config
        v = row.get("realized_vol")
        if v is None:
            v = row.get("atr_pct")
        if v is None or v != v:
            return VolatilityRegime.NORMAL
        if v >= c.vol_extreme:
            return VolatilityRegime.EXTREME
        if v >= c.vol_elevated:
            return VolatilityRegime.ELEVATED
        if v <= c.vol_low:
            return VolatilityRegime.LOW
        return VolatilityRegime.NORMAL

    def _liquidity(self, row: dict) -> LiquidityRegime:
        c = self.config
        spread_bps = row.get("spread_bps")
        rel_vol = row.get("rel_volume")
        if spread_bps is not None and spread_bps >= c.spread_insufficient_bps:
            return LiquidityRegime.INSUFFICIENT
        if rel_vol is not None and rel_vol < c.min_rel_volume:
            return LiquidityRegime.DEGRADED
        if spread_bps is not None and spread_bps >= c.spread_degraded_bps:
            return LiquidityRegime.DEGRADED
        return LiquidityRegime.NORMAL

    def _confidence(self, score: float, row: dict) -> float:
        """Heuristic agreement score in [0,1] — NOT a calibrated probability.
        Blends |directional_score| magnitude with ADX strength."""
        s = 0.0 if score != score else min(1.0, abs(score) / 100.0)
        adx = row.get("adx", 0.0) or 0.0
        a = min(1.0, max(0.0, adx / 50.0)) if adx == adx else 0.0
        return round(0.6 * s + 0.4 * a, 4)

    # -- hysteresis for the direction dimension ------------------------------
    def _apply_hysteresis(self, scope: str, instant: RegimeDirection) -> tuple:
        st = self._states.setdefault(scope, _ScopeState())
        prev_active = st.active
        if instant == st.active:
            st.candidate = instant
            st.streak = 0
        elif st.active == RegimeDirection.UNCERTAIN:
            # Leaving the unknown baseline requires no persistence — adopt the first
            # clear reading immediately. Persistence only dampens flips BETWEEN
            # established (concrete) directions.
            st.active = instant
            st.candidate = instant
            st.streak = 0
        elif instant == st.candidate:
            st.streak += 1
            if st.streak >= self.config.min_persistence_updates:
                st.active = instant
                st.streak = 0
        else:
            st.candidate = instant
            st.streak = 1
        return st.active, prev_active

    # -- public assessment ---------------------------------------------------
    def assess_instrument(self, symbol: str, row: dict, *, now_ns: int,
                          level: RegimeLevel = RegimeLevel.INSTRUMENT,
                          underlying_score: Optional[float] = None,
                          regime_cfg: Optional[amps.RegimeConfig] = None) -> RegimeAssessment:
        """Assess one instrument/benchmark/sector from a feature ``row`` (keys:
        pdi/mdi/adx/vwap_dist/bb_dist/short_mom plus optional bb_width,
        atr_pct, atr_expansion_ratio, realized_vol, momentum, momentum_accel,
        rel_volume, spread_bps). ``underlying_score`` (for inverse/geared ETFs) is
        the directional score of the UNDERLYING benchmark, reported separately."""
        rc = regime_cfg or amps.RegimeConfig()
        drow = {
            "pdi": row.get("pdi"), "mdi": row.get("mdi"),
            "adx": row.get("adx", 0.0) or 0.0,
            "vwap_dist": row.get("vwap_dist", 0.0) or 0.0,
            "bb_dist": row.get("bb_dist", 0.0) or 0.0,
            "short_mom": row.get("short_mom", row.get("momentum", 0.0)) or 0.0,
        }
        score = amps.directional_score(drow, rc)
        adx = row.get("adx", float("nan"))
        instant_dir = self._direction_from_score(score, adx)
        active_dir, prev_active = self._apply_hysteresis(f"{level.value}:{scope_key(symbol)}",
                                                         instant_dir)
        underlying_dir = None
        if underlying_score is not None:
            underlying_dir = self._direction_from_score(underlying_score,
                                                        row.get("underlying_adx", float("nan")))
        dims = RegimeDimensions(
            direction=active_dir,
            trend=self._trend(row),
            volatility=self._volatility(row),
            liquidity=self._liquidity(row),
            confidence=self._confidence(score, row),
            directional_score=score,
            underlying_direction=underlying_dir)
        age = now_ns - int(row.get("observation_ns", now_ns))
        stale = age > self.config.max_age_ns or age < 0
        return RegimeAssessment(
            level=level, scope=scope_key(symbol), dimensions=dims, as_of_ns=now_ns,
            valid=not stale, stale=stale,
            transitioned=(active_dir != prev_active),
            previous_direction=prev_active,
            inputs_used=sorted(k for k in row if row.get(k) is not None))

    def assess_broad(self, ref_rows: dict[str, dict], breadth: Optional[dict] = None,
                     *, now_ns: int, regime_cfg: Optional[amps.RegimeConfig] = None
                     ) -> RegimeAssessment:
        """Broad-market assessment from the benchmark reference rows (SPY/QQQ/DIA/
        IWM) plus an optional GPU breadth snapshot. Direction is the mean benchmark
        directional score, nudged by breadth participation when available."""
        rc = regime_cfg or amps.RegimeConfig()
        scores = []
        for row in ref_rows.values():
            drow = {"pdi": row.get("pdi"), "mdi": row.get("mdi"),
                    "adx": row.get("adx", 0.0) or 0.0,
                    "vwap_dist": row.get("vwap_dist", 0.0) or 0.0,
                    "bb_dist": row.get("bb_dist", 0.0) or 0.0,
                    "short_mom": row.get("short_mom", 0.0) or 0.0}
            s = amps.directional_score(drow, rc)
            if s == s:
                scores.append(s)
        score = sum(scores) / len(scores) if scores else float("nan")
        # breadth participation tilt: (bullish-bearish)/valid in [-1,1] -> +/-15 pts
        if breadth and breadth.get("available") and breadth.get("valid"):
            valid = max(1, int(breadth["valid"]))
            tilt = (int(breadth.get("bullish", 0)) - int(breadth.get("bearish", 0))) / valid
            if score == score:
                score = max(-100.0, min(100.0, score + 15.0 * tilt))
        avg_adx = (sum(r.get("adx", 0.0) or 0.0 for r in ref_rows.values())
                   / max(1, len(ref_rows)))
        instant_dir = self._direction_from_score(score, avg_adx)
        active_dir, prev_active = self._apply_hysteresis("broad:MARKET", instant_dir)
        # volatility/trend/liquidity for broad: aggregate from refs
        agg = {"adx": avg_adx,
               "realized_vol": _mean_opt(ref_rows, "realized_vol"),
               "atr_pct": _mean_opt(ref_rows, "atr_pct"),
               "bb_width": _mean_opt(ref_rows, "bb_width"),
               "momentum": _mean_opt(ref_rows, "short_mom"),
               "rel_volume": _mean_opt(ref_rows, "rel_volume")}
        dims = RegimeDimensions(
            direction=active_dir, trend=self._trend(agg),
            volatility=self._volatility(agg), liquidity=self._liquidity(agg),
            confidence=self._confidence(score, {"adx": avg_adx}),
            directional_score=score)
        return RegimeAssessment(
            level=RegimeLevel.BROAD, scope="MARKET", dimensions=dims, as_of_ns=now_ns,
            transitioned=(active_dir != prev_active), previous_direction=prev_active,
            inputs_used=sorted(ref_rows.keys()) + (["breadth"] if breadth else []))


def scope_key(symbol: str) -> str:
    return str(symbol).upper().strip()


def _mean_opt(rows: dict[str, dict], key: str) -> Optional[float]:
    vals = [r[key] for r in rows.values() if r.get(key) is not None and r[key] == r[key]]
    return sum(vals) / len(vals) if vals else None
