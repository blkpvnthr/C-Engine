"""Composite candidate ranking for the real-time scanner.

Blends the factors from the approved Section-4 list into a single [0,1] score
and applies the hard liquidity/tradability filters that decide ENTRY-universe
membership. Reuses the engine's own ``amps_v4_regime.directional_score`` for
relative strength and ``universe.classify_symbol`` for direction-aware regime
alignment (inverse ETFs favored in bearish regimes). Every factor is bounded;
missing order-flow contributes 0 (never fabricated as bullish).

This module is pure and deterministic over pre-computed ``CandidateFeatures``;
the orchestrator is responsible for building those features from IEX bars/stream.
The scores are honest heuristic rankings, NOT validated alpha.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import amps_v4_regime
import universe
from market_scan.config import MarketScanConfig


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


@dataclass
class CandidateFeatures:
    symbol: str
    price: Optional[float] = None
    avg_daily_volume: Optional[float] = None          # historical ADV (shares)
    avg_daily_dollar_volume: Optional[float] = None   # historical ADDV ($)
    rel_volume: Optional[float] = None                # intraday volume expansion (1.0 = normal)
    spread_pct: Optional[float] = None                # bid/ask spread / mid (fraction)
    quote_age_s: Optional[float] = None
    directional_row: dict = field(default_factory=dict)  # pdi/mdi/adx/vwap_dist/short_mom/bb_dist
    snapshot: Any = None                              # OrderFlowSnapshot or None
    vwap_dist: Optional[float] = None                 # + = above session VWAP
    bb_mid_dist: Optional[float] = None               # + = above Bollinger midpoint


@dataclass
class MarketContext:
    regime_label: str = ""
    regime_score: float = 0.0                         # [-100,100] broad-market directional score
    spy_score: float = 0.0
    qqq_score: float = 0.0
    cfg: Optional[amps_v4_regime.RegimeConfig] = None


@dataclass
class RankWeights:
    rel_strength: float = 0.30
    momentum: float = 0.15
    volume_expansion: float = 0.10
    liquidity: float = 0.10
    order_flow: float = 0.20
    regime_align: float = 0.15


@dataclass
class RankedCandidate:
    symbol: str
    score: float
    eligible: bool
    reason: str
    detail: dict = field(default_factory=dict)


def passes_liquidity(f: CandidateFeatures, config: MarketScanConfig) -> tuple[bool, str]:
    """Hard filters for ENTRY-universe membership. Historical price/volume are
    required; live spread/quote-age are checked only when present (the entry-time
    gate re-verifies them against the live quote)."""
    if f.price is None or f.price < config.min_price:
        return False, "price"
    if f.avg_daily_volume is None or f.avg_daily_volume < config.min_avg_daily_volume:
        return False, "adv"
    if (f.avg_daily_dollar_volume is None
            or f.avg_daily_dollar_volume < config.min_avg_daily_dollar_volume):
        return False, "addv"
    if f.spread_pct is not None and f.spread_pct > config.max_spread_pct:
        return False, "spread"
    if f.quote_age_s is not None and f.quote_age_s > config.max_quote_age_seconds:
        return False, "quote_age"
    return True, "ok"


def _direction_sign(symbol: str) -> float:
    """+1 for long instruments, -1 for inverse ETFs (negative leverage)."""
    try:
        lev = universe.classify_symbol(symbol).leverage
    except Exception:  # noqa: BLE001
        lev = 1.0
    return -1.0 if lev < 0 else 1.0


def _order_flow_component(snapshot: Any) -> float:
    """[0,1] from live order flow. 0 when no usable snapshot (never bullish)."""
    if snapshot is None or not getattr(snapshot, "ok", False):
        return 0.0
    buy_p = float(getattr(snapshot, "buy_pressure", 0.0) or 0.0)
    cvd_tr = max(0.0, float(getattr(snapshot, "cvd_trend", 0.0) or 0.0))
    se = float(getattr(snapshot, "seller_exhaustion", 0.0) or 0.0)
    se_l1 = float(getattr(snapshot, "seller_exhaustion_l1", 0.0) or 0.0)
    exhaustion = max(se, se_l1)
    return _clip01(0.45 * buy_p + 0.25 * cvd_tr + 0.30 * exhaustion)


def score_candidate(f: CandidateFeatures, market: MarketContext,
                    weights: Optional[RankWeights] = None) -> tuple[float, dict]:
    """Composite [0,1] score. Returns (score, per-factor detail)."""
    w = weights or RankWeights()
    cfg = market.cfg or amps_v4_regime.RegimeConfig()

    # Relative strength vs the market (SPY/QQQ): directional score minus market avg.
    rs = amps_v4_regime.directional_score(f.directional_row, cfg)
    if rs != rs:  # NaN -> neutral-low
        rel_strength = 0.3
    else:
        market_avg = 0.5 * (market.spy_score + market.qqq_score)
        rel_strength = _clip01(0.5 + (rs - market_avg) / 200.0)

    # Momentum from short_mom + structure (above VWAP / BB-mid).
    short_mom = float(f.directional_row.get("short_mom", 0.0) or 0.0)
    struct = 0.0
    if (f.vwap_dist or 0.0) > 0:
        struct += 0.5
    if (f.bb_mid_dist or 0.0) > 0:
        struct += 0.5
    momentum = _clip01(0.5 * (0.5 + 0.5 * max(-1.0, min(1.0, short_mom * 20.0))) + 0.5 * struct)

    # Intraday volume expansion.
    rv = f.rel_volume if f.rel_volume is not None else 1.0
    volume_expansion = _clip01((rv - 0.5) / 1.5)

    # Liquidity: dollar volume (diminishing) + spread tightness.
    addv = float(f.avg_daily_dollar_volume or 0.0)
    addv_norm = addv / (addv + 1.0e8) if addv > 0 else 0.0
    if f.spread_pct is None:
        spread_norm = 0.5
    else:
        spread_norm = 1.0 - _clip01(f.spread_pct / max(1e-9, 0.02))
    liquidity = _clip01(0.6 * addv_norm + 0.4 * spread_norm)

    order_flow = _order_flow_component(f.snapshot)

    # Direction-aware regime alignment (inverse ETFs favored when regime bearish).
    dir_sign = _direction_sign(f.symbol)
    regime_align = _clip01(0.5 + 0.5 * max(-1.0, min(1.0, (market.regime_score * dir_sign) / 100.0)))

    score = (w.rel_strength * rel_strength + w.momentum * momentum
             + w.volume_expansion * volume_expansion + w.liquidity * liquidity
             + w.order_flow * order_flow + w.regime_align * regime_align)
    total_w = (w.rel_strength + w.momentum + w.volume_expansion + w.liquidity
               + w.order_flow + w.regime_align)
    score = _clip01(score / total_w if total_w > 0 else 0.0)

    detail = {
        "rel_strength": round(rel_strength, 4), "momentum": round(momentum, 4),
        "volume_expansion": round(volume_expansion, 4), "liquidity": round(liquidity, 4),
        "order_flow": round(order_flow, 4), "regime_align": round(regime_align, 4),
        "directional_score": None if rs != rs else round(rs, 2),
    }
    return score, detail


def rank(features, *, config: MarketScanConfig, market: MarketContext,
         weights: Optional[RankWeights] = None) -> list[RankedCandidate]:
    """Score + filter + sort. Eligible candidates lead, each group by score desc."""
    out: list[RankedCandidate] = []
    for f in features:
        eligible, reason = passes_liquidity(f, config)
        score, detail = score_candidate(f, market, weights)
        out.append(RankedCandidate(f.symbol, score, eligible, reason, detail))
    out.sort(key=lambda r: (r.eligible, r.score), reverse=True)
    return out
