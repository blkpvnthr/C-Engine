#!/usr/bin/env python3
"""
amps_v4_regime.py -- AMPS V2/V4 market-regime + underlying-momentum + directional-eligibility.

Core idea (the new value): only go LONG a leveraged/inverse ETF when (a) the ETF's OWN AMPS
long conditions pass AND (b) the ETF's UNDERLYING momentum is aligned with the ETF's orientation
-- a bullish ETF requires a bullish underlying; an INVERSE ETF requires a BEARISH underlying
(because the inverse ETF itself rises when the underlying falls). The ETF's own indicators must
be bullish for a long regardless of orientation.

Reference instruments (SPY/QQQ/DIA/IWM/SMH/NVDA/TSLA/AAPL/AMZN/MSFT) are used ONLY for signal
generation; they are never added to the executable trade universe.

Pure + testable: directional score, regime classification, hysteresis, eligibility. No I/O.
Reuses the feature rows produced by amps_v2_backtest.compute_features (pdi/mdi/adx/bb/vwap/...).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

# ---------------------------------------------------------------- instrument registry (sec 3)
# dir: "bull" (long underlying) | "bear" (inverse). lev: signed leverage. SMH proxies the
# semiconductor benchmark for SOXL/SOXS (documented; SOXX is an acceptable alternative).
REGISTRY: dict[str, dict] = {
    "UPRO": {"ref": "SPY", "dir": "bull", "lev": 3, "cat": "index", "pair": "SPY"},
    "SPXU": {"ref": "SPY", "dir": "bear", "lev": -3, "cat": "index", "pair": "SPY"},
    "TQQQ": {"ref": "QQQ", "dir": "bull", "lev": 3, "cat": "index", "pair": "QQQ"},
    "SQQQ": {"ref": "QQQ", "dir": "bear", "lev": -3, "cat": "index", "pair": "QQQ"},
    "UDOW": {"ref": "DIA", "dir": "bull", "lev": 3, "cat": "index", "pair": "DIA"},
    "SDOW": {"ref": "DIA", "dir": "bear", "lev": -3, "cat": "index", "pair": "DIA"},
    "TNA": {"ref": "IWM", "dir": "bull", "lev": 3, "cat": "index", "pair": "IWM"},
    "TZA": {"ref": "IWM", "dir": "bear", "lev": -3, "cat": "index", "pair": "IWM"},
    "SOXL": {"ref": "SMH", "dir": "bull", "lev": 3, "cat": "semi", "pair": "SMH"},
    "SOXS": {"ref": "SMH", "dir": "bear", "lev": -3, "cat": "semi", "pair": "SMH"},
    "NVDL": {"ref": "NVDA", "dir": "bull", "lev": 2, "cat": "single", "pair": "NVDA"},
    "NVDD": {"ref": "NVDA", "dir": "bear", "lev": -1, "cat": "single", "pair": "NVDA"},
    "TSLL": {"ref": "TSLA", "dir": "bull", "lev": 2, "cat": "single", "pair": "TSLA"},
    "TSLQ": {"ref": "TSLA", "dir": "bear", "lev": -1, "cat": "single", "pair": "TSLA"},
    "AAPU": {"ref": "AAPL", "dir": "bull", "lev": 2, "cat": "single", "pair": "AAPL"},
    "AAPD": {"ref": "AAPL", "dir": "bear", "lev": -1, "cat": "single", "pair": "AAPL"},
    "AMZU": {"ref": "AMZN", "dir": "bull", "lev": 2, "cat": "single", "pair": "AMZN"},
    "AMZD": {"ref": "AMZN", "dir": "bear", "lev": -1, "cat": "single", "pair": "AMZN"},
    "MSFU": {"ref": "MSFT", "dir": "bull", "lev": 2, "cat": "single", "pair": "MSFT"},
    "MSFD": {"ref": "MSFT", "dir": "bear", "lev": -1, "cat": "single", "pair": "MSFT"},
}
TRADE_UNIVERSE = list(REGISTRY.keys())
REFERENCES = sorted({v["ref"] for v in REGISTRY.values()})

# regime labels
CONFIRMED_BULLISH = "CONFIRMED_BULLISH"
EMERGING_BULLISH = "EMERGING_BULLISH"
CONFIRMED_BEARISH = "CONFIRMED_BEARISH"
EMERGING_BEARISH = "EMERGING_BEARISH"
SIDEWAYS = "SIDEWAYS"
HIGH_VOLATILITY = "HIGH_VOLATILITY"
UNCERTAIN = "UNCERTAIN"


@dataclass
class RegimeConfig:
    weights: dict = field(default_factory=lambda: {
        "vwap_pos": 0.20, "di_sep": 0.20, "adx_strength": 0.15, "short_mom": 0.15,
        "vwap_slope": 0.10, "bb_align": 0.10, "cross_index": 0.10})
    confirmed_thresh: float = 60.0
    emerging_thresh: float = 30.0
    min_adx_trend: float = 20.0
    high_vol_realized: float = 0.004       # per-bar realized vol -> HIGH_VOLATILITY flag
    underlying_align_thresh: float = 30.0  # |score| needed for directional alignment
    divergence_thresh: float = 50.0        # stronger confidence needed for a divergence override
    # hysteresis
    min_confirm_bars: int = 2              # regime must persist N decision bars before flipping
    activation: float = 35.0
    deactivation: float = 20.0


def _clamp100(x: float) -> float:
    return max(-100.0, min(100.0, float(x)))


def directional_score(row: dict, cfg: RegimeConfig, cross_index_agree: float = 0.0) -> float:
    """Directional score in [-100, +100] for one reference instrument from a feature row.
    ADX only scales the DI-direction strength (it does NOT set direction itself)."""
    pdi = row.get("pdi"); mdi = row.get("mdi"); adx = row.get("adx")
    if pdi is None or mdi is None or pdi != pdi or mdi != mdi:
        return float("nan")
    vwap_dist = row.get("vwap_dist", 0.0) or 0.0
    vwap_slope = row.get("vwap_slope", 0.0) or 0.0
    bb_dist = row.get("bb_dist", 0.0) or 0.0
    short_mom = row.get("short_mom", 0.0) or 0.0
    di_sep = pdi - mdi
    w = cfg.weights
    vwap_pos = 1.0 if vwap_dist > 0 else (-1.0 if vwap_dist < 0 else 0.0)
    di_dir = max(-1.0, min(1.0, di_sep / 25.0))
    adx_conf = max(0.0, min(1.0, ((adx or 0) - cfg.min_adx_trend) / 15.0))
    adx_strength = di_dir * adx_conf         # ADX scales magnitude, DI sets sign
    mom = max(-1.0, min(1.0, short_mom / (abs(short_mom) + 1e-9))) if short_mom else 0.0
    vslope = 1.0 if vwap_slope > 0 else (-1.0 if vwap_slope < 0 else 0.0)
    bb_align = 1.0 if bb_dist > 0 else (-1.0 if bb_dist < 0 else 0.0)
    score = 100.0 * (w["vwap_pos"] * vwap_pos + w["di_sep"] * di_dir + w["adx_strength"] * adx_strength
                     + w["short_mom"] * mom + w["vwap_slope"] * vslope + w["bb_align"] * bb_align
                     + w["cross_index"] * max(-1.0, min(1.0, cross_index_agree)))
    return _clamp100(score)


def classify_regime(score: float, realized_vol: float, cfg: RegimeConfig,
                    disagreement: float = 0.0) -> str:
    if score != score:
        return UNCERTAIN
    if realized_vol is not None and realized_vol == realized_vol and realized_vol > cfg.high_vol_realized:
        return HIGH_VOLATILITY
    if score >= cfg.confirmed_thresh:
        return CONFIRMED_BULLISH
    if score >= cfg.emerging_thresh:
        return EMERGING_BULLISH
    if score <= -cfg.confirmed_thresh:
        return CONFIRMED_BEARISH
    if score <= -cfg.emerging_thresh:
        return EMERGING_BEARISH
    # near zero: sideways if trend-coherent, else uncertain
    return SIDEWAYS if disagreement < 0.5 else UNCERTAIN


def market_regime(ref_rows: dict[str, dict], cfg: RegimeConfig) -> tuple[str, float, float]:
    """Broad-market regime from SPY/QQQ (primary) + DIA/IWM (confirmation). Returns
    (label, score, confidence). Confidence = cross-index agreement."""
    scores = {}
    for r in ("SPY", "QQQ", "DIA", "IWM"):
        row = ref_rows.get(r)
        if row:
            s = directional_score(row, cfg)
            if s == s:
                scores[r] = s
    if "SPY" not in scores and "QQQ" not in scores:
        return UNCERTAIN, float("nan"), 0.0
    prim = [scores[r] for r in ("SPY", "QQQ") if r in scores]
    conf_sources = list(scores.values())
    blended = (0.7 * (sum(prim) / len(prim)) + 0.3 * (sum(conf_sources) / len(conf_sources))) if prim else (sum(conf_sources) / len(conf_sources))
    signs = [1 if s > 0 else -1 for s in conf_sources]
    agree = abs(sum(signs)) / len(signs) if signs else 0.0  # 1.0 = all agree
    rv = max((ref_rows.get(r, {}).get("realized_vol", 0.0) or 0.0) for r in scores) if scores else 0.0
    label = classify_regime(blended, rv, cfg, disagreement=1.0 - agree)
    return label, round(blended, 2), round(agree, 3)


@dataclass
class RegimeState:
    """Hysteresis: a candidate label must persist `min_confirm_bars` before it becomes active."""
    active_label: str = UNCERTAIN
    candidate_label: str = UNCERTAIN
    candidate_count: int = 0
    bars_in_active: int = 0

    def update(self, new_label: str, cfg: RegimeConfig) -> str:
        if new_label == self.active_label:
            self.candidate_label = new_label; self.candidate_count = 0
            self.bars_in_active += 1
            return self.active_label
        if new_label == self.candidate_label:
            self.candidate_count += 1
        else:
            self.candidate_label = new_label; self.candidate_count = 1
        if self.candidate_count >= cfg.min_confirm_bars:
            self.active_label = new_label; self.candidate_count = 0; self.bars_in_active = 1
        return self.active_label


def underlying_aligned(score: float, orientation: str, cfg: RegimeConfig) -> bool:
    """A bull ETF needs a bullish underlying (score >= +thresh); a bear ETF needs a bearish
    underlying (score <= -thresh)."""
    if score != score:
        return False
    if orientation == "bull":
        return score >= cfg.underlying_align_thresh
    return score <= -cfg.underlying_align_thresh


def regime_permits(symbol: str, regime_label: str, underlying_score: float, cfg: RegimeConfig
                   ) -> tuple[bool, str]:
    """Does the broad regime + underlying permit a long in this ETF? Divergence override allowed
    with stronger underlying confidence. Returns (ok, reason)."""
    orient = REGISTRY[symbol]["dir"]
    aligned = underlying_aligned(underlying_score, orient, cfg)
    if regime_label == UNCERTAIN:
        return False, "regime_uncertain"
    if regime_label == HIGH_VOLATILITY:
        return False, "high_volatility_no_entry"
    # regime-aligned path
    bull_regime = regime_label in (CONFIRMED_BULLISH, EMERGING_BULLISH)
    bear_regime = regime_label in (CONFIRMED_BEARISH, EMERGING_BEARISH)
    if orient == "bull" and bull_regime and aligned:
        return True, "regime_aligned_bull"
    if orient == "bear" and bear_regime and aligned:
        return True, "regime_aligned_bear"
    if regime_label == SIDEWAYS:
        # only exceptional high-confidence, underlying-aligned opportunities
        if aligned and abs(underlying_score) >= cfg.divergence_thresh:
            return True, "sideways_high_confidence"
        return False, "sideways_low_confidence"
    # divergence override: underlying strongly disagrees with the broad regime
    if aligned and abs(underlying_score) >= cfg.divergence_thresh:
        return True, "divergence_override"
    return False, "regime_underlying_misaligned"


def eligible_for_long(symbol: str, etf_amps_gates_ok: bool, underlying_score: float,
                      regime_label: str, cfg: RegimeConfig) -> tuple[bool, str]:
    """Final directional eligibility for a LONG entry in `symbol`.
    Requires the ETF's OWN AMPS long gates AND underlying-directional alignment AND regime."""
    if symbol not in REGISTRY:
        return False, "unknown_symbol"
    if not etf_amps_gates_ok:
        return False, "etf_amps_gates_fail"
    permit, reason = regime_permits(symbol, regime_label, underlying_score, cfg)
    return permit, reason
