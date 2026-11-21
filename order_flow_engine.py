#!/usr/bin/env python3
"""
order_flow_engine.py -- real-time order-flow metrics from Alpaca IEX trades and quotes.

PURPOSE
    Detect buyer/seller aggression, momentum shifts, and exhaustion to INFORM EXITS
    (see profit_exit_engine.py). This module only observes and classifies; it never
    places, modifies, or cancels an order.

HONESTY / DATA SCOPE (important -- do not misrepresent)
    * Alpaca's free tier is the IEX feed: a SINGLE exchange, not the consolidated SIP
      tape. IEX is a small, variable share of total US volume. Every metric here is
      therefore an IEX-ONLY proxy for market-wide order flow, labelled data_source="IEX".
    * Quotes are LEVEL 1 (best bid/ask + their sizes), NOT full order-book depth. Quote
      imbalance here is an L1 signal only.
    These are deliberately surfaced in the snapshot (`data_source`, `caveat`) so no
    downstream consumer treats IEX executions as consolidated activity or L1 as depth.

WHAT IT COMPUTES (per symbol)
    * Aggressive buy/sell volume via timestamp-aligned quote classification (Lee-Ready).
    * Rolling signed volume delta over 10 / 30 / 60 / 300 s.
    * Cumulative Volume Delta (CVD) and CVD-vs-price divergence.
    * Executed order-flow imbalance, buy/sell ratio.
    * Level-1 quote imbalance, spread (abs + bps) and a liquidity classification.
    * Buying/selling pressure classification and interpretable [0,1] signals
      (sell_pressure, buy_pressure, cvd_deterioration, buyer_exhaustion) for the exit engine.

ROBUSTNESS
    Stale-data detection, duplicate protection, out-of-order tolerance, bounded memory
    (time-trimmed deques). The REST/stream adapters are fail-safe: any error yields an
    INSUFFICIENT_DATA snapshot rather than raising into the trading loop.
"""
from __future__ import annotations

import logging
import math
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Any, Deque, Iterable, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

DATA_SOURCE = "IEX"
_CAVEAT = ("IEX-only single-exchange tape (not consolidated SIP); quotes are Level 1 "
           "best bid/ask, not full depth. Treat as a proxy, not market-wide volume.")


# ======================================================================
# Config
# ======================================================================
@dataclass
class OrderFlowConfig:
    windows_s: Tuple[int, ...] = (10, 30, 60, 300)
    cvd_window_s: int = 300                 # series retained for divergence/peak tracking
    divergence_lookback_s: int = 120        # window for price-vs-CVD divergence
    max_trades: int = 20000                 # hard memory bound per symbol
    max_quotes: int = 20000
    retain_s: int = 360                     # trim events older than this (>= max window)
    stale_trade_s: float = 15.0             # no trade within this -> stale
    stale_quote_s: float = 15.0
    min_trades_for_signal: int = 8          # below this -> INSUFFICIENT_DATA
    wide_spread_bps: float = 25.0           # above this -> illiquid/caution
    out_of_order_tolerance_s: float = 2.0   # accept slightly-late events up to this lag
    max_subscriptions: int = 40             # cap on symbols polled per cycle (IEX fair-use)
    # --- L1 microstructure primitives (seller-exhaustion extension) ---
    climax_short_s: float = 10.0            # short window for a volume spike
    climax_baseline_s: float = 300.0        # baseline window the spike is measured against
    climax_full_ratio: float = 3.0          # short vol >= Nx its expected share -> full climax
    replenish_window_s: float = 30.0        # window for bid/ask size replenishment
    absorption_hold_tol_pct: float = 0.002  # price move still counted as "held" (0.2%)


# ======================================================================
# Snapshot
# ======================================================================
@dataclass
class OrderFlowSnapshot:
    symbol: str
    ts: str                                  # ISO time of snapshot
    data_source: str = DATA_SOURCE
    caveat: str = _CAVEAT
    ok: bool = False                         # enough data to be meaningful
    status: str = "INSUFFICIENT_DATA"        # OK | STALE | INSUFFICIENT_DATA
    is_stale: bool = True
    last_trade_age_s: Optional[float] = None
    last_quote_age_s: Optional[float] = None
    n_trades_60s: int = 0

    # raw microstructure
    bid: Optional[float] = None
    ask: Optional[float] = None
    mid: Optional[float] = None
    spread: Optional[float] = None
    spread_bps: Optional[float] = None
    liquidity: str = "unknown"               # tight | normal | wide | unknown

    buy_vol_60s: float = 0.0
    sell_vol_60s: float = 0.0
    delta_10s: float = 0.0
    delta_30s: float = 0.0
    delta_60s: float = 0.0
    delta_300s: float = 0.0
    cvd: float = 0.0
    cvd_peak: float = 0.0
    cvd_trough: float = 0.0
    cvd_trend: float = 0.0                    # [-1,1] direction of CVD over the window
    executed_imbalance: float = 0.0          # [-1,1] (buy-sell)/(buy+sell) over 60s
    buy_sell_ratio: float = 1.0
    quote_imbalance: float = 0.0             # [-1,1] (bidsz-asksz)/(bidsz+asksz) L1
    cvd_divergence: float = 0.0              # <0 bearish (price up, CVD down)
    divergence_flag: str = "none"            # bearish | bullish | none

    pressure: str = "NEUTRAL"                # STRONG_BUY|BUY|NEUTRAL|SELL|STRONG_SELL
    pressure_score: float = 0.0              # [-1,1] (+ buy, - sell)

    # interpretable [0,1] signals consumed by the exit engine (sell-side) and the
    # buy-opportunity engine (buy-side). All normalized; missing data -> 0 (never bullish).
    sell_pressure: float = 0.0
    buy_pressure: float = 0.0
    cvd_deterioration: float = 0.0
    buyer_exhaustion: float = 0.0
    cvd_improvement: float = 0.0             # buy-side mirror of cvd_deterioration
    seller_exhaustion: float = 0.0          # buy-side mirror of buyer_exhaustion

    # L1 microstructure primitives (IEX best bid/ask only; NOT depth). Honest
    # heuristic proxies that extend the CVD-based exhaustion signals above. All
    # bounded; missing data -> neutral/0 (never fabricated as a signal).
    microprice: Optional[float] = None       # size-weighted fair price
    microprice_tilt: float = 0.0             # [-1,1] + = upward (buy) pressure vs mid
    absorption: float = 0.0                  # [0,1] net selling soaked up w/o price decline
    bid_replenishment: float = 0.0           # [0,1] resting bid size refilling after dips
    ask_replenishment: float = 0.0           # [0,1] resting ask size refilling after dips
    volume_climax: float = 0.0               # [0,1] short-window volume spike vs baseline
    volume_climax_side: int = 0              # -1 sell climax | +1 buy climax | 0 none
    seller_exhaustion_l1: float = 0.0        # [0,1] L1 composite (absorption+replenish+tilt)
    buyer_exhaustion_l1: float = 0.0         # [0,1] L1 mirror

    def as_dict(self) -> dict[str, Any]:
        from dataclasses import asdict
        return asdict(self)


# ======================================================================
# Pure classification / metric functions (unit-tested, no state)
# ======================================================================
def classify_trade(price: float, bid: Optional[float], ask: Optional[float],
                   last_tick_sign: int = 0) -> int:
    """Lee-Ready trade sign: +1 aggressive buy, -1 aggressive sell, 0 unknown.

    Quote rule first (lifted offer / hit bid), then midpoint, then tick fallback.
    """
    if price <= 0:
        return 0
    if bid is not None and ask is not None and ask > 0 and bid > 0 and ask >= bid:
        if price >= ask:
            return 1
        if price <= bid:
            return -1
        mid = 0.5 * (bid + ask)
        if price > mid:
            return 1
        if price < mid:
            return -1
        return last_tick_sign if last_tick_sign in (1, -1) else 0
    # No usable quote -> fall back to tick direction.
    return last_tick_sign if last_tick_sign in (1, -1) else 0


def signed_delta(events: Sequence[Tuple[float, int, float]], now: float, window_s: float) -> float:
    """Sum signed volume (sign*size) within [now-window_s, now]. events: (ts, sign, size)."""
    lo = now - window_s
    return float(sum(sign * size for ts, sign, size in events if lo <= ts <= now))


def buy_sell_volumes(events: Sequence[Tuple[float, int, float]], now: float,
                     window_s: float) -> Tuple[float, float]:
    lo = now - window_s
    buy = sell = 0.0
    for ts, sign, size in events:
        if ts < lo or ts > now:
            continue
        if sign > 0:
            buy += size
        elif sign < 0:
            sell += size
    return buy, sell


def executed_imbalance(buy_vol: float, sell_vol: float) -> float:
    tot = buy_vol + sell_vol
    if tot <= 0:
        return 0.0
    return max(-1.0, min(1.0, (buy_vol - sell_vol) / tot))


def buy_sell_ratio(buy_vol: float, sell_vol: float) -> float:
    if sell_vol <= 0:
        return float("inf") if buy_vol > 0 else 1.0
    return buy_vol / sell_vol


def quote_imbalance(bid_size: float, ask_size: float) -> float:
    tot = bid_size + ask_size
    if tot <= 0:
        return 0.0
    return max(-1.0, min(1.0, (bid_size - ask_size) / tot))


def spread_metrics(bid: float, ask: float) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Return (mid, spread, spread_bps) or (None,None,None) if invalid."""
    if bid is None or ask is None or bid <= 0 or ask <= 0 or ask < bid:
        return None, None, None
    mid = 0.5 * (bid + ask)
    spread = ask - bid
    spread_bps = (spread / mid) * 1e4 if mid > 0 else None
    return mid, spread, spread_bps


def cvd_divergence(price_series: Sequence[Tuple[float, float]],
                   cvd_series: Sequence[Tuple[float, float]], now: float,
                   lookback_s: float) -> float:
    """Signed divergence over the lookback window, in [-1, 1].

    Compares the direction of price vs the direction of CVD. Negative = bearish
    divergence (price up while CVD flat/down -> distribution / buyer exhaustion);
    positive = bullish divergence. 0 when aligned or insufficient data.
    """
    lo = now - lookback_s
    ps = [(t, v) for t, v in price_series if t >= lo]
    cs = [(t, v) for t, v in cvd_series if t >= lo]
    if len(ps) < 3 or len(cs) < 3:
        return 0.0
    p0, p1 = ps[0][1], ps[-1][1]
    c0, c1 = cs[0][1], cs[-1][1]
    if p0 <= 0:
        return 0.0
    price_chg = (p1 - p0) / p0
    cvd_range = max(abs(c1 - c0), 1.0)
    cvd_dir = (c1 - c0) / cvd_range  # normalized direction of CVD move
    # Divergence present only when the two disagree in sign.
    price_dir = 1.0 if price_chg > 0 else (-1.0 if price_chg < 0 else 0.0)
    if price_dir == 0.0:
        return 0.0
    if (price_dir > 0 and cvd_dir < 0) or (price_dir < 0 and cvd_dir > 0):
        # magnitude scaled by how strongly price moved and CVD contradicted
        mag = min(1.0, abs(price_chg) * 50.0) * min(1.0, abs(cvd_dir))
        return max(-1.0, min(1.0, -price_dir * mag))  # bearish when price up
    return 0.0


def classify_pressure(executed_imb: float, delta_60s: float, q_imb: float) -> Tuple[str, float]:
    """Blend executed imbalance, short delta sign and L1 quote imbalance into a
    pressure label + score in [-1,1] (+ buy, - sell)."""
    delta_sign = 1.0 if delta_60s > 0 else (-1.0 if delta_60s < 0 else 0.0)
    score = 0.55 * executed_imb + 0.30 * (0.5 * q_imb) + 0.15 * delta_sign
    score = max(-1.0, min(1.0, score))
    if score >= 0.5:
        return "STRONG_BUY", score
    if score >= 0.15:
        return "BUY", score
    if score <= -0.5:
        return "STRONG_SELL", score
    if score <= -0.15:
        return "SELL", score
    return "NEUTRAL", score


def norm_sell_pressure(executed_imb: float, q_imb: float, delta_60s: float, buy_vol: float,
                       sell_vol: float) -> float:
    """[0,1]: 0 = all buying, 1 = heavy selling. Blends executed + quote imbalance."""
    sell_share = 0.0
    tot = buy_vol + sell_vol
    if tot > 0:
        sell_share = sell_vol / tot
    s = 0.6 * sell_share + 0.25 * (0.5 - 0.5 * q_imb) + 0.15 * (1.0 if delta_60s < 0 else 0.0)
    return max(0.0, min(1.0, s))


def norm_cvd_deterioration(cvd: float, cvd_peak: float, delta_30s: float, delta_300s: float) -> float:
    """[0,1]: how far CVD has given back from its peak + near-term delta turning negative."""
    giveback = 0.0
    denom = max(abs(cvd_peak), 1.0)
    if cvd_peak > 0 and cvd < cvd_peak:
        giveback = min(1.0, (cvd_peak - cvd) / denom)
    turn = 0.0
    if delta_300s > 0 and delta_30s < 0:          # recent flow flipped vs the trend
        turn = min(1.0, abs(delta_30s) / max(abs(delta_300s), 1.0))
    elif delta_30s < 0:
        turn = 0.5
    return max(0.0, min(1.0, 0.65 * giveback + 0.35 * turn))


def norm_buyer_exhaustion(cvd_div: float, delta_10s: float, delta_60s: float,
                          buy_sell_r: float) -> float:
    """[0,1]: price pushing but buying is drying up. Combines bearish CVD divergence,
    short-window buy deceleration, and a falling buy/sell ratio."""
    diverge = max(0.0, -cvd_div)  # only bearish divergence contributes
    # deceleration: 10s delta much weaker than the 60s average rate
    decel = 0.0
    avg60 = delta_60s / 6.0  # per-10s average over the 60s window
    if avg60 > 0:
        decel = max(0.0, min(1.0, 1.0 - (delta_10s / avg60))) if delta_10s < avg60 else 0.0
    ratio_weak = 0.0
    if math.isfinite(buy_sell_r):
        if buy_sell_r < 1.0:
            ratio_weak = min(1.0, 1.0 - buy_sell_r)
    return max(0.0, min(1.0, 0.5 * diverge + 0.3 * decel + 0.2 * ratio_weak))


# -- buy-side mirrors (consumed by buy_opportunity_engine.py) -----------------------
def norm_buy_pressure(executed_imb: float, q_imb: float, delta_60s: float, buy_vol: float,
                      sell_vol: float) -> float:
    """[0,1]: 0 = all selling, 1 = heavy buying. Mirror of norm_sell_pressure."""
    buy_share = 0.0
    tot = buy_vol + sell_vol
    if tot > 0:
        buy_share = buy_vol / tot
    s = 0.6 * buy_share + 0.25 * (0.5 + 0.5 * q_imb) + 0.15 * (1.0 if delta_60s > 0 else 0.0)
    return max(0.0, min(1.0, s))


def norm_cvd_improvement(cvd: float, cvd_trough: float, delta_30s: float, delta_300s: float) -> float:
    """[0,1]: how far CVD has RECOVERED from its trough + near-term delta turning positive.
    Mirror of norm_cvd_deterioration."""
    recovery = 0.0
    denom = max(abs(cvd_trough), 1.0)
    if cvd_trough < 0 and cvd > cvd_trough:
        recovery = min(1.0, (cvd - cvd_trough) / denom)
    turn = 0.0
    if delta_300s < 0 and delta_30s > 0:          # flow flipped positive vs a selling trend
        turn = min(1.0, abs(delta_30s) / max(abs(delta_300s), 1.0))
    elif delta_30s > 0:
        turn = 0.5
    return max(0.0, min(1.0, 0.65 * recovery + 0.35 * turn))


def norm_seller_exhaustion(cvd_div: float, delta_10s: float, delta_60s: float,
                           buy_sell_r: float) -> float:
    """[0,1]: price dropping but SELLING is drying up. Combines bullish CVD divergence
    (price down, CVD up -> cvd_div>0), selling deceleration, and a rising buy/sell ratio.
    Mirror of norm_buyer_exhaustion."""
    diverge = max(0.0, cvd_div)   # only bullish divergence contributes
    decel = 0.0
    avg60 = delta_60s / 6.0       # per-10s average over the 60s window (negative when selling)
    if avg60 < 0:                 # only meaningful when net selling
        decel = max(0.0, min(1.0, 1.0 - (delta_10s / avg60))) if delta_10s > avg60 else 0.0
    ratio_strong = 0.0
    if math.isfinite(buy_sell_r) and buy_sell_r > 1.0:
        ratio_strong = min(1.0, buy_sell_r - 1.0)
    return max(0.0, min(1.0, 0.5 * diverge + 0.3 * decel + 0.2 * ratio_strong))


def cvd_trend_direction(cvd_series: Sequence[Tuple[float, float]], now: float,
                        window_s: float) -> float:
    """[-1,1] normalized direction of CVD over the window (positive = accumulating)."""
    lo = now - window_s
    pts = [(t, v) for t, v in cvd_series if t >= lo]
    if len(pts) < 2:
        return 0.0
    c0, c1 = pts[0][1], pts[-1][1]
    rng = max(abs(c1), abs(c0), 1.0)
    return max(-1.0, min(1.0, (c1 - c0) / rng))


def classify_liquidity(spread_bps: Optional[float], wide_bps: float) -> str:
    if spread_bps is None:
        return "unknown"
    if spread_bps <= wide_bps * 0.4:
        return "tight"
    if spread_bps <= wide_bps:
        return "normal"
    return "wide"


# ----------------------------------------------------------------------
# L1 microstructure primitives (seller-exhaustion extension)
# ----------------------------------------------------------------------
# IEX quotes are best bid/ask + sizes only (Level 1). These are honest,
# bounded heuristic proxies -- NOT order-book depth, NOT validated alpha.
# L2/iceberg detection is UNAVAILABLE on IEX.

def microprice(bid: Optional[float], ask: Optional[float],
               bid_size: float, ask_size: float) -> Optional[float]:
    """Size-weighted fair price. Leans toward the side with the LARGER resting
    size (where pressure is building): heavy bid -> microprice above mid.
    Returns None for invalid/crossed quotes; falls back to mid if sizes are 0."""
    if bid is None or ask is None or bid <= 0 or ask <= 0 or ask < bid:
        return None
    tot = (bid_size or 0.0) + (ask_size or 0.0)
    if tot <= 0:
        return 0.5 * (bid + ask)
    return (bid * ask_size + ask * bid_size) / tot


def microprice_tilt(micro: Optional[float], mid: Optional[float],
                    spread: Optional[float]) -> float:
    """[-1,1] position of the microprice within the spread. + = upward (buy)
    pressure, - = downward (sell) pressure. 0 when undefined."""
    if micro is None or mid is None or not spread or spread <= 0:
        return 0.0
    return max(-1.0, min(1.0, (micro - mid) / (0.5 * spread)))


def absorption_score(buy_vol: float, sell_vol: float, price_change_pct: float,
                     hold_tol_pct: float = 0.002) -> float:
    """[0,1] net selling that price ABSORBED without declining (demand soaking
    supply). High when sell flow dominates yet price holds or rises; decays as
    price falls. 0 when flow is net buying."""
    net_sell = sell_vol - buy_vol
    total = buy_vol + sell_vol
    if net_sell <= 0 or total <= 0:
        return 0.0
    intensity = min(1.0, net_sell / total)
    if price_change_pct >= 0:
        hold = 1.0
    elif hold_tol_pct <= 0:
        hold = 0.0
    else:
        hold = max(0.0, 1.0 - (abs(price_change_pct) / hold_tol_pct))
    return max(0.0, min(1.0, intensity * hold))


def distribution_score(buy_vol: float, sell_vol: float, price_change_pct: float,
                       hold_tol_pct: float = 0.002) -> float:
    """[0,1] buyer-side mirror of :func:`absorption_score`: net buying that
    FAILS to lift price (supply capping demand)."""
    net_buy = buy_vol - sell_vol
    total = buy_vol + sell_vol
    if net_buy <= 0 or total <= 0:
        return 0.0
    intensity = min(1.0, net_buy / total)
    if price_change_pct <= 0:
        cap = 1.0
    elif hold_tol_pct <= 0:
        cap = 0.0
    else:
        cap = max(0.0, 1.0 - (price_change_pct / hold_tol_pct))
    return max(0.0, min(1.0, intensity * cap))


def replenishment_score(size_series: Sequence[Tuple[float, float]], now: float,
                        window_s: float) -> float:
    """[0,1] fraction of resting-size change that is REFILL (positive) within the
    window -- a proxy for persistent resting liquidity on one side of the book.
    0 when fewer than two observations or the side only bled down."""
    pts = [(t, s) for t, s in size_series if t >= now - window_s]
    if len(pts) < 2:
        return 0.0
    up = 0.0
    tot = 0.0
    for (_t0, s0), (_t1, s1) in zip(pts, pts[1:]):
        d = s1 - s0
        tot += abs(d)
        if d > 0:
            up += d
    if tot <= 0:
        return 0.0
    return max(0.0, min(1.0, up / tot))


def volume_climax_score(short_vol: float, baseline_vol: float, short_s: float,
                        baseline_s: float, full_ratio: float = 3.0) -> float:
    """[0,1] how far the short-window volume exceeds its proportional share of a
    longer baseline (a capitulation/climax spike). 0 at or below the expected
    pace, 1 at >= ``full_ratio`` x expected."""
    if baseline_vol <= 0 or baseline_s <= 0 or short_s <= 0:
        return 0.0
    expected = baseline_vol * (short_s / baseline_s)
    if expected <= 0:
        return 0.0
    ratio = short_vol / expected
    if ratio <= 1.0:
        return 0.0
    if full_ratio <= 1.0:
        return 1.0
    return max(0.0, min(1.0, (ratio - 1.0) / (full_ratio - 1.0)))


def seller_exhaustion_l1(absorption: float, bid_replenishment: float,
                         microprice_tilt: float) -> float:
    """[0,1] composite: selling is being absorbed, bids are refilling, and the
    microprice tilts up -> sellers are exhausting. Only upward tilt contributes."""
    return max(0.0, min(1.0,
                        0.4 * absorption + 0.3 * bid_replenishment
                        + 0.3 * max(0.0, microprice_tilt)))


def buyer_exhaustion_l1(distribution: float, ask_replenishment: float,
                        microprice_tilt: float) -> float:
    """[0,1] mirror of :func:`seller_exhaustion_l1` for buyer exhaustion."""
    return max(0.0, min(1.0,
                        0.4 * distribution + 0.3 * ask_replenishment
                        + 0.3 * max(0.0, -microprice_tilt)))


def window_price_change_pct(price_series: Sequence[Tuple[float, float]], now: float,
                            window_s: float) -> float:
    """Fractional price change over [now-window_s, now]; 0 if insufficient data."""
    pts = [(t, p) for t, p in price_series if t >= now - window_s]
    if len(pts) < 2:
        return 0.0
    p0, p1 = pts[0][1], pts[-1][1]
    if p0 <= 0:
        return 0.0
    return (p1 - p0) / p0


# ======================================================================
# Per-symbol stateful accumulator
# ======================================================================
class SymbolOrderFlow:
    """Accumulates trades/quotes for one symbol and produces snapshots.

    Events are stored as (epoch_seconds, ...) and time-trimmed to the configured
    retention. Duplicate and badly out-of-order events are rejected.
    """

    def __init__(self, symbol: str, config: Optional[OrderFlowConfig] = None):
        self.symbol = symbol.upper()
        self.cfg = config or OrderFlowConfig()
        # signed trades: (ts, sign, size)
        self._trades: Deque[Tuple[float, int, float]] = deque(maxlen=self.cfg.max_trades)
        # cvd series: (ts, cvd_value)
        self._cvd: Deque[Tuple[float, float]] = deque(maxlen=self.cfg.max_trades)
        # price series (last trade price): (ts, price)
        self._price: Deque[Tuple[float, float]] = deque(maxlen=self.cfg.max_trades)
        self._cvd_running = 0.0
        self._cvd_peak = 0.0
        self._cvd_trough = 0.0
        self._last_tick_sign = 0
        self._last_trade_ts: Optional[float] = None
        self._last_quote_ts: Optional[float] = None
        self._last_price: Optional[float] = None
        self._seen_trade_keys: Deque[Tuple[float, float, float]] = deque(maxlen=512)
        # latest quote
        self._bid: Optional[float] = None
        self._ask: Optional[float] = None
        self._bid_size: float = 0.0
        self._ask_size: float = 0.0
        # quote size history for bid/ask replenishment: (ts, bid_size, ask_size)
        self._quotes: Deque[Tuple[float, float, float]] = deque(maxlen=self.cfg.max_quotes)

    # ---- ingestion ----
    def on_quote(self, ts: float, bid: float, ask: float, bid_size: float, ask_size: float) -> None:
        if self._last_quote_ts is not None and ts < self._last_quote_ts - self.cfg.out_of_order_tolerance_s:
            return  # badly out of order
        self._last_quote_ts = max(ts, self._last_quote_ts or ts)
        if bid and ask and bid > 0 and ask > 0:
            self._bid, self._ask = float(bid), float(ask)
            self._bid_size, self._ask_size = float(bid_size or 0.0), float(ask_size or 0.0)
            self._quotes.append((ts, self._bid_size, self._ask_size))

    def on_trade(self, ts: float, price: float, size: float, trade_id: Any = None) -> None:
        if price <= 0 or size <= 0:
            return
        if self._last_trade_ts is not None and ts < self._last_trade_ts - self.cfg.out_of_order_tolerance_s:
            return  # badly out of order
        key = (round(ts, 4), round(float(price), 6), round(float(size), 6))
        if key in self._seen_trade_keys:
            return  # duplicate
        self._seen_trade_keys.append(key)

        sign = classify_trade(price, self._bid, self._ask, self._last_tick_sign)
        if sign == 0 and self._last_price is not None:
            sign = 1 if price > self._last_price else (-1 if price < self._last_price else self._last_tick_sign)
        if sign in (1, -1):
            self._last_tick_sign = sign

        self._trades.append((ts, sign, float(size)))
        self._cvd_running += sign * float(size)
        self._cvd_peak = max(self._cvd_peak, self._cvd_running)
        self._cvd_trough = min(self._cvd_trough, self._cvd_running)
        self._cvd.append((ts, self._cvd_running))
        self._price.append((ts, float(price)))
        self._last_trade_ts = max(ts, self._last_trade_ts or ts)
        self._last_price = float(price)
        self._trim(ts)

    def _trim(self, now: float) -> None:
        lo = now - self.cfg.retain_s
        for dq in (self._trades, self._cvd, self._price):
            while dq and dq[0][0] < lo:
                dq.popleft()
        while self._quotes and self._quotes[0][0] < lo:
            self._quotes.popleft()

    # ---- snapshot ----
    def snapshot(self, now: Optional[float] = None) -> OrderFlowSnapshot:
        now = now if now is not None else datetime.now(timezone.utc).timestamp()
        snap = OrderFlowSnapshot(symbol=self.symbol, ts=datetime.fromtimestamp(now, timezone.utc).isoformat())
        snap.last_trade_age_s = (now - self._last_trade_ts) if self._last_trade_ts else None
        snap.last_quote_age_s = (now - self._last_quote_ts) if self._last_quote_ts else None

        # quote-derived
        if self._bid and self._ask:
            snap.bid, snap.ask = self._bid, self._ask
            snap.mid, snap.spread, snap.spread_bps = spread_metrics(self._bid, self._ask)
            snap.liquidity = classify_liquidity(snap.spread_bps, self.cfg.wide_spread_bps)
            snap.quote_imbalance = quote_imbalance(self._bid_size, self._ask_size)
            snap.microprice = microprice(self._bid, self._ask, self._bid_size, self._ask_size)
            snap.microprice_tilt = microprice_tilt(snap.microprice, snap.mid, snap.spread)

        trades = list(self._trades)
        snap.n_trades_60s = sum(1 for t, _s, _sz in trades if t >= now - 60)

        # staleness / sufficiency gates
        stale = ((snap.last_trade_age_s is None or snap.last_trade_age_s > self.cfg.stale_trade_s))
        if len(trades) < self.cfg.min_trades_for_signal:
            snap.status, snap.is_stale, snap.ok = "INSUFFICIENT_DATA", True, False
            return snap
        if stale:
            snap.status, snap.is_stale, snap.ok = "STALE", True, False
            # still fill metrics below for transparency, but ok stays False

        snap.delta_10s = signed_delta(trades, now, 10)
        snap.delta_30s = signed_delta(trades, now, 30)
        snap.delta_60s = signed_delta(trades, now, 60)
        snap.delta_300s = signed_delta(trades, now, 300)
        snap.buy_vol_60s, snap.sell_vol_60s = buy_sell_volumes(trades, now, 60)
        snap.executed_imbalance = executed_imbalance(snap.buy_vol_60s, snap.sell_vol_60s)
        snap.buy_sell_ratio = buy_sell_ratio(snap.buy_vol_60s, snap.sell_vol_60s)
        snap.cvd = self._cvd_running
        snap.cvd_peak = self._cvd_peak
        snap.cvd_trough = self._cvd_trough
        snap.cvd_trend = cvd_trend_direction(list(self._cvd), now, self.cfg.cvd_window_s)
        snap.cvd_divergence = cvd_divergence(list(self._price), list(self._cvd), now,
                                             self.cfg.divergence_lookback_s)
        snap.divergence_flag = ("bearish" if snap.cvd_divergence < -0.05
                                else "bullish" if snap.cvd_divergence > 0.05 else "none")
        snap.pressure, snap.pressure_score = classify_pressure(
            snap.executed_imbalance, snap.delta_60s, snap.quote_imbalance)

        snap.sell_pressure = norm_sell_pressure(snap.executed_imbalance, snap.quote_imbalance,
                                                snap.delta_60s, snap.buy_vol_60s, snap.sell_vol_60s)
        snap.buy_pressure = norm_buy_pressure(snap.executed_imbalance, snap.quote_imbalance,
                                              snap.delta_60s, snap.buy_vol_60s, snap.sell_vol_60s)
        snap.cvd_deterioration = norm_cvd_deterioration(snap.cvd, snap.cvd_peak,
                                                        snap.delta_30s, snap.delta_300s)
        snap.buyer_exhaustion = norm_buyer_exhaustion(snap.cvd_divergence, snap.delta_10s,
                                                     snap.delta_60s, snap.buy_sell_ratio)
        snap.cvd_improvement = norm_cvd_improvement(snap.cvd, snap.cvd_trough,
                                                    snap.delta_30s, snap.delta_300s)
        snap.seller_exhaustion = norm_seller_exhaustion(snap.cvd_divergence, snap.delta_10s,
                                                        snap.delta_60s, snap.buy_sell_ratio)

        # L1 microstructure primitives (extend the CVD-based exhaustion signals)
        pchg60 = window_price_change_pct(list(self._price), now, 60)
        snap.absorption = absorption_score(snap.buy_vol_60s, snap.sell_vol_60s, pchg60,
                                           self.cfg.absorption_hold_tol_pct)
        _distribution = distribution_score(snap.buy_vol_60s, snap.sell_vol_60s, pchg60,
                                           self.cfg.absorption_hold_tol_pct)
        bid_hist = [(t, bs) for (t, bs, _az) in self._quotes]
        ask_hist = [(t, az) for (t, _bs, az) in self._quotes]
        snap.bid_replenishment = replenishment_score(bid_hist, now, self.cfg.replenish_window_s)
        snap.ask_replenishment = replenishment_score(ask_hist, now, self.cfg.replenish_window_s)
        short_b, short_s = buy_sell_volumes(trades, now, self.cfg.climax_short_s)
        base_b, base_s = buy_sell_volumes(trades, now, self.cfg.climax_baseline_s)
        snap.volume_climax = volume_climax_score(short_b + short_s, base_b + base_s,
                                                 self.cfg.climax_short_s, self.cfg.climax_baseline_s,
                                                 self.cfg.climax_full_ratio)
        short_delta = signed_delta(trades, now, self.cfg.climax_short_s)
        snap.volume_climax_side = 1 if short_delta > 0 else (-1 if short_delta < 0 else 0)
        snap.seller_exhaustion_l1 = seller_exhaustion_l1(snap.absorption, snap.bid_replenishment,
                                                         snap.microprice_tilt)
        snap.buyer_exhaustion_l1 = buyer_exhaustion_l1(_distribution, snap.ask_replenishment,
                                                       snap.microprice_tilt)

        if not stale:
            snap.status, snap.is_stale, snap.ok = "OK", False, True
        return snap


# ======================================================================
# Multi-symbol engine + Alpaca REST adapter (fail-safe)
# ======================================================================
class OrderFlowEngine:
    """Manages per-symbol order flow and (optionally) polls Alpaca IEX trades/quotes.

    The REST poller is fail-safe: any error logs and yields INSUFFICIENT_DATA snapshots
    so it can never raise into the trading loop. Pass data_client=StockHistoricalDataClient.
    """

    def __init__(self, config: Optional[OrderFlowConfig] = None, data_client: Any = None):
        self.cfg = config or OrderFlowConfig()
        self.data_client = data_client
        self._symbols: dict[str, SymbolOrderFlow] = {}
        self._rotation_offset = 0            # round-robin cursor for lower-priority overflow

    def select_priority_symbols(self, priority_groups: Sequence[Sequence[str]],
                                max_symbols: Optional[int] = None) -> list[str]:
        """Pick which symbols to subscribe/poll this cycle, honoring the priority order
        (positions > pending > candidates > universe). Higher-priority groups are always
        kept; the remaining budget is filled from lower-priority symbols, ROTATING across
        calls so coverage is fair and sparse symbols are not misrepresented as covered.
        """
        cap = max_symbols if max_symbols is not None else self.cfg.max_subscriptions
        selected: list[str] = []
        seen: set[str] = set()
        overflow: list[str] = []
        for gi, group in enumerate(priority_groups):
            for raw in group:
                s = str(raw).upper().strip()
                if not s or s in seen:
                    continue
                seen.add(s)
                # The first two groups (positions, pending) are mandatory; never rotate them out.
                if gi <= 1 and len(selected) < cap:
                    selected.append(s)
                else:
                    overflow.append(s)
        free = max(0, cap - len(selected))
        if overflow and free > 0:
            n = len(overflow)
            off = self._rotation_offset % n
            rotated = overflow[off:] + overflow[:off]
            selected.extend(rotated[:free])
            self._rotation_offset = (self._rotation_offset + free) % n
        return selected[:cap]

    def poll_universe(self, priority_groups: Sequence[Sequence[str]], lookback_s: int = 300,
                      max_symbols: Optional[int] = None) -> dict[str, OrderFlowSnapshot]:
        """Prioritized universe poll. Returns snapshots ONLY for the symbols actually
        subscribed this cycle (un-polled symbols are intentionally absent -- never
        fabricated as covered)."""
        chosen = self.select_priority_symbols(priority_groups, max_symbols)
        if not chosen:
            return {}
        return self.poll_rest(chosen, lookback_s=lookback_s)

    def _sym(self, symbol: str) -> SymbolOrderFlow:
        s = symbol.upper()
        if s not in self._symbols:
            self._symbols[s] = SymbolOrderFlow(s, self.cfg)
        return self._symbols[s]

    def prune(self, keep: Iterable[str]) -> int:
        """Evict per-symbol accumulators for symbols not in ``keep``.

        A long-running stream rotates thousands of distinct symbols through a
        single engine; without eviction ``_symbols`` grows unbounded. Callers
        pass the currently-relevant set (subscribed ∪ positions ∪ references).
        Returns the number of symbols evicted.
        """
        keep_set = {str(s).upper() for s in keep}
        drop = [s for s in self._symbols if s not in keep_set]
        for s in drop:
            del self._symbols[s]
        return len(drop)

    # ---- direct ingestion (for stream handlers or tests) ----
    def ingest_trade(self, symbol: str, ts: float, price: float, size: float, trade_id: Any = None) -> None:
        self._sym(symbol).on_trade(ts, price, size, trade_id)

    def ingest_quote(self, symbol: str, ts: float, bid: float, ask: float,
                     bid_size: float, ask_size: float) -> None:
        self._sym(symbol).on_quote(ts, bid, ask, bid_size, ask_size)

    def snapshot(self, symbol: str, now: Optional[float] = None) -> OrderFlowSnapshot:
        return self._sym(symbol).snapshot(now)

    # ---- Alpaca REST poll (IEX) ----
    def poll_rest(self, symbols: Sequence[str], lookback_s: int = 300) -> dict[str, OrderFlowSnapshot]:
        """Fetch recent IEX trades + latest quotes for `symbols` and ingest them.

        Returns {symbol: snapshot}. Fail-safe: on any error a symbol gets whatever
        snapshot its accumulator currently holds (possibly INSUFFICIENT_DATA).
        """
        if self.data_client is not None:
            try:
                self._poll_trades(symbols, lookback_s)
            except Exception as exc:  # noqa: BLE001 - never raise into the loop
                logger.warning("order_flow poll_trades failed: %s", exc)
            try:
                self._poll_quotes(symbols)
            except Exception as exc:  # noqa: BLE001
                logger.warning("order_flow poll_quotes failed: %s", exc)
        return {s.upper(): self.snapshot(s) for s in symbols}

    def _poll_trades(self, symbols: Sequence[str], lookback_s: int) -> None:
        from alpaca.data.requests import StockTradesRequest
        from alpaca.data.timeframe import TimeFrame  # noqa: F401 (import parity)
        try:
            from alpaca.data.enums import DataFeed
            feed = DataFeed.IEX
        except Exception:
            feed = None
        start = datetime.now(timezone.utc) - timedelta(seconds=lookback_s)
        kwargs: dict[str, Any] = {"symbol_or_symbols": list(symbols), "start": start}
        if feed is not None:
            kwargs["feed"] = feed
        req = StockTradesRequest(**kwargs)
        resp = self.data_client.get_stock_trades(req)
        data = getattr(resp, "data", resp) or {}
        for sym in symbols:
            rows = data.get(sym.upper()) or data.get(sym) or []
            for t in rows:
                ts = _epoch(getattr(t, "timestamp", None))
                price = float(getattr(t, "price", 0.0) or 0.0)
                size = float(getattr(t, "size", 0.0) or 0.0)
                tid = getattr(t, "id", None)
                if ts is not None:
                    self._sym(sym).on_trade(ts, price, size, tid)

    def _poll_quotes(self, symbols: Sequence[str]) -> None:
        from alpaca.data.requests import StockLatestQuoteRequest
        try:
            from alpaca.data.enums import DataFeed
            feed = DataFeed.IEX
        except Exception:
            feed = None
        kwargs: dict[str, Any] = {"symbol_or_symbols": list(symbols)}
        if feed is not None:
            kwargs["feed"] = feed
        req = StockLatestQuoteRequest(**kwargs)
        quotes = self.data_client.get_stock_latest_quote(req) or {}
        for sym in symbols:
            q = quotes.get(sym.upper()) or quotes.get(sym)
            if q is None:
                continue
            ts = _epoch(getattr(q, "timestamp", None)) or datetime.now(timezone.utc).timestamp()
            self._sym(sym).on_quote(
                ts,
                float(getattr(q, "bid_price", 0.0) or 0.0),
                float(getattr(q, "ask_price", 0.0) or 0.0),
                float(getattr(q, "bid_size", 0.0) or 0.0),
                float(getattr(q, "ask_size", 0.0) or 0.0),
            )


def _epoch(ts: Any) -> Optional[float]:
    """Coerce an Alpaca timestamp (datetime / pandas Timestamp / ISO str / epoch) to seconds."""
    if ts is None:
        return None
    if isinstance(ts, (int, float)):
        return float(ts)
    try:
        if hasattr(ts, "timestamp"):
            return float(ts.timestamp())
        return datetime.fromisoformat(str(ts).replace("Z", "+00:00")).timestamp()
    except Exception:
        return None
