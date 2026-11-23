"""Level 1 Microstructure Intelligence Engine (Milestone 4 addendum).

Alpaca SIP provides consolidated trades + top-of-book quotes but NO Level-2 depth.
This engine derives L1 microstructure features from genuine SIP observations,
REUSING the existing ``order_flow_engine.SymbolOrderFlow`` (quote imbalance, OFI,
microprice tilt, trade classification, CVD, spread, pressure, seller/buyer
exhaustion, liquidity) and ADDING the features it does not compute: Kyle's lambda
price impact, Amihud illiquidity, trade arrival intensity, trade-size distribution,
and a short-horizon adverse-selection estimate.

Data-integrity rules (addendum):
  * Observed quantities (bid/ask/sizes/spread/last trade) are kept separate from
    INFERRED quantities (OFI/CVD/aggressor side/exhaustion/Kyle/Amihud).
  * NEVER fabricate L2 depth, hidden orders, queue position, liquidity walls,
    participant identity, cancellations, or an aggressor side that cannot be
    determined. Unknown measurements stay ``None``.
  * Crossed/locked quotes, stale quotes/trades, and insufficient data are flagged
    in ``quality`` and lower ``confidence``; they never raise.

Prices here are floats in dollars (converted from market-data ticks at
DEFAULT_PRICE_SCALE = 10_000). These are optimizer/execution INPUTS; they never
override strict entry qualification, strategy maturity, exposure authorization, or
the native financial risk controls.
"""

from __future__ import annotations

import statistics
from collections import deque
from dataclasses import dataclass
from typing import Optional

from order_flow_engine import SymbolOrderFlow

MICROSTRUCTURE_FEATURE_VERSION = "microstructure-1.0.0"
PRICE_SCALE = 10_000  # DEFAULT_PRICE_SCALE; ticks -> dollars


@dataclass(frozen=True)
class MicrostructureFeatures:
    """L1 microstructure snapshot with explicit observed/inferred/quality split."""

    symbol: str
    feature_version: str = MICROSTRUCTURE_FEATURE_VERSION
    observation_ns: int = 0

    # --- OBSERVED (directly from SIP L1) ---
    bid: Optional[float] = None
    ask: Optional[float] = None
    mid: Optional[float] = None
    spread_bps: Optional[float] = None
    bid_size: Optional[int] = None
    ask_size: Optional[int] = None
    last_trade_price: Optional[float] = None

    # --- INFERRED (derived; approximate; never presented as exchange truth) ---
    microprice: Optional[float] = None          # size-weighted mid (L1)
    microprice_displacement_bps: Optional[float] = None
    quote_imbalance: Optional[float] = None     # [-1,1]
    order_flow_imbalance: Optional[float] = None  # OFI proxy (executed imbalance)
    cumulative_volume_delta: Optional[float] = None
    trade_arrival_intensity: Optional[float] = None   # trades/second
    aggressive_imbalance: Optional[float] = None      # signed aggressive volume [-1,1]
    mean_trade_size: Optional[float] = None
    p95_trade_size: Optional[float] = None
    large_trade_fraction: Optional[float] = None
    kyle_lambda: Optional[float] = None         # price impact per unit signed $volume
    amihud_illiquidity: Optional[float] = None  # |ret| / $volume
    liquidity_resilience: Optional[float] = None  # replenishment/absorption proxy [0,1]
    seller_exhaustion: Optional[float] = None
    buyer_exhaustion: Optional[float] = None
    adverse_selection_bps: Optional[float] = None  # post-trade drift against liquidity

    # --- QUALITY ---
    is_stale: bool = True
    quote_age_s: Optional[float] = None
    trade_age_s: Optional[float] = None
    crossed_or_locked: bool = False
    n_trades_window: int = 0
    confidence: float = 0.0                      # heuristic [0,1], NOT calibrated
    quality: str = "INSUFFICIENT_DATA"           # OK | STALE | CROSSED | INSUFFICIENT_DATA

    def fill_probability_features(self) -> dict:
        """Feature contract for a FUTURE fill-probability model (§18). Deterministic
        snapshot of the signals; no model is fit here — the contract is prepared for
        the later supervised/bandit/RL platform."""
        return {
            "feature_version": self.feature_version, "spread_bps": self.spread_bps,
            "quote_imbalance": self.quote_imbalance,
            "order_flow_imbalance": self.order_flow_imbalance,
            "microprice_displacement_bps": self.microprice_displacement_bps,
            "trade_arrival_intensity": self.trade_arrival_intensity,
            "aggressive_imbalance": self.aggressive_imbalance,
            "kyle_lambda": self.kyle_lambda, "amihud_illiquidity": self.amihud_illiquidity,
            "liquidity_resilience": self.liquidity_resilience,
            "adverse_selection_bps": self.adverse_selection_bps,
            "is_stale": self.is_stale, "confidence": self.confidence}


class SymbolMicrostructure:
    """Per-symbol rolling microstructure state (reuses SymbolOrderFlow)."""

    def __init__(self, symbol: str, *, impact_window: int = 200,
                 size_window: int = 200, intensity_window_s: float = 60.0,
                 adverse_horizon_s: float = 5.0, stale_quote_s: float = 5.0,
                 stale_trade_s: float = 30.0) -> None:
        self.symbol = symbol.upper()
        self._flow = SymbolOrderFlow(self.symbol)
        self._impacts: deque = deque(maxlen=impact_window)   # (mid_change, signed_dollar_vol)
        self._amihud: deque = deque(maxlen=impact_window)    # (|ret|, dollar_vol)
        self._sizes: deque = deque(maxlen=size_window)       # trade sizes
        self._arrivals: deque = deque(maxlen=1000)           # trade timestamps (s)
        self._pending_adverse: deque = deque(maxlen=size_window)  # (ts, signed, mid_at_trade)
        self._adverse_bps: deque = deque(maxlen=size_window)
        self._prev_trade_mid: Optional[float] = None
        self._last_quote_ts: Optional[float] = None
        self._last_trade_ts: Optional[float] = None
        self._bid = self._ask = None
        self._bid_size = self._ask_size = None
        self._last_trade_price: Optional[float] = None
        self._intensity_window_s = intensity_window_s
        self._adverse_horizon_s = adverse_horizon_s
        self._stale_quote_s = stale_quote_s
        self._stale_trade_s = stale_trade_s

    def on_quote(self, ts: float, bid: float, ask: float, bid_size: float,
                 ask_size: float) -> None:
        self._flow.on_quote(ts, bid, ask, bid_size, ask_size)
        self._bid, self._ask = bid, ask
        self._bid_size, self._ask_size = bid_size, ask_size
        self._last_quote_ts = ts

    def on_trade(self, ts: float, price: float, size: float) -> None:
        self._flow.on_trade(ts, price, size)
        self._last_trade_ts = ts
        self._last_trade_price = price
        self._sizes.append(size)
        self._arrivals.append(ts)
        # aggressor side via quote-aligned classification (unknown -> 0, not fabricated)
        signed = 0.0
        if self._bid is not None and self._ask is not None:
            mid = 0.5 * (self._bid + self._ask)
            if price >= self._ask:
                signed = 1.0
            elif price <= self._bid:
                signed = -1.0
            elif price > mid:
                signed = 0.5
            elif price < mid:
                signed = -0.5
        dollar_vol = price * size
        # Kyle's lambda / Amihud inputs: TRADE-to-TRADE mid change vs signed $volume.
        cur_mid = 0.5 * (self._bid + self._ask) if (self._bid and self._ask) else None
        if (cur_mid is not None and self._prev_trade_mid is not None
                and self._prev_trade_mid > 0):
            mid_change = cur_mid - self._prev_trade_mid
            self._impacts.append((mid_change, signed * dollar_vol))
            ret = mid_change / self._prev_trade_mid
            if dollar_vol > 0:
                self._amihud.append((abs(ret), dollar_vol))
        if cur_mid is not None:
            self._prev_trade_mid = cur_mid
        # adverse-selection: record trade for later post-horizon mid comparison
        if signed != 0.0 and self._bid and self._ask:
            self._pending_adverse.append((ts, signed, 0.5 * (self._bid + self._ask)))
        self._resolve_adverse(ts)

    def _resolve_adverse(self, now: float) -> None:
        """Resolve adverse-selection for trades whose horizon has elapsed: did the
        mid move in the aggressor's favor (adverse for the resting side)?"""
        if self._bid is None or self._ask is None:
            return
        cur_mid = 0.5 * (self._bid + self._ask)
        horizon = self._adverse_horizon_s
        while self._pending_adverse and (now - self._pending_adverse[0][0]) >= horizon:
            _, signed, mid0 = self._pending_adverse.popleft()
            if mid0 > 0:
                drift_bps = (cur_mid - mid0) / mid0 * 10_000.0
                # adverse to the liquidity provider = drift in the aggressor's direction
                self._adverse_bps.append(drift_bps * (1.0 if signed > 0 else -1.0))

    def _intensity(self, now: float) -> Optional[float]:
        cutoff = now - self._intensity_window_s
        recent = [t for t in self._arrivals if t >= cutoff]
        if not recent:
            return 0.0
        return len(recent) / self._intensity_window_s

    def _kyle_lambda(self) -> Optional[float]:
        if len(self._impacts) < 10:
            return None
        dp = [a for a, _ in self._impacts]
        sv = [b for _, b in self._impacts]
        var = statistics.pvariance(sv) if len(sv) > 1 else 0.0
        if var <= 0:
            return None
        mean_dp, mean_sv = statistics.fmean(dp), statistics.fmean(sv)
        cov = sum((a - mean_dp) * (b - mean_sv) for a, b in zip(dp, sv, strict=False)) / len(dp)
        return cov / var

    def _amihud_illiquidity(self) -> Optional[float]:
        if len(self._amihud) < 5:
            return None
        return statistics.fmean([r / v for r, v in self._amihud if v > 0]) or None

    def features(self, now: float, observation_ns: int = 0) -> MicrostructureFeatures:
        snap = self._flow.snapshot(now)
        quote_age = (now - self._last_quote_ts) if self._last_quote_ts else None
        trade_age = (now - self._last_trade_ts) if self._last_trade_ts else None
        crossed = bool(self._bid is not None and self._ask is not None
                       and self._bid > 0 and self._ask > 0 and self._bid >= self._ask)
        stale = bool(snap.is_stale or (quote_age is not None and quote_age > self._stale_quote_s))

        microprice = None
        micro_disp = None
        if (self._bid and self._ask and self._bid_size is not None
                and self._ask_size is not None and not crossed):
            denom = self._bid_size + self._ask_size
            if denom > 0:
                microprice = (self._bid * self._ask_size + self._ask * self._bid_size) / denom
                mid = 0.5 * (self._bid + self._ask)
                if mid > 0:
                    micro_disp = (microprice - mid) / mid * 10_000.0

        sizes = list(self._sizes)
        mean_sz = statistics.fmean(sizes) if sizes else None
        p95_sz = (sorted(sizes)[min(len(sizes) - 1, int(0.95 * len(sizes)))]
                  if sizes else None)
        large_frac = (sum(1 for s in sizes if mean_sz and s > 3 * mean_sz) / len(sizes)
                      if sizes and mean_sz else None)
        adverse = statistics.fmean(self._adverse_bps) if self._adverse_bps else None

        # confidence: more data + fresh quotes + not crossed -> higher
        n_tr = len(sizes)
        conf = 0.0 if (stale or crossed) else min(1.0, n_tr / 50.0) * 0.7 + 0.3
        quality = ("CROSSED" if crossed else "STALE" if stale
                   else "OK" if snap.ok else "INSUFFICIENT_DATA")

        return MicrostructureFeatures(
            symbol=self.symbol, observation_ns=observation_ns,
            bid=self._bid, ask=self._ask, mid=snap.mid, spread_bps=snap.spread_bps,
            bid_size=int(self._bid_size) if self._bid_size is not None else None,
            ask_size=int(self._ask_size) if self._ask_size is not None else None,
            last_trade_price=self._last_trade_price,
            microprice=microprice, microprice_displacement_bps=micro_disp,
            quote_imbalance=snap.quote_imbalance,
            order_flow_imbalance=snap.executed_imbalance,
            cumulative_volume_delta=snap.cvd,
            trade_arrival_intensity=self._intensity(now),
            aggressive_imbalance=snap.executed_imbalance,
            mean_trade_size=mean_sz, p95_trade_size=p95_sz, large_trade_fraction=large_frac,
            kyle_lambda=self._kyle_lambda(), amihud_illiquidity=self._amihud_illiquidity(),
            liquidity_resilience=_nz_opt(snap.seller_exhaustion, snap.buyer_exhaustion),
            seller_exhaustion=snap.seller_exhaustion, buyer_exhaustion=snap.buyer_exhaustion,
            adverse_selection_bps=adverse,
            is_stale=stale, quote_age_s=quote_age, trade_age_s=trade_age,
            crossed_or_locked=crossed, n_trades_window=n_tr, confidence=round(conf, 4),
            quality=quality)


def _nz_opt(a, b):
    vals = [v for v in (a, b) if v is not None]
    return (sum(vals) / len(vals)) if vals else None


class MicrostructureEngine:
    """Per-symbol L1 microstructure registry fed by SIP quote/trade events.

    Reuses the single SIP connection (events are pushed in from the engine's market
    tap); establishes no new provider connection. Compute-intensive features are
    applied primarily to actionable symbols (callers pass only candidates / held /
    pending symbols)."""

    def __init__(self) -> None:
        self._books: dict[str, SymbolMicrostructure] = {}

    def _book(self, symbol: str) -> SymbolMicrostructure:
        key = symbol.upper()
        book = self._books.get(key)
        if book is None:
            book = SymbolMicrostructure(key)
            self._books[key] = book
        return book

    def on_quote_event(self, event) -> None:
        """Ingest a market_data QuoteEvent (prices in ticks @ scale 10_000)."""
        if getattr(event, "bid_price_ticks", 0) <= 0 or getattr(event, "ask_price_ticks", 0) <= 0:
            return
        ts = event.timestamp_ns / 1e9
        self._book(event.symbol).on_quote(
            ts, event.bid_price_ticks / PRICE_SCALE, event.ask_price_ticks / PRICE_SCALE,
            float(event.bid_size), float(event.ask_size))

    def on_trade_event(self, event) -> None:
        """Ingest a market_data TradeEvent (price in ticks @ scale 10_000)."""
        if getattr(event, "price_ticks", 0) <= 0:
            return
        ts = event.timestamp_ns / 1e9
        self._book(event.symbol).on_trade(ts, event.price_ticks / PRICE_SCALE, float(event.size))

    def features(self, symbol: str, *, now_ns: int) -> Optional[MicrostructureFeatures]:
        book = self._books.get(symbol.upper())
        if book is None:
            return None
        return book.features(now_ns / 1e9, observation_ns=now_ns)

    def tracked_symbols(self) -> list[str]:
        return list(self._books)
