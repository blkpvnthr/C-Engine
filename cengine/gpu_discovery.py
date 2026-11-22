"""Phase B.5 — GPU-backed discovery feature provider.

Bridges three native pieces into the market scanner's ``feature_provider`` seam:

  * DiscoveryEngine (B.2) — authoritative per-symbol market state + bar buffers.
  * technical_features (B.3) — Wilder DMI/ADX/SMA/BB/ATR/VWAP on the GPU.
  * screen_market (B.1) — the mandatory shared entry contract on the GPU.

All numerical work runs on Apple Metal when a device + compiled library are
available, and falls back to the validated native CPU oracle otherwise. The
public :meth:`GpuDiscoveryProvider.candidate_features` matches the scanner's
``Callable[[list[str], float], dict[str, CandidateFeatures]]`` contract, so the
native GPU path can REPLACE the per-symbol Python directional/technical loop
(``MarketScanner._default_features``) while the scanner's own ``rank()`` and the
single ``UniversePublisher`` / ``universe.txt`` writer are reused unchanged.

This module computes the directional/screening portion of each candidate; the
daily liquidity stats (price / ADV / ADDV) that the hard liquidity gate needs
are merged in from ``daily_stats`` supplied by the caller (the scanner's
existing REST daily-stats mechanism), never fabricated here.

GPU screening is advisory discovery context only — it never authorizes orders.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import _cengine_native as native

_TF_1M = native.BarTimeframe.ONE_MINUTE
_TF_5M = native.BarTimeframe.FIVE_MINUTE


def _discover_metallib() -> Optional[str]:
    try:
        from cengine.metal_backend import discover_metallib

        lib = discover_metallib()
        return str(lib) if lib else None
    except Exception:  # noqa: BLE001 — metal is optional; fall back to CPU
        return None


@dataclass(slots=True)
class ScreenRow:
    """Per-symbol native discovery result (technical features + screen output)."""

    symbol: str
    backend: str
    # 5-minute technical features
    di_plus: float
    di_minus: float
    adx: float
    sma: float
    bb_middle: float
    bb_upper: float
    bb_lower: float
    atr: float
    vwap: float
    tech_valid: int
    # screen inputs / outputs
    price: float
    previous_close_1m: float
    momentum_1m_pct: float
    directional_strength: float
    vwap_distance_pct: float
    flags: int

    @property
    def entry_qualified(self) -> bool:
        return bool(self.flags & native.SCREEN_ENTRY_QUALIFIED)

    @property
    def technical_exit(self) -> bool:
        return bool(self.flags & native.SCREEN_TECHNICAL_EXIT)

    @property
    def data_invalid(self) -> bool:
        return bool(self.flags & native.SCREEN_DATA_INVALID)


class GpuDiscoveryProvider:
    """Owns the native DiscoveryEngine and (optional) Metal runtime.

    Thread-safety: the native DiscoveryEngine is internally locked; this wrapper
    adds no mutable Python state beyond the cached symbol-id map, which is only
    rebuilt under :meth:`set_universe`.
    """

    def __init__(
        self,
        *,
        metallib: Optional[str] = None,
        bar_capacity: int = 256,
        window: int = 64,
        price_scale: int = 10_000,
        prefer_metal: bool = True,
        dmi_period: int = 14,
        sma_period: int = 20,
    ) -> None:
        self.engine = native.DiscoveryEngine(bar_capacity, price_scale)
        self._scale = int(price_scale)
        self._window = int(window)
        self._dmi = int(dmi_period)
        self._sma = int(sma_period)
        self._runtime: Optional[Any] = None
        if prefer_metal and getattr(native, "metal_available", False):
            lib = metallib or _discover_metallib()
            if lib:
                rt = native.MetalRuntime(str(lib))
                if rt.available:
                    self._runtime = rt

    @property
    def backend(self) -> str:
        return "metal" if self._runtime is not None else "cpu"

    # -- membership / ingestion ---------------------------------------------
    def set_universe(self, symbols: list[str]) -> int:
        return self.engine.update_membership(list(symbols))

    def _ids(self, symbols: list[str]) -> list[tuple[str, int]]:
        out = []
        for s in symbols:
            sid = self.engine.symbol_id(s)
            if sid is not None:
                out.append((s, sid))
        return out

    def ingest_bars(self, bars: list[tuple], timeframe) -> int:
        """bars: iterable of (symbol, ts_ns, o_ticks, h_ticks, l_ticks, c_ticks, volume)."""
        batch = []
        for sym, ts, o, h, lo, c, vol in bars:
            sid = self.engine.symbol_id(sym)
            if sid is None:
                continue
            batch.append(native.BarObs(symbol_id=sid, timestamp_ns=int(ts),
                                       open_ticks=int(o), high_ticks=int(h),
                                       low_ticks=int(lo), close_ticks=int(c),
                                       volume=int(vol)))
        return self.engine.ingest_bars(batch, timeframe) if batch else 0

    def ingest_trades(self, trades: list[tuple]) -> int:
        """trades: iterable of (symbol, ts_ns, price_ticks, size, sequence)."""
        batch = []
        for sym, ts, px, sz, seq in trades:
            sid = self.engine.symbol_id(sym)
            if sid is None:
                continue
            batch.append(native.TradeObs(symbol_id=sid, timestamp_ns=int(ts),
                                         price_ticks=int(px), size=int(sz),
                                         sequence=int(seq)))
        return self.engine.ingest_trades(batch) if batch else 0

    def ingest_quotes(self, quotes: list[tuple]) -> int:
        """quotes: iterable of (symbol, ts_ns, bid_ticks, ask_ticks, bid_sz, ask_sz, seq)."""
        batch = []
        for sym, ts, bid, ask, bsz, asz, seq in quotes:
            sid = self.engine.symbol_id(sym)
            if sid is None:
                continue
            batch.append(native.QuoteObs(symbol_id=sid, timestamp_ns=int(ts),
                                         bid_ticks=int(bid), ask_ticks=int(ask),
                                         bid_size=int(bsz), ask_size=int(asz),
                                         sequence=int(seq)))
        return self.engine.ingest_quotes(batch) if batch else 0

    # -- native computation --------------------------------------------------
    def _technical(self, bw) -> list:
        counts = list(bw.counts)
        if self._runtime is not None:
            return self._runtime.technical_features(
                list(bw.closes), list(bw.highs), list(bw.lows), list(bw.volumes),
                counts, bw.symbol_count, bw.max_window, self._dmi, self._sma)
        return native.technical_features_cpu(
            list(bw.closes), list(bw.highs), list(bw.lows), list(bw.volumes),
            counts, bw.symbol_count, bw.max_window, self._dmi, self._sma)

    def _screen(self, inputs: list):
        if not inputs:
            return []
        if self._runtime is not None:
            return self._runtime.screen(inputs)
        return native.screen_market_cpu(inputs)

    def market_breadth(self, symbols: list[str]):
        """Compute market breadth over `symbols` via the GPU market_breadth kernel
        (CPU oracle fallback) from each symbol's current screen flags. Returns a
        native.BreadthCounters, or None if no symbols are known."""
        rows = self.compute(symbols)
        flags = [rows[s].flags for s in symbols if s in rows]
        if not flags:
            return None
        if self._runtime is not None:
            return self._runtime.market_breadth(flags)
        return native.market_breadth_cpu(flags)

    def compute(self, symbols: list[str]) -> dict[str, ScreenRow]:
        """Run the native discovery pipeline for `symbols`, returning per-symbol
        technical features + screen output. Symbols unknown to the engine (never
        in the active universe) are skipped."""
        valid = self._ids(symbols)
        if not valid:
            return {}
        ids = [sid for _, sid in valid]
        bw5 = self.engine.export_bar_window(ids, _TF_5M, self._window)
        bw1 = self.engine.export_bar_window(ids, _TF_1M, self._window)
        tech = self._technical(bw5)
        c1 = list(bw1.closes)
        n1 = list(bw1.counts)
        W = bw1.max_window

        inputs = []
        meta = []
        for row, (sym, sid) in enumerate(valid):
            t = tech[row]
            snap = self.engine.snapshot(sid)
            # previous_close_1m = most recent COMPLETED 1m bar close (§30);
            # price = latest observed trade (developing), else last 1m close.
            cnt1 = n1[row]
            prev_close = c1[row * W + cnt1 - 1] if cnt1 >= 1 else 0.0
            if snap.has_trade and snap.last_trade_ticks > 0:
                price = snap.last_trade_ticks / self._scale
            elif cnt1 >= 1:
                price = c1[row * W + cnt1 - 1]
            else:
                price = 0.0
            valid_market = 1 if price > 0.0 else 0
            valid_1m = 1 if (cnt1 >= 1 and prev_close > 0.0) else 0
            si = native.ScreeningInput(
                price=price, previous_close_1m=prev_close,
                di_plus_5m=t.di_plus, di_minus_5m=t.di_minus,
                session_vwap=t.vwap, sma20_5m=t.sma, bb_middle_5m=t.bb_middle,
                valid_market=valid_market,
                valid_direction=1 if (t.valid & native.TECH_DMI) else 0,
                valid_1m=valid_1m,
                valid_vwap=1 if (t.valid & native.TECH_VWAP) else 0,
                valid_sma=1 if (t.valid & native.TECH_SMA) else 0,
                valid_bb=1 if (t.valid & native.TECH_BB) else 0)
            inputs.append(si)
            meta.append((sym, t, price, prev_close))

        outs = self._screen(inputs)
        backend = self.backend
        rows: dict[str, ScreenRow] = {}
        for (sym, t, price, prev_close), o in zip(meta, outs, strict=True):
            rows[sym] = ScreenRow(
                symbol=sym, backend=backend,
                di_plus=t.di_plus, di_minus=t.di_minus, adx=t.adx, sma=t.sma,
                bb_middle=t.bb_middle, bb_upper=t.bb_upper, bb_lower=t.bb_lower,
                atr=t.atr, vwap=t.vwap, tech_valid=t.valid,
                price=price, previous_close_1m=prev_close,
                momentum_1m_pct=o.momentum_1m_pct,
                directional_strength=o.directional_strength,
                vwap_distance_pct=o.vwap_distance_pct, flags=o.flags)
        return rows

    # -- scanner feature_provider seam --------------------------------------
    def candidate_features(
        self, symbols: list[str], now: float,
        daily_stats: Optional[dict[str, dict]] = None,
    ) -> dict[str, Any]:
        """Scanner-compatible provider: dict[symbol] -> CandidateFeatures.

        The native GPU path supplies the directional/structure features; `price`
        and ADV/ADDV are merged from `daily_stats` (the scanner's REST daily
        mechanism) when provided — never fabricated here.
        """
        from market_scan.ranking import CandidateFeatures

        rows = self.compute(symbols)
        daily_stats = daily_stats or {}
        out: dict[str, Any] = {}
        for sym in symbols:
            r = rows.get(sym)
            ds = daily_stats.get(sym, {})
            if r is None:
                out[sym] = CandidateFeatures(symbol=sym, price=ds.get("price"),
                                             avg_daily_volume=ds.get("adv"),
                                             avg_daily_dollar_volume=ds.get("addv"))
                continue
            vwap_dist = ((r.price - r.vwap) / r.vwap) if r.vwap > 0 else None
            bb_dist = ((r.price - r.bb_middle) / r.bb_middle) if r.bb_middle > 0 else None
            directional_row = {
                "pdi": r.di_plus if (r.tech_valid & native.TECH_DMI) else None,
                "mdi": r.di_minus if (r.tech_valid & native.TECH_DMI) else None,
                "adx": r.adx if (r.tech_valid & native.TECH_ADX) else 0.0,
                "vwap_dist": vwap_dist or 0.0,
                "bb_dist": bb_dist or 0.0,
                "short_mom": r.momentum_1m_pct if r.momentum_1m_pct == r.momentum_1m_pct else 0.0,
            }
            out[sym] = CandidateFeatures(
                symbol=sym,
                price=ds.get("price", r.price if r.price > 0 else None),
                avg_daily_volume=ds.get("adv"),
                avg_daily_dollar_volume=ds.get("addv"),
                directional_row=directional_row,
                vwap_dist=vwap_dist,
                bb_mid_dist=bb_dist)
        return out
