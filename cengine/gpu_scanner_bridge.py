"""Phase B.5b — wire the GPU discovery provider into the live MarketScanner.

The scanner's ``feature_provider`` seam is the single, architecture-preserving
plug-in point (one socket, one scanner, one UniversePublisher). This module
provides:

  * BarAggregator — folds completed 1-minute bars (the only bar cadence Alpaca
    IEX streams) into completed 5-minute bars for the DiscoveryEngine's 5m
    buffer, which the technical_features kernel requires (§29).
  * make_composed_feature_provider — a provider that keeps the scanner's base
    features (price / ADV / ADDV / order-flow snapshot / spread, which the hard
    liquidity gate and order-flow score need) and OVERLAYS the native GPU
    directional features (DI+/DI-/ADX/VWAP/Bollinger distances) computed by the
    GpuDiscoveryProvider. GPU work never authorizes orders; it only shapes the
    advisory ranking the scanner already performs.

The overlay is applied only when the GPU technical features are valid for a
symbol, so insufficient history degrades gracefully to the scanner's prior
behavior rather than fabricating directional signal.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

import _cengine_native as native

from cengine.gpu_discovery import GpuDiscoveryProvider

_NS_PER_MIN = 60 * 1_000_000_000


class BarAggregator:
    """Fold completed 1-minute bars into completed N-minute bars (tick units).

    ``add_1m`` returns a completed N-minute bar tuple
    ``(symbol, ts_ns, open, high, low, close, volume)`` when a bucket rolls over,
    else None. All price fields are int64 ticks (passed through unchanged).
    """

    def __init__(self, minutes: int = 5) -> None:
        self._span = max(1, int(minutes)) * _NS_PER_MIN
        self._acc: dict[str, dict] = {}

    def add_1m(self, symbol: str, ts_ns: int, o: int, h: int, lo: int, c: int,
               vol: int) -> Optional[tuple]:
        bucket = int(ts_ns) // self._span
        a = self._acc.get(symbol)
        completed: Optional[tuple] = None
        if a is None or a["bucket"] != bucket:
            if a is not None:
                completed = (symbol, a["ts"], a["o"], a["h"], a["lo"], a["c"], a["v"])
            self._acc[symbol] = {"bucket": bucket, "ts": int(ts_ns), "o": int(o),
                                 "h": int(h), "lo": int(lo), "c": int(c),
                                 "v": int(vol)}
        else:
            a["h"] = max(a["h"], int(h))
            a["lo"] = min(a["lo"], int(lo))
            a["c"] = int(c)
            a["v"] += int(vol)
            a["ts"] = int(ts_ns)
        return completed

    def flush(self, symbol: str) -> Optional[tuple]:
        """Emit the in-progress bucket for `symbol` as a (provisional) bar."""
        a = self._acc.pop(symbol, None)
        if a is None:
            return None
        return (symbol, a["ts"], a["o"], a["h"], a["lo"], a["c"], a["v"])


class EngineGpuDiscovery:
    """Engine-side glue: one market-event tap + the scanner feature_provider.

    Owns a GpuDiscoveryProvider and a 1m->5m aggregator. The live MarketEngine
    feeds every normalized market event through :meth:`on_market_event` (cheap:
    bars are ingested, trades are cached and flushed once per rank tick), and the
    scanner's feature_provider is replaced with :meth:`feature_provider_for`. The
    discovery universe grows monotonically (symbols are never dropped mid-session,
    so bar history / generations are preserved).
    """

    def __init__(self, *, bar_capacity: int = 256, window: int = 64,
                 agg_minutes: int = 5, prefer_metal: bool = True) -> None:
        self.provider = GpuDiscoveryProvider(
            bar_capacity=bar_capacity, window=window, prefer_metal=prefer_metal)
        self._agg = BarAggregator(agg_minutes)
        self._latest_trade: dict[str, tuple[int, int]] = {}
        self._universe: set[str] = set()
        # Latest operationally-consumed market breadth (BreadthCounters | None).
        self.latest_breadth: Any = None
        self.latest_breadth_backend: str = ""
        # Latest GPU candidate compaction: (entry_qualified_ids, total_qualified).
        self.latest_compaction: tuple = ([], 0)

    @property
    def backend(self) -> str:
        return self.provider.backend

    def ensure_symbols(self, symbols) -> None:
        new = {s for s in symbols if s not in self._universe}
        if new:
            self._universe |= new
            self.provider.set_universe(sorted(self._universe))

    def register_universe(self, symbols) -> int:
        """Bulk-register the full master universe into the native DiscoveryEngine
        up front (directive item 3), so symbol ids / bar-ring generations exist
        for every tradable member before the first market event or warm-up batch.

        Unlike the lazy :meth:`ensure_symbols` path (which only registers symbols
        that actually stream or get scanned), this guarantees the GPU breadth /
        compaction / screen kernels see the WHOLE market, not just the subset that
        happened to tick. Returns the number of symbols newly registered. The
        discovery universe still only grows (never shrinks mid-session)."""
        before = len(self._universe)
        self.ensure_symbols(
            str(s).upper().strip() for s in symbols if str(s).strip())
        return len(self._universe) - before

    @property
    def universe_size(self) -> int:
        return len(self._universe)

    def on_market_event(self, event: Any) -> None:
        """Dispatch a normalized BarEvent / TradeEvent (others ignored). Never
        raises — market ingestion must not be perturbed by discovery."""
        try:
            name = type(event).__name__
            sym = getattr(event, "symbol", None)
            if not sym:
                return
            if name == "BarEvent":
                self.ensure_symbols([sym])
                o, h = int(event.open_ticks), int(event.high_ticks)
                lo, c = int(event.low_ticks), int(event.close_ticks)
                vol = int(event.volume)
                ts = int(event.timestamp_ns)
                self.provider.ingest_bars([(sym, ts, o, h, lo, c, vol)],
                                          native.BarTimeframe.ONE_MINUTE)
                rolled = self._agg.add_1m(sym, ts, o, h, lo, c, vol)
                if rolled is not None:
                    self.provider.ingest_bars([rolled], native.BarTimeframe.FIVE_MINUTE)
            elif name == "TradeEvent":
                self._latest_trade[sym] = (int(event.timestamp_ns),
                                           int(event.price_ticks))
        except Exception:  # noqa: BLE001 — discovery must never break ingestion
            pass

    def _flush_trades(self) -> None:
        if not self._latest_trade:
            return
        batch = [(s, ts, px, 1, 0) for s, (ts, px) in self._latest_trade.items()]
        self._latest_trade.clear()
        try:
            self.ensure_symbols([b[0] for b in batch])
            self.provider.ingest_trades(batch)
        except Exception:  # noqa: BLE001
            pass

    def update_breadth(self) -> Any:
        """Compute + store market breadth AND candidate compaction over the full
        active universe (one flag computation feeding both GPU kernels, CPU
        fallback). Operational consumer of market_breadth + compact_entry_candidates
        (invoked each rank tick). Returns the BreadthCounters."""
        syms = sorted(self._universe)
        if not syms:
            return None
        try:
            breadth, compaction = self.provider.breadth_and_compaction(syms)
            self.latest_breadth = breadth
            self.latest_compaction = compaction
            self.latest_breadth_backend = self.provider.backend
        except Exception:  # noqa: BLE001 — breadth/compaction are advisory telemetry
            return self.latest_breadth
        return self.latest_breadth

    def breadth_snapshot(self) -> dict:
        """Serialize the latest breadth for telemetry / live_context / dashboard."""
        b = self.latest_breadth
        if b is None:
            return {"available": False}
        return {"available": True, "backend": self.latest_breadth_backend,
                "total": b.total, "valid": b.valid,
                "entry_qualified": b.entry_qualified,
                "technical_exit": b.technical_exit, "bullish": b.bullish,
                "bearish": b.bearish, "positive_momentum": b.positive_momentum,
                "negative_momentum": b.negative_momentum,
                "above_vwap": b.above_vwap, "below_vwap": b.below_vwap,
                "data_invalid": b.data_invalid}

    def momentum_confirmed(self, symbols) -> set:
        """Return the subset of `symbols` whose GPU structural screen shows
        PRELIMINARY momentum + direction confirmation (directive item 3), the gate
        between a trade subscription and a quote subscription:

          * valid Wilder DMI,
          * DI+ (5m) > DI- (5m)        — directional,
          * developing 1-minute momentum > 0 (price above previous 1m close).

        This REUSES the native screen outputs; it is weaker than the strict final
        entry contract (which also requires a valid VWAP/SMA/BB reference, liquidity,
        and strategy/exposure/risk authorization) — those remain enforced downstream
        before any order. It is advisory: it never authorizes execution."""
        syms = [str(s).upper().strip() for s in symbols if str(s).strip()]
        if not syms:
            return set()
        try:
            rows = self.provider.compute(syms)
        except Exception:  # noqa: BLE001 — promotion gating must not break the loop
            return set()
        out: set = set()
        for s, r in rows.items():
            if not (r.tech_valid & native.TECH_DMI):
                continue
            mom = r.momentum_1m_pct
            if r.di_plus > r.di_minus and mom == mom and mom > 0.0:
                out.add(s)
        return out

    def rank_fn(self):
        """A drop-in replacement for market_scan.ranking.rank that computes the
        composite score on the GPU (CPU fallback)."""
        return make_gpu_rank(self.provider)

    def feature_provider_for(self, scanner: Any) -> Callable[[list[str], float], dict]:
        """Return a feature_provider bound to `scanner` that flushes cached trades,
        overlays GPU directional features on the scanner's base features, and
        updates the operationally-consumed market breadth."""
        composed = make_composed_feature_provider(scanner, self.provider)

        def provider(symbols: list[str], now: float) -> dict:
            self.ensure_symbols(symbols)
            self._flush_trades()
            feats = composed(symbols, now)
            self.update_breadth()  # operational consumer of market_breadth kernel
            return feats

        return provider


def make_gpu_rank(gpu: GpuDiscoveryProvider) -> Callable:
    """Build a rank() replacement that computes the composite candidate score on
    the GPU rank_candidates kernel (CPU fallback), producing RankedCandidate output
    IDENTICAL to market_scan.ranking.rank. The per-candidate raw components are
    computed by REUSING the exact scanner functions (directional_score,
    _order_flow_component, _direction_sign, passes_liquidity), so only the
    composite arithmetic moves to the GPU — no semantic drift (directive §3)."""
    import amps_v4_regime
    from market_scan import ranking as rk

    def gpu_rank(features, *, config, market, weights=None):
        feats = list(features)
        if not feats:
            return []
        w = weights or rk.RankWeights()
        cfg = market.cfg or amps_v4_regime.RegimeConfig()
        inputs = []
        meta = []
        for f in feats:
            rs = amps_v4_regime.directional_score(f.directional_row, cfg)
            has_rs = rs == rs  # not NaN
            spread = f.spread_pct
            elig, reason = rk.passes_liquidity(f, config)
            inputs.append(native.RankInput(
                rs=float(rs) if has_rs else 0.0,
                short_mom=float(f.directional_row.get("short_mom", 0.0) or 0.0),
                vwap_dist=float(f.vwap_dist or 0.0),
                bb_mid_dist=float(f.bb_mid_dist or 0.0),
                rel_volume=float(f.rel_volume if f.rel_volume is not None else 1.0),
                addv=float(f.avg_daily_dollar_volume or 0.0),
                spread_pct=float(spread) if spread is not None else 0.0,
                order_flow=float(rk._order_flow_component(f.snapshot)),
                dir_sign=float(rk._direction_sign(f.symbol)),
                has_rs=1 if has_rs else 0,
                has_spread=1 if spread is not None else 0,
                eligible=1 if elig else 0))
            meta.append((f.symbol, elig, reason, rs if has_rs else float("nan")))
        params = native.RankParams(
            w_rel_strength=w.rel_strength, w_momentum=w.momentum,
            w_volume_expansion=w.volume_expansion, w_liquidity=w.liquidity,
            w_order_flow=w.order_flow, w_regime_align=w.regime_align,
            spy_score=market.spy_score, qqq_score=market.qqq_score,
            regime_score=market.regime_score)
        outs = gpu.rank(inputs, params)
        ranked = []
        for (sym, elig, reason, rs), o in zip(meta, outs, strict=True):
            detail = {"rel_strength": round(o.rel_strength, 4),
                      "momentum": round(o.momentum, 4),
                      "volume_expansion": round(o.volume_expansion, 4),
                      "liquidity": round(o.liquidity, 4),
                      "order_flow": round(o.order_flow, 4),
                      "regime_align": round(o.regime_align, 4),
                      "directional_score": None if rs != rs else round(rs, 2),
                      "backend": gpu.backend}
            ranked.append(rk.RankedCandidate(sym, o.total_score, elig, reason, detail))
        ranked.sort(key=lambda r: (r.eligible, r.score), reverse=True)
        return ranked

    return gpu_rank


def make_composed_feature_provider(
    scanner: Any, gpu: GpuDiscoveryProvider,
) -> Callable[[list[str], float], dict]:
    """Return a feature_provider that overlays GPU directional features on top of
    the scanner's own base features. Assign to ``scanner.feature_provider`` AFTER
    construction. Fail-safe: any GPU error falls back to base features only.
    """
    base_provider = scanner._default_features  # bound: price/ADV/snapshot/spread

    def composed(symbols: list[str], now: float) -> dict:
        base = base_provider(symbols, now) or {}
        try:
            rows = gpu.compute(symbols)
        except Exception:  # noqa: BLE001 — never break the rank loop on GPU error
            return base
        for s in symbols:
            cf = base.get(s)
            r = rows.get(s)
            if cf is None or r is None:
                continue
            if not (r.tech_valid & native.TECH_DMI):
                continue  # insufficient history -> leave base features untouched
            vwap_dist = (r.price - r.vwap) / r.vwap if r.vwap > 0 else None
            bb_dist = (r.price - r.bb_middle) / r.bb_middle if r.bb_middle > 0 else None
            mom = r.momentum_1m_pct if r.momentum_1m_pct == r.momentum_1m_pct else 0.0
            cf.directional_row = {
                "pdi": r.di_plus,
                "mdi": r.di_minus,
                "adx": r.adx if (r.tech_valid & native.TECH_ADX) else 0.0,
                "vwap_dist": vwap_dist or 0.0,
                "bb_dist": bb_dist or 0.0,
                "short_mom": mom,
            }
            if vwap_dist is not None:
                cf.vwap_dist = vwap_dist
            if bb_dist is not None:
                cf.bb_mid_dist = bb_dist
        return base

    return composed
