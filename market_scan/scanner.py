"""MarketScanner — the in-process real-time IEX scanner orchestrator.

Owns a single IEX websocket (via StreamClient) feeding one OrderFlowEngine,
runs a ranking tick (~3s) and a broader discovery tick (~300s), maintains the
MONITORING union (open positions + pending orders + market references) that is
always subscribed and never pruned, ranks the discovered ENTRY candidates
(context-only proxies excluded), and publishes the authoritative universe.txt +
live_context.json through the UniversePublisher.

Heavy/external collaborators (broker, stream, feature computation, discovery,
clock, sleep) are injectable so the orchestration logic is unit-tested with
fakes and no sockets/network. The default feature provider computes ADV/ADDV/
price from Alpaca IEX daily bars + microstructure from the live order-flow
snapshot; it is fail-safe and validated by live connectivity testing, not unit
tests. The engine's own entry gates re-verify every candidate at entry time, so
the ranking need not replicate the engine's exact DMI/VWAP/Bollinger math.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Optional

import amps_v4_regime
import universe
from market_scan import discovery as discovery_mod
from market_scan import ranking as ranking_mod
from market_scan.config import MarketScanConfig
from market_scan.health import STATUS_HEALTHY, STATUS_UNHEALTHY
from market_scan.publisher import UniversePublisher
from market_scan.ranking import CandidateFeatures, MarketContext
from market_scan.subscription_manager import SubscriptionManager
from order_flow_engine import OrderFlowConfig, OrderFlowEngine

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_UNIVERSE_PATH = PROJECT_ROOT / "universe.txt"


def _live_context_path(mode: str) -> Path:
    # Namespace by mode so a paper scanner and a live scanner in the same
    # checkout never clobber each other's health metadata (locked decision #1).
    return Path(__file__).resolve().parent / f"live_context_{mode}.json"


class MarketScanner:
    def __init__(
        self,
        executor: Any,
        config: MarketScanConfig,
        *,
        universe_path: str | Path | None = None,
        context_path: str | Path | None = None,
        stream_client: Any = None,
        feature_provider: Optional[Callable[[list[str], float], dict]] = None,
        rank_fn: Optional[Callable[..., Any]] = None,
        discovery_fn: Optional[Callable[..., Any]] = None,
        sleep_fn=None,
        now_fn=None,
        log_fn: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.executor = executor
        self.broker = getattr(executor, "broker", None)
        self.config = config
        exec_cfg = getattr(executor, "config", None)
        self.mode = "paper" if getattr(exec_cfg, "paper", True) else "live"
        self._log = log_fn or (lambda m: logger.info(m))

        self.ofe = OrderFlowEngine(OrderFlowConfig(
            windows_s=tuple(config.order_flow_windows_s),
            max_subscriptions=config.max_subscriptions,
            stale_quote_s=config.max_quote_age_seconds,
        ))
        # positions, pending, references all mandatory (never rotated out).
        self.sub_mgr = SubscriptionManager(config.max_subscriptions, mandatory_groups=3)
        self.publisher = UniversePublisher(
            universe_path or DEFAULT_UNIVERSE_PATH,
            context_path or _live_context_path(self.mode),
            config,
            mode=self.mode,
        )
        self.stream = stream_client if stream_client is not None else self._make_stream()
        self.feature_provider = feature_provider or self._default_features
        # Ranking backend: defaults to the pure-Python ranking.rank; the engine
        # injects a GPU-backed rank_fn (identical semantics, composite score on
        # Metal) to put market-wide ranking on the GPU (directive §3/§5).
        self.rank_fn = rank_fn or ranking_mod.rank
        self.discovery_fn = discovery_fn or discovery_mod.discover_us_listed

        self._now_fn = now_fn
        self._sleep_fn = sleep_fn
        self._regime_cfg = amps_v4_regime.RegimeConfig()

        self._discovered: list[str] = []
        self._stopped = False
        self._last_discovery = 0.0
        self._force_publish = False
        self._subscribe_ok = False   # last apply_subscriptions() succeeded
        self._last_plan: Any = None
        self._errors: list[str] = []
        self._hist_client: Any = None
        # Cached daily ADV/ADDV/price, refreshed on the discovery cadence (NOT
        # per rank tick) so the ~3s rank loop never makes a blocking REST call.
        self._daily_stats: dict[str, dict] = {}
        # Accumulated per-symbol scores (symbol -> (score, eligible, ts)) so the
        # PUBLISHED entry universe is the stable top-N across rotation, decoupled
        # from the rotating subscription window that only covers part of the
        # discovered universe each tick.
        self._scored: dict[str, tuple] = {}
        # Shared immutable position/pending snapshot (refreshed off-loop) so the
        # rank tick never polls the broker directly (decoupled from the heartbeat).
        self._pos_cache: tuple = ((), (), 0.0)
        self._last_pos_refresh: float = 0.0

    # ------------------------------------------------------------------ setup
    def _make_stream(self):
        from market_scan.stream_client import StreamClient
        sc = StreamClient(
            api_key=getattr(self.broker, "api_key", None),
            secret_key=getattr(self.broker, "secret_key", None),
            feed=self.config.data_feed,
            on_trade=self.ofe.ingest_trade,
            on_quote=self.ofe.ingest_quote,
            reconnect=self.config.auto_reconnect,
            log_fn=self._log,
        )
        sc._now_fn = self._now_fn  # share the clock so is_live() is consistent
        return sc

    def _now(self) -> float:
        if self._now_fn is not None:
            return self._now_fn()
        import time
        return time.time()

    # -------------------------------------------------------------- monitoring
    def _refresh_positions_snapshot(self) -> None:
        """Refresh the shared, immutable position/pending snapshot.

        Blocking broker reads; called OFF the event loop on position_refresh_s
        cadence (not every ~3s rank tick) so the scanner does not redundantly
        poll -- or contend on -- the shared broker/TradingClient used by the
        execution heartbeat. Open-position EXIT risk is unaffected: the executor
        reconciles positions itself via REST; this snapshot only feeds the
        scanner's monitoring/subscription set and order-flow signals.
        """
        positions: list[str] = []
        pending: list[str] = []
        try:
            positions = [str(getattr(p, "symbol", "")).upper()
                         for p in (self.broker.get_all_open_positions() or [])
                         if getattr(p, "symbol", None)]
        except Exception as exc:  # noqa: BLE001
            self._errors.append(f"positions: {exc}")
        try:
            pending = [str(getattr(o, "symbol", "")).upper()
                       for o in (self.broker.list_open_orders() or [])
                       if getattr(o, "symbol", None)]
        except Exception as exc:  # noqa: BLE001
            self._errors.append(f"pending: {exc}")
        # Immutable tuples so readers can never mutate the shared snapshot.
        self._pos_cache = (tuple(positions), tuple(pending), self._now())

    def _open_symbols(self) -> tuple[list[str], list[str]]:
        """Return the cached (positions, pending) snapshot -- NO per-tick broker
        call. The snapshot is refreshed off-loop by
        :meth:`_refresh_positions_snapshot` (seeded once before run()'s loop). A
        single lazy seed covers the cold-cache case; steady-state rank ticks never
        poll the broker. Returns copies so callers cannot mutate shared state.
        """
        positions, pending, ts = self._pos_cache
        if ts == 0.0:  # cold cache (e.g. a direct rank tick before run() seeded it)
            self._refresh_positions_snapshot()
            positions, pending, ts = self._pos_cache
        return list(positions), list(pending)

    def _monitoring_union(self, positions, pending) -> set[str]:
        return set(positions) | set(pending) | {s.upper() for s in self.config.market_refs}

    def _entry_candidates(self) -> list[str]:
        proxies = {s.upper() for s in self.config.market_refs}
        seen: set[str] = set()
        out: list[str] = []
        for s in self._discovered:
            su = str(s).upper()
            if su in proxies or su in seen:
                continue
            seen.add(su)
            out.append(su)
        return out

    # ------------------------------------------------------------------ ticks
    def _discovery_tick(self, now: float) -> None:
        positions, pending = self._open_symbols()
        extra = sorted(self._monitoring_union(positions, pending))
        try:
            res = self.discovery_fn(extra=extra)
            symbols = list(getattr(res, "symbols", []) or [])
            if symbols:
                self._discovered = symbols
            if getattr(res, "error", None):
                self._errors.append(f"discovery: {res.error}")
        except Exception as exc:  # noqa: BLE001 - never kill the loop on discovery
            self._errors.append(f"discovery: {exc}")
        self._last_discovery = now
        self._force_publish = True  # refresh the published rank order on discovery

    async def _rank_tick(self, now: float) -> None:
        positions, pending = self._open_symbols()
        refs = [s.upper() for s in self.config.market_refs]
        entry_pool = self._entry_candidates()

        priority_groups = [positions, pending, refs, entry_pool]
        plan = self.sub_mgr.plan(priority_groups)
        self._last_plan = plan
        try:
            await self.stream.apply_subscriptions(to_add=plan.to_add, to_remove=plan.to_remove)
            self.sub_mgr.commit(plan)
            self._subscribe_ok = True
        except Exception as exc:  # noqa: BLE001
            self._errors.append(f"subscribe: {exc}")
            self._subscribe_ok = False
        # Explicit subscription-overflow policy: a dropped MANDATORY symbol
        # (position/pending/reference) is a loud warning + recorded error; a
        # dropped candidate is routine rotation under the cap.
        if plan.mandatory_dropped:
            msg = (f"[market-scan] SUBSCRIPTION OVERFLOW: cap={self.config.max_subscriptions} "
                   f"exceeded; dropped mandatory {plan.mandatory_dropped} "
                   f"(position exits remain safe via REST reconciliation)")
            self._log(msg)
            self._errors.append(msg)
        elif plan.dropped:
            self._log(f"[market-scan] {len(plan.dropped)} candidate(s) rotated out this cycle (cap)")

        subscribed = list(plan.target)
        # Evict order-flow accumulators for symbols that rotated out of coverage
        # (bounds memory over a long session); always keep monitoring symbols.
        try:
            self.ofe.prune(set(subscribed) | self._monitoring_union(positions, pending))
        except Exception:  # noqa: BLE001
            pass

        try:
            feats = self.feature_provider(subscribed, now) or {}
        except Exception as exc:  # noqa: BLE001
            self._errors.append(f"features: {exc}")
            feats = {}

        market_ctx = self._market_context(feats)
        rank_inputs = [feats[s] for s in entry_pool if s in feats]
        ranked = self.rank_fn(rank_inputs, config=self.config, market=market_ctx)

        # Accumulate scores so the published universe is the STABLE top-N across
        # rotation (not just this 3s window). BUT only record a score when the
        # candidate's OWN market data is fresh, and expire scores whose data has
        # not been refreshed within candidate_score_ttl_s -- a historical score
        # sitting in the rotation cache must never keep a stale candidate live.
        for r in ranked:
            if self._candidate_fresh(feats.get(r.symbol), now):
                self._scored[r.symbol] = (r.score, r.eligible, now)
            # stale-data candidate: leave any prior entry to expire naturally
        discovered_set = set(entry_pool)
        ttl = max(1.0, self.config.candidate_score_ttl_s)
        self._scored = {
            s: v for s, v in self._scored.items()
            if s in discovered_set and (now - v[2]) <= ttl
        }
        eligible = [
            s for s, _v in sorted(
                ((s, v) for s, v in self._scored.items() if v[1]),
                key=lambda kv: kv[1][0], reverse=True,
            )
        ][: self.config.target_universe_size]

        health = self._health(now, plan, market_ctx, positions, pending, feats)
        # Publish per-candidate REPLACEMENT eligibility (the Section-9 gates) so
        # the live engine can emit a gated replacement BUY without recomputing or
        # duplicating the optimizer. Rank order preserved; only all-gates-pass
        # candidates are listed.
        health["replacement_candidates"] = self._replacement_candidates(
            eligible, feats, market_ctx)
        try:
            self.publisher.publish(eligible, now=now, force_rewrite=self._force_publish,
                                   health=health)
        except Exception as exc:  # noqa: BLE001
            self._errors.append(f"publish: {exc}")
        self._force_publish = False

    # ---------------------------------------------------------------- context
    def _market_context(self, feats: dict) -> MarketContext:
        ref_rows: dict[str, dict] = {}
        for r in ("SPY", "QQQ", "DIA", "IWM"):
            cf = feats.get(r)
            if cf is not None and getattr(cf, "directional_row", None):
                ref_rows[r] = cf.directional_row
        label, score, _conf = amps_v4_regime.market_regime(ref_rows, self._regime_cfg)
        spy = amps_v4_regime.directional_score(ref_rows.get("SPY", {}), self._regime_cfg) if "SPY" in ref_rows else 0.0
        qqq = amps_v4_regime.directional_score(ref_rows.get("QQQ", {}), self._regime_cfg) if "QQQ" in ref_rows else 0.0
        spy = 0.0 if spy != spy else spy
        qqq = 0.0 if qqq != qqq else qqq
        sc = 0.0 if (score != score) else float(score)
        return MarketContext(regime_label=label, regime_score=sc, spy_score=spy,
                             qqq_score=qqq, cfg=self._regime_cfg)

    def _replacement_candidates(self, eligible, feats, market_ctx) -> list[dict]:
        """Section-9 per-candidate replacement eligibility, in rank order.

        Evaluates the gates (seller exhaustion, positive momentum, DI+ > DI-,
        price > BB-mid, price > VWAP, liquidity, regime alignment) from live
        features and lists only all-gates-pass candidates so the engine's
        replacement BUY is authorized by the gates, not by universe membership.
        """
        from market_scan.replacement import eligibility_from_features
        out: list[dict] = []
        for sym in eligible:
            f = feats.get(sym)
            if f is None:
                continue
            e = eligibility_from_features(f, market_ctx, self.config)
            # Publish on the SCANNER-side subset (momentum/liquidity/regime/±SE);
            # the engine enforces DI+/DI-/BB-mid/VWAP via confirm_buy_direction.
            if e.scanner_side_eligible:
                out.append({
                    "symbol": sym,
                    "price": getattr(f, "price", None),
                    "gates": {
                        "seller_exhaustion": e.seller_exhaustion,
                        "positive_momentum": e.positive_momentum,
                        "di_plus_gt_minus": e.di_plus_gt_minus,
                        "above_bb_mid": e.above_bb_mid,
                        "above_vwap": e.above_vwap,
                        "liquidity_ok": e.liquidity_ok,
                        "regime_aligned": e.regime_aligned,
                    },
                })
        return out

    def _candidate_fresh(self, cf: Any, now: float) -> bool:
        """Candidate-SPECIFIC freshness: this symbol's own quote AND trade must
        be recent. Never infer a candidate's freshness from other symbols."""
        snap = getattr(cf, "snapshot", None) if cf is not None else None
        if snap is None:
            return False
        q = getattr(snap, "last_quote_age_s", None)
        t = getattr(snap, "last_trade_age_s", None)
        maxage = self.config.max_quote_age_seconds
        if not isinstance(q, (int, float)) or q > maxage:
            return False
        if not isinstance(t, (int, float)) or t > maxage:
            return False
        return True

    def _references_fresh(self, feats: dict, now: float) -> bool:
        """Overall session data health = the MARKET REFERENCES are fresh (so the
        regime read is valid). Requires at least one primary reference (SPY/QQQ)
        to have fresh data -- NOT the freshest symbol anywhere in the universe."""
        for r in ("SPY", "QQQ"):
            if self._candidate_fresh(feats.get(r), now):
                return True
        return False

    def _stream_live(self, now: float) -> bool:
        """Real websocket readiness: the stream must have DELIVERED a message
        recently. Subscription registration alone is not connectivity."""
        is_live = getattr(self.stream, "is_live", None)
        if callable(is_live):
            try:
                return bool(is_live(now, self.config.stream_live_staleness_s))
            except Exception:  # noqa: BLE001
                return False
        # Fallback for stream stubs without is_live: subscription success proxy.
        return bool(self._subscribe_ok)

    def _health(self, now, plan, market_ctx, positions, pending, feats) -> dict:
        live = self._stream_live(now)
        refs_fresh = self._references_fresh(feats, now)
        # Healthy requires an actually-live stream AND fresh reference data. Each
        # PUBLISHED candidate is separately gated on its own freshness (so a
        # fresh reference never vouches for a stale candidate).
        status = STATUS_HEALTHY if (self._subscribe_ok and live and refs_fresh) else STATUS_UNHEALTHY
        if not self._subscribe_ok:
            conn = "disconnected"
        elif live:
            conn = "connected"
        else:
            conn = "connecting"  # subscribed, no inbound data yet
        return {
            "scanner_status": status,
            "market_data_fresh": refs_fresh,
            "stream_live": live,
            "connection_status": conn,
            "regime": market_ctx.regime_label,
            "regime_score": round(market_ctx.regime_score, 2),
            "discovered_count": len(self._discovered),
            "active_subscriptions": plan.active,
            "available_subscriptions": plan.available,
            "subscription_cap": self.config.max_subscriptions,
            "subscription_overflow": bool(plan.overflow),
            "mandatory_dropped": list(plan.mandatory_dropped),
            "monitoring_count": len(self._monitoring_union(positions, pending)),
            "last_discovery_at": self._last_discovery,
            "errors": self._errors[-10:],
            "error_count": len(self._errors),
        }

    # ------------------------------------------------------------------- loop
    async def run(self) -> None:
        import asyncio
        loop = asyncio.get_running_loop()
        sleep = self._sleep_fn or asyncio.sleep

        async def _offload(fn, *a):
            # Run blocking I/O (web discovery, REST daily-stats) off the event
            # loop so the websocket _run_forever() task is never starved.
            try:
                await loop.run_in_executor(None, fn, *a)
            except Exception as exc:  # noqa: BLE001
                self._errors.append(f"offload {getattr(fn, '__name__', fn)}: {exc}")

        # Kick off the websocket run in the background (reconnects itself).
        stream_task = None
        try:
            stream_task = asyncio.create_task(self._run_stream())
        except Exception as exc:  # noqa: BLE001
            self._errors.append(f"stream start: {exc}")

        await _offload(self._refresh_positions_snapshot)
        self._last_pos_refresh = self._now()
        await _offload(self._discovery_tick, self._now())
        await _offload(self._refresh_daily_stats)
        try:
            while not self._stopped:
                now = self._now()
                # Refresh the shared position snapshot off-loop on its own
                # (slower) cadence -- not on every rank tick.
                if now - self._last_pos_refresh >= self.config.position_refresh_s:
                    await _offload(self._refresh_positions_snapshot)
                    self._last_pos_refresh = now
                if now - self._last_discovery >= self.config.discovery_interval_seconds:
                    await _offload(self._discovery_tick, now)
                    await _offload(self._refresh_daily_stats)
                await self._rank_tick(now)
                maybe = sleep(self.config.rank_refresh_seconds)
                if maybe is not None and hasattr(maybe, "__await__"):
                    await maybe
        finally:
            self._stopped = True
            try:
                self.stream.stop()
            except Exception:  # noqa: BLE001
                pass
            if stream_task is not None:
                stream_task.cancel()

    async def _run_stream(self) -> None:
        try:
            await self.stream.run()
        except Exception as exc:  # noqa: BLE001
            self._subscribe_ok = False
            self._errors.append(f"stream: {exc}")

    def stop(self) -> None:
        self._stopped = True
        try:
            self.stream.stop()
        except Exception:  # noqa: BLE001
            pass

    # -------------------------------------------------- default feature provider
    def _default_features(self, symbols: list[str], now: float) -> dict:
        """Best-effort live features: ADV/ADDV/price from IEX daily bars +
        microstructure from the live order-flow snapshot. Fail-safe (never
        raises); symbols lacking data simply get sparse features (and fail the
        liquidity gate -> excluded, never fabricated as eligible)."""
        out: dict[str, CandidateFeatures] = {}
        # Read the cached daily stats (refreshed off-loop on the discovery
        # cadence); NEVER make a blocking REST call from the hot rank loop.
        daily = self._daily_stats
        for s in symbols:
            snap = None
            try:
                snap = self.ofe.snapshot(s, now)
            except Exception:  # noqa: BLE001
                snap = None
            d = daily.get(s, {})
            spread_pct = None
            quote_age = None
            if snap is not None and getattr(snap, "mid", None) and getattr(snap, "spread", None):
                try:
                    spread_pct = float(snap.spread) / float(snap.mid) if snap.mid else None
                except Exception:  # noqa: BLE001
                    spread_pct = None
                quote_age = getattr(snap, "last_quote_age_s", None)
            out[s] = CandidateFeatures(
                symbol=s,
                price=d.get("price"),
                avg_daily_volume=d.get("adv"),
                avg_daily_dollar_volume=d.get("addv"),
                rel_volume=d.get("rel_volume"),
                spread_pct=spread_pct,
                quote_age_s=quote_age,
                directional_row=d.get("directional_row", {}),
                snapshot=snap,
                vwap_dist=d.get("vwap_dist"),
                bb_mid_dist=d.get("bb_mid_dist"),
            )
        return out

    def _refresh_daily_stats(self) -> None:
        """Refresh the cached daily ADV/ADDV/price for the discovered universe.

        Blocking (REST); called off the event loop via run_in_executor on the
        discovery cadence only, so the ~3s rank loop stays non-blocking. Updates
        incrementally and never raises.
        """
        try:
            syms = list(self._discovered)[: max(1, 5 * self.config.max_subscriptions)]
            stats = self._fetch_daily_stats(syms)
            if stats:
                self._daily_stats = stats
        except Exception as exc:  # noqa: BLE001
            self._errors.append(f"daily refresh: {exc}")

    def _historical(self):
        if self._hist_client is None:
            try:
                from alpaca.data.historical import StockHistoricalDataClient
                self._hist_client = StockHistoricalDataClient(
                    self.broker.api_key, self.broker.secret_key)
            except Exception as exc:  # noqa: BLE001
                self._errors.append(f"hist client: {exc}")
                self._hist_client = None
        return self._hist_client

    def _fetch_daily_stats(self, symbols: list[str]) -> dict:
        """20-day ADV/ADDV/last-price from IEX daily bars. Fail-safe -> {}."""
        client = self._historical()
        if client is None or not symbols:
            return {}
        try:
            from datetime import datetime, timedelta, timezone
            from alpaca.data.requests import StockBarsRequest
            from alpaca.data.timeframe import TimeFrame
            try:
                from alpaca.data.enums import DataFeed
                feed = DataFeed.IEX
            except Exception:  # noqa: BLE001
                feed = None
            start = datetime.now(timezone.utc) - timedelta(days=40)
            kwargs: dict[str, Any] = {"symbol_or_symbols": list(symbols),
                                      "timeframe": TimeFrame.Day, "start": start}
            if feed is not None:
                kwargs["feed"] = feed
            resp = client.get_stock_bars(StockBarsRequest(**kwargs))
            data = getattr(resp, "data", resp) or {}
            stats: dict[str, dict] = {}
            for sym in symbols:
                bars = data.get(sym) or data.get(sym.upper()) or []
                closes = [float(getattr(b, "close", 0.0) or 0.0) for b in bars]
                vols = [float(getattr(b, "volume", 0.0) or 0.0) for b in bars]
                if not closes:
                    continue
                window_v = vols[-20:] if len(vols) >= 1 else vols
                window_c = closes[-20:] if len(closes) >= 1 else closes
                adv = sum(window_v) / len(window_v) if window_v else 0.0
                addv = (sum(c * v for c, v in zip(window_c, window_v)) / len(window_v)
                        if window_v else 0.0)
                price = closes[-1]
                short_mom = (closes[-1] - closes[-6]) / closes[-6] if len(closes) >= 6 and closes[-6] else 0.0
                rel_volume = (vols[-1] / adv) if adv > 0 else 1.0
                stats[sym] = {
                    "price": price, "adv": adv, "addv": addv,
                    "rel_volume": rel_volume,
                    "directional_row": {"short_mom": short_mom},
                }
            return stats
        except Exception as exc:  # noqa: BLE001
            self._errors.append(f"daily stats: {exc}")
            return {}
