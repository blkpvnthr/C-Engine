"""Automatic SIP historical warm-up (directive items 4/7).

Reusable PRODUCTION warm-up extracted from the certification harness logic: it
fetches SIP historical bars for the master universe in priority order and ingests
them into the existing native DiscoveryEngine buffers (via GpuDiscoveryProvider),
so GPU feature computation has adequate history. It tracks explicit per-symbol
warm-up status and never fabricates bars or mixes incompatible feeds.

Priority order (directive §6): held positions -> pending orders -> references ->
actionable candidates -> promising preliminary candidates -> remaining universe.

Runs independently of execution-critical protection (the caller schedules it off
the hot path). Fetching is pluggable (`fetch_fn`) so the orchestration — priority,
batching, status, ingestion — is unit-testable offline without network.
"""

from __future__ import annotations

import datetime
from enum import Enum
from typing import Any, Callable, Iterable, Optional

import _cengine_native as native

SCALE = 10_000


class WarmupStatus(str, Enum):
    NOT_REQUESTED = "not_requested"
    DOWNLOADING = "downloading"
    WARMED = "warmed"
    INCOMPLETE = "incomplete"          # fetched but below required depth
    FAILED = "failed"


def _to_ticks(x: float) -> int:
    return int(round(float(x) * SCALE))


def prioritize(*, positions=(), pending=(), references=(), actionable=(),
               preliminary=(), remaining=()) -> list[str]:
    """Deduplicated, priority-ordered warm-up list (§6)."""
    order = []
    seen: set[str] = set()
    for group in (positions, pending, references, actionable, preliminary, remaining):
        for s in group or []:
            s = str(s).upper().strip()
            if s and s not in seen:
                seen.add(s)
                order.append(s)
    return order


class HistoricalWarmup:
    def __init__(self, provider: Any, fetch_fn: Callable[..., dict], *,
                 batch_size: int = 100, min_5m_bars: int = 30,
                 min_1m_bars: int = 2) -> None:
        """`provider` is a GpuDiscoveryProvider; `fetch_fn(symbols, timeframe, start)
        -> {symbol: [bar,...]}` where each bar has t/o/h/l/c/v (dollars)."""
        self._provider = provider
        self._fetch = fetch_fn
        self._batch = max(1, int(batch_size))
        self._min_5m = int(min_5m_bars)
        self._min_1m = int(min_1m_bars)
        self._status: dict[str, WarmupStatus] = {}
        self._bars_ingested = {"5m": 0, "1m": 0}

    def status(self, symbol: str) -> WarmupStatus:
        return self._status.get(str(symbol).upper().strip(), WarmupStatus.NOT_REQUESTED)

    def _ingest(self, bars_by_sym: dict, tf) -> int:
        batch = []
        for sym, bars in bars_by_sym.items():
            if str(sym).startswith("__"):
                continue
            for b in bars:
                try:
                    ts = int(datetime.datetime.fromisoformat(
                        b["t"].replace("Z", "+00:00")).timestamp() * 1e9)
                except Exception:  # noqa: BLE001
                    ts = 0
                batch.append((sym, ts, _to_ticks(b["o"]), _to_ticks(b["h"]),
                              _to_ticks(b["l"]), _to_ticks(b["c"]), int(b.get("v", 0))))
        if not batch:
            return 0
        return self._provider.ingest_bars(batch, tf)

    def warm(self, symbols: Iterable[str], *, start_5m: str, start_1m: str) -> dict:
        """Fetch + ingest history for `symbols` in order, batched. Returns a
        coverage summary. Synchronous orchestration; the caller runs it off the
        hot path (e.g. in a thread/executor)."""
        syms = [str(s).upper().strip() for s in symbols if str(s).strip()]
        self._provider.set_universe(sorted(set(syms) | set(self._provider_universe())))
        for i in range(0, len(syms), self._batch):
            chunk = syms[i:i + self._batch]
            for s in chunk:
                self._status[s] = WarmupStatus.DOWNLOADING
            try:
                bars5 = self._fetch(chunk, "5Min", start_5m)
                bars1 = self._fetch(chunk, "1Min", start_1m)
            except Exception:  # noqa: BLE001 — a batch failure must not abort warm-up
                for s in chunk:
                    self._status[s] = WarmupStatus.FAILED
                continue
            if "__error__" in bars5 or "__error__" in bars1:
                for s in chunk:
                    self._status[s] = WarmupStatus.FAILED
                continue
            self._bars_ingested["5m"] += self._ingest(bars5, native.BarTimeframe.FIVE_MINUTE)
            self._bars_ingested["1m"] += self._ingest(bars1, native.BarTimeframe.ONE_MINUTE)
            for s in chunk:
                n5 = len(bars5.get(s, []))
                n1 = len(bars1.get(s, []))
                if n5 >= self._min_5m and n1 >= self._min_1m:
                    self._status[s] = WarmupStatus.WARMED
                elif n5 > 0 or n1 > 0:
                    self._status[s] = WarmupStatus.INCOMPLETE
                else:
                    self._status[s] = WarmupStatus.FAILED
        return self.coverage()

    def _provider_universe(self) -> list[str]:
        try:
            return list(self._provider.engine.symbols())
        except Exception:  # noqa: BLE001
            return []

    def coverage(self) -> dict:
        counts: dict[str, int] = {}
        for st in self._status.values():
            counts[st.value] = counts.get(st.value, 0) + 1
        return {"requested": len(self._status),
                "warmed": counts.get(WarmupStatus.WARMED.value, 0),
                "incomplete": counts.get(WarmupStatus.INCOMPLETE.value, 0),
                "failed": counts.get(WarmupStatus.FAILED.value, 0),
                "bars_ingested": dict(self._bars_ingested),
                "by_status": counts}


def make_sip_fetch_fn(key: str, sec: str,
                      base: str = "https://data.alpaca.markets") -> Callable[..., dict]:
    """Produce a batched SIP historical-bars fetcher (REST, paginated, feed=sip)."""
    import json
    import urllib.error
    import urllib.parse
    import urllib.request

    def fetch(symbols, timeframe, start, limit=10000) -> dict:
        out: dict[str, list] = {s: [] for s in symbols}
        params = {"symbols": ",".join(symbols), "timeframe": timeframe,
                  "feed": "sip", "start": start, "limit": str(limit),
                  "adjustment": "raw"}
        token: Optional[str] = None
        for _ in range(20):
            q = dict(params)
            if token:
                q["page_token"] = token
            url = f"{base.rstrip('/')}/v2/stocks/bars?{urllib.parse.urlencode(q)}"
            req = urllib.request.Request(url, method="GET")
            req.add_header("APCA-API-KEY-ID", key)
            req.add_header("APCA-API-SECRET-KEY", sec)
            try:
                with urllib.request.urlopen(req, timeout=30) as resp:
                    doc = json.loads(resp.read().decode("utf-8", "replace"))
            except urllib.error.HTTPError as e:
                return {"__error__": f"HTTP {e.code}"}
            except Exception as e:  # noqa: BLE001
                return {"__error__": f"{type(e).__name__}: {e}"}
            for sym, bars in (doc.get("bars") or {}).items():
                out.setdefault(sym, []).extend(bars)
            token = doc.get("next_page_token")
            if not token:
                break
        return out

    return fetch
