"""Historical market-data ingestion into the bronze lakehouse (Milestone 5, §5/§6/§7).

Reuses the EXISTING Alpaca SIP REST fetcher (``cengine.historical_warmup.make_sip_fetch_fn``)
— the same credentials and endpoint the warm-up path already uses. It opens NO new
websocket (the engine owns the single live socket) and performs NO silent SIP->IEX
fallback: if the SIP feed is unavailable/unentitled the batch is recorded as an
explicit failure, never downgraded.

Properties:
  * Bounded      — explicit per-run symbol cap and batch size.
  * Resumable    — per-(feed, timeframe, symbol) high-water checkpoint; a re-run skips
                   bars at/under the checkpoint (and Parquet parts are content-hashed,
                   so even without the checkpoint re-ingestion is idempotent).
  * Tiered       — priority-ordered tiers (held/pending/references/... -> remaining),
                   each with its own timeframe + lookback, matching §6 priority order.
  * Point-in-time — records the universe membership snapshot used for the run so future
                   training reconstructs the as-of universe (no survivorship leakage).

Monetary fields are stored in DEFAULT_PRICE_SCALE=10000 ticks. The bar's event time is
its OPEN; availability time is its CLOSE (a bar is only usable once the interval ends).
"""

from __future__ import annotations

import datetime
import json
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

from cengine.lakehouse.config import DEFAULT_PRICE_SCALE, LakehouseConfig
from cengine.lakehouse.parquet_writer import ParquetWriter

HISTORICAL_VERSION = "historical-ingestion-1.0.0"

# supported bar intervals -> seconds (for availability = open + interval)
_TIMEFRAME_SECONDS = {
    "1Min": 60, "5Min": 300, "15Min": 900, "30Min": 1800,
    "1Hour": 3600, "1Day": 86400,
}


def _to_ticks(x) -> int:
    return int(round(float(x) * DEFAULT_PRICE_SCALE))


def _iso_to_ns(iso: str) -> int:
    try:
        return int(datetime.datetime.fromisoformat(
            str(iso).replace("Z", "+00:00")).timestamp() * 1e9)
    except Exception:  # noqa: BLE001
        return 0


@dataclass
class HistoricalTier:
    """One priority tier of the ingestion plan (§5/§6)."""
    name: str
    symbols: tuple
    timeframe: str = "5Min"
    start: str = ""                 # ISO8601 lookback start


@dataclass
class IngestionReport:
    version: str = HISTORICAL_VERSION
    feed: str = "sip"
    started_ns: int = 0
    completed_ns: int = 0
    symbols_requested: int = 0
    symbols_ingested: int = 0
    bars_written: int = 0
    batches_failed: int = 0
    skipped_resumed: int = 0
    failures: list = field(default_factory=list)
    by_timeframe: dict = field(default_factory=dict)


class HistoricalSIPIngestion:
    """Bounded, resumable, tiered historical bar ingestion into bronze/market_bars."""

    def __init__(self, config: Optional[LakehouseConfig] = None, *,
                 fetch_fn: Optional[Callable[..., dict]] = None,
                 feed: str = "sip", batch_size: int = 100,
                 max_symbols_per_run: int = 5000) -> None:
        if feed.lower() not in ("sip", "delayed_sip"):
            # No silent downgrade: an IEX fetcher must be passed deliberately, not
            # substituted for SIP behind the caller's back.
            raise ValueError(f"historical ingestion is SIP-only here; got feed={feed!r}")
        self.config = config or LakehouseConfig.from_env()
        self.writer = ParquetWriter(self.config)
        self.writer.recover_incomplete()
        self._fetch = fetch_fn
        self.feed = feed.lower()
        self.batch = max(1, int(batch_size))
        self.max_symbols = max(1, int(max_symbols_per_run))
        self._ckpt_path = self.config.checkpoints / "historical_checkpoints.json"

    # -- checkpoints (resumability) ------------------------------------------
    def _load_ckpt(self) -> dict:
        if self._ckpt_path.exists():
            try:
                return json.loads(self._ckpt_path.read_text())
            except (OSError, json.JSONDecodeError):
                return {}
        return {}

    def _save_ckpt(self, ckpt: dict) -> None:
        tmp = self._ckpt_path.with_suffix(".tmp")
        try:
            tmp.write_text(json.dumps(ckpt, indent=2))
            tmp.replace(self._ckpt_path)
        except OSError:
            pass

    def _ckpt_key(self, timeframe: str, symbol: str) -> str:
        return f"{self.feed}|{timeframe}|{symbol}"

    # -- universe snapshot (point-in-time integrity §6) ----------------------
    def record_universe_snapshot(self, symbols, *, as_of_ns: Optional[int] = None,
                                 source: str = "historical_ingestion") -> None:
        """Persist the universe membership used for this run so training can
        reconstruct the as-of universe (prevents survivorship bias)."""
        as_of_ns = as_of_ns if as_of_ns is not None else time.time_ns()
        rows = [{"symbol_id": str(s).upper().strip(), "as_of_ns": as_of_ns,
                 "membership_source": source, "feed": self.feed,
                 "event_timestamp_ns": as_of_ns, "source_feed": self.feed}
                for s in symbols if str(s).strip()]
        if rows:
            self.writer.write_rows("bronze", "universe_membership", rows)

    # -- ingestion -----------------------------------------------------------
    def ingest_tier(self, tier: HistoricalTier, report: IngestionReport,
                    ckpt: dict) -> None:
        if self._fetch is None:
            report.failures.append({"tier": tier.name, "error": "no_fetch_fn"})
            return
        tf_secs = _TIMEFRAME_SECONDS.get(tier.timeframe, 60)
        syms = [str(s).upper().strip() for s in tier.symbols if str(s).strip()]
        for i in range(0, len(syms), self.batch):
            chunk = syms[i:i + self.batch]
            try:
                bars_by_sym = self._fetch(chunk, tier.timeframe, tier.start)
            except Exception as exc:  # noqa: BLE001 — one batch failing never aborts the run
                report.batches_failed += 1
                report.failures.append({"tier": tier.name, "chunk": chunk[:3],
                                        "error": f"{type(exc).__name__}:{exc}"})
                continue
            if "__error__" in bars_by_sym:
                # Explicit SIP failure — NO IEX fallback. Record and move on.
                report.batches_failed += 1
                report.failures.append({"tier": tier.name, "chunk": chunk[:3],
                                        "error": f"sip:{bars_by_sym['__error__']}"})
                continue
            rows = self._bars_to_rows(bars_by_sym, tier.timeframe, tf_secs, ckpt, report)
            if rows:
                results = self.writer.write_rows("bronze", "market_bars", rows)
                written = sum(r.rows for r in results if r.ok)
                report.bars_written += written
                report.by_timeframe[tier.timeframe] = (
                    report.by_timeframe.get(tier.timeframe, 0) + written)

    def _bars_to_rows(self, bars_by_sym: dict, timeframe: str, tf_secs: int,
                      ckpt: dict, report: IngestionReport) -> list:
        rows = []
        for sym, bars in bars_by_sym.items():
            if str(sym).startswith("__"):
                continue
            key = self._ckpt_key(timeframe, sym)
            high_water = int(ckpt.get(key, 0))
            new_high = high_water
            ingested_any = False
            for b in bars:
                open_ns = _iso_to_ns(b.get("t"))
                if open_ns <= 0:
                    continue
                if open_ns <= high_water:
                    report.skipped_resumed += 1
                    continue
                close_ns = open_ns + tf_secs * 1_000_000_000
                rows.append({
                    "symbol_id": sym, "timeframe": timeframe,
                    "event_timestamp_ns": open_ns,
                    "availability_timestamp_ns": close_ns,
                    "bar_open_ticks": _to_ticks(b.get("o", 0)),
                    "bar_high_ticks": _to_ticks(b.get("h", 0)),
                    "bar_low_ticks": _to_ticks(b.get("l", 0)),
                    "bar_close_ticks": _to_ticks(b.get("c", 0)),
                    "volume": int(b.get("v", 0) or 0),
                    "trade_count": int(b.get("n", 0) or 0),
                    "vwap_ticks": _to_ticks(b.get("vw", 0)) if b.get("vw") else None,
                    "price_scale": DEFAULT_PRICE_SCALE, "source_feed": self.feed,
                    "schema_version": HISTORICAL_VERSION})
                new_high = max(new_high, open_ns)
                ingested_any = True
            if ingested_any:
                ckpt[key] = new_high
                report.symbols_ingested += 1
        return rows

    def run(self, tiers, *, record_universe: bool = True,
            now_ns: Optional[int] = None) -> dict:
        """Ingest all tiers in priority order (bounded by max_symbols_per_run)."""
        now_ns = now_ns if now_ns is not None else time.time_ns()
        report = IngestionReport(feed=self.feed, started_ns=now_ns)
        ckpt = self._load_ckpt()
        seen: set = set()
        bounded_tiers = []
        for tier in tiers:
            kept = []
            for s in tier.symbols:
                s = str(s).upper().strip()
                if not s or s in seen:
                    continue
                if len(seen) >= self.max_symbols:
                    break
                seen.add(s)
                kept.append(s)
            if kept:
                bounded_tiers.append(HistoricalTier(
                    name=tier.name, symbols=tuple(kept),
                    timeframe=tier.timeframe, start=tier.start))
        report.symbols_requested = len(seen)
        if record_universe and seen:
            self.record_universe_snapshot(sorted(seen), as_of_ns=now_ns)
        for tier in bounded_tiers:
            self.ingest_tier(tier, report, ckpt)
        self._save_ckpt(ckpt)
        report.completed_ns = time.time_ns()
        return _report_dict(report)


def _report_dict(r: IngestionReport) -> dict:
    return {"version": r.version, "feed": r.feed,
            "symbols_requested": r.symbols_requested,
            "symbols_ingested": r.symbols_ingested, "bars_written": r.bars_written,
            "batches_failed": r.batches_failed, "skipped_resumed": r.skipped_resumed,
            "by_timeframe": r.by_timeframe, "failures": r.failures[:20],
            "duration_s": round((r.completed_ns - r.started_ns) / 1e9, 3)}
