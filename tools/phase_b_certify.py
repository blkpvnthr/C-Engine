"""Phase B operational certification harness (non-trading, SIP historical replay).

Drives GENUINE Alpaca SIP market data through the complete native discovery
pipeline and records evidence:

    SIP historical bars (REST, feed=sip)
      -> DiscoveryEngine ingestion (B.2)
      -> technical_features (B.3, GPU)
      -> screen_market (B.1, GPU)
      -> market_breadth (B.4, GPU)
      -> compact_entry_candidates (B.4, GPU)

This is HISTORICAL REPLAY validation with real SIP observations — it is NOT
real-time streaming certification (directive §12). Real-time session capture is
produced by the live engine during an eligible trading session. No orders are
submitted. Credentials are never printed.

Run:  ./.venv/bin/python tools/phase_b_certify.py
Writes reports/sip_historical_validation.json + reports/phase_b_operational_validation.json.
"""

from __future__ import annotations

import datetime
import json
import os
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import _cengine_native as native

from cengine.gpu_discovery import GpuDiscoveryProvider

REPO = Path(__file__).resolve().parent.parent
REPORTS = REPO / "reports"
DATA_BASE = "https://data.alpaca.markets"
SCALE = 10_000

# A small, liquid certification basket (not the full universe — that is Phase C).
SYMBOLS = ["SPY", "QQQ", "AAPL", "MSFT", "NVDA", "AMD", "TSLA", "META", "AMZN",
           "GOOGL", "AVGO", "NFLX"]


def _load_env() -> None:
    for name in (".env", "atlas-credentials.env"):
        p = REPO / name
        if not p.exists():
            continue
        for line in p.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def _creds() -> tuple[str, str]:
    return (os.environ.get("ALPACA_API_KEY", ""),
            os.environ.get("ALPACA_SECRET_KEY", ""))


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _fetch_bars(symbols, timeframe, start, key, sec, limit=10000) -> dict:
    """GET SIP historical bars for `symbols`; returns {symbol: [bar,...]}. Bars
    carry OHLCV in dollars; we convert to ticks on ingest."""
    out: dict[str, list] = {s: [] for s in symbols}
    params = {"symbols": ",".join(symbols), "timeframe": timeframe,
              "feed": "sip", "start": start, "limit": str(limit), "adjustment": "raw"}
    page_token = None
    for _ in range(20):  # bounded pagination
        q = dict(params)
        if page_token:
            q["page_token"] = page_token
        url = f"{DATA_BASE}/v2/stocks/bars?{urllib.parse.urlencode(q)}"
        req = urllib.request.Request(url, method="GET")
        req.add_header("APCA-API-KEY-ID", key)
        req.add_header("APCA-API-SECRET-KEY", sec)
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                doc = json.loads(resp.read().decode("utf-8", "replace"))
        except urllib.error.HTTPError as e:
            return {"__error__": f"HTTP {e.code}: {e.read(200).decode('utf-8','replace')}"}
        except Exception as e:  # noqa: BLE001
            return {"__error__": f"{type(e).__name__}: {e}"}
        for sym, bars in (doc.get("bars") or {}).items():
            out.setdefault(sym, []).extend(bars)
        page_token = doc.get("next_page_token")
        if not page_token:
            break
    return out


def _to_ticks(x: float) -> int:
    return int(round(float(x) * SCALE))


def _ingest(provider: GpuDiscoveryProvider, bars_by_sym: dict, tf) -> int:
    batch = []
    for sym, bars in bars_by_sym.items():
        if sym.startswith("__"):
            continue
        for b in bars:
            # Alpaca bar: t (RFC3339), o,h,l,c,v
            try:
                ts = int(datetime.datetime.fromisoformat(
                    b["t"].replace("Z", "+00:00")).timestamp() * 1e9)
            except Exception:  # noqa: BLE001
                ts = 0
            batch.append((sym, ts, _to_ticks(b["o"]), _to_ticks(b["h"]),
                          _to_ticks(b["l"]), _to_ticks(b["c"]), int(b.get("v", 0))))
    return provider.ingest_bars(batch, tf) if batch else 0


def main() -> int:
    REPORTS.mkdir(exist_ok=True)
    _load_env()
    key, sec = _creds()
    stamp = _now()
    hist = {"recorded_utc": stamp, "feed": "sip", "provider": "alpaca",
            "basket": SYMBOLS, "kind": "historical_replay",
            "note": "Genuine SIP historical bars; NOT real-time streaming (directive §12)."}

    if not (key and sec):
        hist["verdict"] = "BLOCKED"
        hist["reason"] = "no credentials in .env"
        (REPORTS / "sip_historical_validation.json").write_text(json.dumps(hist, indent=2))
        print("BLOCKED: no credentials")
        return 0

    # ~15 trading days back for 5m (DI/ADX/SMA warm-up); ~3 days for 1m.
    today = datetime.date(2026, 11, 21)
    start_5m = (today - datetime.timedelta(days=20)).isoformat()
    start_1m = (today - datetime.timedelta(days=4)).isoformat()
    bars5 = _fetch_bars(SYMBOLS, "5Min", start_5m, key, sec)
    bars1 = _fetch_bars(SYMBOLS, "1Min", start_1m, key, sec)

    if "__error__" in bars5 or "__error__" in bars1:
        hist["verdict"] = "BLOCKED"
        hist["reason"] = bars5.get("__error__") or bars1.get("__error__")
        (REPORTS / "sip_historical_validation.json").write_text(json.dumps(hist, indent=2))
        print("BLOCKED:", hist["reason"])
        return 0

    bars5_counts = {s: len(bars5.get(s, [])) for s in SYMBOLS}
    bars1_counts = {s: len(bars1.get(s, [])) for s in SYMBOLS}
    hist["bars_5m_fetched"] = sum(bars5_counts.values())
    hist["bars_1m_fetched"] = sum(bars1_counts.values())
    hist["per_symbol_5m"] = bars5_counts
    hist["verdict"] = "OK" if hist["bars_5m_fetched"] > 0 else "EMPTY"
    (REPORTS / "sip_historical_validation.json").write_text(json.dumps(hist, indent=2))

    # Drive the full native pipeline with the real SIP bars.
    provider = GpuDiscoveryProvider(bar_capacity=4096, window=256)
    provider.set_universe(SYMBOLS)
    n5 = _ingest(provider, bars5, native.BarTimeframe.FIVE_MINUTE)
    n1 = _ingest(provider, bars1, native.BarTimeframe.ONE_MINUTE)

    rows = provider.compute(SYMBOLS)
    flags = [rows[s].flags for s in SYMBOLS if s in rows]
    breadth = provider.market_breadth(SYMBOLS)
    rt = provider._runtime  # may be None (CPU)
    if rt is not None and flags:
        ids, total = rt.compact_entry_candidates(flags, len(flags))
    else:
        total = sum(1 for f in flags if f & native.SCREEN_ENTRY_QUALIFIED)
        ids = [i for i, f in enumerate(flags) if f & native.SCREEN_ENTRY_QUALIFIED]

    sample = {s: {"di_plus": round(r.di_plus, 3), "di_minus": round(r.di_minus, 3),
                  "adx": round(r.adx, 3), "vwap": round(r.vwap, 4),
                  "sma": round(r.sma, 4), "price": round(r.price, 4),
                  "entry_qualified": r.entry_qualified,
                  "technical_exit": r.technical_exit,
                  "tech_valid": r.tech_valid}
              for s, r in list(rows.items())}

    op = {
        "recorded_utc": stamp, "feed": "sip", "backend": provider.backend,
        "kind": "historical_replay_not_realtime",
        "pipeline": "SIP bars -> DiscoveryEngine -> technical_features -> "
                    "screen_market -> market_breadth -> compact_entry_candidates",
        "ingested_5m_bars": n5, "ingested_1m_bars": n1,
        "symbols_screened": len(rows),
        "entry_qualified": total,
        "entry_qualified_ids": ids,
        "breadth": (provider and breadth and {
            "total": breadth.total, "valid": breadth.valid,
            "entry_qualified": breadth.entry_qualified,
            "technical_exit": breadth.technical_exit,
            "bullish": breadth.bullish, "bearish": breadth.bearish,
            "above_vwap": breadth.above_vwap, "below_vwap": breadth.below_vwap,
            "data_invalid": breadth.data_invalid}) or None,
        "per_symbol": sample,
        "realtime_certification": "PENDING — requires an eligible live SIP session "
                                  "(this run is historical replay).",
        "orders_submitted": 0,
    }
    (REPORTS / "phase_b_operational_validation.json").write_text(json.dumps(op, indent=2))
    print(f"SIP historical: {hist['bars_5m_fetched']} 5m + {hist['bars_1m_fetched']} 1m bars")
    print(f"pipeline backend={provider.backend} screened={len(rows)} "
          f"entry_qualified={total} breadth_total={breadth.total if breadth else 0}")
    print(f"-> {REPORTS}/sip_historical_validation.json + phase_b_operational_validation.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
