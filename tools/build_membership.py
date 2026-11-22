"""Acquire the master-universe membership from Alpaca and persist a snapshot (§14-17).

Genuinely fetches the Alpaca supported US-equity universe (authoritative), builds
the stable symbol registry, and writes:
  * data/membership_snapshot.json          (versioned cache, for degraded recovery)
  * reports/market_discovery_coverage.json (coverage denominators, §25)

S&P 500 / DJIA constituents are loaded from optional files if present
(CENGINE_SP500_FILE / data/sp500.txt, CENGINE_DJIA_FILE / data/djia.txt) — one
symbol per line. Index constituents are NEVER fabricated; if no source file is
present the flag is simply absent and the snapshot records the gap. On acquisition
failure the tool falls back to the cached snapshot (never to a small ETF universe).

Run:  ./.venv/bin/python tools/build_membership.py   (read-only; no trading)
"""

from __future__ import annotations

import datetime
import json
import os
from pathlib import Path

from cengine.membership import (
    MembershipRegistry,
    _read_symbol_file,
    fetch_alpaca_assets,
)

REPO = Path(__file__).resolve().parent.parent
CACHE = REPO / "data" / "membership_snapshot.json"
REPORT = REPO / "reports" / "market_discovery_coverage.json"


def _load_env() -> None:
    for name in (".env", "atlas-credentials.env"):
        p = REPO / name
        if not p.exists():
            continue
        for line in p.read_text().splitlines():
            if line.strip() and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def _index_source(env_var: str, default_rel: str) -> tuple[list[str], str]:
    path = os.environ.get(env_var) or str(REPO / default_rel)
    syms = _read_symbol_file(path)
    return syms, (path if syms else "not_configured")


def main() -> int:
    _load_env()
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    key = os.environ.get("ALPACA_API_KEY", "")
    sec = os.environ.get("ALPACA_SECRET_KEY", "")
    base = os.environ.get("ALPACA_TRADING_BASE_URL", "https://api.alpaca.markets")

    sp500, sp500_src = _index_source("CENGINE_SP500_FILE", "data/sp500.txt")
    djia, djia_src = _index_source("CENGINE_DJIA_FILE", "data/djia.txt")
    # Configured members = the canonical universe.txt UNION the curated ETF seed
    # (data/configured_etfs.txt) so the master universe genuinely includes the
    # broad/sector/leveraged/inverse ETF categories (directive §3).
    configured = sorted(set(_read_symbol_file(REPO / "universe.txt"))
                        | set(_read_symbol_file(REPO / "data" / "configured_etfs.txt")))

    reg = MembershipRegistry()
    sources = {"universe": "alpaca_assets", "sp500": sp500_src, "djia": djia_src,
               "configured": "universe.txt"}
    degraded = False
    try:
        assets = fetch_alpaca_assets(key, sec, base)
        snap = reg.build(assets, sp500=sp500, djia=djia, configured=configured,
                         now_iso=now, sources=sources)
    except Exception as e:  # noqa: BLE001 — fall back to cache, never to ETF stub
        degraded = True
        if reg.load_cache(CACHE):
            snap = reg.snapshot_meta
            snap.note = f"acquisition failed ({type(e).__name__}); using cache"
        else:
            REPORT.parent.mkdir(exist_ok=True)
            REPORT.write_text(json.dumps({
                "recorded_utc": now, "verdict": "BLOCKED",
                "reason": f"membership acquisition failed and no cache: {e}"}, indent=2))
            print("BLOCKED:", e)
            return 0

    if not degraded:
        reg.persist(CACHE)

    c = snap.counts
    REPORT.parent.mkdir(exist_ok=True)
    REPORT.write_text(json.dumps({
        "recorded_utc": now,
        "membership_version": snap.membership_version,
        "degraded": degraded,
        "sources": sources,
        "index_sources_configured": {"sp500": bool(sp500), "djia": bool(djia)},
        # §25 coverage denominators (distinct; unobserved != evaluated)
        "coverage_denominators": {
            "MASTER_MEMBERSHIP": c.get("master_universe", 0),
            "PROVIDER_SUPPORTED": c.get("alpaca_supported", 0),
            "TRADABLE": c.get("tradable", 0),
            "FRACTIONABLE": c.get("fractionable", 0),
        },
        "group_counts": {
            "nasdaq_member": c.get("nasdaq_member", 0),
            "sp500_member": c.get("sp500_member", 0),
            "djia_member": c.get("djia_member", 0),
            "configured_member": c.get("configured_member", 0),
            "master_universe_deduplicated": c.get("master_universe", 0),
            "master_tradable": c.get("master_tradable", 0),
        },
        "changes": snap.changes,
        "note": ("Master universe = NASDAQ-listed ∪ S&P500 ∪ DJIA ∪ configured, "
                 "deduplicated. Index constituents are loaded from configured source "
                 "files only and never fabricated; absent sources are reported, not "
                 "inferred from exchange. Tradability is distinct from discovery "
                 "membership. Cache: data/membership_snapshot.json."),
    }, indent=2))
    print(f"membership v{snap.membership_version} degraded={degraded}")
    print(f"  provider_supported={c.get('alpaca_supported')} "
          f"nasdaq={c.get('nasdaq_member')} sp500={c.get('sp500_member')} "
          f"djia={c.get('djia_member')} configured={c.get('configured_member')}")
    print(f"  master_universe(dedup)={c.get('master_universe')} "
          f"tradable={c.get('tradable')} fractionable={c.get('fractionable')}")
    print(f"  index sources: sp500={sp500_src} djia={djia_src}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
