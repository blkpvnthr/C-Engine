"""Broad-market capacity benchmark (directive item 6).

Measures whether the native Metal pipeline + ingestion path sustain a larger
simultaneous BAR subscription set, across 500 / 1000 / 2000 / 4096 / FULL tradable
master. Everything here is a SYNTHETIC in-process throughput/latency benchmark
(explicitly labeled): it exercises native registration, bar ingestion, and the GPU
screen/breadth/compaction dispatch over N symbols on real hardware. It does NOT and
CANNOT measure genuine provider subscription confirmation, live observation
coverage, or protective-exit latency under live streaming — those are session-gated
and reported as pending, never fabricated.

Writes reports/full_market_discovery_benchmark.json. Read-only; no trading, no
network, no orders.
"""

from __future__ import annotations

import datetime
import json
import resource
import time
from pathlib import Path

import _cengine_native as native

from cengine.gpu_scanner_bridge import EngineGpuDiscovery

REPO = Path(__file__).resolve().parent.parent
SCALE = 10_000
LEVELS = [500, 1000, 2000, 4096]


def _rss_mb() -> float:
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports ru_maxrss in BYTES; Linux in KILOBYTES. A real RSS in bytes is
    # large (>1e7) even for a modest process; a KB value for the same process is
    # ~1e5-1e6. Disambiguate on magnitude.
    return ru / (1024 * 1024) if ru > 10_000_000 else ru / 1024


def _synthetic_bars(egd: EngineGpuDiscovery, symbols, n_1m: int) -> int:
    """Ingest n_1m synthetic 1m bars per symbol (folding to 5m). Returns bars sent."""
    ns_min = 60 * 1_000_000_000
    sent = 0
    for s in symbols:
        px = 100.0
        batch_1m = []
        for i in range(n_1m):
            o = px
            px *= 1.0009
            batch_1m.append((s, i * ns_min, int(o * SCALE), int(max(o, px) * 1.001 * SCALE),
                             int(min(o, px) * 0.999 * SCALE), int(px * SCALE), 1000 + i))
        egd.provider.ingest_bars(batch_1m, native.BarTimeframe.ONE_MINUTE)
        # fold to 5m in bulk
        batch_5m = []
        for j in range(0, n_1m, 5):
            chunk = batch_1m[j:j + 5]
            if not chunk:
                continue
            o = chunk[0][2]
            h = max(x[3] for x in chunk)
            lo = min(x[4] for x in chunk)
            close = chunk[-1][5]
            v = sum(x[6] for x in chunk)
            ts = chunk[-1][1]
            batch_5m.append((s, ts, o, h, lo, close, v))
        egd.provider.ingest_bars(batch_5m, native.BarTimeframe.FIVE_MINUTE)
        sent += len(batch_1m) + len(batch_5m)
    return sent


def bench_level(n: int, n_1m: int = 60) -> dict:
    egd = EngineGpuDiscovery(bar_capacity=128, window=64)
    syms = [f"BM{i:05d}" for i in range(n)]
    t0 = time.perf_counter()
    registered = egd.register_universe(syms)
    t_reg = time.perf_counter() - t0

    t0 = time.perf_counter()
    bars = _synthetic_bars(egd, syms, n_1m)
    t_ing = time.perf_counter() - t0

    # GPU dispatch: screen + technical_features over all N (one rank-tick equivalent)
    t0 = time.perf_counter()
    rows = egd.provider.compute(syms)
    t_screen = time.perf_counter() - t0

    t0 = time.perf_counter()
    breadth, (ids, total) = egd.provider.breadth_and_compaction(syms)
    t_breadth = time.perf_counter() - t0

    return {
        "symbols": n,
        "backend": egd.backend,
        "registered": registered,
        "register_s": round(t_reg, 4),
        "bars_ingested": bars,
        "ingest_s": round(t_ing, 4),
        "ingest_bars_per_s": round(bars / t_ing) if t_ing > 0 else None,
        "gpu_screen_s": round(t_screen, 4),
        "gpu_screen_per_symbol_us": round(t_screen / n * 1e6, 2) if n else None,
        "gpu_breadth_compaction_s": round(t_breadth, 4),
        "rows_computed": len(rows),
        "entry_qualified": int(getattr(breadth, "entry_qualified", 0)) if breadth else 0,
        "rss_mb_after": round(_rss_mb(), 1),
    }


def main() -> int:
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    # FULL tradable master from the real snapshot, if available.
    full = None
    try:
        from cengine.membership import MembershipRegistry
        reg = MembershipRegistry()
        if reg.load_cache(REPO / "data" / "membership_snapshot.json"):
            full = len(reg.tradable_master_symbols())
    except Exception:  # noqa: BLE001
        full = None
    levels = list(LEVELS)
    if full and full not in levels:
        levels.append(full)

    results = [bench_level(n) for n in sorted(set(levels))]
    # Sustainability heuristic (offline): FULL bar coverage is "sustainable" if the
    # GPU screen dispatch over the full set stays well under a 1s rank cadence and
    # memory is bounded. This governs ONLY the offline bar-capacity recommendation;
    # live sustainability is session-gated.
    full_row = next((r for r in results if r["symbols"] == full), results[-1])
    sustainable = bool(full and full_row["gpu_screen_s"] < 1.0
                       and full_row["gpu_breadth_compaction_s"] < 1.0)
    recommended_bar_cap = (max(8192, full) if sustainable else 4096)

    report = {
        "recorded_utc": now,
        "measurement_type": "SYNTHETIC in-process throughput/latency on real "
                            "hardware (Apple Metal). NOT a live-streaming measurement.",
        "device_backend": results[0]["backend"] if results else "unknown",
        "full_tradable_master": full,
        "levels": results,
        "offline_sustainability": {
            "full_bar_coverage_sustainable_offline": sustainable,
            "basis": "GPU screen + breadth/compaction dispatch over FULL set < 1.0s "
                     "rank cadence; bounded RSS.",
            "recommended_default_bar_cap": recommended_bar_cap,
            "rotation": "AdaptiveSubscriptionManager rotates elective bar coverage "
                        "deterministically (per-channel _rotation cursor) when demand "
                        "exceeds the bar cap; protected instruments are never rotated out.",
        },
        "session_gated_not_measured": [
            "provider subscription confirmation latency (needs live SIP session)",
            "live observation coverage / freshness under streaming load",
            "event-loop ingestion throughput at live SIP message rates",
            "mandatory protective-exit latency under full-market streaming load",
            "dropped/delayed observation counts under load",
        ],
        "invariants": [
            "one SIP connection; trades/quotes stay adaptively allocated by relevance",
            "rotation is deterministic and explicitly reported — NOT equivalent to "
            "continuous full-market streaming",
            "mandatory -0.5% protective exit independent of coverage breadth",
        ],
    }
    out = REPO / "reports" / "full_market_discovery_benchmark.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"wrote {out}")
    for r in results:
        print(f"  n={r['symbols']:>5} screen={r['gpu_screen_s']}s "
              f"breadth={r['gpu_breadth_compaction_s']}s "
              f"ingest={r['ingest_bars_per_s']}b/s rss={r['rss_mb_after']}MB")
    print(f"  full={full} sustainable_offline={sustainable} "
          f"recommended_bar_cap={recommended_bar_cap}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
