"""Standalone lakehouse CLI (Milestone 5, §27).

Zero-infrastructure operational entry point for the local data platform, runnable as:

    python -m cengine.lakehouse <command> [options]

Commands:
  pipeline    Run the batch analytical pipeline (silver -> gold -> quality ->
              attribution -> maintenance -> storage -> DuckDB views). Idempotent.
  historical  Bounded, resumable historical SIP ingestion into bronze (no IEX
              fallback). Reuses the existing Alpaca SIP REST fetcher + credentials.
  storage     Print the storage/footprint/health report.
  compact     Run compaction + bronze retention maintenance.
  quality     Print the data-quality profile.
  warehouse   Register DuckDB views over the lakehouse and print table counts.

Everything is local (Parquet + DuckDB). No cloud, no trading side effects: the CLI
never touches the order path, the broker, or the authoritative financial journal.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Optional

from cengine.lakehouse.config import LakehouseConfig


def _print(obj) -> None:
    print(json.dumps(obj, default=str, indent=2))


def _symbols_from_args(args) -> list:
    syms: list = []
    if getattr(args, "symbols", None):
        syms += [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    if getattr(args, "symbols_file", None) and os.path.exists(args.symbols_file):
        with open(args.symbols_file, encoding="utf-8") as fh:
            for line in fh:
                s = line.strip().upper()
                if s and not s.startswith("#"):
                    syms.append(s)
    if not syms:
        env = os.environ.get("CENGINE_SYMBOLS", "")
        syms += [s.strip().upper() for s in env.split(",") if s.strip()]
    # dedup preserving order
    seen: set = set()
    out = []
    for s in syms:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def cmd_pipeline(args, config: LakehouseConfig) -> int:
    from cengine.lakehouse.pipeline import LakehousePipeline
    out = LakehousePipeline(config).run(compact=not args.no_compact)
    _print(out)
    return 0


def cmd_historical(args, config: LakehouseConfig) -> int:
    from cengine.historical_warmup import make_sip_fetch_fn
    from cengine.lakehouse.historical import HistoricalSIPIngestion, HistoricalTier
    # Reuse the engine's credential loading + mirroring (ALPACA_* / APCA_*).
    try:
        from trade import load_runtime_env
        load_runtime_env()
    except Exception:  # noqa: BLE001 — env file loading is best-effort
        pass
    key = os.environ.get("ALPACA_API_KEY") or os.environ.get("APCA_API_KEY_ID")
    sec = os.environ.get("ALPACA_SECRET_KEY") or os.environ.get("APCA_API_SECRET_KEY")
    if not key or not sec:
        print("error: no Alpaca credentials (set ALPACA_API_KEY / ALPACA_SECRET_KEY "
              "or APCA_API_KEY_ID / APCA_API_SECRET_KEY)", file=sys.stderr)
        return 2
    symbols = _symbols_from_args(args)
    if not symbols:
        print("error: no symbols (use --symbols, --symbols-file, or CENGINE_SYMBOLS)",
              file=sys.stderr)
        return 2
    symbols = symbols[:args.max_symbols]
    fetch = make_sip_fetch_fn(key, sec)
    ing = HistoricalSIPIngestion(config, fetch_fn=fetch, feed=args.feed,
                                 batch_size=args.batch, max_symbols_per_run=args.max_symbols)
    tier = HistoricalTier(name="cli", symbols=tuple(symbols),
                          timeframe=args.timeframe, start=args.start)
    out = ing.run([tier])
    _print(out)
    return 0 if out.get("batches_failed", 0) == 0 or out.get("bars_written", 0) > 0 else 1


def cmd_storage(args, config: LakehouseConfig) -> int:
    from cengine.lakehouse.storage_monitor import StorageMonitor
    rep = StorageMonitor(config).report()
    _print({"health": rep.health, "root": rep.root, "free_disk_gib": rep.free_disk_gib,
            "used_lakehouse_gib": rep.used_lakehouse_gib,
            "accepts_writes": rep.accepts_writes, "accepts_reason": rep.accepts_reason,
            "datasets": rep.datasets, "compaction_candidates": rep.compaction_candidates,
            "recommendations": rep.recommendations})
    return 0


def cmd_compact(args, config: LakehouseConfig) -> int:
    from cengine.lakehouse.maintenance import LakehouseMaintenance
    _print(LakehouseMaintenance(config).run())
    return 0


def cmd_quality(args, config: LakehouseConfig) -> int:
    from cengine.lakehouse.quality import DataQualityValidator
    _print(DataQualityValidator(config).profile())
    return 0


def cmd_warehouse(args, config: LakehouseConfig) -> int:
    from cengine.lakehouse.warehouse import DuckDBWarehouse
    wh = DuckDBWarehouse(config)
    views = wh.register_views()
    _print({"views_registered": views, "counts": wh.table_counts()})
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="cengine.lakehouse",
                                description="Local lakehouse operations (M5)")
    p.add_argument("--root", default=None,
                   help="lakehouse root (overrides CENGINE_LAKEHOUSE_ROOT)")
    sub = p.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("pipeline", help="run the batch analytical pipeline")
    sp.add_argument("--no-compact", action="store_true", help="skip compaction/retention")
    sp.set_defaults(func=cmd_pipeline)

    sh = sub.add_parser("historical", help="bounded resumable SIP historical ingestion")
    sh.add_argument("--symbols", default=None, help="comma-separated symbols")
    sh.add_argument("--symbols-file", default=None, help="file with one symbol per line")
    sh.add_argument("--timeframe", default="5Min",
                    help="bar timeframe (1Min/5Min/15Min/1Hour/1Day)")
    sh.add_argument("--start", required=True, help="ISO8601 lookback start (e.g. 2026-10-01)")
    sh.add_argument("--feed", default="sip", help="sip | delayed_sip (no IEX fallback)")
    sh.add_argument("--batch", type=int, default=100, help="symbols per REST batch")
    sh.add_argument("--max-symbols", type=int, default=500, help="per-run symbol cap")
    sh.set_defaults(func=cmd_historical)

    for name, fn, helptxt in (("storage", cmd_storage, "storage/health report"),
                              ("compact", cmd_compact, "compaction + retention"),
                              ("quality", cmd_quality, "data-quality profile"),
                              ("warehouse", cmd_warehouse, "register views + counts")):
        s = sub.add_parser(name, help=helptxt)
        s.set_defaults(func=fn)
    return p


def main(argv: Optional[list] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.root:
        os.environ["CENGINE_LAKEHOUSE_ROOT"] = args.root
    config = LakehouseConfig.from_env()
    config.ensure_dirs()
    return args.func(args, config)


if __name__ == "__main__":
    raise SystemExit(main())
