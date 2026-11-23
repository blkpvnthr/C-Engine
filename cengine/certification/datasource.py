"""Shared read-only access to the historical lakehouse for certification.

Thin, defensive wrapper around the Parquet ``market_bars`` dataset. Certification is a
*measurement* exercise: this module only ever issues read queries, it validates every ticker
against :func:`safe_ticker` before interpolation, and it returns an explicit
``available=False`` inventory rather than raising when the lakehouse is empty — so a cert run
degrades to INSUFFICIENT_DATA instead of crashing.
"""

from __future__ import annotations

import glob
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from cengine.certification.contracts import safe_ticker

DEFAULT_PRICE_SCALE = 10000
# Bar schema columns we rely on (verified against the live dataset).
TS_COL = "event_timestamp_ns"
AVAIL_COL = "availability_timestamp_ns"
SYMBOL_COL = "symbol_id"


@dataclass
class BarInventory:
    """Snapshot of what the lakehouse actually contains, taken at cert time."""

    available: bool
    glob: str = ""
    file_count: int = 0
    timeframes: dict = field(default_factory=dict)   # tf -> {bars, symbols, min_ns, max_ns}
    per_symbol: list = field(default_factory=list)    # [{symbol, timeframe, bars, min_ns, max_ns}]

    def symbols(self, timeframe: str) -> list:
        return sorted({r["symbol"] for r in self.per_symbol if r["timeframe"] == timeframe})

    def to_dict(self) -> dict:
        return {"available": self.available, "glob": self.glob, "file_count": self.file_count,
                "timeframes": self.timeframes, "per_symbol": self.per_symbol}


def market_bars_glob() -> str:
    from cengine.lakehouse.config import LakehouseConfig
    cfg = LakehouseConfig.from_env()
    base = cfg.dataset_path("bronze", "market_bars")
    return str(Path(base) / "**" / "*.parquet")


def connect():
    import duckdb
    return duckdb.connect()


def _escape(path: str) -> str:
    return path.replace("'", "''")


def bar_inventory() -> BarInventory:
    """Aggregate inventory of the market_bars dataset (one pass, no row materialization)."""
    g = market_bars_glob()
    files = glob.glob(g, recursive=True)
    if not files:
        return BarInventory(available=False, glob=g, file_count=0)
    con = connect()
    gq = _escape(g)
    tf_rows = con.execute(
        f"""SELECT timeframe, count(*) bars, count(distinct {SYMBOL_COL}) syms,
                   min({TS_COL}) mn, max({TS_COL}) mx
            FROM read_parquet('{gq}', union_by_name=true)
            GROUP BY timeframe ORDER BY timeframe"""
    ).fetchall()
    timeframes = {tf: {"bars": int(n), "symbols": int(s), "min_ns": int(mn), "max_ns": int(mx)}
                  for tf, n, s, mn, mx in tf_rows}
    sym_rows = con.execute(
        f"""SELECT {SYMBOL_COL} sym, timeframe, count(*) bars,
                   min({TS_COL}) mn, max({TS_COL}) mx
            FROM read_parquet('{gq}', union_by_name=true)
            GROUP BY {SYMBOL_COL}, timeframe ORDER BY timeframe, bars DESC"""
    ).fetchall()
    per_symbol = [{"symbol": str(sym), "timeframe": tf, "bars": int(n),
                   "min_ns": int(mn), "max_ns": int(mx)} for sym, tf, n, mn, mx in sym_rows]
    return BarInventory(available=True, glob=g, file_count=len(files),
                        timeframes=timeframes, per_symbol=per_symbol)


@lru_cache(maxsize=1)
def cached_inventory() -> BarInventory:
    return bar_inventory()


def load_symbol_bars(symbol: str, timeframe: str, *, start_ns: int | None = None,
                     end_ns: int | None = None, columns: str = "*"):
    """Load one symbol's bars (as an Arrow table) for a timeframe, time-ordered.

    ``symbol`` is validated via :func:`safe_ticker`; ``timeframe`` is checked against an
    allow-list; time bounds are passed as bound parameters.
    """
    sym = safe_ticker(symbol)
    if timeframe not in ("1Min", "5Min", "15Min", "1Hour", "1Day"):
        raise ValueError(f"unexpected timeframe {timeframe!r}")
    g = _escape(market_bars_glob())
    con = connect()
    clauses = [f"{SYMBOL_COL} = ?", "timeframe = ?"]
    params: list = [sym, timeframe]
    if start_ns is not None:
        clauses.append(f"{TS_COL} >= ?")
        params.append(int(start_ns))
    if end_ns is not None:
        clauses.append(f"{TS_COL} < ?")
        params.append(int(end_ns))
    where = " AND ".join(clauses)
    return con.execute(
        f"""SELECT {columns} FROM read_parquet('{g}', union_by_name=true)
            WHERE {where} ORDER BY {TS_COL}""",
        params,
    ).arrow()
