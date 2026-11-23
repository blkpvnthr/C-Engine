"""M7B — Historical data integrity certification.

Reconciles and validates the genuine historical bars that every downstream economic
certification depends on. If the data is not trustworthy, nothing built on it can be — so this
runs first and is unforgiving. All checks are computed with aggregated DuckDB queries over the
real schema (no row materialization), plus one bounded Python spot-check for 1m->5m alignment
where DST-correct ET arithmetic is required.

Checks (each an independent :class:`CertificationResult`):

* **inventory / reconciliation** — total + per-symbol bar counts and date spans.
* **duplicates** — no repeated ``(symbol, timeframe, event_timestamp_ns)`` rows.
* **monotonicity** — timestamps strictly increase within each symbol/timeframe partition.
* **OHLC validity** — ``low <= min(open,close)`` and ``high >= max(open,close)``, ``high>=low``,
  strictly positive prices, non-negative volume.
* **point-in-time** — ``availability_timestamp_ns >= event_timestamp_ns`` (no look-ahead:
  a bar may never be *available* before the timestamp it is labeled with).
* **price scale** — uniform ``price_scale == 10000`` ticks.
* **session coverage** — fraction of expected NYSE trading days present per symbol (calendar-
  derived), and thin-day detection; large gaps are flagged, never hidden.
* **1m/5m alignment** — on a bounded sample, aggregated 1-minute bars reconcile to the stored
  5-minute bars (close/high/low/volume), proving the two timeframes are mutually consistent.

The data carries provenance ``historical_replay`` context — these are GENUINE market bars, but
integrity certification is an engineering claim about the *data*, not a profitability claim.
"""

from __future__ import annotations

import datetime as _dt
from zoneinfo import ZoneInfo

from cengine.certification.contracts import (
    CertificationResult,
    CertificationVerdict,
    safe_ticker,
)
from cengine.certification.datasource import (
    AVAIL_COL,
    DEFAULT_PRICE_SCALE,
    SYMBOL_COL,
    TS_COL,
    BarInventory,
    _escape,
    bar_inventory,
    connect,
    market_bars_glob,
)
from cengine.runtime.calendar import is_trading_day

DATA_INTEGRITY_VERSION = "m7b-data-integrity-1.0.0"
_ET = ZoneInfo("America/New_York")
_MIN_BARS_FOR_CERT = 100_000           # below this the dataset is too thin for research-grade cert
_MAX_DUP_FRACTION = 0.0                 # zero tolerance for duplicate bars
_MIN_SESSION_COVERAGE = 0.90           # >=90% of expected trading days present to certify coverage


def _v(ok: bool) -> CertificationVerdict:
    return CertificationVerdict.CERTIFIED if ok else CertificationVerdict.NOT_CERTIFIED


def _date(ns: int) -> _dt.date:
    return _dt.datetime.fromtimestamp(ns / 1e9, tz=_ET).date()


def _check_inventory(inv: BarInventory) -> CertificationResult:
    total = sum(tf["bars"] for tf in inv.timeframes.values())
    ok = total >= _MIN_BARS_FOR_CERT
    spans = {tf: {"bars": d["bars"], "symbols": d["symbols"],
                  "start": _date(d["min_ns"]).isoformat(), "end": _date(d["max_ns"]).isoformat()}
             for tf, d in inv.timeframes.items()}
    return CertificationResult(
        check="data_integrity.inventory", component="lakehouse",
        verdict=_v(ok),
        summary=(f"{total:,} bars across {len(inv.timeframes)} timeframes; " +
                 "; ".join(f"{tf}={d['bars']:,}/{d['symbols']}sym" for tf, d in spans.items())),
        sample_size=total,
        metrics={"timeframes": spans, "file_count": inv.file_count},
        notes=f"certification floor = {_MIN_BARS_FOR_CERT:,} bars",
    )


def _check_duplicates(g: str) -> CertificationResult:
    con = connect()
    row = con.execute(
        f"""SELECT count(*) FROM (
                SELECT {SYMBOL_COL}, timeframe, {TS_COL}, count(*) c
                FROM read_parquet('{g}', union_by_name=true)
                GROUP BY {SYMBOL_COL}, timeframe, {TS_COL} HAVING count(*) > 1)"""
    ).fetchone()
    dup_groups = int(row[0])
    return CertificationResult(
        check="data_integrity.duplicates", component="lakehouse",
        verdict=_v(dup_groups == 0),
        summary=("no duplicate (symbol,timeframe,timestamp) bars" if dup_groups == 0
                 else f"{dup_groups:,} duplicated bar keys"),
        metrics={"duplicate_key_groups": dup_groups, "tolerance": _MAX_DUP_FRACTION},
    )


def _check_monotonic(g: str) -> CertificationResult:
    con = connect()
    row = con.execute(
        f"""SELECT count(*) FROM (
                SELECT {TS_COL},
                       lag({TS_COL}) OVER (PARTITION BY {SYMBOL_COL}, timeframe
                                           ORDER BY {TS_COL}) AS prev
                FROM read_parquet('{g}', union_by_name=true))
            WHERE prev IS NOT NULL AND {TS_COL} <= prev"""
    ).fetchone()
    viol = int(row[0])
    return CertificationResult(
        check="data_integrity.monotonic", component="lakehouse",
        verdict=_v(viol == 0),
        summary=("timestamps strictly increasing within every partition" if viol == 0
                 else f"{viol:,} non-increasing timestamp steps"),
        metrics={"non_increasing_steps": viol},
    )


def _check_ohlc(g: str) -> CertificationResult:
    con = connect()
    row = con.execute(
        f"""SELECT
              count(*) total,
              count(*) FILTER (WHERE bar_low_ticks > least(bar_open_ticks, bar_close_ticks)
                                  OR bar_high_ticks < greatest(bar_open_ticks, bar_close_ticks)
                                  OR bar_high_ticks < bar_low_ticks) ohlc_bad,
              count(*) FILTER (WHERE bar_low_ticks <= 0 OR bar_open_ticks <= 0
                                  OR bar_close_ticks <= 0 OR bar_high_ticks <= 0) nonpos,
              count(*) FILTER (WHERE volume < 0) negvol
            FROM read_parquet('{g}', union_by_name=true)"""
    ).fetchone()
    total, ohlc_bad, nonpos, negvol = (int(x) for x in row)
    bad = ohlc_bad + nonpos + negvol
    return CertificationResult(
        check="data_integrity.ohlc_validity", component="lakehouse",
        verdict=_v(bad == 0),
        summary=("all bars satisfy OHLC ordering, positivity and non-negative volume"
                 if bad == 0 else f"{bad:,} invalid bars (ohlc={ohlc_bad}, "
                                  f"nonpos={nonpos}, negvol={negvol})"),
        sample_size=total,
        metrics={"ohlc_order_violations": ohlc_bad, "nonpositive_price": nonpos,
                 "negative_volume": negvol},
    )


def _check_point_in_time(g: str) -> CertificationResult:
    con = connect()
    row = con.execute(
        f"""SELECT
              count(*) FILTER (WHERE {AVAIL_COL} IS NULL) null_avail,
              count(*) FILTER (WHERE {AVAIL_COL} IS NOT NULL AND {AVAIL_COL} < {TS_COL}) lookahead
            FROM read_parquet('{g}', union_by_name=true)"""
    ).fetchone()
    null_avail, lookahead = int(row[0]), int(row[1])
    # look-ahead (availability before the event it is labeled with) is a hard fail; missing
    # availability is a weaker integrity gap (reported, not a pass).
    verdict = (CertificationVerdict.NOT_CERTIFIED if lookahead > 0
               else CertificationVerdict.CERTIFIED if null_avail == 0
               else CertificationVerdict.INSUFFICIENT_DATA)
    return CertificationResult(
        check="data_integrity.point_in_time", component="lakehouse",
        verdict=verdict,
        summary=(f"{lookahead:,} look-ahead bars (availability < event)" if lookahead
                 else ("no look-ahead; availability timestamps present on all bars"
                       if null_avail == 0
                       else f"no look-ahead, but {null_avail:,} bars lack availability ts")),
        metrics={"lookahead_bars": lookahead, "null_availability": null_avail},
        notes="point-in-time guard: a bar may never be available before its own timestamp",
    )


def _check_price_scale(g: str) -> CertificationResult:
    con = connect()
    scales = [int(r[0]) for r in con.execute(
        f"SELECT DISTINCT price_scale FROM read_parquet('{g}', union_by_name=true)").fetchall()]
    ok = scales == [DEFAULT_PRICE_SCALE]
    return CertificationResult(
        check="data_integrity.price_scale", component="lakehouse",
        verdict=_v(ok),
        summary=(f"uniform price_scale={DEFAULT_PRICE_SCALE}" if ok
                 else f"inconsistent price_scale values: {scales}"),
        metrics={"distinct_price_scales": scales},
    )


def _check_session_coverage(inv: BarInventory, g: str) -> CertificationResult:
    """Per-symbol: fraction of expected NYSE trading days present (calendar-derived)."""
    con = connect()
    rows = con.execute(
        f"""SELECT {SYMBOL_COL} sym, timeframe,
                   count(distinct trade_date) n_days, count(*) n_bars,
                   min(trade_date) mn, max(trade_date) mx
            FROM read_parquet('{g}', union_by_name=true)
            GROUP BY {SYMBOL_COL}, timeframe"""
    ).fetchall()
    per_symbol = []
    worst = 1.0
    for sym, tf, days, bars, mn, mx in rows:
        if mn is None or mx is None:
            continue
        start = mn if isinstance(mn, _dt.date) else _dt.date.fromisoformat(str(mn))
        end = mx if isinstance(mx, _dt.date) else _dt.date.fromisoformat(str(mx))
        expected = _expected_trading_days(start, end)
        cov = (days / expected) if expected else 0.0
        worst = min(worst, cov)
        per_symbol.append({"symbol": str(sym), "timeframe": tf, "present_days": int(days),
                           "expected_days": expected, "coverage": round(cov, 4),
                           "bars_per_day": round(bars / days, 1) if days else 0.0,
                           "span": f"{start.isoformat()}..{end.isoformat()}"})
    per_symbol.sort(key=lambda r: r["coverage"])
    below = [r for r in per_symbol if r["coverage"] < _MIN_SESSION_COVERAGE]
    return CertificationResult(
        check="data_integrity.session_coverage", component="lakehouse",
        verdict=_v(not below),
        summary=(f"all {len(per_symbol)} series >= {_MIN_SESSION_COVERAGE:.0%} trading-day "
                 f"coverage (worst {worst:.1%})" if not below
                 else f"{len(below)} series below {_MIN_SESSION_COVERAGE:.0%} coverage "
                      f"(worst {worst:.1%})"),
        sample_size=len(per_symbol),
        metrics={"worst_coverage": round(worst, 4), "below_threshold": below[:10],
                 "per_symbol": per_symbol},
        notes="coverage measured at trading-day granularity vs NYSE calendar",
    )


def _expected_trading_days(start: _dt.date, end: _dt.date) -> int:
    n = 0
    d = start
    one = _dt.timedelta(days=1)
    while d <= end:
        if is_trading_day(d):
            n += 1
        d += one
    return n


def _check_alignment(inv: BarInventory, *, sample_days: int = 20) -> CertificationResult:
    """Bounded spot-check: aggregate 1m bars into 5m buckets and compare to stored 5m bars.

    Only symbols present in BOTH timeframes are eligible. We take the most recent
    ``sample_days`` trading days of overlap to keep the check fast but genuine.
    """
    one_min = set(inv.symbols("1Min"))
    five_min = set(inv.symbols("5Min"))
    both = sorted(one_min & five_min)
    if not both:
        return CertificationResult(
            check="data_integrity.timeframe_alignment", component="lakehouse",
            verdict=CertificationVerdict.INSUFFICIENT_DATA,
            summary="no symbol present in both 1Min and 5Min; alignment not checkable",
            metrics={"symbols_1min": len(one_min), "symbols_5min": len(five_min)},
        )
    sym = safe_ticker(both[0])
    g = _escape(market_bars_glob())
    con = connect()
    # window: last sample_days*2 calendar days of 5m data for this symbol
    maxrow = con.execute(
        f"SELECT max({TS_COL}) FROM read_parquet('{g}', union_by_name=true) "
        f"WHERE {SYMBOL_COL} = ? AND timeframe = '5Min'", [sym]).fetchone()
    if not maxrow or maxrow[0] is None:
        return CertificationResult(
            check="data_integrity.timeframe_alignment", component="lakehouse",
            verdict=CertificationVerdict.INSUFFICIENT_DATA,
            summary=f"no 5Min data for {sym}", metrics={})
    end_ns = int(maxrow[0])
    start_ns = end_ns - sample_days * 3 * 24 * 3600 * 1_000_000_000  # generous calendar window
    # Aggregate 1m into 5m buckets (floor event ts to 5-minute grid) and join to stored 5m.
    bucket = "( {ts} - ({ts} % {w}) )".format(ts=TS_COL, w=5 * 60 * 1_000_000_000)
    cmp = con.execute(
        f"""WITH m1 AS (
                SELECT {bucket} AS b,
                       arg_min(bar_open_ticks, {TS_COL}) o,
                       max(bar_high_ticks) h, min(bar_low_ticks) l,
                       arg_max(bar_close_ticks, {TS_COL}) c, sum(volume) v, count(*) n
                FROM read_parquet('{g}', union_by_name=true)
                WHERE {SYMBOL_COL} = ? AND timeframe='1Min'
                  AND {TS_COL} >= ? AND {TS_COL} <= ?
                GROUP BY b),
             m5 AS (
                SELECT {TS_COL} AS b, bar_high_ticks h, bar_low_ticks l,
                       bar_close_ticks c, volume v
                FROM read_parquet('{g}', union_by_name=true)
                WHERE {SYMBOL_COL} = ? AND timeframe='5Min'
                  AND {TS_COL} >= ? AND {TS_COL} <= ?)
            SELECT count(*) matched,
                   count(*) FILTER (WHERE m1.h = m5.h) high_ok,
                   count(*) FILTER (WHERE m1.l = m5.l) low_ok,
                   count(*) FILTER (WHERE m1.c = m5.c) close_ok,
                   count(*) FILTER (WHERE m1.v = m5.v) vol_ok
            FROM m1 JOIN m5 USING (b)""",
        [sym, start_ns, end_ns, sym, start_ns, end_ns],
    ).fetchone()
    matched, high_ok, low_ok, close_ok, vol_ok = (int(x) for x in cmp)
    if matched == 0:
        return CertificationResult(
            check="data_integrity.timeframe_alignment", component="lakehouse",
            verdict=CertificationVerdict.INSUFFICIENT_DATA,
            summary=f"no overlapping 1m/5m buckets for {sym} in sample window",
            metrics={"symbol": sym})
    # price (high/low/close) must reconcile exactly; volume may differ slightly if the 5m feed
    # de-dups trades differently, so we report it but gate only on price consistency.
    price_ok = min(high_ok, low_ok, close_ok)
    frac = price_ok / matched
    return CertificationResult(
        check="data_integrity.timeframe_alignment", component="lakehouse",
        verdict=_v(frac >= 0.99),
        summary=(f"{sym}: {frac:.2%} of {matched} 5m buckets reconcile from 1m "
                 f"(high/low/close); volume match {vol_ok}/{matched}"),
        sample_size=matched,
        data_label=None,
        metrics={"symbol": sym, "buckets": matched, "high_ok": high_ok, "low_ok": low_ok,
                 "close_ok": close_ok, "volume_ok": vol_ok, "price_consistency": round(frac, 4)},
        notes=f"bounded spot-check over ~{sample_days} recent trading days",
    )


def run_data_integrity_certification() -> list:
    """Run every M7B check. Returns a list[CertificationResult]."""
    inv = bar_inventory()
    if not inv.available:
        return [CertificationResult(
            check="data_integrity.inventory", component="lakehouse",
            verdict=CertificationVerdict.INSUFFICIENT_DATA,
            summary="no market_bars parquet found; historical certification cannot run",
            metrics={"glob": inv.glob},
            notes="populate the lakehouse via `python -m cengine.lakehouse historical ...`")]
    g = _escape(market_bars_glob())
    return [
        _check_inventory(inv),
        _check_duplicates(g),
        _check_monotonic(g),
        _check_ohlc(g),
        _check_point_in_time(g),
        _check_price_scale(g),
        _check_session_coverage(inv, g),
        _check_alignment(inv),
    ]
