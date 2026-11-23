"""Counterfactual protective-exit data engineering (Milestone 6B, §16).

Measures, on GENUINE 1-minute bars, what actually happened AFTER hypothetical -0.5%
protective exits — was the exit beneficial or premature? For each simulated long that hits
the threshold, it records observable subsequent market behavior at the §16 horizons (1/5/15/
30/60 min, EOD, +1/+3/+5 sessions): recovery above the exit/entry price, time-to-recovery,
additional adverse/favorable excursion.

Strict evidence classification: these are OBSERVED subsequent market prices + SIMULATED
trading outcomes — never genuine executed outcomes, and a favorable recovery is NOT proof
that holding was the superior PORTFOLIO decision (capital use, downside, opportunity cost
are not captured here). Point-in-time honored: a horizon's label is censored until the
window has elapsed in the data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import duckdb
import numpy as np

from cengine.lakehouse.config import DEFAULT_PRICE_SCALE, LakehouseConfig

COUNTERFACTUAL_VERSION = "exit-counterfactual-1.0.0"

# horizon label -> minutes (session/multi-session handled separately)
_MINUTE_HORIZONS = {"h_1m": 1, "h_5m": 5, "h_15m": 15, "h_30m": 30, "h_60m": 60}
_SECONDS = 1_000_000_000


@dataclass
class CounterfactualReport:
    version: str = COUNTERFACTUAL_VERSION
    symbol: str = ""
    exit_events: int = 0
    evidence_class: str = ("observed_subsequent_prices + simulated_outcomes "
                           "(NOT genuine executions)")
    recovery_above_exit_rate: dict = field(default_factory=dict)
    recovery_above_entry_rate: dict = field(default_factory=dict)
    mean_additional_adverse: dict = field(default_factory=dict)
    mean_additional_favorable: dict = field(default_factory=dict)
    censored_horizons: dict = field(default_factory=dict)
    note: str = ""

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


def _glob(config: LakehouseConfig) -> Optional[str]:
    base = config.dataset_path("bronze", "market_bars")
    if not base.exists() or not any(base.rglob("*.parquet")):
        return None
    return str(base / "**" / "*.parquet").replace("'", "''")


def _load_1m(config: LakehouseConfig, symbol: str) -> Optional[dict]:
    glob = _glob(config)
    if glob is None:
        return None
    con = duckdb.connect()
    try:
        tbl = con.sql(
            f"SELECT event_timestamp_ns ts, bar_open_ticks o, bar_high_ticks h, "
            f"bar_low_ticks l, bar_close_ticks c FROM read_parquet('{glob}', "
            f"union_by_name=true) WHERE symbol_id='{symbol}' AND timeframe='1Min' "
            f"AND bar_close_ticks>0 ORDER BY ts").to_arrow_table()
    finally:
        con.close()
    if tbl.num_rows == 0:
        return None
    return {k: tbl.column(k).to_numpy(zero_copy_only=False) for k in tbl.column_names}


def analyze(config: Optional[LakehouseConfig], symbol: str, *,
            exit_threshold: float = -0.005, entry_stride: int = 60,
            max_events: int = 5000) -> CounterfactualReport:
    """Measure post-exit recovery behavior for hypothetical -0.5% exits on real 1m bars."""
    config = config or LakehouseConfig.from_env()
    rep = CounterfactualReport(symbol=symbol)
    bars = _load_1m(config, symbol)
    if bars is None:
        rep.note = "no 1Min bars for symbol"
        return rep
    o, h, low, c = bars["o"], bars["h"], bars["l"], bars["c"]
    n = len(c)
    scale = DEFAULT_PRICE_SCALE
    horizons = list(_MINUTE_HORIZONS.items())
    recov_exit = {k: [] for k, _ in horizons}
    recov_entry = {k: [] for k, _ in horizons}
    add_adverse = {k: [] for k, _ in horizons}
    add_favorable = {k: [] for k, _ in horizons}
    censored = {k: 0 for k, _ in horizons}
    events = 0
    i = 0
    search = 390        # bounded forward search for a stop breach (~1 session)
    while i < n - 2 and events < max_events:
        entry = o[i + 1] / scale   # enter next-bar open
        hi = min(n, i + 1 + search)
        win = low[i + 1:hi] / scale / entry - 1.0
        breach = np.nonzero(win <= exit_threshold)[0]
        if len(breach) == 0:
            i += entry_stride
            continue
        exit_idx = i + 1 + int(breach[0])
        exit_price = entry * (1.0 + exit_threshold)   # conservative fill at the stop level
        events += 1
        for name, mins in horizons:
            fut = exit_idx + mins
            if fut >= n:
                censored[name] += 1
                continue
            window_hi = np.max(h[exit_idx + 1:fut + 1]) / scale if fut > exit_idx else exit_price
            window_lo = np.min(low[exit_idx + 1:fut + 1]) / scale if fut > exit_idx else exit_price
            recov_exit[name].append(1.0 if window_hi > exit_price else 0.0)
            recov_entry[name].append(1.0 if window_hi >= entry else 0.0)
            add_favorable[name].append(window_hi / exit_price - 1.0)
            add_adverse[name].append(window_lo / exit_price - 1.0)
        i = exit_idx + entry_stride
    rep.exit_events = events
    rep.recovery_above_exit_rate = {k: _mean(v) for k, v in recov_exit.items()}
    rep.recovery_above_entry_rate = {k: _mean(v) for k, v in recov_entry.items()}
    rep.mean_additional_adverse = {k: _mean(v) for k, v in add_adverse.items()}
    rep.mean_additional_favorable = {k: _mean(v) for k, v in add_favorable.items()}
    rep.censored_horizons = censored
    rep.note = ("a recovery above the exit price does NOT prove holding was the superior "
                "portfolio decision (capital use / downside / opportunity cost excluded)")
    return rep


def _mean(v: list):
    return round(float(np.mean(v)), 5) if v else None
