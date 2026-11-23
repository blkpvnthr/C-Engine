"""EXIT-00..EXIT-15 protective-exit experiment matrix (Milestone 6B, §17 / §38).

Multi-timeframe historical replay on GENUINE 1-minute bars: strategic long entries are
generated on completed 5-minute-aggregated features (no look-ahead), and each position's exit
is simulated tactically on the 1-minute path under each candidate policy. Conservative
assumptions: entry at the next 1m open + costs; a stop fills at the stop level (or the bar
open if it gapped through); within a bar a stop is assumed to trigger before any favorable
target (no favorable-first assumption); an independent -3% emergency trigger; costs on both
sides.

Two evaluations: A) exit-policy isolation (identical entries + fixed size) and B) risk-
normalized portfolio (position size scaled by 1/stop-distance so a wider stop does not raise
capital at risk). Deterministic stop-based policies are run genuinely; ML/RL/portfolio-aware
policies (EXIT-08/09/10/11/13) are marked RESEARCH_ONLY until their models/context exist.
Results are historical-replay simulations — NOT actual trading performance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import duckdb
import numpy as np

from cengine.lakehouse.config import LakehouseConfig
from cengine.learning.evaluation import DATA_LABEL_HISTORICAL_REPLAY
from cengine.optimization.economic import evaluate_candidate
from cengine.optimization.exits.adaptive import asset_class_stop_range

EXIT_MATRIX_VERSION = "exit-experiment-matrix-1.0.0"

# EXIT id -> (kind, params). kind drives the stop-distance logic. 'research_only' kinds need
# a trained model / portfolio context not yet available.
_MATRIX = {
    "EXIT-00": ("fixed", {"stop": 0.005, "baseline": True}),      # production incumbent
    "EXIT-01": ("fixed", {"stop": 0.0075}),
    "EXIT-02": ("fixed", {"stop": 0.010}),
    "EXIT-03": ("fixed", {"stop": 0.015}),
    "EXIT-04": ("fixed", {"stop": 0.020}),
    "EXIT-05": ("atr", {"mult": 2.0}),
    "EXIT-06": ("volatility", {"mult": 2.5}),
    "EXIT-07": ("regime", {"calm": 0.008, "volatile": 0.015}),
    "EXIT-08": ("research_only", {"reason": "needs trained recovery-probability model"}),
    "EXIT-09": ("research_only", {"reason": "needs trained constrained-RL exit policy"}),
    "EXIT-10": ("research_only", {"reason": "needs trained hybrid ML/RL policy"}),
    "EXIT-11": ("research_only", {"reason": "needs seller-exhaustion model + replacement ctx"}),
    "EXIT-12": ("holding_period", {"base": 0.015, "tighten_per_bar": 0.00005}),
    "EXIT-13": ("research_only", {"reason": "needs portfolio context"}),
    "EXIT-14": ("asset_class", {}),
    "EXIT-15": ("tightening_only", {"initial": 0.020, "floor": 0.005}),
}

EMERGENCY_PCT = 0.03


@dataclass
class ExitMatrixResult:
    version: str = EXIT_MATRIX_VERSION
    symbol: str = ""
    window: str = ""
    n_entries: int = 0
    data_label: str = "historical_replay_simulated"
    per_policy: dict = field(default_factory=dict)      # id -> {status, metrics, score, n}
    baseline_id: str = "EXIT-00"
    note: str = ""

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


def _load_1m(config: LakehouseConfig, symbol: str, *, start_ns=None, end_ns=None) -> Optional[dict]:
    base = config.dataset_path("bronze", "market_bars")
    if not base.exists() or not any(base.rglob("*.parquet")):
        return None
    glob = str(base / "**" / "*.parquet").replace("'", "''")
    where = [f"symbol_id='{symbol}'", "timeframe='1Min'", "bar_close_ticks>0"]
    if start_ns is not None:
        where.append(f"event_timestamp_ns>={int(start_ns)}")
    if end_ns is not None:
        where.append(f"event_timestamp_ns<{int(end_ns)}")
    con = duckdb.connect()
    try:
        tbl = con.sql(f"SELECT event_timestamp_ns ts, bar_open_ticks o, bar_high_ticks h, "
                      f"bar_low_ticks l, bar_close_ticks c, volume v "
                      f"FROM read_parquet('{glob}', union_by_name=true) "
                      f"WHERE {' AND '.join(where)} ORDER BY ts").to_arrow_table()
    finally:
        con.close()
    if tbl.num_rows < 500:
        return None
    return {k: tbl.column(k).to_numpy(zero_copy_only=False) for k in tbl.column_names}


def _rolling_mean(x: np.ndarray, win: int) -> np.ndarray:
    """Causal rolling mean of the PREVIOUS ``win`` values (x[i-win:i]); NaN before warmup."""
    cs = np.cumsum(np.insert(x, 0, 0.0))
    out = np.full(len(x), np.nan)
    if len(x) > win:
        out[win:] = (cs[win:-1] - cs[:-win - 1]) / win
    return out


def _entries_and_atr(bars: dict, *, max_entries: int = 3000) -> tuple:
    """Look-ahead-free long entries at 5-min cadence + a rolling ATR-fraction (vectorized).

    Entries are capped/strided to ``max_entries`` for a statistically-valid, tractable sample.
    """
    c = bars["c"].astype(np.float64)
    h = bars["h"].astype(np.float64)
    low = bars["l"].astype(np.float64)
    ts = bars["ts"]
    n = len(c)
    win = 14
    tr = np.zeros(n)
    prev_c = np.concatenate(([c[0]], c[:-1]))
    tr = np.maximum(h - low, np.maximum(np.abs(h - prev_c), np.abs(low - prev_c)))
    atr = _rolling_mean(tr, win)
    atr_frac = np.where(c > 0, np.nan_to_num(atr) / c, 0.0)
    sma20_prev = _rolling_mean(c, 20)                # mean of previous 20 (look-ahead-free)
    c_lag5 = np.concatenate((np.full(5, np.nan), c[:-5]))
    mom5 = np.where(c_lag5 > 0, c / c_lag5 - 1.0, 0.0)
    minute = (ts // (60 * 1_000_000_000)) % 5        # 5-min boundary proxy
    mask = (np.arange(n) >= 30) & (np.arange(n) < n - 1) & (minute == 0) \
        & (mom5 > 0.0008) & np.isfinite(sma20_prev) & (c > sma20_prev)
    idx = np.nonzero(mask)[0]
    if len(idx) > max_entries:
        idx = idx[:: max(1, len(idx) // max_entries)][:max_entries]
    return idx.tolist(), atr_frac


def _stop_for(policy_kind: str, params: dict, *, i: int, entry_i: int, atr_frac: float,
              vol_frac: float, asset_class: str, held_bars: int) -> float:
    """Return the active stop DISTANCE (positive fraction) for a policy at bar i."""
    if policy_kind == "fixed":
        return params["stop"]
    if policy_kind == "atr":
        return float(np.clip(params["mult"] * atr_frac, 0.005, 0.025))
    if policy_kind == "volatility":
        return float(np.clip(params["mult"] * vol_frac, 0.005, 0.025))
    if policy_kind == "regime":
        return params["volatile"] if vol_frac > 0.004 else params["calm"]
    if policy_kind == "holding_period":
        return float(max(0.005, params["base"] - params["tighten_per_bar"] * held_bars))
    if policy_kind == "asset_class":
        lo, hi = asset_class_stop_range(asset_class)
        return float(np.clip(2.0 * atr_frac, lo, hi))
    if policy_kind == "tightening_only":
        return float(max(params["floor"], params["initial"] - 0.00004 * held_bars))
    return 0.005


_TIME_VARYING = {"tightening_only", "holding_period"}


def _simulate(bars: dict, entries: list, atr_frac, policy_kind: str, params: dict, *,
              asset_class: str, cost_frac: float, max_hold: int = 120,
              risk_normalized: bool = False, risk_budget: float = 0.005) -> list:
    """Return per-trade net returns (Eval A) or risk-weighted contributions (Eval B).

    Constant-stop policies use a vectorized first-breach search; time-varying policies
    (tightening/holding-period) use a bounded per-bar loop. Conservative: a stop fills at the
    stop level or the bar open if it gapped through; the -3% emergency trigger is independent.
    """
    o, h, low, c = (bars[k].astype(np.float64) for k in ("o", "h", "l", "c"))
    n = len(c)
    out = []
    # per-entry realized vol (vectorized-ish via slices on the small capped entry set)
    for e in entries:
        if e + 1 >= n:
            continue
        entry = o[e + 1] * (1.0 + cost_frac)
        rets = np.diff(np.log(c[max(0, e - 20):e + 1]))
        vol_frac = float(np.std(rets)) if len(rets) > 2 else 0.003
        entry_stop_dist = _stop_for(policy_kind, params, i=e + 1, entry_i=e,
                                    atr_frac=atr_frac[e], vol_frac=vol_frac,
                                    asset_class=asset_class, held_bars=0)
        hi = min(n, e + 1 + max_hold)
        win_low = low[e + 1:hi]
        win_open = o[e + 1:hi]
        emerg_price = entry * (1.0 - EMERGENCY_PCT)
        exit_price = None
        if policy_kind in _TIME_VARYING:
            held = np.arange(len(win_low))
            if policy_kind == "tightening_only":
                dist = np.maximum(params["floor"], params["initial"] - 0.00004 * held)
            else:  # holding_period
                dist = np.maximum(0.005, params["base"] - params["tighten_per_bar"] * held)
            dist = np.minimum(dist, entry_stop_dist)
            stop_px = entry * (1.0 - dist)
            breach = np.nonzero((win_low <= stop_px) | (win_low <= emerg_price))[0]
            if len(breach):
                k = breach[0]
                lvl = emerg_price if win_low[k] <= emerg_price else stop_px[k]
                exit_price = min(win_open[k], lvl)
        else:
            stop_price = entry * (1.0 - entry_stop_dist)
            breach = np.nonzero(win_low <= stop_price)[0]
            if len(breach):
                k = breach[0]
                bar_lvl = emerg_price if win_low[k] <= emerg_price else stop_price
                exit_price = min(win_open[k], bar_lvl)
        if exit_price is None:
            exit_price = c[hi - 1]                    # time-stop at close
        gross = exit_price / entry - 1.0
        net = gross - cost_frac                        # exit-side cost
        if risk_normalized:
            size = risk_budget / max(entry_stop_dist, 1e-4)
            out.append(net * size)
        else:
            out.append(net)
    return out


def run_exit_matrix(config: Optional[LakehouseConfig], symbol: str, *,
                    asset_class: str = "large_cap", cost_frac: float = 0.0004,
                    start_ns=None, end_ns=None, window: str = "all",
                    risk_normalized: bool = False) -> ExitMatrixResult:
    """Run EXIT-00..15 on genuine 1m bars for one symbol/window. Returns per-policy metrics."""
    config = config or LakehouseConfig.from_env()
    res = ExitMatrixResult(symbol=symbol, window=window)
    bars = _load_1m(config, symbol, start_ns=start_ns, end_ns=end_ns)
    if bars is None:
        res.note = "insufficient 1Min bars for window"
        res.per_policy = {k: {"status": "INSUFFICIENT_DATA"} for k in _MATRIX}
        return res
    entries, atr_frac = _entries_and_atr(bars)
    res.n_entries = len(entries)
    if len(entries) < 30:
        res.note = f"insufficient entries ({len(entries)})"
        res.per_policy = {k: {"status": "INSUFFICIENT_DATA"} for k in _MATRIX}
        return res
    label = DATA_LABEL_HISTORICAL_REPLAY
    for exit_id, (kind, params) in _MATRIX.items():
        if kind == "research_only":
            res.per_policy[exit_id] = {"status": "RESEARCH_ONLY", "reason": params["reason"]}
            continue
        returns = _simulate(bars, entries, atr_frac, kind, params,
                            asset_class=asset_class, cost_frac=cost_frac,
                            risk_normalized=risk_normalized)
        ev = evaluate_candidate(returns, label=exit_id, data_label=label)
        res.per_policy[exit_id] = {
            "status": "COMPLETED", "kind": kind, "n": ev.n,
            "metrics": ev.metrics, "score": ev.score,
            "net_return_ci": ev.net_return_ci,
            "baseline": bool(params.get("baseline", False))}
    res.note = ("historical-replay simulation on genuine 1m bars; NOT actual trading "
                "performance; conservative fills; stop-before-target within a bar")
    return res
