"""Incumbent failure attribution (M8 §3 / §16).

Decomposes *why* the M7 incumbent (−0.5% protective exit + momentum entry) loses, on genuine 1m
bars, to seed falsifiable research hypotheses rather than guesses. For each resolved trade it
records how the position ended and the market context at entry:

* **stop-out vs time-stop** — what fraction of trades are killed by the −0.5% protective stop
  rather than reaching the holding horizon? A high stop-out rate with a low above-entry recovery
  is the signature of a stop set *inside the entry's own noise band*.
* **entry-volatility vs stop distance** — the realized per-bar volatility just before entry
  compared to the 0.5% stop. When recent 1-bar volatility is ≈ or > 0.5%, a −0.5% stop sits
  inside one bar of noise, so premature stop-outs are structural, not bad luck.
* **adverse / favorable excursion** — MAE/MFE over the holding window, quantifying how often price
  dips through the stop only to recover (ties to the M7 95.7%-above-exit / 47.7%-above-entry split).

These are measured facts on historical bars (simulated, not live). The output drives the governed
hypothesis generator (entry-volatility gating, higher-timeframe confirmation, selectivity).
"""

from __future__ import annotations

import numpy as np

from cengine.alpha.portfolio_replay import resolve_exit
from cengine.certification.contracts import DATA_LABEL_HISTORICAL_REPLAY, safe_ticker
from cengine.optimization.exits.experiment_matrix import _entries_and_atr, _load_1m

FAILURE_ATTRIBUTION_VERSION = "incumbent-failure-attribution-1.0.0"
_STOP = 0.005
_MAX_HOLD = 120


def _entry_vol(bars: dict, entry_i: int, lookback: int = 20) -> float:
    c = bars["c"].astype(np.float64)
    lo = max(1, entry_i - lookback)
    seg = c[lo:entry_i + 1]
    if len(seg) < 3:
        return 0.0
    rets = np.diff(np.log(np.clip(seg, 1e-9, None)))
    return float(np.std(rets))


def attribute_symbol(bars: dict) -> dict:
    h = bars["h"].astype(np.float64)
    low = bars["l"].astype(np.float64)
    n = len(bars["c"])
    entries, _atr = _entries_and_atr(bars)
    stop_outs = 0
    time_stops = 0
    mae = []          # max adverse excursion (fraction, <=0)
    mfe = []          # max favorable excursion (fraction, >=0)
    entry_vols = []
    vol_ge_stop = 0
    returns = []
    for e in entries:
        res = resolve_exit(bars, e, cost_frac=0.0004, protective_stop=_STOP,
                           emergency_stop=0.03, max_hold=_MAX_HOLD)
        if res is None:
            continue
        exit_idx, entry_price, exit_price, net = res
        returns.append(net)
        # stop-out if exit price is at/below the stop level (gap or stop fill), else time-stop
        if exit_price <= entry_price * (1.0 - _STOP) + 1e-9:
            stop_outs += 1
        else:
            time_stops += 1
        hi = min(n, e + 1 + _MAX_HOLD)
        if hi > e + 1:
            win_hi = float(np.max(h[e + 1:hi]))
            win_lo = float(np.min(low[e + 1:hi]))
            mfe.append(win_hi / entry_price - 1.0)
            mae.append(win_lo / entry_price - 1.0)
        v = _entry_vol(bars, e)
        entry_vols.append(v)
        if v >= _STOP:
            vol_ge_stop += 1
    nt = len(returns)
    if nt == 0:
        return {"n_trades": 0}
    return {
        "n_trades": nt,
        "stop_out_rate": round(stop_outs / nt, 4),
        "time_stop_rate": round(time_stops / nt, 4),
        "mean_entry_bar_vol": round(float(np.mean(entry_vols)), 5),
        "median_entry_bar_vol": round(float(np.median(entry_vols)), 5),
        "frac_entry_vol_ge_stop": round(vol_ge_stop / nt, 4),
        "mean_max_adverse_excursion": round(float(np.mean(mae)), 5),
        "mean_max_favorable_excursion": round(float(np.mean(mfe)), 5),
        "net_expectancy": round(float(np.mean(returns)), 6),
    }


def run_failure_attribution(symbols: list, *, config=None) -> dict:
    if config is None:
        from cengine.lakehouse.config import LakehouseConfig
        config = LakehouseConfig.from_env()
    per_symbol = {}
    agg_trades = 0
    w_stop = w_vol = w_mae = w_mfe = 0.0
    for sym in symbols:
        s = safe_ticker(sym)
        bars = _load_1m(config, s)
        if bars is None:
            per_symbol[s] = {"available": False}
            continue
        a = attribute_symbol(bars)
        per_symbol[s] = a
        if a.get("n_trades", 0):
            agg_trades += a["n_trades"]
            w_stop += a["stop_out_rate"] * a["n_trades"]
            w_vol += a["frac_entry_vol_ge_stop"] * a["n_trades"]
            w_mae += a["mean_max_adverse_excursion"] * a["n_trades"]
            w_mfe += a["mean_max_favorable_excursion"] * a["n_trades"]
    agg = {
        "total_trades": agg_trades,
        "weighted_stop_out_rate": round(w_stop / agg_trades, 4) if agg_trades else None,
        "weighted_frac_entry_vol_ge_stop": round(w_vol / agg_trades, 4) if agg_trades else None,
        "weighted_mean_adverse_excursion": round(w_mae / agg_trades, 5) if agg_trades else None,
        "weighted_mean_favorable_excursion": round(w_mfe / agg_trades, 5) if agg_trades else None,
    }
    findings = _derive_findings(agg, per_symbol)
    return {
        "version": FAILURE_ATTRIBUTION_VERSION, "data_label": DATA_LABEL_HISTORICAL_REPLAY,
        "per_symbol": per_symbol, "aggregate": agg, "findings": findings,
        "note": ("measured on genuine 1m bars (simulated fills); drives governed hypothesis "
                 "generation — not a profitability claim"),
    }


def _derive_findings(agg: dict, per_symbol: dict) -> list:
    out = []
    sor = agg.get("weighted_stop_out_rate")
    mae = agg.get("weighted_mean_adverse_excursion")
    mfe = agg.get("weighted_mean_favorable_excursion")
    if sor is not None and sor >= 0.40:
        out.append({
            "finding": "high_protective_stop_frequency",
            "evidence": f"weighted stop-out rate {sor:.1%} of trades are killed by the −0.5% stop",
            "hypothesis": "require higher-timeframe trend confirmation and/or greater momentum "
                          "separation at entry to reduce the share of trades that stop out",
        })
    # the stop sits inside the TYPICAL multi-bar adverse swing, not 1-bar noise
    if mae is not None and abs(mae) >= _STOP:
        out.append({
            "finding": "stop_inside_normal_adverse_excursion",
            "evidence": f"mean max adverse excursion {mae:.2%} exceeds the −0.5% stop; mean max "
                        f"favorable excursion is only {mfe:.2%} — a near-symmetric swing against "
                        "a tight stop",
            "hypothesis": "entries are taken into normal two-sided intrabar swings wider than the "
                          "stop; selectivity (seller-exhaustion / displacement-from-VWAP / "
                          "regime) that raises MFE relative to MAE should improve net expectancy",
        })
    if mae is not None and mfe is not None and abs(mae) > mfe:
        out.append({
            "finding": "adverse_exceeds_favorable_excursion",
            "evidence": f"adverse excursion {mae:.2%} is larger in magnitude than favorable "
                        f"{mfe:.2%} — the entry edge is negative before the stop even acts",
            "hypothesis": "the momentum entry has no directional edge over the hold window; "
                          "entry-quality filters (not exit tuning) are the lever",
        })
    return out
