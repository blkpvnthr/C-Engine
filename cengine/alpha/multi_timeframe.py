"""Multi-timeframe confirmation study (M8 §6).

Tests whether requiring **higher-timeframe trend confirmation** at entry improves the EXECUTABLE
capital-constrained portfolio (not merely per-trade expectancy). Confirmation gates are built from
the look-ahead-free, completed-candle higher-TF features (15-min and 60-min trend sign, trend
persistence vs the 100-bar mean) and applied as entry filters; the surviving entries are run
through the portfolio accountant and compared against the ungated baseline.

Incremental value is judged against the existing 1m/5m momentum entry contract. Developing candles
are never used to qualify an earlier decision. Historical-replay on genuine bars — simulated, not
live, never a profitability claim.
"""

from __future__ import annotations

import numpy as np

from cengine.alpha.features import compute_features
from cengine.alpha.portfolio_replay import PortfolioReplayConfig, replay_portfolio
from cengine.certification.contracts import DATA_LABEL_HISTORICAL_REPLAY, safe_ticker
from cengine.optimization.exits.experiment_matrix import _entries_and_atr, _load_1m

MULTI_TIMEFRAME_VERSION = "multi-timeframe-confirmation-1.0.0"

# confirmation gates: name -> predicate over the per-bar feature dict at the entry index
_GATES = {
    "baseline_all_entries": None,
    "htf15_up": lambda f, i: f["htf_trend_15"][i] > 0,
    "htf60_up": lambda f, i: f["htf_trend_60"][i] > 0,
    "htf15_and_htf60_up": lambda f, i: f["htf_trend_15"][i] > 0 and f["htf_trend_60"][i] > 0,
    "aligned_up_not_extended": lambda f, i: (f["htf_trend_60"][i] > 0
                                             and np.isfinite(f["trend_persist"][i])
                                             and f["trend_persist"][i] < 0.01),
}


def _gated_entries(bars: dict, entries: list, gate) -> list:
    if gate is None:
        return entries
    feats = compute_features(bars)
    out = []
    for e in entries:
        try:
            if gate(feats, e):
                out.append(e)
        except (IndexError, TypeError):
            continue
    return out


def run_multi_timeframe(symbols: list, *, config=None,
                        replay_config: PortfolioReplayConfig | None = None) -> dict:
    if config is None:
        from cengine.lakehouse.config import LakehouseConfig
        config = LakehouseConfig.from_env()
    bars_by_symbol = {}
    entries_by_symbol = {}
    for sym in symbols:
        s = safe_ticker(sym)
        bars = _load_1m(config, s)
        if bars is None:
            continue
        bars_by_symbol[s] = bars
        entries_by_symbol[s] = _entries_and_atr(bars)[0]
    if not bars_by_symbol:
        return {"version": MULTI_TIMEFRAME_VERSION, "available": False, "note": "no bars"}

    cfg = replay_config or PortfolioReplayConfig()
    results = {}
    base_signals = sum(len(v) for v in entries_by_symbol.values())
    for gate_name, gate in _GATES.items():
        gated = {s: _gated_entries(bars_by_symbol[s], entries_by_symbol[s], gate)
                 for s in bars_by_symbol}
        n_sig = sum(len(v) for v in gated.values())
        pf = replay_portfolio(bars_by_symbol, gated, config=cfg)
        results[gate_name] = {
            "entry_signals": n_sig,
            "signal_fraction": round(n_sig / base_signals, 4) if base_signals else 0.0,
            "trades_taken": pf.n_trades_taken,
            "net_expectancy_per_trade": round(pf.net_expectancy_per_trade, 6),
            "total_return": round(pf.total_return, 4),
            "max_drawdown": round(pf.max_drawdown, 4),
            "sharpe_annualized": round(pf.sharpe_annualized, 3),
            "profit_factor": round(pf.profit_factor, 4),
        }
    base = results["baseline_all_entries"]
    best = max((k for k in results if k != "baseline_all_entries"),
               key=lambda k: results[k]["net_expectancy_per_trade"], default=None)
    improved = (best is not None
                and results[best]["net_expectancy_per_trade"] > base["net_expectancy_per_trade"])
    any_positive = any(r["net_expectancy_per_trade"] > 0 for r in results.values())
    if best is None:
        conclusion = "insufficient gates to evaluate"
    elif any_positive:
        conclusion = (f"a higher-timeframe-confirmed subset reached POSITIVE expectancy "
                      f"(best gate '{best}' {results[best]['net_expectancy_per_trade']:+.5f} vs "
                      f"baseline {base['net_expectancy_per_trade']:+.5f}) — development evidence "
                      "only, requires independent certification")
    elif improved:
        conclusion = (f"higher-timeframe confirmation IMPROVES but does not turn positive the "
                      f"executable portfolio (best gate '{best}' "
                      f"{results[best]['net_expectancy_per_trade']:+.5f} vs baseline "
                      f"{base['net_expectancy_per_trade']:+.5f})")
    else:
        conclusion = (f"higher-timeframe confirmation does not improve the executable portfolio "
                      f"(baseline {base['net_expectancy_per_trade']:+.5f}); negative result "
                      "preserved")
    return {
        "version": MULTI_TIMEFRAME_VERSION, "available": True,
        "data_label": DATA_LABEL_HISTORICAL_REPLAY, "symbols": list(bars_by_symbol),
        "by_gate": results, "best_gate": best, "best_improves_expectancy": improved,
        "any_gate_positive_expectancy": any_positive, "conclusion": conclusion,
        "note": ("portfolio-level (executable-equity) comparison on genuine bars; incremental "
                 "value vs the 1m/5m momentum contract; simulated, not live"),
    }
