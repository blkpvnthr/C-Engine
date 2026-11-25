"""Protective-exit research (M8 §14).

The production mandatory protective exit (``unrealized_return < -0.005``), its independent
event-driven path and the 5-second periodic backstop are **preserved and never modified here**; the
inactive −3% candidate stays inactive. This module researches whether *better entries* (not a looser
exit) reduce premature stop-outs, keeping any alternative hard-stop as an **isolated
counterfactual**.

Research, on genuine 1m bars:

* **stop-frequency vs entry selectivity** — does higher-timeframe-confirmed / less-extended entry
  selection reduce the fraction of trades killed by the −0.5% stop? (Reuses the entry features +
  the shared exit resolver.)
* **alternative-stop counterfactuals** — a grid of wider isolated hard stops (research-only; the
  production −0.5% remains the incumbent and is never auto-changed). Reuses the M7 exit-matrix
  logic.

Simulated historical-replay; not live; no change to production protection.
"""

from __future__ import annotations

import numpy as np

from cengine.alpha.features import compute_features
from cengine.alpha.portfolio_replay import resolve_exit
from cengine.certification.contracts import DATA_LABEL_HISTORICAL_REPLAY, safe_ticker
from cengine.optimization.exits.experiment_matrix import _entries_and_atr, _load_1m

PROTECTIVE_EXIT_RESEARCH_VERSION = "protective-exit-research-1.0.0"
_PROD_STOP = 0.005
_ALT_STOPS = (0.0075, 0.010, 0.015, 0.020)   # ISOLATED counterfactuals only


def _stop_out_fraction(bars: dict, entries: list, stop: float) -> tuple:
    outs = 0
    rets = []
    for e in entries:
        res = resolve_exit(bars, e, cost_frac=0.0004, protective_stop=stop, emergency_stop=0.03,
                           max_hold=120)
        if res is None:
            continue
        _, entry_price, exit_price, net = res
        rets.append(net)
        if exit_price <= entry_price * (1.0 - stop) + 1e-9:
            outs += 1
    n = len(rets)
    return (outs / n if n else 0.0), (float(np.mean(rets)) if n else 0.0), n


def run_protective_exit_research(symbols: list, *, config=None) -> dict:
    if config is None:
        from cengine.lakehouse.config import LakehouseConfig
        config = LakehouseConfig.from_env()
    all_entries = {}
    sel_entries = {}
    bars_by_symbol = {}
    for sym in symbols:
        s = safe_ticker(sym)
        bars = _load_1m(config, s)
        if bars is None:
            continue
        bars_by_symbol[s] = bars
        ent = _entries_and_atr(bars)[0]
        all_entries[s] = ent
        feats = compute_features(bars)
        sel_entries[s] = [e for e in ent
                          if np.isfinite(feats["trend_persist"][e])
                          and feats["trend_persist"][e] < 0.01 and feats["htf_trend_60"][e] > 0]
    if not bars_by_symbol:
        return {"version": PROTECTIVE_EXIT_RESEARCH_VERSION, "available": False, "note": "no bars"}

    # entry selectivity vs stop frequency (production −0.5% stop)
    def _agg(entries_by_symbol):
        sor = exp = 0.0
        n = 0
        for s, bars in bars_by_symbol.items():
            f, e, cnt = _stop_out_fraction(bars, entries_by_symbol.get(s, []), _PROD_STOP)
            sor += f * cnt
            exp += e * cnt
            n += cnt
        return (round(sor / n, 4) if n else None, round(exp / n, 6) if n else None, n)

    all_sor, all_exp, all_n = _agg(all_entries)
    sel_sor, sel_exp, sel_n = _agg(sel_entries)

    # isolated alternative-stop counterfactuals (research-only; production unchanged)
    alt = {}
    for stop in (_PROD_STOP, *_ALT_STOPS):
        tot_sor = tot_exp = 0.0
        n = 0
        for s, bars in bars_by_symbol.items():
            f, e, cnt = _stop_out_fraction(bars, all_entries.get(s, []), stop)
            tot_sor += f * cnt
            tot_exp += e * cnt
            n += cnt
        alt[f"stop_{stop:.4f}"] = {
            "stop_out_fraction": round(tot_sor / n, 4) if n else None,
            "net_expectancy": round(tot_exp / n, 6) if n else None,
            "is_production": stop == _PROD_STOP, "isolated_counterfactual": stop != _PROD_STOP}

    return {
        "version": PROTECTIVE_EXIT_RESEARCH_VERSION, "available": True,
        "data_label": DATA_LABEL_HISTORICAL_REPLAY,
        "production_exit_preserved": True, "emergency_3pct_candidate_active": False,
        "entry_selectivity_vs_stop_frequency": {
            "all_entries": {"stop_out_fraction": all_sor, "net_expectancy": all_exp, "n": all_n},
            "selective_entries": {"stop_out_fraction": sel_sor, "net_expectancy": sel_exp,
                                  "n": sel_n},
            "selectivity_reduces_stop_frequency": (sel_sor is not None and all_sor is not None
                                                   and sel_sor < all_sor),
        },
        "alternative_stop_counterfactuals_isolated": alt,
        "conclusion": _conclude(all_sor, sel_sor, all_exp, sel_exp, alt),
        "note": ("production −0.5% exit + event-driven path + 5s backstop UNCHANGED; −3% candidate "
                 "inactive; alternative stops are isolated counterfactuals; simulated, not live"),
    }


def _conclude(all_sor, sel_sor, all_exp, sel_exp, alt) -> str:
    parts = []
    if sel_sor is not None and all_sor is not None:
        if sel_sor < all_sor:
            parts.append(f"entry selectivity reduces the stop-out rate "
                         f"{all_sor:.1%}->{sel_sor:.1%} "
                         f"(expectancy {all_exp:+.5f}->{sel_exp:+.5f})")
        else:
            parts.append(f"entry selectivity does not reduce the stop-out rate ({all_sor:.1%})")
    # best isolated alt stop by expectancy
    best = max((k for k in alt if alt[k]["net_expectancy"] is not None),
               key=lambda k: alt[k]["net_expectancy"], default=None)
    if best:
        be = alt[best]["net_expectancy"]
        parts.append(f"best isolated hard stop '{best}' expectancy {be:+.5f} "
                     + ("(positive — but an isolated counterfactual, not a portfolio result, and "
                        "production stays −0.5%)" if be > 0 else "(still negative)"))
    parts.append("production protection unchanged; no auto-modification")
    return "; ".join(parts) + "."
