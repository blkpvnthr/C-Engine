"""Independent validation protocol (M8 §20).

Separates training / development-validation / research data from independent certification evidence
and protects the latter from optimization:

* **Holdout integrity** — the original M7 sealed holdout (the 5 canonical symbols' 1-minute series)
  is treated as **consumed** and is never reused as new independent evidence. All M8 development
  (entry-quality, multi-timeframe, regime, family, supervised, bandit) used ONLY those 5 canonical
  1-minute symbols, so the 26 five-minute-only instruments form a cohort **untouched** by both M7
  certification and M8 development.

* **Fresh cohort** — those 26 untouched instruments (broad/sector/leveraged/inverse/defensive) are
  the available independent cohort (§15). Because **no development challenger reached the frozen
  acceptance gates** (every M8 result is negative), there is nothing to *certify* — so the cohort is
  not consumed by a selection search. We run exactly **one pre-registered confirmatory test** (the
  production incumbent, not a searched configuration) to check whether the negative finding
  generalizes to fresh instruments; confirming a negative result is falsification, not
  data-snooping.

* **Multiple-testing audit** — counts the development experiments and states the data-snooping /
  selection-bias caveat so no best-of-many-development-configs is ever described as certified.

If a positive challenger ever qualifies in development, independent certification on this fresh
cohort (or on prospective paper observations) is the required next step — deferred until then.
"""

from __future__ import annotations

import numpy as np

from cengine.alpha.portfolio_replay import PortfolioReplayConfig, replay_portfolio
from cengine.certification.contracts import DATA_LABEL_HISTORICAL_REPLAY, safe_ticker
from cengine.certification.datasource import bar_inventory
from cengine.optimization.exits.experiment_matrix import _entries_and_atr

VALIDATION_VERSION = "alpha-independent-validation-1.0.0"
CANONICAL = ["AAPL", "MSFT", "NVDA", "SPY", "QQQ"]


def fresh_cohort() -> list:
    """5-minute-only instruments untouched by M7 holdout + M8 1-minute development."""
    inv = bar_inventory()
    if not inv.available:
        return []
    one = set(inv.symbols("1Min"))
    five = set(inv.symbols("5Min"))
    return sorted(five - one)


def _load_5m_batch(symbols: list) -> dict:
    """Load many symbols' 5-minute bars in ONE DuckDB scan (avoids N full-glob scans)."""
    from cengine.certification.datasource import _escape, connect, market_bars_glob
    safe = [safe_ticker(s) for s in symbols]
    if not safe:
        return {}
    g = _escape(market_bars_glob())
    inlist = ", ".join(f"'{s}'" for s in safe)   # all allow-list-validated
    con = connect()
    tbl = con.sql(
        f"""SELECT symbol_id, event_timestamp_ns ts, bar_open_ticks o, bar_high_ticks h,
                   bar_low_ticks l, bar_close_ticks c, volume v
            FROM read_parquet('{g}', union_by_name=true)
            WHERE symbol_id IN ({inlist}) AND timeframe='5Min' AND bar_close_ticks>0
            ORDER BY symbol_id, ts"""
    ).to_arrow_table()
    syms = tbl.column("symbol_id").to_numpy(zero_copy_only=False)
    cols = {k: tbl.column(k).to_numpy(zero_copy_only=False)
            for k in ("ts", "o", "h", "l", "c", "v")}
    out = {}
    for s in safe:
        mask = syms == s
        if mask.sum() >= 500:
            out[s] = {k: v[mask] for k, v in cols.items()}
    return out


def holdout_integrity() -> dict:
    inv = bar_inventory()
    fresh = fresh_cohort()
    return {
        "m7_canonical_holdout": {"symbols": CANONICAL, "timeframe": "1Min", "status": "consumed",
                                 "reused_as_new_holdout": False},
        "m8_development_symbols": CANONICAL,
        "m8_development_timeframe": "1Min",
        "fresh_cohort_symbols": fresh,
        "fresh_cohort_timeframe": "5Min",
        "fresh_cohort_untouched_by_development": True,
        "fresh_cohort_size": len(fresh),
        "lakehouse_available": inv.available,
        "note": ("fresh cohort shares no symbol with the 1m development/holdout universe; it is "
                 "reserved and only a single pre-registered incumbent test is run on it"),
    }


def multiple_testing_audit(development_experiment_count: int) -> dict:
    # rough count of independent development comparisons that touched the canonical 1m data
    return {
        "development_experiment_families": {
            "entry_quality_filter_scans": "11 features x 8 thresholds x 2 directions",
            "multi_timeframe_gates": 5,
            "regime_buckets": "~9",
            "strategy_family_archetypes": 5,
            "supervised_models": 4,
            "bandit_contexts": "regime x family",
        },
        "approx_total_development_comparisons": development_experiment_count,
        "selection_bias_warning": (
            "the best figure among many development comparisons is optimistic; no best-of-search "
            "configuration is described as certified. A Bonferroni-style correction would widen "
            "every development confidence interval substantially."),
        "data_snooping_controls": [
            "chronological TRAIN/VAL splits (no future leakage)",
            "thresholds fit on TRAIN, measured on VAL",
            "sealed M7 holdout never accessed",
            "fresh cohort reserved; single pre-registered incumbent test only",
        ],
    }


def confirmatory_fresh_cohort_test(*, max_symbols: int = 26) -> dict:
    """ONE pre-registered test: the production incumbent on the fresh 5m cohort (not a search)."""
    fresh = fresh_cohort()[:max_symbols]
    if not fresh:
        return {"status": "unavailable", "note": "no fresh cohort symbols"}
    bars_by_symbol = _load_5m_batch(fresh)
    entries_by_symbol = {s: _entries_and_atr(b)[0] for s, b in bars_by_symbol.items()}
    if not bars_by_symbol:
        return {"status": "unavailable", "note": "no loadable fresh bars"}
    # 5-minute bars -> shorter max_hold (24 bars = 2h) so the holding horizon matches 1m's 120min
    cfg = PortfolioReplayConfig(max_hold_bars=24)
    pf = replay_portfolio(bars_by_symbol, entries_by_symbol, config=cfg)
    per_trade = [r for r in [pf.net_expectancy_per_trade] if r is not None]
    return {
        "status": "completed", "pre_registered": "production incumbent (−0.5% exit + momentum)",
        "data_label": DATA_LABEL_HISTORICAL_REPLAY, "symbols": list(bars_by_symbol),
        "n_symbols": len(bars_by_symbol), "trades_taken": pf.n_trades_taken,
        "net_expectancy_per_trade": round(pf.net_expectancy_per_trade, 6),
        "profit_factor": round(pf.profit_factor, 4), "total_return": round(pf.total_return, 4),
        "max_drawdown": round(pf.max_drawdown, 4),
        "sharpe_annualized": round(pf.sharpe_annualized, 3),
        "positive": pf.net_expectancy_per_trade > 0,
        "interpretation": ("single pre-registered confirmatory test; a negative result here "
                           "independently corroborates the development finding on fresh "
                           "instruments (not a certification of a positive edge)"),
        "_per_trade": per_trade,
    }


def run_validation_protocol(*, development_experiment_count: int = 0,
                            any_challenger_accepted: bool = False) -> dict:
    integrity = holdout_integrity()
    audit = multiple_testing_audit(development_experiment_count)
    confirm = confirmatory_fresh_cohort_test()
    if any_challenger_accepted:
        cert_status = "pending_fresh_cohort_certification"
        cert_note = ("a challenger reached the gates in development; run its independent "
                     "certification on the fresh cohort before any claim")
    else:
        cert_status = "deferred"
        cert_note = ("no development challenger reached the frozen acceptance gates; there is no "
                     "positive edge to certify, so independent certification is deferred to future "
                     "qualifying challengers / prospective paper observations (§20)")
    return {
        "version": VALIDATION_VERSION, "holdout_integrity": integrity,
        "multiple_testing_audit": audit, "confirmatory_fresh_cohort_test": confirm,
        "independent_certification_status": cert_status,
        "independent_certification_note": cert_note,
        "np_finite_guard": bool(np.isfinite(confirm.get("net_expectancy_per_trade", 0.0))
                                if confirm.get("status") == "completed" else True),
    }
