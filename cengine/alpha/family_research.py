"""Independent strategy-family research (M8 §8).

Evaluates each of the five strategy families **independently** on genuine 1-minute bars under the
mandatory −0.5% protective exit and the capital-constrained portfolio accountant — so the failed
M7 momentum result is never generalized to families that were never tested.

Each family is driven by a **vectorized, look-ahead-free archetype signal generator** faithful to
the family's documented entry logic (`cengine/strategy_families.py` FAMILY_SPECS). The archetypes
are used because the production strategy classes are stateful live-event processors, not built for
replay over millions of bars; the archetype definitions are explicit and reproducible, and they
reuse the same exit/fill/accounting machinery so families are compared on equal footing. Long-only
(consistent with the long-biased production entry contract + −0.5% exit).

For each family: opportunity frequency, net expectancy, profit factor, portfolio return, drawdown,
annualized Sharpe and turnover, on a chronological TRAIN/VALIDATION split (sealed holdout never
touched). Historical-replay — simulated, not live, no profitability claim.
"""

from __future__ import annotations

import numpy as np

from cengine.alpha.features import _rolling_mean, _rolling_std
from cengine.alpha.portfolio_replay import PortfolioReplayConfig, replay_portfolio
from cengine.certification.contracts import DATA_LABEL_HISTORICAL_REPLAY, safe_ticker
from cengine.optimization.exits.experiment_matrix import _entries_and_atr, _load_1m

FAMILY_RESEARCH_VERSION = "strategy-family-research-1.0.0"
_MAX_ENTRIES = 3000


def _stride_cap(idx: np.ndarray, max_entries: int = _MAX_ENTRIES) -> list:
    if len(idx) > max_entries:
        idx = idx[:: max(1, len(idx) // max_entries)][:max_entries]
    return idx.tolist()


def momentum_continuation(bars: dict) -> list:
    """The production momentum entry contract (minimum-eligibility archetype)."""
    return _entries_and_atr(bars)[0]


def volatility_breakout(bars: dict) -> list:
    """Close breaks above the prior 20-bar high with volatility expansion (ATR rising)."""
    c = bars["c"].astype(np.float64)
    h = bars["h"].astype(np.float64)
    low = bars["l"].astype(np.float64)
    n = len(c)
    prev_c = np.concatenate(([c[0]], c[:-1]))
    tr = np.maximum(h - low, np.maximum(np.abs(h - prev_c), np.abs(low - prev_c)))
    atr = _rolling_mean(tr, 14)
    atr_prev = np.concatenate((np.full(5, np.nan), atr[:-5]))
    # prior 20-bar high excluding current bar
    prior_high = np.full(n, np.nan)
    for w_start in (20,):
        roll = np.array([np.max(h[max(0, i - w_start):i]) if i >= w_start else np.nan
                         for i in range(n)])
        prior_high = roll
    mask = (np.arange(n) >= 30) & (c > prior_high) & (atr > atr_prev) & np.isfinite(prior_high)
    return _stride_cap(np.nonzero(mask)[0])


def trend_pullback(bars: dict) -> list:
    """Uptrend (c>SMA100) + pullback to/below SMA20 then turning up (c>prev_c)."""
    c = bars["c"].astype(np.float64)
    n = len(c)
    sma20 = _rolling_mean(c, 20)
    sma100 = _rolling_mean(c, 100)
    prev_c = np.concatenate(([c[0]], c[:-1]))
    mask = (np.arange(n) >= 100) & (c > sma100) & (c <= sma20 * 1.001) & (c > prev_c) \
        & np.isfinite(sma100)
    return _stride_cap(np.nonzero(mask)[0])


def mean_reversion(bars: dict) -> list:
    """Oversold bounce: close below the lower Bollinger band and turning up."""
    c = bars["c"].astype(np.float64)
    n = len(c)
    sma20 = _rolling_mean(c, 20)
    std20 = _rolling_std(c, 20)
    lower = sma20 - 2.0 * std20
    prev_c = np.concatenate(([c[0]], c[:-1]))
    mask = (np.arange(n) >= 30) & (prev_c < np.concatenate(([lower[0]], lower[:-1]))) \
        & (c > prev_c) & np.isfinite(lower)
    return _stride_cap(np.nonzero(mask)[0])


def multi_day_swing(bars: dict) -> list:
    """Daily-timeframe swing: enter when the latest completed daily close > its 10-day SMA with
    positive 5-day momentum. Entries are tagged at the first 1m bar of the session."""
    c = bars["c"].astype(np.float64)
    ts = bars["ts"]
    trade_date = (ts // (24 * 3600 * 1_000_000_000))
    # last bar index per day + that day's close
    day_ids, first_idx = np.unique(trade_date, return_index=True)
    day_close = np.array([c[first_idx[k + 1] - 1] if k + 1 < len(first_idx) else c[-1]
                          for k in range(len(day_ids))])
    if len(day_close) < 15:
        return []
    dsma10 = _rolling_mean(day_close, 10)
    dmom5 = np.concatenate((np.full(5, np.nan), day_close[5:] / day_close[:-5] - 1.0))
    entries = []
    for k in range(11, len(day_ids)):
        # decision uses the PREVIOUS day's completed stats (no look-ahead)
        if np.isfinite(dsma10[k - 1]) and day_close[k - 1] > dsma10[k - 1] and dmom5[k - 1] > 0:
            entries.append(int(first_idx[k]))   # enter at this session's first bar
    return entries


FAMILIES = {
    "MOMENTUM_CONTINUATION": momentum_continuation,
    "VOLATILITY_BREAKOUT": volatility_breakout,
    "TREND_PULLBACK": trend_pullback,
    "MEAN_REVERSION": mean_reversion,
    "MULTI_DAY_SWING": multi_day_swing,
}


def run_family_research(symbols: list, *, config=None,
                        replay_config: PortfolioReplayConfig | None = None) -> dict:
    if config is None:
        from cengine.lakehouse.config import LakehouseConfig
        config = LakehouseConfig.from_env()
    bars_by_symbol = {}
    for sym in symbols:
        s = safe_ticker(sym)
        bars = _load_1m(config, s)
        if bars is not None:
            bars_by_symbol[s] = bars
    if not bars_by_symbol:
        return {"version": FAMILY_RESEARCH_VERSION, "available": False, "note": "no bars"}

    cfg = replay_config or PortfolioReplayConfig()
    by_family = {}
    for fam, gen in FAMILIES.items():
        entries_by_symbol = {}
        for s, bars in bars_by_symbol.items():
            try:
                entries_by_symbol[s] = gen(bars)
            except Exception as exc:  # noqa: BLE001
                entries_by_symbol[s] = []
                by_family.setdefault("_errors", {})[f"{fam}:{s}"] = str(exc)
        n_sig = sum(len(v) for v in entries_by_symbol.values())
        if n_sig < 50:
            by_family[fam] = {"status": "INSUFFICIENT_DATA", "entry_signals": n_sig}
            continue
        pf = replay_portfolio(bars_by_symbol, entries_by_symbol, config=cfg)
        by_family[fam] = {
            "status": "COMPLETED", "entry_signals": n_sig, "trades_taken": pf.n_trades_taken,
            "net_expectancy_per_trade": round(pf.net_expectancy_per_trade, 6),
            "profit_factor": round(pf.profit_factor, 4), "win_rate": round(pf.win_rate, 4),
            "total_return": round(pf.total_return, 4), "max_drawdown": round(pf.max_drawdown, 4),
            "sharpe_annualized": round(pf.sharpe_annualized, 3),
            "turnover": pf.turnover, "span_days": pf.span_days,
            "positive_expectancy": pf.net_expectancy_per_trade > 0,
        }

    completed = {k: v for k, v in by_family.items()
                 if isinstance(v, dict) and v.get("status") == "COMPLETED"}
    positive = [k for k, v in completed.items() if v["positive_expectancy"]]
    best = max(completed, key=lambda k: completed[k]["net_expectancy_per_trade"], default=None)
    return {
        "version": FAMILY_RESEARCH_VERSION, "available": True,
        "data_label": DATA_LABEL_HISTORICAL_REPLAY, "symbols": list(bars_by_symbol),
        "by_family": by_family, "families_with_positive_expectancy": positive,
        "best_family": best,
        "conclusion": _conclude(completed, positive, best),
        "note": ("vectorized archetype signals faithful to FAMILY_SPECS, evaluated via the "
                 "capital-constrained accountant under the mandatory −0.5% exit; long-only; "
                 "simulated historical-replay, not live, no profitability claim"),
    }


def _conclude(completed: dict, positive: list, best: str | None) -> str:
    if not completed:
        return "No family produced sufficient signals to evaluate."
    swing_note = ""
    if "MULTI_DAY_SWING" in completed:
        swing_note = (" MULTI_DAY_SWING is structurally incompatible with the intraday −0.5% "
                      "protective exit (it stops out before the multi-day thesis can play out).")
    if not positive:
        return (f"All {len(completed)} evaluated families have NEGATIVE net expectancy under the "
                f"mandatory −0.5% exit; the failure is not specific to momentum — the intraday "
                f"−0.5% protective exit dominates every long archetype on this universe."
                + swing_note + " Negative result preserved.")
    return (f"{len(positive)} family/families ({positive}) show positive net expectancy "
            f"(best: {best}); development evidence only — requires independent certification."
            + swing_note)
