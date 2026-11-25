"""Portfolio-aware alpha selection (M8 §13).

Compares deploying capital against **holding cash**, making NO_TRADE an explicit, first-class
baseline. Three portfolios are run through the capital-constrained accountant on the identical
genuine bars:

* **all_eligible** — take every eligible momentum entry (the naive policy);
* **selective_htf** — take only higher-timeframe-confirmed entries (fewer, hopefully better);
* **cash / NO_TRADE** — deploy nothing; equity stays flat at the risk-free-zero baseline.

Because the entries carry negative expected edge, the expected and observed result is that **cash
dominates** both active policies — which is exactly why a valid NO_TRADE decision must exist. The
comparison quantifies how much capital the active policies destroy versus simply holding cash.
Historical-replay — simulated, not live, no profitability claim.
"""

from __future__ import annotations

from cengine.alpha.features import compute_features
from cengine.alpha.portfolio_replay import PortfolioReplayConfig, replay_portfolio
from cengine.certification.contracts import DATA_LABEL_HISTORICAL_REPLAY, safe_ticker
from cengine.optimization.exits.experiment_matrix import _entries_and_atr, _load_1m

PORTFOLIO_ALPHA_VERSION = "portfolio-alpha-selection-1.0.0"


def _htf_confirmed(bars: dict, entries: list) -> list:
    feats = compute_features(bars)
    out = []
    for e in entries:
        try:
            if feats["htf_trend_60"][e] > 0 and feats["htf_trend_15"][e] > 0:
                out.append(e)
        except IndexError:
            continue
    return out


def run_portfolio_alpha(symbols: list, *, config=None,
                        replay_config: PortfolioReplayConfig | None = None) -> dict:
    if config is None:
        from cengine.lakehouse.config import LakehouseConfig
        config = LakehouseConfig.from_env()
    bars_by_symbol = {}
    all_entries = {}
    sel_entries = {}
    for sym in symbols:
        s = safe_ticker(sym)
        bars = _load_1m(config, s)
        if bars is None:
            continue
        bars_by_symbol[s] = bars
        ent = _entries_and_atr(bars)[0]
        all_entries[s] = ent
        sel_entries[s] = _htf_confirmed(bars, ent)
    if not bars_by_symbol:
        return {"version": PORTFOLIO_ALPHA_VERSION, "available": False, "note": "no bars"}

    cfg = replay_config or PortfolioReplayConfig()
    pf_all = replay_portfolio(bars_by_symbol, all_entries, config=cfg)
    pf_sel = replay_portfolio(bars_by_symbol, sel_entries, config=cfg)

    def _summ(pf) -> dict:
        return {"trades_taken": pf.n_trades_taken, "total_return": round(pf.total_return, 4),
                "final_equity": pf.final_equity, "max_drawdown": round(pf.max_drawdown, 4),
                "sharpe_annualized": round(pf.sharpe_annualized, 3),
                "net_expectancy_per_trade": round(pf.net_expectancy_per_trade, 6)}

    cash = {"trades_taken": 0, "total_return": 0.0, "final_equity": cfg.initial_capital,
            "max_drawdown": 0.0, "sharpe_annualized": 0.0, "net_expectancy_per_trade": 0.0}
    portfolios = {"all_eligible": _summ(pf_all), "selective_htf": _summ(pf_sel),
                  "cash_no_trade": cash}
    best = max(portfolios, key=lambda k: portfolios[k]["total_return"])
    cash_dominates = best == "cash_no_trade"
    return {
        "version": PORTFOLIO_ALPHA_VERSION, "available": True,
        "data_label": DATA_LABEL_HISTORICAL_REPLAY, "symbols": list(bars_by_symbol),
        "portfolios": portfolios, "best_policy_by_total_return": best,
        "cash_dominates_active": cash_dominates,
        "no_trade_is_valid_decision": True,
        "conclusion": (
            f"Cash (NO_TRADE) {'DOMINATES' if cash_dominates else 'does not dominate'} the active "
            f"policies: all_eligible {portfolios['all_eligible']['total_return']:+.1%}, "
            f"selective_htf {portfolios['selective_htf']['total_return']:+.1%}, cash +0.0%. "
            + ("Deploying capital into these negative-edge entries destroys capital versus holding "
               "cash — NO_TRADE is the correct portfolio decision here." if cash_dominates
               else "An active policy beat cash on total return (investigate before any claim).")),
        "note": "capital-constrained portfolio comparison on genuine bars; cash is an explicit "
                "baseline; simulated, not live, no profitability claim",
    }
