#!/usr/bin/env python3
"""Portfolio allocation economic comparison (Milestone 3, §21).

Compares allocation policies on a walk-forward basis and reports net performance
metrics. IMPORTANT (honesty, per CLAUDE.md + §24): this uses EXPLICITLY SYNTHETIC
return series (a seeded multi-regime generator), clearly labeled. It is a
methodology + relative-behavior comparison, NOT a claim of economic edge. Genuine
historical / out-of-sample / SIP-vs-IEX walk-forward validation on real bars is
DEFERRED (data/session-gated) and must not be conflated with these synthetic runs.

Policies compared: equal-weight, inverse-volatility, native risk-based (the M3
deterministic allocator), and regularized mean-variance (CVXPY, if available).
"""

from __future__ import annotations

import json
from pathlib import Path

import _cengine_native as native
import numpy as np

REPORTS = Path(__file__).resolve().parents[1] / "reports"
ASSETS = 8
PERIODS = 1000
LOOKBACK = 120
REBALANCE = 20
COST_BPS = 5.0           # round-trip transaction cost assumption (bps of turnover)
PER_ASSET_CAP = 0.30
GROSS_CAP = 1.0


def synthetic_returns(seed: int = 20260101) -> np.ndarray:
    """Seeded multi-regime synthetic returns (observations x assets). LABELED SYNTHETIC."""
    rng = np.random.default_rng(seed)
    # distinct per-asset drifts + vols; a shared market factor creates correlation;
    # two volatility regimes alternate to stress regime behavior.
    betas = rng.uniform(0.3, 1.3, size=ASSETS)
    idio_vol = rng.uniform(0.006, 0.02, size=ASSETS)
    drift = rng.normal(0.0003, 0.0003, size=ASSETS)
    out = np.zeros((PERIODS, ASSETS))
    for t in range(PERIODS):
        regime_vol = 0.008 if (t // 100) % 2 == 0 else 0.018
        market = rng.normal(0.0002, regime_vol)
        for a in range(ASSETS):
            out[t, a] = drift[a] + betas[a] * market + rng.normal(0.0, idio_vol[a])
    return out


def _project(weights: np.ndarray) -> np.ndarray:
    w = np.clip(weights, 0.0, PER_ASSET_CAP)
    gross = w.sum()
    if gross > GROSS_CAP and gross > 0:
        w = w * (GROSS_CAP / gross)
    return w


def equal_weight(cov, var):
    n = len(var)
    return _project(np.full(n, 1.0 / n))


def inverse_vol(cov, var):
    inv = np.array([1.0 / np.sqrt(v) if v > 0 else 0.0 for v in var])
    s = inv.sum()
    return _project(inv / s if s > 0 else inv)


def native_risk_based(cov, var):
    res = native.allocate_risk_based(
        [float(v) for v in var], cov.astype(np.float32).flatten().tolist(),
        len(var), PER_ASSET_CAP, GROSS_CAP)
    return np.array(res.weights) if res.valid else equal_weight(cov, var)


def mean_variance(cov, var, mu):
    try:
        import cvxpy as cp
    except Exception:  # noqa: BLE001
        return None
    n = len(var)
    w = cp.Variable(n)
    obj = cp.Maximize(mu @ w - 5.0 * cp.quad_form(w, cov))
    cons = [w >= 0, w <= PER_ASSET_CAP, cp.sum(w) <= GROSS_CAP]
    try:
        cp.Problem(obj, cons).solve()
        if w.value is None:
            return None
        return _project(np.maximum(0.0, np.asarray(w.value)))
    except Exception:  # noqa: BLE001
        return None


def backtest(returns: np.ndarray, policy, *, use_mu: bool = False) -> dict:
    prev_w = np.zeros(ASSETS)
    pnl = []
    turnover_total = 0.0
    cost_total = 0.0
    for t in range(LOOKBACK, PERIODS):
        if (t - LOOKBACK) % REBALANCE == 0:
            window = returns[t - LOOKBACK:t]
            cov = np.cov(window, rowvar=False, ddof=1) + 1e-8 * np.eye(ASSETS)
            var = np.diag(cov)
            if use_mu:
                mu = window.mean(axis=0)
                w = policy(cov, var, mu)
                if w is None:
                    return {"available": False}
            else:
                w = policy(cov, var)
            turnover = float(np.abs(w - prev_w).sum())
            turnover_total += turnover
            cost_total += turnover * (COST_BPS / 10_000.0)
            prev_w = w
        gross_ret = float(prev_w @ returns[t])
        pnl.append(gross_ret)
    pnl = np.array(pnl)
    # subtract amortized costs across the holding days of each rebalance
    net = pnl.copy()
    reb_days = max(1, (PERIODS - LOOKBACK))
    net = net - (cost_total / reb_days)
    ann = 252
    mean_d = float(net.mean())
    vol_d = float(net.std(ddof=1)) if len(net) > 1 else 0.0
    sharpe = (mean_d / vol_d * np.sqrt(ann)) if vol_d > 0 else 0.0
    downside = net[net < 0]
    dstd = downside.std(ddof=1) if len(downside) > 1 else 0.0
    sortino = (mean_d / dstd * np.sqrt(ann)) if dstd > 0 else 0.0
    equity = np.cumprod(1.0 + net)
    peak = np.maximum.accumulate(equity)
    max_dd = float(((peak - equity) / peak).max()) if len(equity) else 0.0
    return {
        "available": True,
        "net_mean_daily": round(mean_d, 8),
        "net_vol_daily": round(vol_d, 8),
        "net_sharpe_annualized": round(sharpe, 4),
        "sortino_annualized": round(sortino, 4),
        "max_drawdown": round(max_dd, 4),
        "total_turnover": round(turnover_total, 4),
        "total_cost_fraction": round(cost_total, 6),
        "final_equity_multiple": round(float(equity[-1]) if len(equity) else 1.0, 4),
    }


def main() -> None:
    returns = synthetic_returns()
    results = {
        "equal_weight": backtest(returns, equal_weight),
        "inverse_volatility": backtest(returns, inverse_vol),
        "native_risk_based": backtest(returns, native_risk_based),
        "mean_variance_cvxpy": backtest(returns, mean_variance, use_mu=True),
    }
    report = {
        "report": "portfolio_economic_comparison",
        "milestone": "M3 §21 — allocation policy comparison",
        "data_source": "SYNTHETIC (seeded multi-regime generator) — LABELED, NOT real market data",
        "scope_honesty": "Methodology + relative-behavior comparison only. This is NOT "
                         "an economic-edge claim. Genuine historical / walk-forward / OOS "
                         "and SIP-vs-IEX validation on real bars is DEFERRED (data/session-"
                         "gated) and must not be conflated with these synthetic runs.",
        "config": {"assets": ASSETS, "periods": PERIODS, "lookback": LOOKBACK,
                   "rebalance_every": REBALANCE, "cost_bps": COST_BPS,
                   "per_asset_cap": PER_ASSET_CAP, "gross_cap": GROSS_CAP},
        "policies": results,
        "note": "Fixed market-data subscription cost (Alpaca Algo Trader Plus, $99/mo) is "
                "an account-level cost measured separately; it is not amortized into these "
                "per-strategy synthetic returns.",
    }
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "portfolio_economic_comparison.json").write_text(json.dumps(report, indent=2))
    for name, r in results.items():
        if r.get("available"):
            print(f"{name:22} sharpe={r['net_sharpe_annualized']:+.2f} "
                  f"maxDD={r['max_drawdown']:.3f} turnover={r['total_turnover']:.1f}")
    print(f"-> {REPORTS}/portfolio_economic_comparison.json")


if __name__ == "__main__":
    main()
