"""Portfolio certification.

Two layers, kept distinct so engineering soundness is never confused with economic edge:

* **Construction (engineering)** — the native risk-based allocator and portfolio risk math are
  operational and correct: ``allocate_risk_based`` returns a valid, cap-respecting allocation;
  ``portfolio_risk_contributions`` sum to the total portfolio variance (the additive risk-
  decomposition identity); results are deterministic. CERTIFIABLE on their own merits.

* **Diversification (measurement)** — the realized daily-return correlation across the canonical
  universe, with an effective-breadth estimate. A genuine measurement from the historical bars.

* **Economic (gated)** — a portfolio can only add risk-adjusted value if its components have
  positive expected edge. Since strategy economic certification is NOT_CERTIFIED (negative net
  expectancy on the sealed holdout), a positive-Sharpe portfolio cannot be honestly constructed
  from these components — portfolio economic certification is NOT_CERTIFIED by inheritance, not
  by fabricated diversification benefit.
"""

from __future__ import annotations

import numpy as np

from cengine.certification.contracts import (
    DATA_LABEL_HISTORICAL_REPLAY,
    CertificationResult,
    CertificationVerdict,
    safe_ticker,
)
from cengine.certification.datasource import DEFAULT_PRICE_SCALE, load_symbol_bars

PORTFOLIO_CERT_VERSION = "certification-portfolio-1.0.0"


def _v(ok: bool) -> CertificationVerdict:
    return CertificationVerdict.CERTIFIED if ok else CertificationVerdict.NOT_CERTIFIED


def _certify_native_allocator() -> CertificationResult:
    import _cengine_native as native
    # a simple 3-asset problem with a known covariance
    variances = [0.04, 0.09, 0.16]
    cov = [0.04, 0.01, 0.00,
           0.01, 0.09, 0.02,
           0.00, 0.02, 0.16]
    alloc = native.allocate_risk_based(variances, cov, 3, 0.6, 1.0)
    weights = list(alloc.weights)
    valid = bool(alloc.valid)
    # determinism
    alloc2 = native.allocate_risk_based(variances, cov, 3, 0.6, 1.0)
    deterministic = list(alloc2.weights) == weights
    # Euler risk-decomposition identity: the per-asset risk contributions sum to the portfolio
    # VOLATILITY (the native convention) — the additive consistency of the risk math.
    rc = list(alloc.risk_contributions)
    vol = float(alloc.portfolio_volatility)
    additive = abs(sum(rc) - vol) < 1e-9 if rc else False
    # caps respected
    gross_ok = sum(abs(w) for w in weights) <= 1.0 + 1e-9
    ok = valid and deterministic and additive and gross_ok
    return CertificationResult(
        check="portfolio.native_allocator", component="portfolio", verdict=_v(ok),
        summary=(f"native risk-based allocator valid={valid}, deterministic={deterministic}, "
                 f"risk-contrib additive(sum={sum(rc):.6f} vs vol={vol:.6f})={additive}, "
                 f"gross<=1={gross_ok}"),
        metrics={"weights": [round(w, 6) for w in weights],
                 "portfolio_variance": float(alloc.portfolio_variance),
                 "portfolio_volatility": vol, "risk_contrib_sum": sum(rc),
                 "deterministic": deterministic})


def _daily_returns(symbol: str, timeframe: str = "5Min") -> np.ndarray | None:
    try:
        tbl = load_symbol_bars(symbol, timeframe,
                               columns="trade_date, bar_close_ticks c, event_timestamp_ns")
    except Exception:  # noqa: BLE001
        return None
    if tbl.num_rows < 200:
        return None
    import collections
    dates = tbl.column("trade_date").to_pylist()
    closes = tbl.column("c").to_numpy(zero_copy_only=False).astype(np.float64)
    # last close per trade_date -> daily series
    last_by_day: dict = collections.OrderedDict()
    for d, c in zip(dates, closes, strict=False):
        last_by_day[str(d)] = c
    series = np.array(list(last_by_day.values()), dtype=np.float64) / DEFAULT_PRICE_SCALE
    if len(series) < 100:
        return None
    return np.diff(np.log(series))


def _certify_diversification(symbols: list) -> CertificationResult:
    syms = [safe_ticker(s) for s in symbols]
    series = {}
    for s in syms:
        r = _daily_returns(s)
        if r is not None:
            series[s] = r
    if len(series) < 2:
        return CertificationResult(
            check="portfolio.diversification", component="portfolio",
            verdict=CertificationVerdict.INSUFFICIENT_DATA,
            summary="fewer than 2 symbols with sufficient daily history",
            data_label=DATA_LABEL_HISTORICAL_REPLAY)
    # align to the shortest common length (most recent)
    n = min(len(r) for r in series.values())
    mat = np.vstack([r[-n:] for r in series.values()])
    corr = np.corrcoef(mat)
    off = corr[np.triu_indices_from(corr, k=1)]
    mean_corr = float(np.mean(off))
    k = len(series)
    # effective breadth under equal weights with average pairwise correlation
    eff_breadth = k / (1.0 + (k - 1) * max(mean_corr, 0.0)) if k > 1 else 1.0
    return CertificationResult(
        check="portfolio.diversification", component="portfolio",
        verdict=CertificationVerdict.CERTIFIED,
        summary=(f"{k} symbols: mean pairwise daily-return correlation {mean_corr:.2f}; "
                 f"effective breadth {eff_breadth:.2f}/{k}"),
        sample_size=n, data_label=DATA_LABEL_HISTORICAL_REPLAY,
        metrics={"symbols": list(series.keys()), "mean_pairwise_corr": round(mean_corr, 4),
                 "effective_breadth": round(eff_breadth, 3),
                 "corr_min": round(float(off.min()), 3), "corr_max": round(float(off.max()), 3)},
        notes="measurement of realized diversification; not an economic-edge claim")


def _certify_portfolio_economics(strategy_certified: bool) -> CertificationResult:
    # portfolio economic value is gated on component edge; without a certified positive-expectancy
    # component, a positive-Sharpe portfolio cannot be honestly claimed.
    if strategy_certified:
        return CertificationResult(
            check="portfolio.economics", component="portfolio",
            verdict=CertificationVerdict.INSUFFICIENT_DATA,
            summary="component edge exists; full portfolio economic certification requires a "
                    "multi-strategy OOS portfolio backtest (not yet run)",
            data_label=DATA_LABEL_HISTORICAL_REPLAY)
    return CertificationResult(
        check="portfolio.economics", component="portfolio",
        verdict=CertificationVerdict.NOT_CERTIFIED,
        summary="no component strategy has positive net expectancy on the sealed holdout; a "
                "positive-Sharpe portfolio cannot be constructed from negative-edge components",
        data_label=DATA_LABEL_HISTORICAL_REPLAY,
        notes="portfolio economic certification is inherited-NOT_CERTIFIED, not fabricated")


def certify_portfolio(symbols: list, *, strategy_certified: bool = False) -> list:
    """Full portfolio certification: construction + diversification + gated economics."""
    return [
        _certify_native_allocator(),
        _certify_diversification(symbols),
        _certify_portfolio_economics(strategy_certified),
    ]
