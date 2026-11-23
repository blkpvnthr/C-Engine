"""Tests for §21 capital/cost sensitivity, §22 stress, portfolio and execution certification.

Economic paths run on deterministic synthetic bars (monkeypatched loaders) so they stay fast;
the native allocator, cost model and execution planner are exercised for real.
"""

from __future__ import annotations

import numpy as np

from cengine.certification import portfolio_cert as pc
from cengine.certification import sensitivity as sens
from cengine.certification.contracts import CertificationVerdict
from cengine.certification.execution_cert import certify_execution
from cengine.certification.holdout import SymbolHoldout


def _synthetic_bars(n: int = 2500, seed: int = 3) -> dict:
    rng = np.random.default_rng(seed)
    steps = rng.normal(0.0, 0.0015, size=n)
    price = 100.0 * np.exp(np.cumsum(steps))
    c = (price * 10000).astype(np.int64)
    o = np.concatenate(([c[0]], c[:-1]))
    spread = np.maximum(1, (np.abs(steps) * price * 10000)).astype(np.int64)
    h = np.maximum(o, c) + spread
    low = np.minimum(o, c) - spread
    ts = (np.arange(n, dtype=np.int64)) * (60 * 1_000_000_000)
    return {"ts": ts, "o": o, "h": h, "l": np.maximum(low, 1), "c": c,
            "v": np.full(n, 1000, dtype=np.int64)}


# --------------------------------------------------------------------------- §21 sensitivity


def test_cost_sensitivity_curve_is_monotonic(monkeypatch):
    bars = _synthetic_bars()
    monkeypatch.setattr(sens, "_load_1m", lambda *a, **k: bars)
    monkeypatch.setattr(sens, "derive_symbol_holdout",
                        lambda s, config=None: SymbolHoldout(s, (0, 6), (7, 8), (9, 10), 3, "fp"))
    results = sens.certify_capital_sensitivity(["TEST"])
    sym_res = results[0]
    curve = sym_res.metrics["cost_curve"]
    # expectancy must be non-increasing as cost rises
    exps = [p["net_expectancy"] for p in curve]
    assert all(exps[i] >= exps[i + 1] for i in range(len(exps) - 1))
    # a portfolio roll-up and an explicit market-impact-capacity INSUFFICIENT_DATA result exist
    checks = {r.check: r.verdict for r in results}
    assert checks["capital_sensitivity.market_impact_capacity"] == \
        CertificationVerdict.INSUFFICIENT_DATA
    assert "capital_sensitivity.portfolio" in checks


def test_stress_window_ns_helper_is_ordered():
    assert sens._ns("2020-02-19") < sens._ns("2020-04-07") < sens._ns("2022-01-03")


def test_capital_sensitivity_insufficient_without_data(monkeypatch):
    monkeypatch.setattr(sens, "derive_symbol_holdout", lambda s, config=None: None)
    results = sens.certify_capital_sensitivity(["TEST"])
    assert results[0].verdict == CertificationVerdict.INSUFFICIENT_DATA


# --------------------------------------------------------------------------- portfolio


def test_native_allocator_risk_decomposition_additive():
    r = pc._certify_native_allocator()
    assert r.verdict == CertificationVerdict.CERTIFIED
    assert r.metrics["deterministic"] is True
    # risk contributions sum to the portfolio volatility (Euler identity)
    assert abs(r.metrics["risk_contrib_sum"] - r.metrics["portfolio_volatility"]) < 1e-9


def test_portfolio_economics_gated_on_component_edge():
    not_cert = pc._certify_portfolio_economics(strategy_certified=False)
    assert not_cert.verdict == CertificationVerdict.NOT_CERTIFIED
    gated = pc._certify_portfolio_economics(strategy_certified=True)
    assert gated.verdict == CertificationVerdict.INSUFFICIENT_DATA


def test_diversification_measures_correlation(monkeypatch):
    rng = np.random.default_rng(0)
    base = rng.normal(0, 0.01, 300)
    monkeypatch.setattr(
        pc, "_daily_returns",
        lambda s, timeframe="5Min": base + rng.normal(0, 0.01, 300))
    r = pc._certify_diversification(["AAA", "BBB", "CCC"])
    assert r.verdict == CertificationVerdict.CERTIFIED
    assert 0.0 <= r.metrics["effective_breadth"] <= 3.0
    assert "mean_pairwise_corr" in r.metrics


# --------------------------------------------------------------------------- execution


def test_execution_certification():
    results = {r.check: r for r in certify_execution()}
    assert results["execution.planner"].verdict == CertificationVerdict.CERTIFIED
    assert results["execution.cost_model"].verdict == CertificationVerdict.CERTIFIED
    # conservative fills verified structurally against the real simulator source
    cf = results["execution.conservative_fills"]
    assert cf.verdict == CertificationVerdict.CERTIFIED
    assert cf.metrics["stop_before_target"] and cf.metrics["gap_through_at_open"]
    # realized quality is honestly INSUFFICIENT_DATA (no executed orders)
    assert results["execution.realized_quality"].verdict == CertificationVerdict.INSUFFICIENT_DATA
