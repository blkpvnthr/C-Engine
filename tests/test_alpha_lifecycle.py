"""Tests for M8 objective/gates, validation protocol, lifecycle, dashboard, benchmarks and the
runtime AlphaDiscoveryService."""

from __future__ import annotations

import numpy as np

from cengine.alpha import benchmarks as bm
from cengine.alpha import dashboard as dash
from cengine.alpha import lifecycle as lc
from cengine.alpha import objective as obj
from cengine.alpha import portfolio_alpha as pa
from cengine.alpha import protective_exit_research as per
from cengine.alpha import validation as val


def _walk(n=6000, seed=3) -> dict:
    rng = np.random.default_rng(seed)
    steps = rng.normal(0, 0.0015, n)
    price = (10000 * np.exp(np.cumsum(steps))).astype(np.int64)
    o = np.concatenate(([price[0]], price[:-1]))
    spread = np.maximum(1, (np.abs(steps) * price)).astype(np.int64)
    return {"ts": np.arange(n, dtype=np.int64) * 60_000_000_000,
            "o": o, "h": np.maximum(o, price) + spread, "l": np.minimum(o, price) - spread,
            "c": price, "v": rng.integers(50, 500, n).astype(np.int64)}


# --------------------------------------------------------------------------- objective + gates


def test_challenger_gates_fingerprint_frozen():
    assert obj.ChallengerGates().fingerprint() == obj.FROZEN_CHALLENGER_GATES_FINGERPRINT


def test_evaluate_challenger_accept_reject_defer():
    strong = {"net_expectancy": 0.002, "profit_factor": 1.5, "sharpe_annualized": 1.4,
              "sortino_annualized": 1.8, "max_drawdown": 0.05}
    acc = obj.evaluate_challenger(strong, effective_sample=500, beats_cash=True,
                                  beats_incumbent=True, beats_simple_baselines=True)
    assert acc["decision"] == "accept"
    weak = {"net_expectancy": -0.001, "profit_factor": 0.6, "sharpe_annualized": -0.2,
            "sortino_annualized": -0.3, "max_drawdown": 0.8}
    rej = obj.evaluate_challenger(weak, effective_sample=500, beats_cash=False,
                                 beats_incumbent=False, beats_simple_baselines=False)
    assert rej["decision"] == "reject"
    deferred = obj.evaluate_challenger(strong, effective_sample=5, beats_cash=True,
                                       beats_incumbent=True, beats_simple_baselines=True)
    assert deferred["decision"] == "defer"


def test_challenger_registry_records_and_summarizes():
    reg = obj.ChallengerRegistry()
    reg.record("a", {"net_expectancy": -0.001}, {"decision": "reject", "objective_J": -0.8,
                                                 "failures": ["x"]}, provenance="t")
    reg.record("b", {"net_expectancy": 0.002}, {"decision": "accept", "objective_J": 0.1,
                                                "failures": []}, provenance="t")
    s = reg.summary()
    assert s["total"] == 2 and s["by_decision"]["accept"] == 1 and s["accepted"] == ["b"]


# --------------------------------------------------------------------------- validation protocol


def test_holdout_integrity_marks_m7_consumed(monkeypatch):
    class _Inv:
        available = True

        def symbols(self, tf):
            return (["AAPL", "MSFT", "NVDA", "SPY", "QQQ"] if tf == "1Min"
                    else ["AAPL", "SPY", "XLK", "SOXL", "TQQQ"])
    monkeypatch.setattr(val, "bar_inventory", lambda: _Inv())
    hi = val.holdout_integrity()
    assert hi["m7_canonical_holdout"]["status"] == "consumed"
    assert hi["m7_canonical_holdout"]["reused_as_new_holdout"] is False
    assert set(hi["fresh_cohort_symbols"]) == {"XLK", "SOXL", "TQQQ"}


def test_validation_protocol_defers_without_accepted(monkeypatch):
    monkeypatch.setattr(val, "confirmatory_fresh_cohort_test",
                        lambda **k: {"status": "completed", "net_expectancy_per_trade": -0.001,
                                     "positive": False})
    monkeypatch.setattr(val, "holdout_integrity", lambda: {"fresh_cohort_size": 10})
    r = val.run_validation_protocol(development_experiment_count=400, any_challenger_accepted=False)
    assert r["independent_certification_status"] == "deferred"


# --------------------------------------------------------------------------- lifecycle


def test_generate_hypotheses_from_findings():
    findings = [{"finding": "x", "evidence": "e", "hypothesis": "h"}]
    hyps = lc.generate_hypotheses(findings)
    assert len(hyps) == 1
    assert "falsifiable_test" in hyps[0]


def test_alpha_cycle_demonstrates_accept_reject_defer():
    def fake_attr(symbols, config=None):
        return {"findings": [{"finding": "f", "evidence": "e", "hypothesis": "h"}],
                "aggregate": {"total_trades": 5000}}

    def fake_challengers(symbols, config=None):
        return [
            {"name": "incumbent_momentum", "metrics": {"net_expectancy": -0.001,
             "profit_factor": 0.6, "sharpe_annualized": -0.2, "sortino_annualized": -0.3,
             "max_drawdown": 0.8}, "effective_sample": 500, "total_return": -0.8,
             "provenance": "t"},
            {"name": "winner", "metrics": {"net_expectancy": 0.002, "profit_factor": 1.5,
             "sharpe_annualized": 1.4, "sortino_annualized": 1.8, "max_drawdown": 0.05},
             "effective_sample": 500, "total_return": 0.3, "provenance": "t"},
            {"name": "deferred_rl", "metrics": {}, "effective_sample": 0, "force_defer": True,
             "provenance": "t"},
        ]

    cyc = lc.run_alpha_cycle(["AAA"], now_ns=1, challengers_fn=fake_challengers,
                             attribution_fn=fake_attr)
    d = cyc["decisions"]
    assert d["accept"] == 1 and d["reject"] == 1 and d["defer"] == 1
    assert "winner" in cyc["accepted"]
    assert len(cyc["hypotheses"]) == 1


# --------------------------------------------------------------------------- dashboard


def test_dashboard_separates_dev_and_certification():
    cyc = {"challenger_registry": {"challengers": []},
           "decisions": {"reject": 6, "defer": 2, "accept": 0},
           "hypotheses": [{"hypothesis": "h"}], "incumbent_net_expectancy": -0.001,
           "data_sufficient": True}
    st = dash.build_status(cyc, validation={"independent_certification_status": "deferred"},
                           memory_size=14)
    assert st["EXPERIMENTS_REJECTED"] == 6
    assert st["INDEPENDENT_CERTIFICATION_STATUS"] == "deferred"
    txt = dash.render(st)
    assert "INDEPENDENT CERTIFICATION" in txt
    assert "development" in txt.lower()


# --------------------------------------------------------------------------- benchmarks


def test_metal_benchmarks_measure_completed():
    r = bm.run_metal_benchmarks()
    assert "benchmarks" in r and len(r["benchmarks"]) == 2
    for b in r["benchmarks"]:
        assert b["cpu_seconds"] > 0
        assert b["faster"] in ("cpu", "gpu")


# --------------------------------------------------------------------------- portfolio/exit (smoke)


def test_portfolio_alpha_cash_is_baseline(monkeypatch):
    bars = _walk()
    monkeypatch.setattr(pa, "_load_1m", lambda cfg, s: bars)
    r = pa.run_portfolio_alpha(["AAA", "BBB"], config=object())
    assert r["available"] is True
    assert "cash_no_trade" in r["portfolios"]
    assert r["no_trade_is_valid_decision"] is True


def test_protective_exit_research_preserves_production(monkeypatch):
    bars = _walk()
    monkeypatch.setattr(per, "_load_1m", lambda cfg, s: bars)
    r = per.run_protective_exit_research(["AAA"], config=object())
    assert r["production_exit_preserved"] is True
    assert r["emergency_3pct_candidate_active"] is False


# --------------------------------------------------------------------------- runtime integration


def test_alpha_discovery_service_run_once(monkeypatch):
    import cengine.alpha.lifecycle as life
    from cengine.runtime.continuous import AlphaDiscoveryService
    monkeypatch.setattr(life, "run_alpha_cycle",
                        lambda symbols, now_ns=None, config=None, memory=None: {
                            "decisions": {"accept": 0, "reject": 2, "defer": 1}, "accepted": []})
    svc = AlphaDiscoveryService(symbols=["AAA"])
    out = svc.run_once(now_ns=1)
    assert out["ok"] is True
    assert out["decisions"]["reject"] == 2
    assert out["cycle"] == 1
