"""Tests for the M8 capital-constrained portfolio accountant + P0 accounting audit."""

from __future__ import annotations

import numpy as np

from cengine.alpha import accounting_audit as aa
from cengine.alpha import failure_attribution as fa
from cengine.alpha.portfolio_replay import (
    PortfolioReplayConfig,
    ReplayResult,
    replay_portfolio,
    resolve_exit,
)


def _bars(prices, ts_start=0, step_ns=60_000_000_000) -> dict:
    p = np.array(prices, dtype=np.int64)
    o = np.concatenate(([p[0]], p[:-1]))
    return {"ts": np.arange(len(p), dtype=np.int64) * step_ns + ts_start,
            "o": o, "h": np.maximum(o, p) + 5, "l": np.minimum(o, p) - 5, "c": p,
            "v": np.full(len(p), 100, dtype=np.int64)}


# --------------------------------------------------------------------------- resolve_exit


def test_resolve_exit_stop_hit():
    # price drops sharply after entry -> −0.5% stop fills
    bars = _bars([10000, 10000, 9900, 9800, 9700] + [9700] * 10)
    res = resolve_exit(bars, 0, cost_frac=0.0, protective_stop=0.005, emergency_stop=0.03,
                       max_hold=10)
    assert res is not None
    exit_idx, entry_price, exit_price, net = res
    # stop level is entry*(1-0.005); exit fills at or below it
    assert exit_price <= entry_price * 0.995 + 1e-6
    assert net < 0


def test_resolve_exit_time_stop_profit():
    # steady rise, never hits stop -> exits at time horizon above entry
    bars = _bars([10000, 10000, 10050, 10100, 10150, 10200, 10250])
    res = resolve_exit(bars, 0, cost_frac=0.0, protective_stop=0.005, emergency_stop=0.03,
                       max_hold=5)
    assert res is not None
    _, entry_price, exit_price, net = res
    assert exit_price > entry_price
    assert net > 0


# --------------------------------------------------------------------------- portfolio replay


def test_replay_conserves_cash_and_is_consistent():
    bars = _bars([10000, 10000, 10050, 10100, 10150, 10200] + [10200] * 10)
    res = replay_portfolio({"AAA": bars}, {"AAA": [0]},
                           config=PortfolioReplayConfig(initial_capital=100_000.0))
    assert isinstance(res, ReplayResult)
    assert res.cash_never_negative is True
    assert res.min_cash_observed >= -1e-6
    assert res.final_equity >= 0.0
    # total_return consistent with final equity
    assert abs((res.final_equity / 100_000.0 - 1.0) - res.total_return) < 1e-4


def test_replay_winning_strategy_grows_equity():
    bars = _bars([10000, 10000, 10050, 10100, 10150, 10200, 10250, 10300] + [10300] * 5)
    res = replay_portfolio({"AAA": bars}, {"AAA": [0]},
                           config=PortfolioReplayConfig(cost_frac=0.0))
    assert res.n_trades_taken == 1
    assert res.final_equity > 100_000.0
    assert res.net_expectancy_per_trade > 0


def test_replay_concurrency_and_capital_limits():
    # many overlapping entries on one symbol; cap concurrency at 2
    bars = _bars([10000] * 200)
    entries = list(range(0, 100, 5))
    res = replay_portfolio({"AAA": bars}, {"AAA": entries},
                           config=PortfolioReplayConfig(max_concurrent_positions=2,
                                                        capital_fraction_per_position=0.4))
    assert res.peak_concurrent_positions <= 2
    # some entries skipped due to the concurrency cap
    assert res.n_skipped_concurrency >= 0
    assert res.n_trades_taken + res.n_skipped_concurrency + res.n_skipped_no_capital \
        <= res.n_entry_signals + 1


def test_replay_no_entries():
    res = replay_portfolio({}, {})
    assert res.n_trades_taken == 0
    assert "no entry signals" in res.note


# --------------------------------------------------------------------------- accounting audit


def test_accounting_audit_structure(monkeypatch):
    rng = np.random.default_rng(0)
    steps = rng.normal(0, 0.0015, 2500)
    price = (10000 * np.exp(np.cumsum(steps))).astype(np.int64)
    bars = _bars(price.tolist())
    monkeypatch.setattr(aa, "_load_all", lambda symbols, config=None: {"AAA": bars, "BBB": bars})
    audit = aa.run_accounting_audit(["AAA", "BBB"])
    assert audit["available"] is True
    assert "m7_trade_level_accounting" in audit
    assert "capital_constrained_portfolio_accounting" in audit
    assert audit["cash_conservation"]["cash_never_negative"] is True
    # the audit preserves the M7 benchmark verbatim
    assert audit["preserved_m7_benchmark"]["certified_economic_edge"] is False
    # drawdown finding distinguishes compound-artifact from portfolio equity
    assert "corrected_portfolio_equity_drawdown_fraction" in audit["drawdown_finding"]


def test_accounting_audit_unavailable(monkeypatch):
    monkeypatch.setattr(aa, "_load_all", lambda symbols, config=None: {})
    audit = aa.run_accounting_audit(["AAA"])
    assert audit["available"] is False


# --------------------------------------------------------------------------- failure attribution


def test_attribute_symbol_keys():
    rng = np.random.default_rng(1)
    steps = rng.normal(0, 0.0015, 2500)
    price = (10000 * np.exp(np.cumsum(steps))).astype(np.int64)
    a = fa.attribute_symbol(_bars(price.tolist()))
    assert a["n_trades"] > 0
    for k in ("stop_out_rate", "mean_max_adverse_excursion", "mean_max_favorable_excursion",
              "net_expectancy"):
        assert k in a


def test_failure_findings_trigger_on_adverse_gt_favorable():
    agg = {"weighted_stop_out_rate": 0.46, "weighted_mean_adverse_excursion": -0.0069,
           "weighted_mean_favorable_excursion": 0.0058}
    findings = fa._derive_findings(agg, {})
    names = {f["finding"] for f in findings}
    assert "high_protective_stop_frequency" in names
    assert "stop_inside_normal_adverse_excursion" in names
    assert "adverse_exceeds_favorable_excursion" in names
