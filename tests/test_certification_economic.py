"""Tests for the M7 economic-certification spine: frozen gates, sealed holdout, baselines, harness.

Uses a deterministic synthetic 1-minute bar series so the real replay machinery
(_entries_and_atr / _simulate / evaluate_candidate) runs end-to-end without touching the
lakehouse. The frozen-gate anti-retrofit mechanism is tested directly.
"""

from __future__ import annotations

import numpy as np

from cengine.certification import backtest as bt
from cengine.certification import baselines as bl
from cengine.certification.contracts import CertificationVerdict
from cengine.certification.gates import (
    FROZEN_GATES_FINGERPRINT,
    FrozenEconomicGates,
    default_gates,
    evaluate_gates,
    verify_frozen,
)
from cengine.certification.holdout import HoldoutManifest, SymbolHoldout
from cengine.optimization.exits.experiment_matrix import _entries_and_atr


def _synthetic_bars(n: int = 2500, seed: int = 7) -> dict:
    """Deterministic geometric random walk on a 1-minute grid, in ticks (price_scale=10000)."""
    rng = np.random.default_rng(seed)
    steps = rng.normal(0.0, 0.0015, size=n)
    price = 100.0 * np.exp(np.cumsum(steps))
    c = (price * 10000).astype(np.int64)
    o = np.concatenate(([c[0]], c[:-1]))
    spread = np.maximum(1, (np.abs(steps) * price * 10000)).astype(np.int64)
    h = np.maximum(o, c) + spread
    low = np.minimum(o, c) - spread
    ts = (np.arange(n, dtype=np.int64)) * (60 * 1_000_000_000)
    v = np.full(n, 1000, dtype=np.int64)
    return {"ts": ts, "o": o.astype(np.int64), "h": h.astype(np.int64),
            "l": np.maximum(low, 1).astype(np.int64), "c": c, "v": v}


# --------------------------------------------------------------------------- frozen gates


def test_frozen_gates_fingerprint_matches():
    assert verify_frozen() == FROZEN_GATES_FINGERPRINT
    assert default_gates().fingerprint() == FROZEN_GATES_FINGERPRINT


def test_changing_a_gate_changes_fingerprint_and_trips_verify():
    tighter = FrozenEconomicGates(min_profit_factor=1.5)
    assert tighter.fingerprint() != FROZEN_GATES_FINGERPRINT
    # verify_frozen only enforces the DEFAULT; an explicit custom gates object is allowed
    assert verify_frozen(tighter) == tighter.fingerprint()


def test_evaluate_gates_distinguishes_insufficient_fail_pass():
    good = {"net_expectancy": 0.002, "profit_factor": 1.4, "sharpe": 0.3,
            "win_rate": 0.5, "max_drawdown": 0.1}
    # insufficient sample
    r = evaluate_gates(good, n_trades=5, effective_sample=3.0,
                       beats_buy_and_hold=True, beats_random_placebo=True)
    assert r["insufficient"] and not r["passed"]
    # genuine pass
    r = evaluate_gates(good, n_trades=100, effective_sample=80.0,
                       beats_buy_and_hold=True, beats_random_placebo=True)
    assert r["passed"] and not r["failures"]
    # a losing strategy fails on multiple gates
    bad = {"net_expectancy": -0.001, "profit_factor": 0.6, "sharpe": -0.2,
           "win_rate": 0.34, "max_drawdown": 0.9}
    r = evaluate_gates(bad, n_trades=3000, effective_sample=2000.0,
                       beats_buy_and_hold=False, beats_random_placebo=False)
    assert not r["passed"] and not r["insufficient"]
    assert any("expectancy_not_positive" in f for f in r["failures"])
    assert "does_not_beat_buy_and_hold" in r["failures"]


# --------------------------------------------------------------------------- sealed holdout


def test_holdout_seal_logs_access_and_audits(monkeypatch):
    from cengine.certification import holdout as ho

    def fake_derive(symbol, config=None):
        return SymbolHoldout(symbol=symbol.upper(), train=(0, 60), validation=(63, 80),
                             holdout=(83, 100), embargo_days=3, fingerprint="fp-" + symbol.upper())

    monkeypatch.setattr(ho, "derive_symbol_holdout", fake_derive)
    seal = ho.HoldoutSeal()
    win = seal.open("aapl", reason="scoring")
    assert win == (83, 100)
    seal.open("msft", reason="scoring")
    audit = seal.audit()
    assert audit["granted"] == 2 and audit["symbols_touched"] == 2
    assert audit["symbols_touched_more_than_once"] == {}
    # a second touch of the same symbol is visible in the audit (would flag tuning-on-holdout)
    seal.open("aapl", reason="scoring")
    assert seal.audit()["symbols_touched_more_than_once"] == {"AAPL": 2}


def test_holdout_manifest_fingerprint_is_deterministic():
    hs = [SymbolHoldout("AAPL", (0, 6), (7, 8), (9, 10), 3, "a"),
          SymbolHoldout("MSFT", (0, 6), (7, 8), (9, 10), 3, "b")]
    m1 = HoldoutManifest(symbols=hs)
    m2 = HoldoutManifest(symbols=list(hs))
    assert m1.fingerprint() == m2.fingerprint()
    assert m1.get("aapl").symbol == "AAPL"


# --------------------------------------------------------------------------- baselines + harness


def test_baselines_run_on_synthetic_bars():
    bars = _synthetic_bars()
    entries, atr = _entries_and_atr(bars)
    assert len(entries) >= 30
    evals = bl.evaluate_baselines(bars, entries, atr, cost_frac=0.0004, symbol="TEST")
    # the hierarchy must include the isolating + placebo + flat baselines
    assert "hold_to_horizon" in evals
    assert "random_placebo" in evals
    assert "always_flat" in evals
    # always_flat has zero net return by construction
    assert abs(evals["always_flat"].metrics.get("net_return", 0.0)) < 1e-9


def test_run_window_backtest_end_to_end(monkeypatch):
    bars = _synthetic_bars()
    monkeypatch.setattr(bt, "_load_1m", lambda *a, **k: bars)
    res = bt.run_window_backtest("TEST", "holdout", 0, 10, cost_frac=0.0004)
    assert res.n_entries >= 30
    assert res.incumbent is not None
    assert "gate_fingerprint" in res.gate_result
    # a verdict is derivable and is one of the honest outcomes
    v = bt._verdict_from(res)
    assert v in (CertificationVerdict.CERTIFIED, CertificationVerdict.NOT_CERTIFIED,
                 CertificationVerdict.INSUFFICIENT_DATA)


def test_run_window_backtest_insufficient_data(monkeypatch):
    monkeypatch.setattr(bt, "_load_1m", lambda *a, **k: None)
    res = bt.run_window_backtest("TEST", "holdout", 0, 10)
    assert res.n_entries == 0
    assert bt._verdict_from(res) == CertificationVerdict.INSUFFICIENT_DATA


def test_certify_strategy_economics_rollup(monkeypatch):
    bars = _synthetic_bars()
    monkeypatch.setattr(bt, "_load_1m", lambda *a, **k: bars)
    # make every symbol's holdout resolvable without the lakehouse
    monkeypatch.setattr(
        bt.HoldoutSeal, "open",
        lambda self, symbol, reason: (0, 10),
    )
    res = bt.certify_strategy_economics(["TEST1", "TEST2"])
    # one result per symbol + a portfolio roll-up
    assert len(res) == 3
    assert res[-1].check == "strategy_economics.portfolio"
    assert res[-1].verdict in (CertificationVerdict.CERTIFIED,
                               CertificationVerdict.NOT_CERTIFIED,
                               CertificationVerdict.INSUFFICIENT_DATA)
