"""Tests for M8 strategy-family (§8), supervised ML (§10), microstructure (§9) and bandit/RL
(§11/§12) research."""

from __future__ import annotations

import numpy as np

from cengine.alpha import bandit_rl_research as brl
from cengine.alpha import family_research as fr
from cengine.alpha import microstructure_research as ms
from cengine.alpha import supervised_filter as sf


def _walk(n=6000, seed=9) -> dict:
    rng = np.random.default_rng(seed)
    steps = rng.normal(0, 0.0015, n)
    price = (10000 * np.exp(np.cumsum(steps))).astype(np.int64)
    o = np.concatenate(([price[0]], price[:-1]))
    spread = np.maximum(1, (np.abs(steps) * price)).astype(np.int64)
    return {"ts": np.arange(n, dtype=np.int64) * 60_000_000_000,
            "o": o, "h": np.maximum(o, price) + spread, "l": np.minimum(o, price) - spread,
            "c": price, "v": rng.integers(50, 500, n).astype(np.int64)}


# --------------------------------------------------------------------------- §8 families


def test_family_generators_return_entries():
    bars = _walk()
    for name, gen in fr.FAMILIES.items():
        entries = gen(bars)
        assert isinstance(entries, list)
        # indices are in-range
        assert all(0 <= e < len(bars["c"]) for e in entries), name


def test_family_research_runs(monkeypatch):
    bars = _walk()
    monkeypatch.setattr(fr, "_load_1m", lambda cfg, s: bars)
    r = fr.run_family_research(["AAA", "BBB"], config=object())
    assert r["available"] is True
    assert "by_family" in r and "conclusion" in r
    # the five families are represented
    fams = {k for k in r["by_family"] if k != "_errors"}
    assert "MOMENTUM_CONTINUATION" in fams
    assert "MEAN_REVERSION" in fams


# --------------------------------------------------------------------------- §10 supervised ML


def test_supervised_filter_rank_ic_helper():
    a = np.arange(100.0)
    assert sf._rank_ic(a, a) > 0.99          # perfectly rank-correlated
    assert sf._rank_ic(a, -a) < -0.99


def test_supervised_filter_runs_and_scores_economically(monkeypatch):
    bars = _walk()
    monkeypatch.setattr(sf, "_load_1m", lambda cfg, s: bars)
    r = sf.run_supervised_filter(["AAA", "BBB", "CCC"], config=object())
    assert "conclusion" in r
    if r.get("available"):
        assert "baseline_val_expectancy" in r
        assert "models" in r
        # at least one model produced an economic (top-decile) evaluation
        assert any("selected_val_expectancy" in m for m in r["models"].values())


# --------------------------------------------------------------------------- §9 microstructure


def test_microstructure_reports_insufficient_l1(monkeypatch):
    bars = _walk()
    monkeypatch.setattr(ms, "_load_1m", lambda cfg, s: bars)

    class _Cfg:
        def dataset_path(self, layer, ds):
            return "/nonexistent/path/" + ds
    r = ms.run_microstructure_research(["AAA", "BBB"], config=_Cfg())
    assert r["genuine_l1_tick_data_available"] is False
    assert r["genuine_l1_features_status"] == "INSUFFICIENT_DATA"
    assert "quote_imbalance" in r["genuine_l1_features_deferred"]


# --------------------------------------------------------------------------- §11/§12 bandit + RL


def test_contextual_bandit_refuses_ips(monkeypatch):
    bars = _walk()
    monkeypatch.setattr(brl, "_load_1m", lambda cfg, s: bars)
    r = brl.run_contextual_bandit_research(["AAA", "BBB"], config=object())
    assert r["ips_or_dr_possible"] is False
    assert r["certification_status"] == "INSUFFICIENT_DATA"


def test_offline_rl_insufficient(monkeypatch):
    bars = _walk()
    monkeypatch.setattr(brl, "_load_1m", lambda cfg, s: bars)
    r = brl.run_offline_rl_research(["AAA"], config=object())
    assert r["genuine_execution_transitions"] == 0
    assert r["certification_status"] == "INSUFFICIENT_DATA"
    assert r["sequential_transition_data_available"] is False
