"""Tests for M8 entry-quality (§5), multi-timeframe (§6) and regime-conditioned (§7) research."""

from __future__ import annotations

import numpy as np

from cengine.alpha import entry_quality as eq
from cengine.alpha import features as fx
from cengine.alpha import multi_timeframe as mt
from cengine.alpha import regime_research as rr


def _walk(n=6000, seed=5) -> dict:
    rng = np.random.default_rng(seed)
    steps = rng.normal(0, 0.0015, n)
    price = (10000 * np.exp(np.cumsum(steps))).astype(np.int64)
    o = np.concatenate(([price[0]], price[:-1]))
    spread = np.maximum(1, (np.abs(steps) * price)).astype(np.int64)
    return {"ts": np.arange(n, dtype=np.int64) * 60_000_000_000,
            "o": o, "h": np.maximum(o, price) + spread, "l": np.minimum(o, price) - spread,
            "c": price, "v": rng.integers(50, 500, n).astype(np.int64)}


# --------------------------------------------------------------------------- features


def test_features_have_all_names_and_are_finite_after_warmup():
    f = fx.compute_features(_walk())
    for name in fx.FEATURE_NAMES:
        assert name in f
        assert np.isfinite(f[name][500:]).any()


def test_features_are_look_ahead_free():
    bars = _walk()
    f_full = fx.compute_features(bars)
    # perturb a FUTURE bar; features at an earlier index must be unchanged
    bars2 = {k: v.copy() for k, v in bars.items()}
    i = 2000
    bars2["c"][i + 500] = int(bars2["c"][i + 500] * 1.5)
    bars2["h"][i + 500] = int(bars2["h"][i + 500] * 1.5)
    f_pert = fx.compute_features(bars2)
    for name in ("mom5", "disp_sma20", "rel_volume", "htf_trend_60", "trend_persist"):
        a, b = f_full[name][i], f_pert[name][i]
        assert (np.isnan(a) and np.isnan(b)) or a == b, name


def test_build_entry_dataset_shapes():
    bars = _walk()
    from cengine.optimization.exits.experiment_matrix import _entries_and_atr
    entries, _ = _entries_and_atr(bars)
    ds = fx.build_entry_dataset(bars, entries)
    assert len(ds["y"]) == len(ds["entry_idx"])
    for name in fx.FEATURE_NAMES:
        assert len(ds["X"][name]) == len(ds["y"])


# --------------------------------------------------------------------------- entry quality


def test_best_rule_for_feature_picks_threshold():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 1000)
    # y positively correlated with x -> keeping high x should raise mean
    y = x * 0.001 + rng.normal(0, 0.001, 1000)
    rule = eq._best_rule_for_feature(x, y)
    assert rule is not None
    assert rule["direction"] in (">=", "<=")
    assert rule["train_keep_fraction"] >= eq._MIN_KEEP_FRACTION


def test_entry_quality_runs_and_concludes(monkeypatch):
    bars = _walk()
    monkeypatch.setattr(eq, "_load_1m", lambda cfg, s: bars)
    r = eq.run_entry_quality(["AAA", "BBB", "CCC"], config=object())
    assert "conclusion" in r
    if r.get("available"):
        assert "baseline_val_expectancy" in r
        assert "single_filter_ablation" in r
        assert "multiple_testing_caveat" in r


# --------------------------------------------------------------------------- multi-timeframe


def test_gated_entries_filters():
    bars = _walk()
    entries = list(range(200, 400, 10))
    gated = mt._gated_entries(bars, entries, mt._GATES["htf60_up"])
    assert len(gated) <= len(entries)
    assert mt._gated_entries(bars, entries, None) == entries


def test_multi_timeframe_runs(monkeypatch):
    bars = _walk()
    monkeypatch.setattr(mt, "_load_1m", lambda cfg, s: bars)
    r = mt.run_multi_timeframe(["AAA", "BBB"], config=object())
    assert r["available"] is True
    assert "baseline_all_entries" in r["by_gate"]
    assert "conclusion" in r


# --------------------------------------------------------------------------- regime


def test_regime_label_buckets():
    assert rr._regime_label(0.02, 0.001, 0.0005, 0.002) == "up_midvol"
    assert rr._regime_label(-0.02, 0.003, 0.0005, 0.002) == "down_highvol"
    assert rr._regime_label(0.0, 0.0001, 0.0005, 0.002) == "range_lowvol"
    assert rr._regime_label(float("nan"), 0.001, 0.0005, 0.002) == "unknown"


def test_regime_research_runs(monkeypatch):
    bars = _walk()
    monkeypatch.setattr(rr, "_load_1m", lambda cfg, s: bars)
    r = rr.run_regime_research(["AAA", "BBB"], config=object())
    assert "conclusion" in r
    if r.get("available"):
        assert "per_regime" in r
        assert "abstention_policy_val_expectancy" in r
