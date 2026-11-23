"""Tests for the protective-exit certification (§25 contract + recovery reproduction) and the
§26 entry-contract certification.

The §25 mandatory-exit contract cert runs the REAL production code (fast, deterministic). The
recovery reproduction and entry-contract cert are exercised on synthetic bars so they stay
hermetic.
"""

from __future__ import annotations

import numpy as np

from cengine.certification import backtest as bt
from cengine.certification import protective_exit_cert as pe
from cengine.certification.contracts import CertificationVerdict


def _synthetic_bars(n: int = 2500, seed: int = 11) -> dict:
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


# --------------------------------------------------------------------------- §25 contract


def test_mandatory_exit_contract_certifies_real_code():
    results = pe.certify_mandatory_exit()
    checks = {r.check: r.verdict for r in results}
    assert checks["protective_exit.contract_math"] == CertificationVerdict.CERTIFIED
    assert checks["protective_exit.independence"] == CertificationVerdict.CERTIFIED
    assert checks["protective_exit.emergency_candidate"] == CertificationVerdict.CERTIFIED


def test_emergency_clamp_never_widens_risk():
    import _cengine_native as native
    for proposed, current in [(-0.01, -0.03), (-0.05, -0.03), (0.0, -0.03), (-0.1, -0.05)]:
        out = native.clamp_emergency_threshold(proposed, current)
        # result must never be MORE NEGATIVE (riskier) than current
        assert out >= current - 1e-12


# --------------------------------------------------------------------------- recovery reproduction


def test_block_bootstrap_proportion_brackets_mean():
    rng = np.random.default_rng(0)
    x = (rng.random(500) < 0.9).astype(float)   # ~90% ones
    lo, hi = pe._moving_block_bootstrap_proportion(x)
    assert lo is not None and hi is not None
    assert lo <= x.mean() <= hi
    assert 0.0 <= lo < hi <= 1.0


def test_block_bootstrap_too_short_returns_none():
    assert pe._moving_block_bootstrap_proportion(np.array([1.0, 0.0])) == (None, None)


def test_reproduce_recovery_pools_and_cis(monkeypatch):
    def fake_events(symbol, config=None, **k):
        rng = np.random.default_rng(abs(hash(symbol)) % 1000)
        n = 400
        return {"available": True, "events": n,
                "recov_exit": {h: (rng.random(n) < 0.95).astype(float).tolist()
                               for h in pe._HORIZONS},
                "recov_entry": {h: (rng.random(n) < 0.48).astype(float).tolist()
                                for h in pe._HORIZONS},
                "censored": {h: 0 for h in pe._HORIZONS}, "entry_stride": 60}

    monkeypatch.setattr(pe, "_symbol_recovery_events", fake_events)
    rep = pe.reproduce_recovery(["AAA", "BBB"])
    assert rep.total_exit_events == 800
    r60 = rep.recovery_above_exit["h_60m"]
    assert 0.90 < r60["rate"] < 1.0
    assert r60["ci95_block_bootstrap"][0] is not None
    # above-entry rate is the harder bar and must be materially lower
    assert rep.recovery_above_entry["h_60m"]["rate"] < r60["rate"]


def test_certify_recovery_reproduction_insufficient(monkeypatch):
    monkeypatch.setattr(pe, "_symbol_recovery_events",
                        lambda s, config=None, **k: {"available": False})
    out = pe.certify_recovery_reproduction(["AAA"])
    assert out[0].verdict == CertificationVerdict.INSUFFICIENT_DATA


def test_certify_recovery_reproduction_certifies_high_stable_rate(monkeypatch):
    def fake_events(symbol, config=None, **k):
        n = 600
        return {"available": True, "events": n,
                "recov_exit": {h: [1.0] * int(n * 0.95) + [0.0] * (n - int(n * 0.95))
                               for h in pe._HORIZONS},
                "recov_entry": {h: [1.0] * int(n * 0.48) + [0.0] * (n - int(n * 0.48))
                                for h in pe._HORIZONS},
                "censored": {h: 0 for h in pe._HORIZONS}, "entry_stride": 60}

    monkeypatch.setattr(pe, "_symbol_recovery_events", fake_events)
    out = pe.certify_recovery_reproduction(["AAA"])
    assert out[0].verdict == CertificationVerdict.CERTIFIED
    assert "MEASUREMENT reproduction only" in out[0].notes


# --------------------------------------------------------------------------- §26 entry contract


def test_entry_contract_certifies_deterministic_causal(monkeypatch):
    bars = _synthetic_bars()
    monkeypatch.setattr(bt, "_load_1m", lambda *a, **k: bars)
    out = bt.certify_entry_contract(["TEST"])
    assert out[0].verdict == CertificationVerdict.CERTIFIED
    assert out[0].metrics["deterministic"] is True
    assert out[0].metrics["causal"] is True


def test_entry_contract_insufficient(monkeypatch):
    monkeypatch.setattr(bt, "_load_1m", lambda *a, **k: None)
    out = bt.certify_entry_contract(["TEST"])
    assert out[0].verdict == CertificationVerdict.INSUFFICIENT_DATA
