"""Regime-conditioned alpha research + abstention policy (M8 §7).

Buckets eligible entries by market regime and measures per-regime net expectancy, then derives an
explicit **abstention policy**: trade only in regimes with a positive TRAIN expectancy and a
sufficient sample, abstain elsewhere. The chronological TRAIN/VALIDATION split keeps the policy
honest — the regimes to trade are chosen on TRAIN and the policy is measured once on VALIDATION.

Regime is a lightweight proxy aligned with ``MarketRegimeEngine``'s dimensions — **direction**
(trend persistence vs the 100-bar mean) x **volatility** (ATR-fraction terciles) — computed from
the look-ahead-free feature set. The full streaming ``MarketRegimeEngine.assess_instrument`` is the
production authority; this batch proxy is used for the historical study and is documented as such
(it is not a new regime engine). Historical-replay — simulated, not live, no profitability claim.
"""

from __future__ import annotations

import numpy as np

from cengine.alpha.features import build_entry_dataset, compute_features
from cengine.certification.contracts import DATA_LABEL_HISTORICAL_REPLAY, safe_ticker
from cengine.optimization.exits.experiment_matrix import _entries_and_atr, _load_1m

REGIME_RESEARCH_VERSION = "regime-conditioned-research-1.0.0"
_MIN_REGIME_SAMPLE = 100


def _regime_label(trend_persist: float, atr_frac: float, vol_lo: float, vol_hi: float) -> str:
    if not np.isfinite(trend_persist) or not np.isfinite(atr_frac):
        return "unknown"
    direction = "up" if trend_persist > 0.002 else ("down" if trend_persist < -0.002 else "range")
    vol = "lowvol" if atr_frac <= vol_lo else ("highvol" if atr_frac >= vol_hi else "midvol")
    return f"{direction}_{vol}"


def run_regime_research(symbols: list, *, config=None) -> dict:
    if config is None:
        from cengine.lakehouse.config import LakehouseConfig
        config = LakehouseConfig.from_env()
    # pool entries with regime labels + realized labels, per-symbol chronological split
    rows = []  # (ts, regime, y, split)  split in {"train","val"}
    atr_all = []
    per_symbol = {}
    for sym in symbols:
        s = safe_ticker(sym)
        bars = _load_1m(config, s)
        if bars is None:
            per_symbol[s] = {"available": False}
            continue
        entries = _entries_and_atr(bars)[0]
        ds = build_entry_dataset(bars, entries)
        feats = compute_features(bars)
        atr_all.append(feats["atr_frac"][np.isfinite(feats["atr_frac"])])
        n = len(ds["y"])
        per_symbol[s] = {"available": True, "n": n}
        if n < 50:
            continue
        cut = int(n * 0.70)
        tp = ds["X"]["trend_persist"]
        af = feats["atr_frac"][np.array(ds["entry_idx"], dtype=np.int64)]
        for j in range(n):
            rows.append([int(ds["ts"][j]), tp[j], af[j], ds["y"][j],
                         "train" if j < cut else "val"])
    if not rows:
        return {"version": REGIME_RESEARCH_VERSION, "available": False, "note": "no entries",
                "per_symbol": per_symbol}

    atr_cat = np.concatenate(atr_all) if atr_all else np.array([0.0])
    vol_lo = float(np.quantile(atr_cat, 0.33))
    vol_hi = float(np.quantile(atr_cat, 0.66))

    # assign regimes
    train_reg: dict = {}
    val_reg: dict = {}
    for _ts, tp, af, y, split in rows:
        reg = _regime_label(tp, af, vol_lo, vol_hi)
        bucket = train_reg if split == "train" else val_reg
        bucket.setdefault(reg, []).append(y)

    per_regime = {}
    tradeable = []
    for reg in sorted(set(train_reg) | set(val_reg)):
        tr = np.array(train_reg.get(reg, []), dtype=np.float64)
        va = np.array(val_reg.get(reg, []), dtype=np.float64)
        tr_exp = float(tr.mean()) if len(tr) else None
        va_exp = float(va.mean()) if len(va) else None
        trade_this = (tr_exp is not None and tr_exp > 0 and len(tr) >= _MIN_REGIME_SAMPLE)
        if trade_this:
            tradeable.append(reg)
        per_regime[reg] = {
            "train_n": int(len(tr)), "val_n": int(len(va)),
            "train_expectancy": round(tr_exp, 6) if tr_exp is not None else None,
            "val_expectancy": round(va_exp, 6) if va_exp is not None else None,
            "train_tradeable": trade_this,
        }

    # abstention policy: trade only tradeable regimes; evaluate on VAL
    all_val = np.array([r[3] for r in rows if r[4] == "val"], dtype=np.float64)
    policy_val = np.array(
        [r[3] for r in rows if r[4] == "val"
         and _regime_label(r[1], r[2], vol_lo, vol_hi) in tradeable], dtype=np.float64)
    baseline_val_exp = float(all_val.mean()) if len(all_val) else None
    policy_val_exp = float(policy_val.mean()) if len(policy_val) else None
    policy_positive = policy_val_exp is not None and policy_val_exp > 0

    return {
        "version": REGIME_RESEARCH_VERSION, "available": True,
        "data_label": DATA_LABEL_HISTORICAL_REPLAY,
        "regime_proxy": "direction(trend_persist) x volatility(atr_frac terciles); "
                        "MarketRegimeEngine is the production authority (streaming)",
        "volatility_terciles": {"low<=": round(vol_lo, 5), "high>=": round(vol_hi, 5)},
        "per_regime": per_regime,
        "tradeable_regimes_by_train": tradeable,
        "baseline_val_expectancy":
            round(baseline_val_exp, 6) if baseline_val_exp is not None else None,
        "abstention_policy_val_expectancy":
            round(policy_val_exp, 6) if policy_val_exp is not None else None,
        "abstention_policy_val_sample": int(len(policy_val)),
        "abstention_policy_positive": policy_positive,
        "per_symbol": per_symbol,
        "conclusion": _conclude(baseline_val_exp, policy_val_exp, tradeable, policy_positive),
        "note": "regime proxy on genuine bars; abstention chosen on TRAIN, measured once on VAL; "
                "simulated, not live, no profitability claim",
    }


def _conclude(base, policy, tradeable, positive) -> str:
    if not tradeable:
        return ("No regime had positive TRAIN expectancy with sufficient sample — the abstention "
                "policy abstains everywhere (NO_TRADE). Negative result preserved.")
    if positive:
        return (f"A regime-abstention policy (trade only {tradeable}) reached positive VALIDATION "
                f"expectancy {policy:+.5f} vs baseline {base:+.5f} — development evidence only, "
                "requires independent certification on a fresh cohort.")
    return (f"Regime abstention (trade only {tradeable}) improves on baseline {base:+.5f} but "
            f"stays non-positive on VALIDATION ({policy:+.5f}); regime conditioning does not "
            "rescue the entry. Negative result preserved.")
