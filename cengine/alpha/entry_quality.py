"""Entry-quality optimization (M8 §5).

The momentum entry contract is treated as a **minimum eligibility gate, not evidence of
profitability** (the failure attribution shows its adverse excursion exceeds its favorable one —
no directional edge). This module asks, rigorously and with discipline against overfitting:
*can any additional entry-qualification filter turn a subset of eligible entries economically
positive out-of-sample?*

Protocol:

* per symbol, split the eligible entries **chronologically** into TRAIN (first 70%) and
  VALIDATION (last 30%); pool trains and pool vals across symbols. (The sealed M7 holdout is
  consumed and is never touched here — this is development research only.)
* **baseline** = validation net expectancy of ALL eligible entries.
* for each candidate filter, choose the keep-direction + threshold that maximizes TRAIN net
  expectancy subject to a minimum kept fraction, then measure its VALIDATION expectancy — honest
  out-of-sample, the threshold is never fit on validation.
* **ablation / combination** = greedy forward selection on TRAIN with a **complexity penalty** per
  added filter, then a single VALIDATION measurement of the chosen combination.
* report every result including the negatives, and flag the **multiple-testing** selection bias
  (N filters scanned ⇒ the best is optimistic); nothing here is an independent certification.

Historical-replay on genuine bars — simulated, not live, never a profitability claim.
"""

from __future__ import annotations

import numpy as np

from cengine.alpha.features import FEATURE_NAMES, build_entry_dataset
from cengine.certification.contracts import DATA_LABEL_HISTORICAL_REPLAY, safe_ticker
from cengine.optimization.exits.experiment_matrix import _entries_and_atr, _load_1m

ENTRY_QUALITY_VERSION = "entry-quality-optimization-1.0.0"
_MIN_KEEP_FRACTION = 0.15      # a filter must retain >=15% of entries to be considered
_MIN_VAL_SAMPLE = 100
_COMPLEXITY_PENALTY = 0.00005  # per added filter, subtracted from train expectancy (net units)
_QUANTILE_GRID = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)


def _split_pool(symbols: list, config=None) -> dict:
    """Build pooled TRAIN/VAL feature matrices with per-symbol chronological splits."""
    if config is None:
        from cengine.lakehouse.config import LakehouseConfig
        config = LakehouseConfig.from_env()
    Xtr = {f: [] for f in FEATURE_NAMES}
    Xva = {f: [] for f in FEATURE_NAMES}
    ytr, yva = [], []
    per_symbol = {}
    for sym in symbols:
        s = safe_ticker(sym)
        bars = _load_1m(config, s)
        if bars is None:
            per_symbol[s] = {"available": False}
            continue
        entries, _atr = _entries_and_atr(bars)
        ds = build_entry_dataset(bars, entries)
        n = len(ds["y"])
        if n < 50:
            per_symbol[s] = {"available": True, "n": n, "note": "too few entries"}
            continue
        cut = int(n * 0.70)
        for f in FEATURE_NAMES:
            Xtr[f].append(ds["X"][f][:cut])
            Xva[f].append(ds["X"][f][cut:])
        ytr.append(ds["y"][:cut])
        yva.append(ds["y"][cut:])
        per_symbol[s] = {"available": True, "n": n, "train": cut, "val": n - cut}
    X_train = {f: np.concatenate(Xtr[f]) if Xtr[f] else np.array([]) for f in FEATURE_NAMES}
    X_val = {f: np.concatenate(Xva[f]) if Xva[f] else np.array([]) for f in FEATURE_NAMES}
    y_train = np.concatenate(ytr) if ytr else np.array([])
    y_val = np.concatenate(yva) if yva else np.array([])
    return {"X_train": X_train, "X_val": X_val, "y_train": y_train, "y_val": y_val,
            "per_symbol": per_symbol}


def _best_rule_for_feature(x: np.ndarray, y: np.ndarray) -> dict | None:
    """Pick keep-direction + threshold maximizing TRAIN expectancy (>= min keep fraction)."""
    finite = np.isfinite(x)
    x, y = x[finite], y[finite]
    if len(y) < _MIN_VAL_SAMPLE:
        return None
    best = None
    for q in _QUANTILE_GRID:
        thr = float(np.quantile(x, q))
        for direction in (">=", "<="):
            mask = x >= thr if direction == ">=" else x <= thr
            keep_frac = mask.mean()
            if keep_frac < _MIN_KEEP_FRACTION or mask.sum() < _MIN_VAL_SAMPLE:
                continue
            exp = float(y[mask].mean())
            if best is None or exp > best["train_expectancy"]:
                best = {"threshold": thr, "direction": direction,
                        "train_expectancy": exp, "train_keep_fraction": round(keep_frac, 4)}
    return best


def _apply_rule(x: np.ndarray, rule: dict) -> np.ndarray:
    return x >= rule["threshold"] if rule["direction"] == ">=" else x <= rule["threshold"]


def run_entry_quality(symbols: list, *, config=None) -> dict:
    pool = _split_pool(symbols, config=config)
    y_tr, y_va = pool["y_train"], pool["y_val"]
    if len(y_va) < _MIN_VAL_SAMPLE:
        return {"version": ENTRY_QUALITY_VERSION, "available": False,
                "note": f"insufficient validation entries ({len(y_va)})",
                "per_symbol": pool["per_symbol"]}

    base_val = float(y_va.mean())
    base_tr = float(y_tr.mean())

    # single-filter ablation (OOS)
    single = {}
    for f in FEATURE_NAMES:
        rule = _best_rule_for_feature(pool["X_train"][f], y_tr)
        if rule is None:
            single[f] = {"status": "insufficient"}
            continue
        xv = pool["X_val"][f]
        maskv = _apply_rule(xv, rule) & np.isfinite(xv)
        if maskv.sum() < _MIN_VAL_SAMPLE:
            single[f] = {"status": "insufficient_val", **rule}
            continue
        val_exp = float(y_va[maskv].mean())
        single[f] = {"status": "ok", **rule, "val_expectancy": round(val_exp, 6),
                     "val_keep_fraction": round(float(maskv.mean()), 4),
                     "val_improvement_vs_base": round(val_exp - base_val, 6),
                     "val_positive": val_exp > 0}

    # greedy forward selection on TRAIN with a complexity penalty, then one VAL measurement
    combo = _greedy_combo(pool, y_tr)
    combo_val = _evaluate_combo(pool, combo["filters"], y_va) if combo["filters"] else None

    improving = [f for f, r in single.items()
                 if r.get("status") == "ok" and r.get("val_positive")]
    best_single = max(
        (r for r in single.values() if r.get("status") == "ok"),
        key=lambda r: r.get("val_expectancy", -1e9), default=None)

    return {
        "version": ENTRY_QUALITY_VERSION, "available": True,
        "data_label": DATA_LABEL_HISTORICAL_REPLAY,
        "n_train": int(len(y_tr)), "n_val": int(len(y_va)),
        "baseline_train_expectancy": round(base_tr, 6),
        "baseline_val_expectancy": round(base_val, 6),
        "single_filter_ablation": single,
        "filters_with_positive_val_expectancy": improving,
        "best_single_filter_val_expectancy":
            round(best_single["val_expectancy"], 6) if best_single else None,
        "combination": {"selected": combo["filters"], "train_expectancy": combo["train_expectancy"],
                        "val": combo_val},
        "multiple_testing_caveat": (
            f"{len(FEATURE_NAMES)} filters x {len(_QUANTILE_GRID)} thresholds x 2 directions were "
            "scanned on TRAIN; the best VAL figure is selection-biased and is development evidence "
            "only — NOT an independent certification."),
        "per_symbol": pool["per_symbol"],
        "conclusion": _conclude(base_val, improving, combo_val),
        "note": "historical-replay on genuine bars; simulated, not live; no profitability claim",
    }


def _greedy_combo(pool: dict, y_tr: np.ndarray) -> dict:
    selected: list = []
    selected_rules: dict = {}
    base = float(y_tr.mean())
    current_exp = base
    mask = np.ones(len(y_tr), dtype=bool)
    remaining = list(FEATURE_NAMES)
    while remaining:
        best_f = None
        best_gain = 0.0
        best_rule = None
        best_mask = None
        for f in remaining:
            x = pool["X_train"][f]
            rule = _best_rule_for_feature(x, y_tr)
            if rule is None:
                continue
            cand_mask = mask & _apply_rule(x, rule) & np.isfinite(x)
            if cand_mask.sum() < _MIN_VAL_SAMPLE:
                continue
            exp = float(y_tr[cand_mask].mean())
            gain = exp - current_exp - _COMPLEXITY_PENALTY
            if gain > best_gain:
                best_gain, best_f, best_rule, best_mask = gain, f, rule, cand_mask
                best_exp = exp
        if best_f is None:
            break
        selected.append(best_f)
        selected_rules[best_f] = best_rule
        mask = best_mask
        current_exp = best_exp
        remaining.remove(best_f)
    return {"filters": selected, "rules": selected_rules,
            "train_expectancy": round(current_exp, 6) if selected else round(base, 6),
            "train_keep_fraction": round(float(mask.mean()), 4)}


def _evaluate_combo(pool: dict, filters: list, y_va: np.ndarray) -> dict:
    if not filters:
        return {"status": "no_filters"}
    combo = _greedy_combo({"X_train": pool["X_train"], "y_train": pool["y_train"]},
                          pool["y_train"])
    rules = combo["rules"]
    mask = np.ones(len(y_va), dtype=bool)
    for f in filters:
        rule = rules.get(f)
        if rule is None:
            continue
        xv = pool["X_val"][f]
        mask &= _apply_rule(xv, rule) & np.isfinite(xv)
    if mask.sum() < _MIN_VAL_SAMPLE:
        return {"status": "insufficient_val", "kept": int(mask.sum())}
    val_exp = float(y_va[mask].mean())
    return {"status": "ok", "val_expectancy": round(val_exp, 6),
            "val_keep_fraction": round(float(mask.mean()), 4), "val_positive": val_exp > 0}


def _conclude(base_val: float, improving: list, combo_val: dict | None) -> str:
    combo_pos = bool(combo_val and combo_val.get("val_positive"))
    if not improving and not combo_pos:
        return (f"No entry filter produced positive out-of-sample net expectancy (baseline "
                f"{base_val:+.5f}); entry-quality filtering does not rescue the momentum entry on "
                "this universe. Negative result preserved.")
    return (f"{len(improving)} filter(s) reached positive VAL net expectancy"
            + (" and the greedy combination did too" if combo_pos else "")
            + "; this is selection-biased development evidence requiring independent "
              "certification on a fresh cohort before any claim.")
