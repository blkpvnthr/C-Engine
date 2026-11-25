"""Supervised ML opportunity filtering (M8 §10).

Trains models to distinguish economically attractive entries from technically-eligible-but-
unattractive ones, using MLX/Metal as the primary backend and sklearn tree/linear/constant
baselines. The label is the realized net return of each eligible momentum entry under the −0.5%
protective exit (from the shared replay resolver), so the model is scored on **economics**, not
classification accuracy.

Protocol (overfitting-disciplined): per-symbol chronological TRAIN/VALIDATION split, pooled;
features standardized and NaN-imputed with TRAIN medians (no leakage). Each model predicts the net
return on VALIDATION; we then **select the top-decile predicted entries** and measure that subset's
actual mean net return. The economic test is: *does model selection produce positive out-of-sample
net expectancy, and does it beat the baselines economically?* Rank-IC is reported for context but
is never the acceptance criterion. The sealed M7 holdout is never touched. Historical-replay —
simulated, not live, no profitability claim.
"""

from __future__ import annotations

import numpy as np

from cengine.alpha.features import FEATURE_NAMES, build_entry_dataset
from cengine.certification.contracts import DATA_LABEL_HISTORICAL_REPLAY, safe_ticker
from cengine.optimization.exits.experiment_matrix import _entries_and_atr, _load_1m

SUPERVISED_FILTER_VERSION = "supervised-opportunity-filter-1.0.0"
_MIN_VAL = 200
_SELECT_TOP_FRACTION = 0.10


def _assemble(symbols: list, config=None) -> dict:
    if config is None:
        from cengine.lakehouse.config import LakehouseConfig
        config = LakehouseConfig.from_env()
    Xtr, Xva, ytr, yva = [], [], [], []
    per_symbol = {}
    for sym in symbols:
        s = safe_ticker(sym)
        bars = _load_1m(config, s)
        if bars is None:
            per_symbol[s] = {"available": False}
            continue
        entries = _entries_and_atr(bars)[0]
        ds = build_entry_dataset(bars, entries)
        n = len(ds["y"])
        if n < 100:
            per_symbol[s] = {"available": True, "n": n, "note": "too few"}
            continue
        feat = np.column_stack([ds["X"][f] for f in FEATURE_NAMES])
        cut = int(n * 0.70)
        Xtr.append(feat[:cut])
        ytr.append(ds["y"][:cut])
        Xva.append(feat[cut:])
        yva.append(ds["y"][cut:])
        per_symbol[s] = {"available": True, "n": n, "train": cut, "val": n - cut}
    if not Xtr:
        return {"ok": False, "per_symbol": per_symbol}
    X_train = np.vstack(Xtr)
    X_val = np.vstack(Xva)
    y_train = np.concatenate(ytr)
    y_val = np.concatenate(yva)
    # impute NaN with TRAIN medians (no leakage)
    med = np.nanmedian(X_train, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    X_train = np.where(np.isfinite(X_train), X_train, med)
    X_val = np.where(np.isfinite(X_val), X_val, med)
    return {"ok": True, "X_train": X_train.astype(np.float32), "X_val": X_val.astype(np.float32),
            "y_train": y_train.astype(np.float32), "y_val": y_val.astype(np.float32),
            "per_symbol": per_symbol}


def _economic_eval(pred_val: np.ndarray, y_val: np.ndarray) -> dict:
    """Select the top-decile predicted entries; report the subset's actual mean net return."""
    n = len(y_val)
    k = max(1, int(n * _SELECT_TOP_FRACTION))
    order = np.argsort(-pred_val)
    top = order[:k]
    sel_exp = float(y_val[top].mean())
    # rank-IC (Spearman via rank correlation)
    ic = _rank_ic(pred_val, y_val)
    return {"selected_n": int(k), "selected_val_expectancy": round(sel_exp, 6),
            "selected_positive": sel_exp > 0, "rank_ic": round(ic, 4)}


def _rank_ic(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3:
        return 0.0
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    ra -= ra.mean()
    rb -= rb.mean()
    denom = np.sqrt((ra * ra).sum() * (rb * rb).sum())
    return float((ra * rb).sum() / denom) if denom > 0 else 0.0


def run_supervised_filter(symbols: list, *, config=None) -> dict:
    data = _assemble(symbols, config=config)
    if not data.get("ok") or len(data["y_val"]) < _MIN_VAL:
        return {"version": SUPERVISED_FILTER_VERSION, "available": False,
                "note": "insufficient labeled data", "per_symbol": data.get("per_symbol", {})}
    Xtr, Xva = data["X_train"], data["X_val"]
    ytr, yva = data["y_train"], data["y_val"]
    base_val_exp = float(yva.mean())

    models = {}
    # primary: MLX on Metal
    try:
        from cengine.learning.supervised.base import TrainConfig
        from cengine.learning.supervised.mlx_models import MLXRegressor
        cfg = TrainConfig(epochs=40, hidden=(32, 16), patience=6)
        mlx = MLXRegressor(Xtr.shape[1], config=cfg)
        fit = mlx.fit(Xtr, ytr, X_val=Xva, y_val=yva)
        models["mlx_regressor"] = {"pred": mlx.predict(Xva),
                                   "device": getattr(fit, "device", "?")}
    except Exception as exc:  # noqa: BLE001
        models["mlx_regressor"] = {"error": str(exc)}
    # baselines
    try:
        from cengine.learning.supervised.baselines import (
            ConstantBaseline,
            SklearnRegressor,
        )
        hgbr = SklearnRegressor(kind="hgbr")
        hgbr.fit(Xtr, ytr)
        models["sklearn_hgbr_tree"] = {"pred": hgbr.predict(Xva)}
        ridge = SklearnRegressor(kind="ridge", standardize=True)
        ridge.fit(Xtr, ytr)
        models["linear_ridge"] = {"pred": ridge.predict(Xva)}
        const = ConstantBaseline()
        const.fit(Xtr, ytr)
        models["constant_baseline"] = {"pred": const.predict(Xva)}
    except Exception as exc:  # noqa: BLE001
        models["baselines_error"] = str(exc)

    evals = {}
    for name, m in models.items():
        if "pred" not in m:
            evals[name] = {k: v for k, v in m.items() if k != "pred"}
            continue
        ev = _economic_eval(np.asarray(m["pred"], dtype=np.float64), yva)
        ev["improves_vs_base"] = ev["selected_val_expectancy"] - base_val_exp
        if "device" in m:
            ev["device"] = m["device"]
        evals[name] = ev

    economically_useful = [n for n, e in evals.items()
                           if e.get("selected_positive")]
    best = max((n for n in evals if "selected_val_expectancy" in evals[n]),
               key=lambda n: evals[n]["selected_val_expectancy"], default=None)
    return {
        "version": SUPERVISED_FILTER_VERSION, "available": True,
        "data_label": DATA_LABEL_HISTORICAL_REPLAY,
        "n_train": int(len(ytr)), "n_val": int(len(yva)),
        "baseline_val_expectancy": round(base_val_exp, 6),
        "select_top_fraction": _SELECT_TOP_FRACTION,
        "models": evals, "models_with_positive_selection": economically_useful,
        "best_model": best,
        "conclusion": _conclude(base_val_exp, evals, economically_useful, best),
        "per_symbol": data["per_symbol"],
        "note": ("MLX/Metal primary + sklearn tree/linear/constant baselines; scored on ECONOMICS "
                 "(top-decile OOS net expectancy), not accuracy/AUC; simulated historical-replay, "
                 "not live, no profitability claim"),
    }


def _conclude(base, evals, useful, best) -> str:
    if not useful:
        be = evals.get(best, {}).get("selected_val_expectancy") if best else None
        return (f"No model's top-decile selection produced positive out-of-sample net expectancy "
                f"(baseline {base:+.5f}; best '{best}' {be:+.5f} if measurable). Supervised "
                "opportunity filtering does not create an edge the entries do not contain. "
                "Negative result preserved.")
    return (f"Model(s) {useful} produced a positive top-decile OOS net expectancy; this is "
            "selection-biased development evidence requiring independent certification before any "
            "economic claim.")
