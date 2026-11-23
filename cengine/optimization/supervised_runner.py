"""Supervised-model optimization runner for the MetaOptimizer (Milestone 6B, §11/§18).

Autonomously tunes the forward-return model's family/architecture/features/horizon/
hyperparameters over the approved ``supervised_model`` search space, trained + evaluated OUT-OF-
SAMPLE on genuine market bars, and scored ECONOMICALLY (a paper long-if-predicted-positive
net-return series) against the sklearn baseline. Reuses the Milestone 6 models (MLX GPU +
sklearn); a candidate must beat the baseline OOS to be accepted (shadow-only). No profitability
is claimed — forward-return edge is marginal and the result is historical-replay, not live.
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np

from cengine.lakehouse.config import LakehouseConfig
from cengine.learning.evaluation import DATA_LABEL_HISTORICAL_REPLAY
from cengine.learning.features.builder import build_bar_features
from cengine.learning.supervised.base import TrainConfig
from cengine.learning.supervised.baselines import SklearnRegressor
from cengine.optimization.economic import evaluate_candidate


def _paper_returns(pred: np.ndarray, realized: np.ndarray, cost: float = 0.0004) -> list:
    """Long-if-predicted-positive paper net returns on realized forward returns (research)."""
    taken = pred > 0
    return list(realized[taken] - cost) if taken.any() else []


def _fit_predict(model_family: str, cfg: TrainConfig, Xtr, ytr, Xva, yva, Xte):
    """Fit the configured model and return OOS predictions + the device used."""
    if model_family.startswith("mlx"):
        try:
            from cengine.learning.supervised.mlx_models import (
                MLXQuantileRegressor,
                MLXRegressor,
                mlx_is_available,
            )
            if mlx_is_available():
                m = (MLXQuantileRegressor(Xtr.shape[1], config=cfg)
                     if model_family == "mlx_quantile"
                     else MLXRegressor(Xtr.shape[1], config=cfg))
                fr = m.fit(Xtr, ytr, X_val=Xva, y_val=yva)
                return m.predict(Xte), fr.device
        except Exception:  # noqa: BLE001 — fall back to a CPU model
            pass
    kind = "ridge" if model_family == "ridge" else "hgbr"
    m = SklearnRegressor(kind=kind)
    m.fit(Xtr, ytr)
    return m.predict(Xte), "cpu"


def make_supervised_runner(*, config: Optional[LakehouseConfig] = None):
    """Build a MetaOptimizer runner for the 'supervised_model' space on genuine bars."""
    config = config or LakehouseConfig.from_env()
    cache: dict = {}

    def _matrix(horizon: int):
        if horizon not in cache:
            cache[horizon] = build_bar_features(config, horizon_bars=horizon)
        return cache[horizon]

    def runner(space: str, cfg_dict: dict, now_ns: int):
        if space != "supervised_model":
            return None
        horizon = int(cfg_dict.get("horizon_bars", 3))
        m = _matrix(horizon)
        if m is None or m.n < 2000:
            return None
        order = np.argsort(m.timestamps_ns)
        X, y = m.X[order], m.y_return[order]
        n = len(X)
        tr, va = int(n * 0.6), int(n * 0.8)       # chronological; last 20% = OOS
        cfg = TrainConfig(
            epochs=25, batch_size=int(cfg_dict.get("batch_size", 4096)),
            learning_rate=float(cfg_dict.get("learning_rate", 1e-3)),
            hidden=tuple([int(cfg_dict.get("hidden_width", 64))]
                         * int(cfg_dict.get("hidden_depth", 2))),
            weight_decay=float(cfg_dict.get("weight_decay", 1e-4)))
        family = str(cfg_dict.get("model_family", "mlx_quantile"))
        t0 = time.perf_counter()
        try:
            cand_pred, device = _fit_predict(family, cfg, X[:tr], y[:tr],
                                             X[tr:va], y[tr:va], X[va:])
            base_pred, _ = _fit_predict("sklearn_hgbr", cfg, X[:tr], y[:tr],
                                        X[tr:va], y[tr:va], X[va:])
        except Exception:  # noqa: BLE001
            return None
        realized = y[va:]
        cand = evaluate_candidate(_paper_returns(cand_pred, realized), label="candidate",
                                  data_label=DATA_LABEL_HISTORICAL_REPLAY)
        inc = evaluate_candidate(_paper_returns(base_pred, realized), label="incumbent",
                                 data_label=DATA_LABEL_HISTORICAL_REPLAY)
        return {"evaluation": cand.to_dict(),
                "baselines": {"incumbent": inc.to_dict()},
                "data_label": DATA_LABEL_HISTORICAL_REPLAY,
                "runtime_s": round(time.perf_counter() - t0, 3), "device": device,
                "feature_definition": f"bar_features_h{horizon}",
                "reward_definition": "paper_long_if_pred_positive_net"}

    return runner
