"""Reconstruct a verified LearningModel from a registry ModelCard (Milestone 6, §18/§27).

Loads model weights ONLY through the checksum-verifying ArtifactStore (corrupt/incompatible
blobs are rejected, never loaded) and reconstructs the correct model class from the card's
framework + model_type. MLX models are reconstructed with their saved architecture; sklearn/
numpy baselines are rehydrated from their joblib blob. A load failure raises a typed error so
the inference service can fall back safely rather than feed a bad model into decisions.
"""

from __future__ import annotations

import json
from typing import Optional

from cengine.lakehouse.config import LakehouseConfig
from cengine.learning.registry.artifacts import (
    ArtifactRef,
    ArtifactStore,
    ArtifactVerificationError,
)
from cengine.learning.supervised.base import TrainConfig


class ModelLoadError(RuntimeError):
    """Raised when a model cannot be reconstructed/verified from its card."""


def _artifact_ref(card: dict, name: str) -> Optional[ArtifactRef]:
    arts = card.get("artifacts") or {}
    ref = arts.get(name)
    if not ref:
        return None
    return ArtifactRef(path=ref["path"], checksum=ref["checksum"], bytes=ref["bytes"],
                       kind=ref["kind"], framework=ref.get("framework", ""),
                       framework_version=ref.get("framework_version", ""))


def load_model(card: dict, *, config: Optional[LakehouseConfig] = None):
    """Reconstruct a fitted LearningModel from a ModelCard dict. Raises ModelLoadError."""
    config = config or LakehouseConfig.from_env()
    store = ArtifactStore(config)
    framework = card.get("framework", "")
    model_type = card.get("model_type", "")
    ref = _artifact_ref(card, "weights") or _artifact_ref(card, "model")
    if ref is None:
        raise ModelLoadError(f"no weights artifact on card {card.get('model_id')}")
    try:
        params = store.load_npz(ref)
    except ArtifactVerificationError as exc:
        raise ModelLoadError(str(exc)) from exc

    try:
        if framework == "mlx":
            model = _build_mlx(model_type, params)
        elif framework == "sklearn":
            model = _build_sklearn(model_type)
        elif framework == "numpy":
            model = _build_numpy(model_type)
        else:
            raise ModelLoadError(f"unknown framework {framework!r}")
        model.load_params(params)
    except ModelLoadError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise ModelLoadError(f"reconstruct failed ({framework}/{model_type}): {exc}") from exc
    return model


def _build_mlx(model_type: str, params: dict):
    from cengine.learning.supervised.mlx_models import (
        MLXBinaryClassifier,
        MLXQuantileRegressor,
        MLXRegressor,
        mlx_is_available,
    )
    if not mlx_is_available():
        raise ModelLoadError("mlx unavailable for an mlx-framework model")
    meta_raw = params.get("meta::config")
    if meta_raw is None:
        raise ModelLoadError("mlx model missing meta::config")
    meta = json.loads(bytes(meta_raw.astype("uint8")).decode())
    cfg = TrainConfig(hidden=tuple(meta.get("hidden", (64, 32))),
                      quantiles=tuple(meta.get("quantiles", (0.1, 0.5, 0.9))))
    n_features = int(meta["n_features"])
    if model_type == "mlx_regressor":
        return MLXRegressor(n_features, config=cfg)
    if model_type == "mlx_quantile_regressor":
        return MLXQuantileRegressor(n_features, config=cfg)
    if model_type == "mlx_classifier":
        return MLXBinaryClassifier(n_features, config=cfg)
    raise ModelLoadError(f"unknown mlx model_type {model_type!r}")


def _build_sklearn(model_type: str):
    from cengine.learning.supervised.baselines import SklearnClassifier, SklearnRegressor
    if model_type == "sklearn_classifier":
        return SklearnClassifier()
    return SklearnRegressor()


def _build_numpy(model_type: str):
    from cengine.learning.supervised.baselines import ConstantBaseline, HistoricalMeanBaseline
    if model_type == "historical_mean_baseline":
        return HistoricalMeanBaseline()
    return ConstantBaseline()
