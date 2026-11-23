"""Supervised learning: MLX GPU neural models + independent sklearn/numpy baselines (§6/§7)."""

from cengine.learning.supervised.base import (
    LearningModel,
    Standardizer,
    TrainConfig,
    regression_metrics,
)
from cengine.learning.supervised.baselines import (
    ConstantBaseline,
    HistoricalMeanBaseline,
    SklearnClassifier,
    SklearnRegressor,
)

__all__ = [
    "ConstantBaseline",
    "HistoricalMeanBaseline",
    "LearningModel",
    "SklearnClassifier",
    "SklearnRegressor",
    "Standardizer",
    "TrainConfig",
    "regression_metrics",
]
