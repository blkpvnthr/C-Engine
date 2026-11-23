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
from cengine.learning.supervised.trainer import TrainingReport, train_forward_return_model

__all__ = [
    "ConstantBaseline",
    "HistoricalMeanBaseline",
    "LearningModel",
    "SklearnClassifier",
    "SklearnRegressor",
    "Standardizer",
    "TrainConfig",
    "TrainingReport",
    "regression_metrics",
    "train_forward_return_model",
]
