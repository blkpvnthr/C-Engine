"""Base learning-model contract + shared utilities (Milestone 6, §6/§7).

A ``LearningModel`` exposes a uniform fit/predict/uncertainty/serialize interface so the
trainer, registry, inference service and champion/challenger harness treat MLX neural
models and sklearn/numpy baselines identically. Preprocessing (standardization) is FIT ON
TRAINING DATA ONLY — a point-in-time requirement (§5). Metrics helpers are shared so every
model and baseline is scored the same way.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class TrainConfig:
    epochs: int = 60
    batch_size: int = 1024
    learning_rate: float = 1e-3
    hidden: tuple = (64, 32)
    weight_decay: float = 1e-4
    seed: int = 7
    quantiles: tuple = (0.1, 0.5, 0.9)
    patience: int = 8                 # early-stop patience on validation loss
    standardize: bool = True


class Standardizer:
    """Z-score standardizer fit on TRAINING data only (no leakage)."""

    def __init__(self) -> None:
        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "Standardizer":
        self.mean = X.mean(axis=0).astype(np.float32)
        std = X.std(axis=0).astype(np.float32)
        std[std < 1e-8] = 1.0
        self.std = std
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean is None:
            return X.astype(np.float32)
        return ((X - self.mean) / self.std).astype(np.float32)

    def to_dict(self) -> dict:
        return {"mean": None if self.mean is None else self.mean.tolist(),
                "std": None if self.std is None else self.std.tolist()}

    @classmethod
    def from_dict(cls, d: dict) -> "Standardizer":
        s = cls()
        if d.get("mean") is not None:
            s.mean = np.asarray(d["mean"], dtype=np.float32)
            s.std = np.asarray(d["std"], dtype=np.float32)
        return s


@dataclass
class FitResult:
    train_loss: float = 0.0
    val_loss: Optional[float] = None
    epochs_run: int = 0
    device: str = "cpu"
    history: list = field(default_factory=list)


class LearningModel(ABC):
    """Uniform model interface for trainer / registry / inference."""

    framework: str = "numpy"
    model_type: str = "base"
    device: str = "cpu"

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray, *,
            X_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None) -> FitResult: ...

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray: ...

    def predict_with_uncertainty(self, X: np.ndarray) -> tuple:
        """Return (prediction, uncertainty). Default: zero uncertainty."""
        p = self.predict(X)
        return p, np.zeros_like(p)

    @abstractmethod
    def params(self) -> dict:
        """Serializable numpy-array params (for the artifact store)."""

    @abstractmethod
    def load_params(self, params: dict) -> None: ...


# -- shared metrics ----------------------------------------------------------

def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    err = y_pred - y_true
    mse = float(np.mean(err ** 2))
    mae = float(np.mean(np.abs(err)))
    var = float(np.var(y_true))
    r2 = float(1.0 - mse / var) if var > 0 else 0.0
    # directional accuracy: did we get the sign right?
    dir_acc = float(np.mean((y_pred > 0) == (y_true > 0)))
    # rank information coefficient (Spearman-like via rank correlation)
    ic = _rank_corr(y_true, y_pred)
    return {"mse": mse, "rmse": float(np.sqrt(mse)), "mae": mae, "r2": r2,
            "directional_accuracy": dir_acc, "rank_ic": ic, "n": int(len(y_true))}


def classification_metrics(y_true: np.ndarray, p_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=np.float64)
    p = np.clip(np.asarray(p_pred, dtype=np.float64), 1e-7, 1 - 1e-7)
    brier = float(np.mean((p - y_true) ** 2))
    logloss = float(-np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p)))
    pred = (p >= 0.5).astype(np.float64)
    acc = float(np.mean(pred == y_true))
    base_rate = float(np.mean(y_true))
    return {"brier": brier, "log_loss": logloss, "accuracy": acc,
            "base_rate": base_rate, "n": int(len(y_true))}


def _rank_corr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3:
        return 0.0
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    ra -= ra.mean()
    rb -= rb.mean()
    denom = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / denom) if denom > 0 else 0.0
