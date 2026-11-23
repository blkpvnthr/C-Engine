"""Independent non-neural baselines (Milestone 6, §6).

Every learned (MLX GPU) model must beat an appropriate baseline before promotion. These
run on the CPU and are deliberately simple + robust: a gradient-boosted tree (sklearn
HistGradientBoosting — a 'suitable gradient-boosted tree implementation' with no extra
dependency), linear/logistic models, and naive constant / historical-mean predictors.

Serialization note (security): sklearn estimators are serialized with joblib (pickle),
which executes code on load. This is safe here because artifacts are LOCALLY PRODUCED by
this same process and are published + loaded ONLY through the ArtifactStore, which verifies
a sha256 content checksum before any deserialization (a tampered/corrupted blob is
rejected, never unpickled). No artifact is ever loaded from an untrusted/remote source.
MLX weights use npz with allow_pickle=False (see registry/artifacts.py).
"""

from __future__ import annotations

import numpy as np

from cengine.learning.supervised.base import FitResult, LearningModel, Standardizer


class ConstantBaseline(LearningModel):
    """Predicts the training-mean target — the floor every model must beat."""

    framework = "numpy"
    model_type = "constant_baseline"

    def __init__(self) -> None:
        self._mean = 0.0

    def fit(self, X, y, *, X_val=None, y_val=None) -> FitResult:
        self._mean = float(np.mean(y))
        return FitResult(train_loss=float(np.mean((y - self._mean) ** 2)), device="cpu")

    def predict(self, X) -> np.ndarray:
        return np.full(len(X), self._mean, dtype=np.float32)

    def params(self) -> dict:
        return {"mean": np.array([self._mean], dtype=np.float32)}

    def load_params(self, params: dict) -> None:
        self._mean = float(np.asarray(params["mean"]).ravel()[0])


class HistoricalMeanBaseline(LearningModel):
    """Predicts the per-instrument historical mean target (needs a symbol index)."""

    framework = "numpy"
    model_type = "historical_mean_baseline"

    def __init__(self) -> None:
        self._global = 0.0
        self._by_symbol: dict = {}

    def fit(self, X, y, *, symbols=None, X_val=None, y_val=None) -> FitResult:
        self._global = float(np.mean(y))
        if symbols is not None:
            for s in np.unique(symbols):
                self._by_symbol[str(s)] = float(np.mean(y[symbols == s]))
        return FitResult(train_loss=float(np.mean((y - self._global) ** 2)), device="cpu")

    def predict(self, X, symbols=None) -> np.ndarray:
        if symbols is None:
            return np.full(len(X), self._global, dtype=np.float32)
        return np.array([self._by_symbol.get(str(s), self._global) for s in symbols],
                        dtype=np.float32)

    def params(self) -> dict:
        return {"global": np.array([self._global], dtype=np.float32)}

    def load_params(self, params: dict) -> None:
        self._global = float(np.asarray(params["global"]).ravel()[0])


class SklearnRegressor(LearningModel):
    """Gradient-boosted (or ridge) regression baseline."""

    framework = "sklearn"
    model_type = "sklearn_regressor"

    def __init__(self, *, kind: str = "hgbr", standardize: bool = False,
                 random_state: int = 7) -> None:
        self.kind = kind
        self.random_state = random_state
        self._std = Standardizer() if standardize else None
        self._model = None

    def _build(self):
        if self.kind == "ridge":
            from sklearn.linear_model import Ridge
            return Ridge(alpha=1.0, random_state=self.random_state)
        from sklearn.ensemble import HistGradientBoostingRegressor
        return HistGradientBoostingRegressor(
            max_iter=200, learning_rate=0.05, max_depth=4,
            l2_regularization=1.0, random_state=self.random_state)

    def fit(self, X, y, *, X_val=None, y_val=None) -> FitResult:
        if self._std is not None:
            X = self._std.fit(X).transform(X)
        self._model = self._build()
        self._model.fit(X, y)
        pred = self._model.predict(X)
        return FitResult(train_loss=float(np.mean((pred - y) ** 2)), device="cpu")

    def predict(self, X) -> np.ndarray:
        if self._std is not None:
            X = self._std.transform(X)
        return self._model.predict(X).astype(np.float32)

    def params(self) -> dict:
        import io

        import joblib
        buf = io.BytesIO()
        joblib.dump({"model": self._model, "std": self._std.to_dict()
                     if self._std else None, "kind": self.kind}, buf)
        return {"joblib": np.frombuffer(buf.getvalue(), dtype=np.uint8)}

    def load_params(self, params: dict) -> None:
        import io

        import joblib
        d = joblib.load(io.BytesIO(bytes(np.asarray(params["joblib"], dtype=np.uint8))))
        self._model = d["model"]
        self.kind = d.get("kind", self.kind)
        self._std = Standardizer.from_dict(d["std"]) if d.get("std") else None


class SklearnClassifier(LearningModel):
    """Gradient-boosted (or logistic) probability-of-positive baseline."""

    framework = "sklearn"
    model_type = "sklearn_classifier"

    def __init__(self, *, kind: str = "hgbc", standardize: bool = False,
                 random_state: int = 7) -> None:
        self.kind = kind
        self.random_state = random_state
        self._std = Standardizer() if standardize else None
        self._model = None

    def _build(self):
        if self.kind == "logistic":
            from sklearn.linear_model import LogisticRegression
            return LogisticRegression(max_iter=500, C=1.0)
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(
            max_iter=200, learning_rate=0.05, max_depth=4,
            l2_regularization=1.0, random_state=self.random_state)

    def fit(self, X, y, *, X_val=None, y_val=None) -> FitResult:
        if self._std is not None:
            X = self._std.fit(X).transform(X)
        self._model = self._build()
        self._model.fit(X, y.astype(int))
        return FitResult(train_loss=0.0, device="cpu")

    def predict(self, X) -> np.ndarray:
        """Return P(y=1)."""
        if self._std is not None:
            X = self._std.transform(X)
        proba = self._model.predict_proba(X)
        return proba[:, 1].astype(np.float32)

    def params(self) -> dict:
        import io

        import joblib
        buf = io.BytesIO()
        joblib.dump({"model": self._model, "std": self._std.to_dict()
                     if self._std else None, "kind": self.kind}, buf)
        return {"joblib": np.frombuffer(buf.getvalue(), dtype=np.uint8)}

    def load_params(self, params: dict) -> None:
        import io

        import joblib
        d = joblib.load(io.BytesIO(bytes(np.asarray(params["joblib"], dtype=np.uint8))))
        self._model = d["model"]
        self.kind = d.get("kind", self.kind)
        self._std = Standardizer.from_dict(d["std"]) if d.get("std") else None
