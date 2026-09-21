"""Advisory next-session tide classifier with explicit train/inference separation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class TidePrediction:
    direction: int
    confidence: float | None
    model: str


class NextSessionTideModel:
    def __init__(self, algorithm: str, **parameters: object) -> None:
        if algorithm not in {"knn", "svm"}:
            raise ValueError("algorithm must be 'knn' or 'svm'")
        try:
            from sklearn.neighbors import KNeighborsClassifier
            from sklearn.svm import SVC
        except ImportError as exc:
            raise RuntimeError("install cengine[research] for predictive models") from exc
        self.algorithm = algorithm
        self.model = (
            KNeighborsClassifier(**parameters)
            if algorithm == "knn"
            else SVC(probability=True, **parameters)
        )
        self._fitted = False

    def fit(self, features: np.ndarray, next_session_direction: np.ndarray) -> None:
        if features.ndim != 2 or next_session_direction.ndim != 1:
            raise ValueError("features must be 2D and labels 1D")
        if len(features) != len(next_session_direction):
            raise ValueError("feature/label length mismatch")
        self.model.fit(features, next_session_direction)
        self._fitted = True

    def predict(self, features: np.ndarray) -> TidePrediction:
        if not self._fitted:
            raise RuntimeError("model must be fitted before inference")
        direction = int(self.model.predict(features.reshape(1, -1))[0])
        confidence = None
        if hasattr(self.model, "predict_proba"):
            confidence = float(np.max(self.model.predict_proba(features.reshape(1, -1))[0]))
        return TidePrediction(direction, confidence, self.algorithm)
