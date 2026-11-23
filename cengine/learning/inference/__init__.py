"""Versioned inference contract + GPU/CPU inference service (Milestone 6, §27 / addendum §15)."""

from cengine.learning.inference.contract import (
    INFERENCE_CONTRACT_VERSION,
    Applicability,
    InferenceRequest,
    InferenceResponse,
    abstain,
)
from cengine.learning.inference.loader import ModelLoadError, load_model
from cengine.learning.inference.service import LearningInferenceService

__all__ = [
    "INFERENCE_CONTRACT_VERSION",
    "Applicability",
    "InferenceRequest",
    "InferenceResponse",
    "LearningInferenceService",
    "ModelLoadError",
    "abstain",
    "load_model",
]
