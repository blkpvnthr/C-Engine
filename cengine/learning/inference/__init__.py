"""Versioned inference contract + GPU/CPU inference service (Milestone 6, §27 / addendum §15)."""

from cengine.learning.inference.contract import (
    INFERENCE_CONTRACT_VERSION,
    Applicability,
    InferenceRequest,
    InferenceResponse,
    abstain,
)

__all__ = [
    "INFERENCE_CONTRACT_VERSION",
    "Applicability",
    "InferenceRequest",
    "InferenceResponse",
    "abstain",
]
