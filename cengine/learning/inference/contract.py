"""Versioned inference request/response contract (Milestone 6, §27).

Every inference carries its model/feature-schema versions and both the observation time
and the availability time, so a prediction can be reproduced, point-in-time audited, and
rejected when stale or schema-incompatible. A response is ALWAYS returned — on any
failure the service abstains with a reason and a safe deterministic fallback, never
raising into the trading path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

INFERENCE_CONTRACT_VERSION = "inference-contract-1.0.0"


class Applicability(str, Enum):
    APPLICABLE = "applicable"
    OUT_OF_DISTRIBUTION = "out_of_distribution"
    UNSUPPORTED_INSTRUMENT = "unsupported_instrument"
    UNSUPPORTED_REGIME = "unsupported_regime"
    SCHEMA_INCOMPATIBLE = "schema_incompatible"
    STALE = "stale"
    NO_MODEL = "no_model"
    LOW_CONFIDENCE = "low_confidence"


@dataclass(frozen=True)
class InferenceRequest:
    task: str                               # e.g. 'forward_return_prediction'
    symbol_id: str
    features: tuple                         # ordered float features (schema-aligned)
    feature_schema_version: str
    observation_timestamp_ns: int
    availability_timestamp_ns: int
    model_id: Optional[str] = None          # None -> service picks the champion
    model_version: Optional[str] = None
    strategy_family: Optional[str] = None
    regime_version: Optional[str] = None
    portfolio_version: Optional[str] = None
    prediction_horizon: Optional[str] = None


@dataclass(frozen=True)
class InferenceResponse:
    task: str
    symbol_id: str
    prediction: Optional[float]
    prediction_uncertainty: Optional[float]
    applicability: str
    data_quality: str
    model_maturity: str
    model_id: Optional[str]
    model_version: Optional[str]
    device: str                             # 'gpu' | 'cpu' | 'none'
    inference_timestamp_ns: int
    expiration_timestamp_ns: int
    abstention_reason: Optional[str] = None
    extra: dict = field(default_factory=dict)

    @property
    def abstained(self) -> bool:
        return self.prediction is None or self.abstention_reason is not None

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


def abstain(request: InferenceRequest, *, reason: str,
            applicability: str = Applicability.NO_MODEL.value,
            model_maturity: str = "untrained", now_ns: int = 0,
            model_id: Optional[str] = None,
            model_version: Optional[str] = None) -> InferenceResponse:
    """Build a safe abstention response (no prediction) — the universal fallback."""
    return InferenceResponse(
        task=request.task, symbol_id=request.symbol_id, prediction=None,
        prediction_uncertainty=None, applicability=applicability,
        data_quality="abstained", model_maturity=model_maturity,
        model_id=model_id, model_version=model_version, device="none",
        inference_timestamp_ns=now_ns, expiration_timestamp_ns=now_ns,
        abstention_reason=reason)
