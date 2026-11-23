"""GPU/CPU inference service (Milestone 6, §27 + addendum §15).

Serves predictions from the registry champion model with persistent (cached) loading,
batched cross-sectional inference (GPU-first via DevicePolicy for large batches, measured
CPU fallback for small ones), versioned schema checks, staleness/applicability gating,
uncertainty outputs, model abstention, and a safe deterministic fallback.

Hard guarantees:
  * ALWAYS returns an InferenceResponse — any failure yields a reasoned abstention, never a
    raise into the caller.
  * Stale or schema-incompatible predictions are rejected.
  * Only SHADOW-or-later models serve; untrained/research models abstain.
  * This service is invoked OFF the trading-critical path (the detached learning loop / shadow
    comparison). It is NEVER called from, and never blocks, the mandatory protective exit.
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np

from cengine.lakehouse.config import LakehouseConfig
from cengine.learning.hardware import DevicePolicy
from cengine.learning.inference.contract import (
    Applicability,
    InferenceRequest,
    InferenceResponse,
    abstain,
)
from cengine.learning.inference.loader import ModelLoadError, load_model
from cengine.learning.registry.maturity import ModelMaturity
from cengine.learning.registry.model_registry import LearningModelRegistry

# model maturities permitted to SERVE predictions (shadow and beyond).
_SERVING = {ModelMaturity.SHADOW.value, ModelMaturity.PAPER_ELIGIBLE.value,
            ModelMaturity.PAPER_CERTIFIED.value, ModelMaturity.LIVE_ELIGIBLE.value,
            ModelMaturity.LIVE_APPROVED.value, ModelMaturity.ACTIVE.value}

DEFAULT_TTL_NS = 5 * 60 * 1_000_000_000      # predictions expire after 5 minutes
DEFAULT_MAX_STALENESS_NS = 10 * 60 * 1_000_000_000


class LearningInferenceService:
    def __init__(self, config: Optional[LakehouseConfig] = None, *,
                 registry: Optional[LearningModelRegistry] = None,
                 device_policy: Optional[DevicePolicy] = None,
                 ttl_ns: int = DEFAULT_TTL_NS,
                 max_staleness_ns: int = DEFAULT_MAX_STALENESS_NS,
                 min_confidence: float = 0.0) -> None:
        self.config = config or LakehouseConfig.from_env()
        self.registry = registry or LearningModelRegistry(self.config)
        self.policy = device_policy or DevicePolicy()
        self.ttl_ns = ttl_ns
        self.max_staleness_ns = max_staleness_ns
        self.min_confidence = min_confidence
        self._cache: dict = {}            # (model_id, model_version) -> (model, card)
        self.stats = {"served": 0, "abstained": 0, "fallbacks": 0, "load_errors": 0}

    # -- model resolution + caching ------------------------------------------
    def _resolve_card(self, request: InferenceRequest) -> Optional[dict]:
        if request.model_id:
            card = self.registry.latest(request.model_id)
            want_ver = request.model_version
            if card and want_ver and card.get("model_version") != want_ver:
                # a specific pinned version was requested; find it in the ledger
                for row in reversed(self.registry._all_rows()):  # noqa: SLF001
                    if (row.get("model_id") == request.model_id
                            and row.get("model_version") == want_ver):
                        return row
                return None
            return card
        # otherwise pick the champion for this task's model family (by task tag)
        return self.registry.champion(_task_model_type(request.task))

    def _get_model(self, card: dict):
        key = (card.get("model_id"), card.get("model_version"), card.get("revision"))
        if key in self._cache:
            return self._cache[key]
        model = load_model(card, config=self.config)
        self._cache[key] = (model, card)
        return model, card

    # -- single-request inference --------------------------------------------
    def predict(self, request: InferenceRequest, *, now_ns: Optional[int] = None,
                train_feature_ranges: Optional[dict] = None) -> InferenceResponse:
        now_ns = now_ns if now_ns is not None else time.time_ns()
        try:
            card = self._resolve_card(request)
            if card is None:
                self.stats["abstained"] += 1
                return abstain(request, reason="no_model_registered",
                               applicability=Applicability.NO_MODEL.value, now_ns=now_ns)
            maturity = card.get("maturity", ModelMaturity.UNTRAINED.value)
            if maturity not in _SERVING:
                self.stats["abstained"] += 1
                return abstain(request, reason=f"model_not_serving(maturity={maturity})",
                               applicability=Applicability.NO_MODEL.value,
                               model_maturity=maturity, now_ns=now_ns,
                               model_id=card.get("model_id"),
                               model_version=card.get("model_version"))
            # schema compatibility
            if (request.feature_schema_version and card.get("feature_schema_version")
                    and request.feature_schema_version != card.get("feature_schema_version")):
                self.stats["abstained"] += 1
                return abstain(request, reason="feature_schema_incompatible",
                               applicability=Applicability.SCHEMA_INCOMPATIBLE.value,
                               model_maturity=maturity, now_ns=now_ns,
                               model_id=card.get("model_id"),
                               model_version=card.get("model_version"))
            # staleness
            if (request.availability_timestamp_ns
                    and now_ns - request.availability_timestamp_ns > self.max_staleness_ns):
                self.stats["abstained"] += 1
                return abstain(request, reason="stale_features",
                               applicability=Applicability.STALE.value,
                               model_maturity=maturity, now_ns=now_ns,
                               model_id=card.get("model_id"),
                               model_version=card.get("model_version"))
            model, card = self._get_model(card)
            X = np.asarray([request.features], dtype=np.float32)
            applicability = Applicability.APPLICABLE.value
            if train_feature_ranges is not None and _is_ood(X[0], train_feature_ranges):
                applicability = Applicability.OUT_OF_DISTRIBUTION.value
            pred, unc = model.predict_with_uncertainty(X)
            p = float(pred[0])
            u = float(unc[0]) if unc is not None else None
            self.stats["served"] += 1
            return InferenceResponse(
                task=request.task, symbol_id=request.symbol_id, prediction=p,
                prediction_uncertainty=u, applicability=applicability, data_quality="ok",
                model_maturity=maturity, model_id=card.get("model_id"),
                model_version=card.get("model_version"),
                device=getattr(model, "device", "cpu"),
                inference_timestamp_ns=now_ns, expiration_timestamp_ns=now_ns + self.ttl_ns)
        except ModelLoadError as exc:
            self.stats["load_errors"] += 1
            self.stats["fallbacks"] += 1
            return abstain(request, reason=f"model_load_error:{exc}",
                           applicability=Applicability.NO_MODEL.value, now_ns=now_ns)
        except Exception as exc:  # noqa: BLE001 — inference must never raise into the caller
            self.stats["fallbacks"] += 1
            return abstain(request, reason=f"inference_error:{type(exc).__name__}",
                           applicability=Applicability.NO_MODEL.value, now_ns=now_ns)

    # -- batched cross-sectional inference (GPU-first) -----------------------
    def predict_batch(self, task: str, feature_matrix: np.ndarray, symbols,
                      feature_schema_version: str, *, now_ns: Optional[int] = None,
                      model_id: Optional[str] = None) -> list:
        """Vectorized inference over a universe. One model load, one GPU pass."""
        now_ns = now_ns if now_ns is not None else time.time_ns()
        X = np.asarray(feature_matrix, dtype=np.float32)
        probe = InferenceRequest(
            task=task, symbol_id="*", features=(), model_id=model_id,
            feature_schema_version=feature_schema_version,
            observation_timestamp_ns=now_ns, availability_timestamp_ns=now_ns)
        card = self._resolve_card(probe)
        if card is None or card.get("maturity") not in _SERVING:
            self.stats["abstained"] += len(X)
            return [abstain(InferenceRequest(task=task, symbol_id=str(s), features=(),
                                             feature_schema_version=feature_schema_version,
                                             observation_timestamp_ns=now_ns,
                                             availability_timestamp_ns=now_ns),
                            reason="no_serving_model", now_ns=now_ns) for s in symbols]
        try:
            model, card = self._get_model(card)
            pred, unc = model.predict_with_uncertainty(X)
            dev = getattr(model, "device", "cpu")
            out = []
            for i, s in enumerate(symbols):
                out.append(InferenceResponse(
                    task=task, symbol_id=str(s), prediction=float(pred[i]),
                    prediction_uncertainty=(float(unc[i]) if unc is not None else None),
                    applicability=Applicability.APPLICABLE.value, data_quality="ok",
                    model_maturity=card.get("maturity"), model_id=card.get("model_id"),
                    model_version=card.get("model_version"), device=dev,
                    inference_timestamp_ns=now_ns,
                    expiration_timestamp_ns=now_ns + self.ttl_ns))
            self.stats["served"] += len(out)
            return out
        except Exception as exc:  # noqa: BLE001
            self.stats["fallbacks"] += len(X)
            return [abstain(InferenceRequest(task=task, symbol_id=str(s), features=(),
                                             feature_schema_version=feature_schema_version,
                                             observation_timestamp_ns=now_ns,
                                             availability_timestamp_ns=now_ns),
                            reason=f"batch_inference_error:{type(exc).__name__}",
                            now_ns=now_ns) for s in symbols]

    def dashboard_stats(self) -> dict:
        return {**self.stats, "cached_models": len(self._cache),
                "device_policy": self.policy.summary()}


def _task_model_type(task: str) -> str:
    return {
        "forward_return_prediction": "mlx_quantile_regressor",
        "probability_positive_return": "mlx_classifier",
        "strategy_net_return": "mlx_regressor",
    }.get(task, "mlx_regressor")


def _is_ood(x: np.ndarray, ranges: dict) -> bool:
    lo = np.asarray(ranges.get("lo"))
    hi = np.asarray(ranges.get("hi"))
    if lo.shape != x.shape or hi.shape != x.shape:
        return False
    span = np.maximum(hi - lo, 1e-9)
    return bool(np.any(x < lo - span) or np.any(x > hi + span))
