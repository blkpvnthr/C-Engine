"""Universe-selection decision recorder (§9 persistence adapter).

Translates a :class:`RotationResult` + :class:`UniverseLayers` into the generic
nonblocking :class:`cengine.decision_events.DecisionEventSink` contract, so
universe-selection decisions flow through the existing bronze → silver → gold →
feature-store pipeline exactly like every other decision event. The sink's
``emit`` is O(1) and never raises, so wiring this as the manager's event emitter
cannot perturb rotation or the trading loop.

No fabrication: only observed selection decisions are recorded here; downstream
price/trade performance is linked later via the existing per-symbol
REALIZED_OUTCOME events (joined in the gold layer), never invented for symbols
that were not monitored.
"""

from __future__ import annotations

from typing import Optional

from .layers import UniverseLayers
from .rotation import RotationResult


class UniverseDecisionRecorder:
    """Callable adapter matching ``DynamicUniverseManager`` event-emitter shape."""

    def __init__(self, sink, *, regime_version_fn=None, model_version: Optional[str] = None):
        self._sink = sink
        self._regime_version_fn = regime_version_fn
        self._model_version = model_version

    def __call__(
        self,
        result: RotationResult,
        layers: UniverseLayers,
        scores: dict,
        now_s: float,
    ) -> None:
        if self._sink is None:
            return
        regime_version = None
        if self._regime_version_fn is not None:
            try:
                regime_version = self._regime_version_fn()
            except Exception:  # noqa: BLE001 - telemetry must never raise
                regime_version = None
        self._sink.record_universe_rotation(
            now_ns=int(now_s * 1e9),
            generation=layers.version,
            active=list(result.active),
            added=list(result.added),
            removed=list(result.removed),
            active_baseline=list(result.active_baseline),
            active_opportunity=list(result.active_opportunity),
            protected=list(result.protected),
            stale_baseline=list(result.stale_baseline),
            degraded=result.degraded,
            reasons=dict(result.reasons),
            scores=dict(scores or {}),
            baseline_set=set(layers.baseline),
            regime_version=regime_version,
            model_version=self._model_version,
        )
