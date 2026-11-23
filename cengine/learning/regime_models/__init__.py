"""Probabilistic regime models (Milestone 6 learning layer).

A probabilistic *extension* of the deterministic ``cengine.regime_engine.MarketRegimeEngine``:
the deterministic engine remains the transparent baseline + fallback that labels regimes,
and predicted probabilities NEVER overwrite observed historical regime labels. The first
and simplest estimator here is a first-order Markov transition-frequency model over the
observed regime dimensions in the silver ``market_regimes`` dataset — evaluated before any
more complex classifier (CLAUDE.md "Simplicity Wins"). Recommendation-only research infra:
no orders, no broker, and never on/around the mandatory -0.5% protective-exit path.
"""

from __future__ import annotations

from cengine.learning.regime_models.transition_model import (
    DIRECTION_STATES,
    LIQUIDITY_STATES,
    REGIME_DIMENSIONS,
    REGIME_TRANSITION_MODEL_VERSION,
    TREND_STATES,
    VOLATILITY_STATES,
    DimensionSpec,
    RegimeTransitionModel,
    TransitionFitResult,
    TransitionMatrix,
    fit_regime_transition_model,
)

__all__ = [
    "REGIME_TRANSITION_MODEL_VERSION",
    "DimensionSpec",
    "REGIME_DIMENSIONS",
    "DIRECTION_STATES",
    "TREND_STATES",
    "VOLATILITY_STATES",
    "LIQUIDITY_STATES",
    "TransitionMatrix",
    "TransitionFitResult",
    "RegimeTransitionModel",
    "fit_regime_transition_model",
]
