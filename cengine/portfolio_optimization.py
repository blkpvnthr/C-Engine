"""Advisory-only regime-aware Markowitz allocation and portfolio risk controls."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class AllocationConstraints:
    gross_limit: float
    per_asset_limit: float
    risk_aversion: float

    def __post_init__(self) -> None:
        if not (0 < self.gross_limit <= 1):
            raise ValueError("gross_limit must be in (0, 1]")
        if not (0 < self.per_asset_limit <= self.gross_limit):
            raise ValueError("invalid per_asset_limit")
        if self.risk_aversion <= 0:
            raise ValueError("risk_aversion must be positive")


@dataclass(frozen=True, slots=True)
class AllocationAdvice:
    symbols: tuple[str, ...]
    weights: tuple[float, ...]
    regime_probability: float
    model_version: str


class RegimeMarkowitzAllocator:
    """Produces advice only; it has no venue or OrderManager dependency."""

    def __init__(self, constraints: AllocationConstraints) -> None:
        self.constraints = constraints

    def optimize(
        self,
        symbols: tuple[str, ...],
        expected_returns: np.ndarray,
        covariance: np.ndarray,
        risk_on_probability: float,
    ) -> AllocationAdvice:
        if not 0 <= risk_on_probability <= 1:
            raise ValueError("regime probability must be in [0, 1]")
        if expected_returns.shape != (len(symbols),):
            raise ValueError("expected-return shape mismatch")
        if covariance.shape != (len(symbols), len(symbols)):
            raise ValueError("covariance shape mismatch")
        try:
            import cvxpy as cp
        except ImportError as exc:
            raise RuntimeError("install cengine[research] for CVXPY allocation") from exc
        weights = cp.Variable(len(symbols))
        regime_returns = expected_returns * (2 * risk_on_probability - 1)
        objective = cp.Maximize(
            regime_returns @ weights
            - self.constraints.risk_aversion * cp.quad_form(weights, covariance)
        )
        constraints = [
            cp.norm1(weights) <= self.constraints.gross_limit,
            weights <= self.constraints.per_asset_limit,
            weights >= -self.constraints.per_asset_limit,
        ]
        problem = cp.Problem(objective, constraints)
        problem.solve()
        if weights.value is None or problem.status not in {"optimal", "optimal_inaccurate"}:
            raise RuntimeError(f"allocation failed: {problem.status}")
        return AllocationAdvice(
            symbols,
            tuple(float(x) for x in weights.value),
            risk_on_probability,
            "regime-markowitz-v1",
        )


@dataclass(frozen=True, slots=True)
class PortfolioRiskLimits:
    max_gross_weight: float
    max_single_weight: float
    max_predicted_volatility: float


class PortfolioRiskManager:
    def validate(
        self,
        advice: AllocationAdvice,
        covariance: np.ndarray,
        limits: PortfolioRiskLimits,
    ) -> None:
        weights = np.asarray(advice.weights)
        if np.abs(weights).sum() > limits.max_gross_weight:
            raise RuntimeError("allocation exceeds gross risk limit")
        if np.abs(weights).max(initial=0.0) > limits.max_single_weight:
            raise RuntimeError("allocation exceeds single-asset risk limit")
        volatility = float(np.sqrt(max(0.0, weights @ covariance @ weights)))
        if volatility > limits.max_predicted_volatility:
            raise RuntimeError("allocation exceeds predicted-volatility limit")
