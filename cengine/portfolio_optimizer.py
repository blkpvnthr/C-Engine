"""Portfolio optimizer orchestration + recommendation contract (Milestone 3, §7/§10/§11).

Chooses the allocation policy, applies hard constraints, sizes target weights into
requested quantities, and emits versioned :class:`PortfolioRecommendation` objects.
Reuses the existing CVXPY :class:`RegimeMarkowitzAllocator` for mean-variance when
reliable expected returns are supplied, and the native C++ risk-based allocator
(``_cengine_native.allocate_risk_based``) otherwise (§7 deterministic fallback).

Hard rules honored:
  * Heuristic strategy scores are NEVER converted into fabricated expected returns.
    When reliable μ is absent the allocation is explicitly risk-based/deterministic.
  * No optimization objective overrides a hard constraint (§8 enforced here).
  * Transaction costs are not double-counted.
  * Recommendations are not orders; they carry explicit unknown (None) economics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional

import numpy as np

from cengine.portfolio_analytics import CovarianceResult, PortfolioAnalytics
from cengine.portfolio_constraints import PortfolioConstraints, enforce

try:
    import _cengine_native as _native
except Exception:  # noqa: BLE001
    _native = None

OPTIMIZER_VERSION = "portfolio-optimizer-1.0.0"


class AllocationPolicy(str, Enum):
    MEAN_VARIANCE = "mean_variance"            # CVXPY regime Markowitz (needs reliable μ)
    RISK_BASED = "risk_based"                  # native inverse-vol (deterministic fallback)
    EQUAL_WEIGHT = "equal_weight"


class RecommendationEligibility(str, Enum):
    PROPOSED = "proposed"                      # passed constraints; ready for reservation/routing
    REJECTED = "rejected"                      # a hard constraint or sizing failed


@dataclass
class PortfolioRecommendation:
    portfolio_decision_id: str
    source_strategy_decision_id: Optional[str]
    symbol_id: str
    instrument_generation: Optional[int]
    portfolio_version: str
    strategy_id: str
    strategy_family: Optional[str]
    regime_version: Optional[str]
    feature_version: Optional[str]
    optimizer_version: str
    configuration_version: str
    observation_timestamp: int
    decision_timestamp: int
    expiration_timestamp: int
    current_weight: float
    target_weight: float
    requested_quantity: int
    requested_notional_ticks: int
    capital_required_ticks: int
    # explicit unknowns stay None — never fabricated
    expected_incremental_net_return: Optional[float] = None
    expected_risk_contribution: Optional[float] = None
    estimated_transaction_cost: Optional[float] = None
    effective_exposure_delta_ticks: Optional[int] = None
    constraint_status: str = "ok"
    binding_constraints: list = field(default_factory=list)
    allocation_policy: str = AllocationPolicy.RISK_BASED.value
    return_basis: str = "deterministic_risk_based"
    eligibility: RecommendationEligibility = RecommendationEligibility.PROPOSED
    rejection_reason: Optional[str] = None


@dataclass(frozen=True)
class OptimizerConfig:
    constraints: PortfolioConstraints = field(default_factory=PortfolioConstraints)
    risk_aversion: float = 5.0
    per_asset_cap: float = 0.20
    gross_cap: float = 1.0
    ttl_ns: int = 120 * 1_000_000_000
    allow_fractional: bool = False
    configuration_version: str = "portfolio-cfg-1"


class PortfolioOptimizer:
    """Produces constrained, sized portfolio recommendations from candidates."""

    def __init__(self, config: Optional[OptimizerConfig] = None, *,
                 analytics: Optional[PortfolioAnalytics] = None) -> None:
        self.config = config or OptimizerConfig()
        self.analytics = analytics or PortfolioAnalytics()

    def target_weights(
        self, symbols: tuple, cov: CovarianceResult, *,
        expected_returns: Optional[np.ndarray] = None,
        regime_probability: float = 0.5,
    ) -> tuple[dict, AllocationPolicy]:
        """Compute raw target weights before hard-constraint enforcement.

        Mean-variance (CVXPY) only when reliable expected returns are supplied AND
        CVXPY is importable; otherwise the native risk-based policy.
        """
        n = len(symbols)
        if n == 0:
            return {}, AllocationPolicy.RISK_BASED
        variances = [float(cov.covariance[i, i]) for i in range(n)]

        if expected_returns is not None and len(expected_returns) == n:
            try:
                from cengine.portfolio_optimization import (
                    AllocationConstraints,
                    RegimeMarkowitzAllocator,
                )
                allocator = RegimeMarkowitzAllocator(AllocationConstraints(
                    gross_limit=min(1.0, self.config.gross_cap),
                    per_asset_limit=min(self.config.gross_cap, self.config.per_asset_cap),
                    risk_aversion=self.config.risk_aversion))
                advice = allocator.optimize(symbols, np.asarray(expected_returns, dtype=float),
                                            cov.covariance, regime_probability)
                # long-only portfolio contract: clamp any negative MV weight to 0
                weights = {s: max(0.0, float(w))
                           for s, w in zip(advice.symbols, advice.weights, strict=False)}
                return weights, AllocationPolicy.MEAN_VARIANCE
            except Exception:  # noqa: BLE001 — CVXPY missing/failed -> risk-based fallback
                pass

        # Native risk-based deterministic allocation.
        if _native is not None and hasattr(_native, "allocate_risk_based"):
            res = _native.allocate_risk_based(
                variances, cov.covariance.astype(np.float32).flatten().tolist(), n,
                self.config.per_asset_cap, self.config.gross_cap)
            if res.valid:
                return {symbols[i]: float(res.weights[i]) for i in range(n)}, \
                    AllocationPolicy.RISK_BASED
        # Pure-numpy inverse-vol fallback (native unavailable).
        inv = np.array([1.0 / np.sqrt(v) if v > 0 else 0.0 for v in variances])
        total = inv.sum()
        if total <= 0:
            return {}, AllocationPolicy.EQUAL_WEIGHT
        weights = {symbols[i]: float(min(self.config.per_asset_cap, inv[i] / total))
                   for i in range(n)}
        return weights, AllocationPolicy.RISK_BASED

    def optimize(
        self, *, symbols: tuple, cov: CovarianceResult, state, now_ns: int,
        expected_returns: Optional[np.ndarray] = None,
        regime_probability: float = 0.5,
        current_weights: Optional[dict] = None,
        source_decision_ids: Optional[dict] = None,
        strategy_meta: Optional[dict] = None,
        price_ticks: Optional[Callable[[str], int]] = None,
        transaction_cost_bps: Optional[dict] = None,
        authorized: Optional[Callable[[str], bool]] = None,
        sector_of: Optional[Callable[[str], str]] = None,
        family_of: Optional[Callable[[str], str]] = None,
        effective_leverage: Optional[float] = None,
    ) -> list[PortfolioRecommendation]:
        """Full optimization: policy -> constraints -> sizing -> recommendations."""
        raw, policy = self.target_weights(
            symbols, cov, expected_returns=expected_returns,
            regime_probability=regime_probability)

        report = enforce(
            raw, constraints=self.config.constraints, covariance=cov.covariance,
            symbols=symbols, current_weights=current_weights,
            drawdown=state.drawdown_fraction, effective_leverage=effective_leverage,
            authorized=authorized, sector_of=sector_of, family_of=family_of)

        decision_ts = now_ns
        recs: list[PortfolioRecommendation] = []
        source_decision_ids = source_decision_ids or {}
        strategy_meta = strategy_meta or {}
        transaction_cost_bps = transaction_cost_bps or {}
        current_weights = current_weights or {}

        if not report.feasible:
            # One rejection record capturing the infeasible optimization (no orders).
            recs.append(self._rejected(symbols, state, now_ns, decision_ts, policy,
                                       report.rejection_reason or "infeasible"))
            return recs

        equity = max(0, state.account_equity_ticks)
        risk_contribs = {}
        if report.weights:
            wv = np.array([report.weights.get(s, 0.0) for s in symbols])
            rc = self.analytics.risk_contributions(wv, cov.covariance)
            risk_contribs = {symbols[i]: float(rc[i]) for i in range(len(symbols))}

        for sym in symbols:
            tgt = float(report.weights.get(sym, 0.0))
            cur = float(current_weights.get(sym, 0.0))
            meta = strategy_meta.get(sym, {})
            px = int(price_ticks(sym)) if price_ticks is not None else 0
            notional = int(tgt * equity)
            if px > 0:
                qty_f = notional / px
                qty = qty_f if self.config.allow_fractional else int(qty_f)
            else:
                qty = 0
            cost_bps = transaction_cost_bps.get(sym)
            est_cost = (abs(tgt - cur) * (cost_bps / 10_000.0)) if cost_bps is not None else None
            recs.append(PortfolioRecommendation(
                portfolio_decision_id=f"pf:{sym}:{now_ns}",
                source_strategy_decision_id=source_decision_ids.get(sym),
                symbol_id=sym, instrument_generation=meta.get("instrument_generation"),
                portfolio_version=state.portfolio_version,
                strategy_id=meta.get("strategy_id", "portfolio"),
                strategy_family=meta.get("strategy_family"),
                regime_version=meta.get("regime_version"),
                feature_version=meta.get("feature_version"),
                optimizer_version=OPTIMIZER_VERSION,
                configuration_version=self.config.configuration_version,
                observation_timestamp=state.observation_timestamp,
                decision_timestamp=decision_ts,
                expiration_timestamp=now_ns + self.config.ttl_ns,
                current_weight=cur, target_weight=tgt,
                requested_quantity=qty if isinstance(qty, int) else int(qty),
                requested_notional_ticks=notional,
                capital_required_ticks=max(0, notional) if tgt > cur else 0,
                expected_incremental_net_return=None,  # not fabricated
                expected_risk_contribution=risk_contribs.get(sym),
                estimated_transaction_cost=est_cost,
                effective_exposure_delta_ticks=None,
                constraint_status="ok" if not report.binding else "binding",
                binding_constraints=list(report.binding),
                allocation_policy=policy.value,
                return_basis=("calibrated_mean_variance"
                              if policy is AllocationPolicy.MEAN_VARIANCE
                              else "deterministic_risk_based"),
                eligibility=(RecommendationEligibility.PROPOSED if qty and tgt > 0
                             else RecommendationEligibility.REJECTED),
                rejection_reason=(None if qty and tgt > 0 else "zero_target_or_quantity"),
            ))
        return recs

    def _rejected(self, symbols, state, now_ns, decision_ts, policy, reason):
        return PortfolioRecommendation(
            portfolio_decision_id=f"pf:INFEASIBLE:{now_ns}",
            source_strategy_decision_id=None, symbol_id="",
            instrument_generation=None, portfolio_version=state.portfolio_version,
            strategy_id="portfolio", strategy_family=None, regime_version=None,
            feature_version=None, optimizer_version=OPTIMIZER_VERSION,
            configuration_version=self.config.configuration_version,
            observation_timestamp=state.observation_timestamp, decision_timestamp=decision_ts,
            expiration_timestamp=now_ns + self.config.ttl_ns,
            current_weight=0.0, target_weight=0.0, requested_quantity=0,
            requested_notional_ticks=0, capital_required_ticks=0,
            allocation_policy=policy.value, constraint_status="infeasible",
            eligibility=RecommendationEligibility.REJECTED, rejection_reason=reason)
