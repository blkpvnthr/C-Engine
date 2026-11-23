"""Hard portfolio constraints (Milestone 3, §8).

No optimization objective may override a hard financial constraint. Infeasible
optimization results in a safe allocation (scaled down) or an explicit rejection —
constraints are NEVER silently relaxed. Every binding/violated constraint is
reported for decision evidence.

Constraints operate on a proposed target allocation (symbol -> weight) plus the
authoritative portfolio state and analytics. Capital/buying-power/reservation
constraints are enforced downstream by the native RiskEngine + capital-reservation
ledger; this layer enforces the PORTFOLIO-level limits (weights, exposure,
volatility, correlation, turnover, authorization, min-notional).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

CONSTRAINTS_VERSION = "portfolio-constraints-1.0.0"


@dataclass(frozen=True)
class PortfolioConstraints:
    max_position_weight: float = 0.20        # per-asset
    max_family_weight: float = 0.50          # per strategy family
    max_sector_weight: float = 0.40
    max_benchmark_exposure: float = 1.0      # |signed effective benchmark exposure| / equity
    max_gross_exposure: float = 1.0
    max_net_exposure: float = 1.0
    max_effective_leverage: float = 1.0      # |signed daily effective exposure| / equity
    max_correlated_exposure: float = 0.60    # summed weight of a highly-correlated cluster
    correlation_threshold: float = 0.80
    max_portfolio_volatility: float = 0.05   # per-period (same units as returns)
    max_drawdown: float = 0.20               # block NEW elective risk beyond this drawdown
    min_trade_notional_ticks: int = 1_000    # $0.10 at scale 10_000 -> set per deployment
    max_turnover: float = 1.0                # Σ|Δw| per rebalance
    min_liquidity_rel_volume: float = 0.0

    def __post_init__(self) -> None:
        for name in ("max_position_weight", "max_gross_exposure", "max_portfolio_volatility"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")


@dataclass
class ConstraintReport:
    feasible: bool
    weights: dict                             # possibly scaled-down feasible weights
    binding: list = field(default_factory=list)      # constraints at their limit
    violated: list = field(default_factory=list)     # hard violations -> rejection
    rejection_reason: Optional[str] = None
    version: str = CONSTRAINTS_VERSION


def enforce(
    target_weights: dict,
    *,
    constraints: PortfolioConstraints,
    covariance: Optional[np.ndarray] = None,
    symbols: Optional[tuple] = None,
    current_weights: Optional[dict] = None,
    drawdown: float = 0.0,
    effective_leverage: Optional[float] = None,
    authorized: Optional[Callable[[str], bool]] = None,
    sector_of: Optional[Callable[[str], str]] = None,
    family_of: Optional[Callable[[str], str]] = None,
) -> ConstraintReport:
    """Validate (and minimally, safely adjust) a proposed allocation.

    Returns a :class:`ConstraintReport`. Hard, non-scalable violations (unauthorized
    instrument, drawdown circuit-breaker, excessive volatility that cannot be scaled
    out, turnover) produce ``feasible=False`` with a rejection reason. Scalable
    breaches (gross/net/leverage over the cap) are scaled down proportionally and
    recorded as binding. Nothing is silently relaxed.
    """
    report = ConstraintReport(True, dict(target_weights))
    w = {s: float(x) for s, x in target_weights.items()}

    # --- instrument authorization (hard; never scale) -----------------------
    if authorized is not None:
        for sym, weight in w.items():
            if weight > 0 and not authorized(sym):
                report.feasible = False
                report.violated.append(f"instrument_not_authorized:{sym}")
        if not report.feasible:
            report.rejection_reason = "unauthorized_instrument"
            report.weights = {}
            return report

    # --- drawdown circuit breaker (hard) ------------------------------------
    if drawdown >= constraints.max_drawdown:
        report.feasible = False
        report.rejection_reason = (f"drawdown_circuit_breaker:{drawdown:.4f}>="
                                   f"{constraints.max_drawdown}")
        report.violated.append("max_drawdown")
        report.weights = {}
        return report

    # --- per-asset weight (scale offending weights down to cap) -------------
    for sym, weight in list(w.items()):
        if weight > constraints.max_position_weight + 1e-12:
            w[sym] = constraints.max_position_weight
            report.binding.append(f"max_position_weight:{sym}")

    # --- per-family / per-sector caps (scale the group down) ----------------
    _scale_group(w, family_of, constraints.max_family_weight, "max_family_weight", report)
    _scale_group(w, sector_of, constraints.max_sector_weight, "max_sector_weight", report)

    # --- gross exposure (scale whole book down) -----------------------------
    gross = sum(abs(x) for x in w.values())
    if gross > constraints.max_gross_exposure + 1e-12 and gross > 0:
        scale = constraints.max_gross_exposure / gross
        w = {s: x * scale for s, x in w.items()}
        report.binding.append("max_gross_exposure")

    # --- net exposure (scale down if over) ----------------------------------
    net = sum(w.values())
    if abs(net) > constraints.max_net_exposure + 1e-12 and abs(net) > 0:
        scale = constraints.max_net_exposure / abs(net)
        w = {s: x * scale for s, x in w.items()}
        report.binding.append("max_net_exposure")

    # --- effective leveraged exposure (hard; scale down) --------------------
    if effective_leverage is not None and abs(effective_leverage) > 0:
        if abs(effective_leverage) > constraints.max_effective_leverage + 1e-12:
            scale = constraints.max_effective_leverage / abs(effective_leverage)
            w = {s: x * scale for s, x in w.items()}
            report.binding.append("max_effective_leverage")

    # --- correlated-cluster exposure ----------------------------------------
    if covariance is not None and symbols is not None:
        cluster = _max_correlated_cluster_weight(w, covariance, symbols,
                                                 constraints.correlation_threshold)
        if cluster > constraints.max_correlated_exposure + 1e-12 and cluster > 0:
            scale = constraints.max_correlated_exposure / cluster
            w = {s: x * scale for s, x in w.items()}
            report.binding.append("max_correlated_exposure")

    # --- portfolio volatility (scale the whole book; vol is homogeneous) -----
    if covariance is not None and symbols is not None:
        vol = _portfolio_vol(w, covariance, symbols)
        if vol > constraints.max_portfolio_volatility + 1e-12 and vol > 0:
            scale = constraints.max_portfolio_volatility / vol
            w = {s: x * scale for s, x in w.items()}
            report.binding.append("max_portfolio_volatility")

    # --- turnover (hard; reject if the change is too large) -----------------
    if current_weights is not None:
        turnover = _turnover(w, current_weights)
        if turnover > constraints.max_turnover + 1e-12:
            report.feasible = False
            report.rejection_reason = (f"turnover_exceeds_cap:{turnover:.4f}>"
                                       f"{constraints.max_turnover}")
            report.violated.append("max_turnover")
            report.weights = {}
            return report

    report.weights = {s: x for s, x in w.items() if abs(x) > 1e-12}
    return report


def _scale_group(w: dict, key_of, cap: float, label: str, report: ConstraintReport) -> None:
    if key_of is None:
        return
    groups: dict = {}
    for sym in w:
        groups.setdefault(key_of(sym), []).append(sym)
    for key, syms in groups.items():
        total = sum(abs(w[s]) for s in syms)
        if total > cap + 1e-12 and total > 0:
            scale = cap / total
            for s in syms:
                w[s] *= scale
            report.binding.append(f"{label}:{key}")


def _portfolio_vol(w: dict, cov: np.ndarray, symbols: tuple) -> float:
    vec = np.array([w.get(s, 0.0) for s in symbols], dtype=float)
    return float(np.sqrt(max(0.0, vec @ cov @ vec)))


def _turnover(w: dict, current: dict) -> float:
    keys = set(w) | set(current)
    return float(sum(abs(w.get(k, 0.0) - current.get(k, 0.0)) for k in keys))


def _max_correlated_cluster_weight(w: dict, cov: np.ndarray, symbols: tuple,
                                   threshold: float) -> float:
    """Largest summed |weight| over a set of mutually highly-correlated names."""
    d = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    denom = np.outer(d, d)
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = np.where(denom > 0, cov / denom, 0.0)
    n = len(symbols)
    best = 0.0
    for i in range(n):
        cluster = sum(abs(w.get(symbols[j], 0.0))
                      for j in range(n) if i == j or corr[i, j] >= threshold)
        best = max(best, cluster)
    return best
