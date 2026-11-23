"""Portfolio optimization -> capital reservation -> execution composition (§17).

Ties the GPU portfolio analytics, native/CVXPY optimizer, atomic capital
reservation, and the EXISTING execution authority into one path:

    strategy recommendations (+ return history + portfolio state)
      -> PortfolioAnalytics (GPU covariance)
      -> PortfolioOptimizer (constrained target weights + sized recommendations)
      -> CapitalReservationLedger (atomic, prevents double-spend)
      -> CandidateRouter (EXISTING: exposure gate -> native RiskEngine -> OrderManager)
      -> fill reconciliation + decision evidence

It creates NO parallel execution authority: it routes through the SAME OrderManager
via its own CandidateRouter translator (optimizer-sized quantities). Default
authorization is NONE -> routes nothing (advisory). The independent -0.5% protective
exit is never touched.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from cengine.capital_reservation import CapitalReservationLedger
from cengine.decision_events import DecisionEventSink
from cengine.portfolio_analytics import PortfolioAnalytics, build_return_matrix
from cengine.portfolio_optimizer import (
    OptimizerConfig,
    PortfolioOptimizer,
    RecommendationEligibility,
)
from cengine.strategy_maturity import ExecutionAuthorization
from order_manager import Side
from strategies import SignalSide, StrategyName, TradeCandidate

PORTFOLIO_PIPELINE_VERSION = "portfolio-pipeline-1.0.0"


class PortfolioQuantityPolicy:
    """Entry/exit quantity policy backed by the optimizer's per-run sizing map."""

    def __init__(self) -> None:
        self._entry: dict = {}
        self._exit: dict = {}

    def set_quantities(self, entry: dict, exit_: Optional[dict] = None) -> None:
        self._entry = {s.upper(): int(q) for s, q in entry.items()}
        self._exit = {s.upper(): int(q) for s, q in (exit_ or {}).items()}

    def quantity_for(self, candidate) -> int:
        return self._entry.get(candidate.symbol.upper(), 0)

    def quantity_to_close(self, candidate) -> int:
        return self._exit.get(candidate.symbol.upper(), 0)


@dataclass
class PortfolioPipelineResult:
    recommendations: list = field(default_factory=list)
    reserved: int = 0
    routed: int = 0
    skipped: int = 0
    covariance_source: str = "cpu"
    allocation_policy: str = "risk_based"
    dispositions: list = field(default_factory=list)
    excluded_symbols: dict = field(default_factory=dict)


class PortfolioPipeline:
    def __init__(self, *, router, quantity_policy: PortfolioQuantityPolicy,
                 optimizer: Optional[PortfolioOptimizer] = None,
                 analytics: Optional[PortfolioAnalytics] = None,
                 reservation_ledger: Optional[CapitalReservationLedger] = None,
                 sink: Optional[DecisionEventSink] = None,
                 authorization: ExecutionAuthorization = ExecutionAuthorization.NONE,
                 min_observations: int = 20,
                 correlation_index: Optional[dict] = None) -> None:
        self._router = router
        self._quantity = quantity_policy
        # correlation_id -> portfolio_decision_id, so an OrderManager audit handler
        # can attribute ack/fill/risk events back to the portfolio decision (§18).
        self.correlation_index = correlation_index if correlation_index is not None else {}
        self.analytics = analytics or PortfolioAnalytics()
        self.optimizer = optimizer or PortfolioOptimizer(
            OptimizerConfig(), analytics=self.analytics)
        self.ledger = reservation_ledger or CapitalReservationLedger()
        self.sink = sink or DecisionEventSink()
        self.authorization = authorization
        self.min_observations = int(min_observations)

    async def run(
        self, *, symbols: tuple, returns_by_symbol: dict, state, now_ns: int,
        price_ticks: Callable[[str], int],
        expected_returns: Optional[np.ndarray] = None,
        regime_probability: float = 0.5,
        current_weights: Optional[dict] = None,
        source_decision_ids: Optional[dict] = None,
        strategy_meta: Optional[dict] = None,
        transaction_cost_bps: Optional[dict] = None,
        authorized: Optional[Callable[[str], bool]] = None,
        sector_of: Optional[Callable[[str], str]] = None,
        family_of: Optional[Callable[[str], str]] = None,
        ewma_lambda: Optional[float] = None,
    ) -> PortfolioPipelineResult:
        """One portfolio pass. Emits decision evidence; routes only authorized,
        reserved recommendations through the existing authority. Never raises."""
        result = PortfolioPipelineResult()
        self.sink.record_portfolio_state(state, now_ns)

        allocatable, why = state.allocatable()
        if not allocatable:
            result.dispositions.append(("portfolio", "not_allocatable:" + why))
            return result

        rm = build_return_matrix(returns_by_symbol, min_observations=self.min_observations)
        result.excluded_symbols = dict(rm.excluded)
        if rm.assets < 1 or rm.observations < 2:
            result.dispositions.append(("portfolio", "insufficient_return_history"))
            return result

        cov = self.analytics.covariance(rm, ewma_lambda=ewma_lambda)
        result.covariance_source = cov.source

        # Align expected returns (if any) to the aligned symbol set.
        er = None
        if expected_returns is not None:
            idx = {s: i for i, s in enumerate(symbols)}
            if all(s in idx for s in rm.symbols):
                er = np.array([expected_returns[idx[s]] for s in rm.symbols], dtype=float)

        recs = self.optimizer.optimize(
            symbols=rm.symbols, cov=cov, state=state, now_ns=now_ns,
            expected_returns=er, regime_probability=regime_probability,
            current_weights=current_weights, source_decision_ids=source_decision_ids,
            strategy_meta=strategy_meta, price_ticks=price_ticks,
            transaction_cost_bps=transaction_cost_bps, authorized=authorized,
            sector_of=sector_of, family_of=family_of)
        result.recommendations = recs
        if recs:
            result.allocation_policy = recs[0].allocation_policy
        binding = sorted({c for r in recs for c in r.binding_constraints})
        self.sink.record_portfolio_optimization(
            now_ns=now_ns, policy=result.allocation_policy,
            covariance_source=cov.source, covariance_version=cov.version,
            return_basis=(recs[0].return_basis if recs else "n/a"),
            binding=binding, candidate_count=len(rm.symbols))
        for rec in recs:
            self.sink.record_portfolio_recommendation(rec, now_ns)

        # Reserve + route proposed BUY recommendations (increase to target).
        base_available = max(0, state.available_cash_ticks)
        for rec in recs:
            if rec.eligibility is not RecommendationEligibility.PROPOSED:
                result.skipped += 1
                continue
            if rec.target_weight <= rec.current_weight or rec.requested_quantity <= 0:
                result.skipped += 1
                continue
            px = int(price_ticks(rec.symbol_id))
            ok, reason = self.ledger.try_reserve(
                decision_id=rec.portfolio_decision_id, symbol=rec.symbol_id,
                side=Side.BUY, quantity=rec.requested_quantity, price_ticks=px,
                base_available_ticks=base_available)
            self.sink.record_capital_reservation(
                decision_id=rec.portfolio_decision_id, symbol_id=rec.symbol_id,
                notional_ticks=rec.requested_quantity * px, accepted=ok, reason=reason,
                now_ns=now_ns)
            if not ok:
                result.dispositions.append((rec.symbol_id, "reservation_failed:" + reason))
                result.skipped += 1
                continue
            result.reserved += 1

            if self.authorization is ExecutionAuthorization.NONE:
                result.dispositions.append((rec.symbol_id, "advisory_authorization_none"))
                self.ledger.mark_canceled(rec.portfolio_decision_id)  # release (not routed)
                continue

            disp = await self._route(rec, px, now_ns)
            result.dispositions.append((rec.symbol_id, disp))
            if disp == "routed":
                result.routed += 1
            else:
                self.ledger.mark_canceled(rec.portfolio_decision_id)
                result.skipped += 1
        return result

    async def _route(self, rec, price_ticks: int, now_ns: int) -> str:
        from order_manager import ExecutionRejected, RiskRejected
        self._quantity.set_quantities({rec.symbol_id: rec.requested_quantity})
        candidate = TradeCandidate(
            strategy=StrategyName.PORTFOLIO, symbol=rec.symbol_id, side=SignalSide.LONG,
            timestamp_ns=now_ns, reference_price_ticks=price_ticks,
            reason=f"portfolio:{rec.allocation_policy}:{rec.portfolio_decision_id}")
        correlation_id = f"{rec.symbol_id}:{now_ns}:{StrategyName.PORTFOLIO.value}"
        self.correlation_index[correlation_id] = rec.portfolio_decision_id
        try:
            submitted, conflicted = await self._router.route((candidate,))
        except RiskRejected:
            self.ledger.mark_rejected(rec.portfolio_decision_id)
            return "risk_rejected"
        except ExecutionRejected:
            self.ledger.mark_rejected(rec.portfolio_decision_id)
            return "venue_rejected"
        except Exception as exc:  # noqa: BLE001 — pipeline must never crash the loop
            return f"error:{type(exc).__name__}"
        if conflicted:
            return "conflicted"
        if submitted == 0:
            return "exposure_blocked_or_pending"
        self.ledger.mark_submitted(rec.portfolio_decision_id)
        return "routed"
