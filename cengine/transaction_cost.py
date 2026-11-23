"""Transaction-cost estimation + analysis (Milestone 4, §7/§18).

Two distinct, explicitly-separated concerns:

* ``TransactionCostEstimator`` — EXPECTED pre-trade cost from microstructure
  observations (half-spread crossing, market impact, fees, size-vs-liquidity),
  with deterministic, explicitly-conservative fallbacks when microstructure is
  unavailable. Expected costs are labeled and never mixed with realized costs.
* ``TransactionCostAnalysis`` — REALIZED post-trade outcome (implementation
  shortfall, realized slippage, latencies) computed from actual fills + genuine
  timestamps. Side-aware. Never fabricates counterfactual fills.

To avoid double-counting (§7): the estimator reports cost in bps; callers that
already fold a per-family cost into expected returns must not add it again — the
estimator is the single execution-time cost model and is kept separate from the
portfolio-optimizer's pre-trade bps estimate (which is used for ranking only).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from order_manager import Side

TCE_VERSION = "transaction-cost-1.0.0"


@dataclass(frozen=True)
class CostEstimate:
    """Expected execution cost, component-separated, in basis points of notional."""

    spread_cost_bps: float
    market_impact_bps: float
    fee_bps: float
    total_cost_bps: float
    basis: str                    # "microstructure" | "conservative_fallback"
    is_conservative_fallback: bool
    version: str = TCE_VERSION

    def total_cost_ticks(self, notional_ticks: int) -> int:
        return int(round(notional_ticks * self.total_cost_bps / 10_000.0))


@dataclass(frozen=True)
class ExecutionOutcome:
    """Realized transaction-cost analysis for one completed/terminal order (§18)."""

    symbol: str
    side: str
    decision_timestamp_ns: int
    arrival_timestamp_ns: Optional[int]
    submission_timestamp_ns: Optional[int]
    acknowledgement_timestamp_ns: Optional[int]
    first_fill_timestamp_ns: Optional[int]
    final_fill_timestamp_ns: Optional[int]
    decision_price_ticks: Optional[int]
    arrival_price_ticks: Optional[int]
    submitted_price_ticks: Optional[int]
    average_fill_price_ticks: Optional[float]
    quoted_spread_bps: Optional[float]
    expected_slippage_bps: Optional[float]
    realized_slippage_bps: Optional[float]
    implementation_shortfall_bps: Optional[float]
    requested_quantity: int
    filled_quantity: int
    unfilled_quantity: int
    acknowledgement_latency_ns: Optional[int]
    fill_latency_ns: Optional[int]
    cancellation_count: int
    repricing_count: int
    adverse_price_movement_bps: Optional[float]
    version: str = TCE_VERSION


class TransactionCostEstimator:
    """Deterministic expected-cost model. Conservative when data is missing."""

    def __init__(self, *, fee_bps: float = 0.0,
                 conservative_spread_bps: float = 10.0,
                 conservative_impact_bps: float = 5.0,
                 impact_coefficient: float = 0.1) -> None:
        # Alpaca US equities are commission-free; fee_bps defaults 0 (regulatory
        # fees are negligible and conservatively ignorable). Fallbacks are used only
        # when microstructure is unavailable and are deliberately pessimistic.
        self.fee_bps = float(fee_bps)
        self.conservative_spread_bps = float(conservative_spread_bps)
        self.conservative_impact_bps = float(conservative_impact_bps)
        self.impact_coefficient = float(impact_coefficient)

    def estimate(self, *, side: Side, quantity: int, features=None,
                 crossing: bool = True) -> CostEstimate:
        """Expected cost. ``features`` is a MicrostructureFeatures (or None).

        crossing=True assumes a marketable order paying the full half-spread to
        cross; passive placement callers pass crossing=False (half-spread credit
        approximated as a reduced spread cost)."""
        if features is None or features.spread_bps is None or features.is_stale:
            # Deterministic conservative fallback (§7).
            spread_cost = self.conservative_spread_bps * (1.0 if crossing else 0.5)
            total = spread_cost + self.conservative_impact_bps + self.fee_bps
            return CostEstimate(round(spread_cost, 4),
                                round(self.conservative_impact_bps, 4),
                                round(self.fee_bps, 4), round(total, 4),
                                "conservative_fallback", True)

        spread_bps = max(0.0, float(features.spread_bps))
        spread_cost = spread_bps * (0.5 if crossing else 0.25)  # cross half / rest quarter

        # Market impact: prefer Kyle's lambda (price impact per unit signed $volume);
        # else a conservative size-vs-liquidity sqrt model; else the fallback constant.
        impact_bps = None
        mid = features.mid
        if features.kyle_lambda is not None and mid and mid > 0 and quantity > 0:
            notional = quantity * mid
            impact_frac = abs(features.kyle_lambda) * notional / mid  # price move / mid
            impact_bps = min(100.0, impact_frac * 10_000.0)
        if impact_bps is None:
            intensity = features.trade_arrival_intensity or 0.0
            mean_sz = features.mean_trade_size or 0.0
            recent_vol = max(1.0, intensity * 60.0 * mean_sz)  # ~1min volume proxy
            participation = min(1.0, quantity / recent_vol) if recent_vol > 0 else 1.0
            impact_bps = self.impact_coefficient * 100.0 * math.sqrt(participation)
        total = spread_cost + impact_bps + self.fee_bps
        return CostEstimate(round(spread_cost, 4), round(impact_bps, 4),
                            round(self.fee_bps, 4), round(total, 4), "microstructure", False)


def _bps(side: Side, fill: float, reference: float) -> Optional[float]:
    """Side-aware cost in bps: positive = worse than reference (paid more on a BUY,
    received less on a SELL)."""
    if reference is None or fill is None or reference <= 0:
        return None
    sign = 1.0 if side is Side.BUY else -1.0
    return sign * (fill - reference) / reference * 10_000.0


def analyze_execution(order, *, side: Side, decision_timestamp_ns: int,
                      decision_price_ticks: Optional[int],
                      arrival_price_ticks: Optional[int] = None,
                      submitted_price_ticks: Optional[int] = None,
                      arrival_timestamp_ns: Optional[int] = None,
                      submission_timestamp_ns: Optional[int] = None,
                      acknowledgement_timestamp_ns: Optional[int] = None,
                      first_fill_timestamp_ns: Optional[int] = None,
                      final_fill_timestamp_ns: Optional[int] = None,
                      quoted_spread_bps: Optional[float] = None,
                      expected_slippage_bps: Optional[float] = None,
                      cancellation_count: int = 0,
                      repricing_count: int = 0) -> ExecutionOutcome:
    """Compute realized TCA from a terminal/managed order + genuine timestamps.

    ``order`` is an order_manager.ManagedOrder. Unknown prices/timestamps stay None
    (no fabricated fills). Implementation shortfall vs decision price; realized
    slippage vs arrival price; adverse movement = arrival->decision drift.
    """
    avg = order.average_fill_price_ticks
    filled = order.cumulative_filled_quantity
    target = order.approved_quantity or order.intent.quantity
    impl_shortfall = _bps(side, avg, decision_price_ticks) if avg is not None else None
    realized_slip = (_bps(side, avg, arrival_price_ticks)
                     if (avg is not None and arrival_price_ticks is not None) else None)
    adverse = (_bps(side, arrival_price_ticks, decision_price_ticks)
               if (arrival_price_ticks is not None and decision_price_ticks is not None)
               else None)
    ack_latency = (acknowledgement_timestamp_ns - submission_timestamp_ns
                   if (acknowledgement_timestamp_ns and submission_timestamp_ns) else None)
    fill_latency = (final_fill_timestamp_ns - acknowledgement_timestamp_ns
                    if (final_fill_timestamp_ns and acknowledgement_timestamp_ns) else None)
    return ExecutionOutcome(
        symbol=order.intent.symbol, side=side.value,
        decision_timestamp_ns=decision_timestamp_ns,
        arrival_timestamp_ns=arrival_timestamp_ns,
        submission_timestamp_ns=submission_timestamp_ns,
        acknowledgement_timestamp_ns=acknowledgement_timestamp_ns,
        first_fill_timestamp_ns=first_fill_timestamp_ns,
        final_fill_timestamp_ns=final_fill_timestamp_ns,
        decision_price_ticks=decision_price_ticks, arrival_price_ticks=arrival_price_ticks,
        submitted_price_ticks=submitted_price_ticks, average_fill_price_ticks=avg,
        quoted_spread_bps=quoted_spread_bps, expected_slippage_bps=expected_slippage_bps,
        realized_slippage_bps=(round(realized_slip, 4) if realized_slip is not None else None),
        implementation_shortfall_bps=(round(impl_shortfall, 4)
                                      if impl_shortfall is not None else None),
        requested_quantity=target, filled_quantity=filled,
        unfilled_quantity=max(0, target - filled),
        acknowledgement_latency_ns=ack_latency, fill_latency_ns=fill_latency,
        cancellation_count=cancellation_count, repricing_count=repricing_count,
        adverse_price_movement_bps=(round(adverse, 4) if adverse is not None else None))


__all__ = ["CostEstimate", "ExecutionOutcome", "TransactionCostEstimator",
           "analyze_execution"]
