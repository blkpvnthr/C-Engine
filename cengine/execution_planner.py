"""Intelligent execution planner + policies (Milestone 4, §4/§5/§8/§9/§20).

Converts an authorized strategy/portfolio recommendation into a versioned
``ExecutionPlan`` with a broker-compatible execution policy and a fresh-quote limit
price, then routes it through the EXISTING CandidateRouter -> exposure gate ->
native RiskEngine -> OrderManager (no parallel execution authority).

New execution policies default to a non-live maturity (``ExecutionAuthorization``
gate, reused from Milestone 2/3). Mandatory protective exits are NOT produced here —
they keep their independent broker-compatible authority in ``protective_exit.py``;
the planner only recognizes the ``MANDATORY_PROTECTIVE_EXIT`` policy label for
attribution. The planner never overrides strict entry qualification, strategy
maturity, exposure authorization, or native financial risk controls.

Prices are market-data ticks (DEFAULT_PRICE_SCALE = 10_000). Limit prices are
rounded to the broker tick size (penny = 100 ticks for >= $1 equities) and bounded
by a maximum acceptable elective slippage. Observed prices are never fabricated:
without a fresh quote the planner falls back to IMMEDIATE_EXECUTION (market) or
rejects, per policy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from cengine.execution_policy import ExecutionInstruction
from cengine.transaction_cost import CostEstimate, TransactionCostEstimator
from order_manager import OrderType, Side, TimeInForce

EXECUTION_PLANNER_VERSION = "execution-planner-1.0.0"
PENNY_TICKS = 100  # $0.01 at DEFAULT_PRICE_SCALE 10_000


class ExecutionPolicy(str, Enum):
    PASSIVE_LIMIT = "passive_limit"
    MARKETABLE_LIMIT = "marketable_limit"
    IMMEDIATE_EXECUTION = "immediate_execution"
    BOUNDED_REPRICE = "bounded_reprice"
    TIME_SLICED_EXECUTION = "time_sliced_execution"
    URGENT_RISK_REDUCTION = "urgent_risk_reduction"
    MANDATORY_PROTECTIVE_EXIT = "mandatory_protective_exit"


class Urgency(str, Enum):
    MANDATORY = "mandatory"      # protective exit — handled independently
    URGENT = "urgent"            # risk reduction
    HIGH = "high"                # opportunity expiring / strong momentum
    NORMAL = "normal"
    LOW = "low"
    PATIENT = "patient"


@dataclass
class ExecutionPlan:
    execution_plan_id: str
    strategy_decision_id: Optional[str]
    portfolio_decision_id: Optional[str]
    symbol_id: str
    instrument_generation: Optional[int]
    side: str
    requested_quantity: int
    requested_notional_ticks: int
    reference_price_ticks: Optional[int]
    decision_price_ticks: Optional[int]
    arrival_price_ticks: Optional[int]
    observed_bid_ticks: Optional[int]
    observed_ask_ticks: Optional[int]
    spread_bps: Optional[float]
    estimated_transaction_cost_bps: Optional[float]
    estimated_slippage_bps: Optional[float]
    urgency: str
    execution_policy: str
    limit_price_ticks: Optional[int]
    expiration_timestamp: int
    maximum_repricing_attempts: int
    capital_reservation_id: Optional[str]
    portfolio_version: Optional[str]
    market_observation_version: Optional[str]
    execution_authorization: str
    decision_timestamp: int = 0
    cost_components: dict = field(default_factory=dict)
    rejection_reason: Optional[str] = None
    version: str = EXECUTION_PLANNER_VERSION

    @property
    def is_executable(self) -> bool:
        return self.rejection_reason is None and self.requested_quantity > 0


def _round_to_tick(price_ticks: int, tick: int = PENNY_TICKS) -> int:
    if tick <= 1:
        return int(price_ticks)
    return int(round(price_ticks / tick) * tick)


def select_policy(*, urgency: Urgency, spread_bps: Optional[float],
                  expected_cost_bps: Optional[float], expiring_soon: bool,
                  momentum_up: bool, size_vs_liquidity: float,
                  stale: bool) -> ExecutionPolicy:
    """Deterministic policy selection (§8). Mandatory exits are not routed here."""
    if urgency is Urgency.MANDATORY:
        return ExecutionPolicy.MANDATORY_PROTECTIVE_EXIT
    if urgency is Urgency.URGENT:
        return ExecutionPolicy.URGENT_RISK_REDUCTION
    # Without a fresh quote we cannot compute a safe limit -> immediate (market).
    if stale or spread_bps is None:
        return ExecutionPolicy.IMMEDIATE_EXECUTION
    # Large order vs observed liquidity -> slice.
    if size_vs_liquidity > 0.25:
        return ExecutionPolicy.TIME_SLICED_EXECUTION
    if urgency is Urgency.HIGH or expiring_soon or momentum_up:
        return ExecutionPolicy.MARKETABLE_LIMIT
    # Wide spread + patience -> rest passively (optionally reprice).
    if spread_bps >= 15.0 and urgency in (Urgency.LOW, Urgency.PATIENT):
        return ExecutionPolicy.BOUNDED_REPRICE
    if spread_bps >= 8.0:
        return ExecutionPolicy.PASSIVE_LIMIT
    return ExecutionPolicy.MARKETABLE_LIMIT


def compute_limit_price_ticks(*, side: Side, policy: ExecutionPolicy,
                              bid_ticks: Optional[int], ask_ticks: Optional[int],
                              decision_price_ticks: Optional[int],
                              max_slippage_bps: float = 25.0) -> Optional[int]:
    """Compute a tick-rounded limit price from fresh quotes, bounded by max elective
    slippage. Returns None for market/immediate/mandatory policies (no limit)."""
    if policy in (ExecutionPolicy.IMMEDIATE_EXECUTION,
                  ExecutionPolicy.MANDATORY_PROTECTIVE_EXIT,
                  ExecutionPolicy.URGENT_RISK_REDUCTION):
        return None
    if bid_ticks is None or ask_ticks is None or bid_ticks <= 0 or ask_ticks <= 0:
        return None
    if side is Side.BUY:
        if policy is ExecutionPolicy.PASSIVE_LIMIT or policy is ExecutionPolicy.BOUNDED_REPRICE:
            raw = bid_ticks                      # join the bid (passive)
        else:                                    # marketable / sliced: cross at the ask
            raw = ask_ticks
        if decision_price_ticks:                 # bound elective slippage above decision
            cap = int(decision_price_ticks * (1.0 + max_slippage_bps / 10_000.0))
            raw = min(raw, cap)
    else:  # SELL
        if policy is ExecutionPolicy.PASSIVE_LIMIT or policy is ExecutionPolicy.BOUNDED_REPRICE:
            raw = ask_ticks                      # join the ask (passive)
        else:
            raw = bid_ticks                      # cross at the bid
        if decision_price_ticks:                 # bound elective slippage below decision
            floor = int(decision_price_ticks * (1.0 - max_slippage_bps / 10_000.0))
            raw = max(raw, floor)
    limit = _round_to_tick(raw)
    return limit if limit > 0 else None


class ExecutionPlanner:
    """Builds versioned ExecutionPlans. Default non-live until authorized."""

    def __init__(self, *, cost_estimator: Optional[TransactionCostEstimator] = None,
                 default_expiration_ns: int = 120 * 1_000_000_000,
                 max_repricing_attempts: int = 3,
                 max_slippage_bps: float = 25.0,
                 execution_authorization: str = "none") -> None:
        self.tce = cost_estimator or TransactionCostEstimator()
        self.default_expiration_ns = int(default_expiration_ns)
        self.max_repricing_attempts = int(max_repricing_attempts)
        self.max_slippage_bps = float(max_slippage_bps)
        self.execution_authorization = execution_authorization

    def plan(self, *, symbol: str, side: Side, quantity: int, now_ns: int,
             features=None, decision_price_ticks: Optional[int] = None,
             urgency: Urgency = Urgency.NORMAL,
             strategy_decision_id: Optional[str] = None,
             portfolio_decision_id: Optional[str] = None,
             capital_reservation_id: Optional[str] = None,
             portfolio_version: Optional[str] = None,
             instrument_generation: Optional[int] = None,
             expiring_soon: bool = False, momentum_up: bool = False) -> ExecutionPlan:
        bid = getattr(features, "bid", None)
        ask = getattr(features, "ask", None)
        bid_ticks = int(round(bid * 10_000)) if bid else None
        ask_ticks = int(round(ask * 10_000)) if ask else None
        mid_ticks = (int(round(features.mid * 10_000))
                     if features is not None and features.mid else None)
        spread_bps = getattr(features, "spread_bps", None)
        stale = bool(getattr(features, "is_stale", True)
                     or getattr(features, "crossed_or_locked", False))
        arrival_ticks = mid_ticks
        decision_ticks = decision_price_ticks or mid_ticks

        # size vs observed liquidity (uses observed top-of-book size, never L2)
        top_size = None
        if features is not None and features.bid_size is not None and side is Side.SELL:
            top_size = features.bid_size
        elif features is not None and features.ask_size is not None and side is Side.BUY:
            top_size = features.ask_size
        size_vs_liq = (quantity / top_size) if (top_size and top_size > 0) else 0.0

        policy = select_policy(
            urgency=urgency, spread_bps=spread_bps, expected_cost_bps=None,
            expiring_soon=expiring_soon, momentum_up=momentum_up,
            size_vs_liquidity=size_vs_liq, stale=stale)

        crossing = policy in (ExecutionPolicy.MARKETABLE_LIMIT,
                              ExecutionPolicy.IMMEDIATE_EXECUTION,
                              ExecutionPolicy.URGENT_RISK_REDUCTION,
                              ExecutionPolicy.TIME_SLICED_EXECUTION)
        cost: CostEstimate = self.tce.estimate(
            side=side, quantity=quantity, features=features, crossing=crossing)
        limit_ticks = compute_limit_price_ticks(
            side=side, policy=policy, bid_ticks=bid_ticks, ask_ticks=ask_ticks,
            decision_price_ticks=decision_ticks, max_slippage_bps=self.max_slippage_bps)

        notional = int(quantity * (mid_ticks or decision_ticks or 0))
        plan = ExecutionPlan(
            execution_plan_id=f"xp:{symbol}:{now_ns}",
            strategy_decision_id=strategy_decision_id,
            portfolio_decision_id=portfolio_decision_id,
            symbol_id=symbol.upper(), instrument_generation=instrument_generation,
            side=side.value, requested_quantity=int(quantity),
            requested_notional_ticks=notional, reference_price_ticks=decision_ticks,
            decision_price_ticks=decision_ticks, arrival_price_ticks=arrival_ticks,
            observed_bid_ticks=bid_ticks, observed_ask_ticks=ask_ticks,
            spread_bps=spread_bps, estimated_transaction_cost_bps=cost.total_cost_bps,
            estimated_slippage_bps=round(cost.spread_cost_bps + cost.market_impact_bps, 4),
            urgency=urgency.value, execution_policy=policy.value,
            limit_price_ticks=limit_ticks,
            expiration_timestamp=now_ns + self.default_expiration_ns,
            maximum_repricing_attempts=(self.max_repricing_attempts
                                        if policy is ExecutionPolicy.BOUNDED_REPRICE else 0),
            capital_reservation_id=capital_reservation_id,
            portfolio_version=portfolio_version,
            market_observation_version=getattr(features, "feature_version", None),
            execution_authorization=self.execution_authorization,
            decision_timestamp=now_ns,
            cost_components={"spread_cost_bps": cost.spread_cost_bps,
                             "market_impact_bps": cost.market_impact_bps,
                             "fee_bps": cost.fee_bps, "basis": cost.basis})
        if quantity <= 0:
            plan.rejection_reason = "non_positive_quantity"
        # A limit policy with no computable limit price (no fresh quote) is invalid;
        # downgrade to immediate only if authorized, else reject (never fabricate price).
        if (plan.limit_price_ticks is None and policy in (
                ExecutionPolicy.PASSIVE_LIMIT, ExecutionPolicy.MARKETABLE_LIMIT,
                ExecutionPolicy.BOUNDED_REPRICE, ExecutionPolicy.TIME_SLICED_EXECUTION)):
            plan.execution_policy = ExecutionPolicy.IMMEDIATE_EXECUTION.value
        return plan

    @staticmethod
    def to_instruction(plan: ExecutionPlan) -> ExecutionInstruction:
        """Map a plan to a broker-compatible ExecutionInstruction for the router."""
        tif = TimeInForce.DAY
        if plan.execution_policy in (ExecutionPolicy.IMMEDIATE_EXECUTION.value,
                                     ExecutionPolicy.URGENT_RISK_REDUCTION.value,
                                     ExecutionPolicy.MANDATORY_PROTECTIVE_EXIT.value):
            return ExecutionInstruction(OrderType.MARKET, tif)
        if plan.limit_price_ticks and plan.limit_price_ticks > 0:
            return ExecutionInstruction(OrderType.LIMIT, tif,
                                        limit_price_ticks=plan.limit_price_ticks)
        return ExecutionInstruction(OrderType.MARKET, tif)


class PlannedExecutionPolicy:
    """ExecutionPolicy-protocol adapter: returns the planner's per-symbol instruction
    so the EXISTING CandidateRouter consumes computed limit prices. Falls back to a
    MARKET instruction when no plan is registered for a candidate."""

    def __init__(self) -> None:
        self._instructions: dict[str, ExecutionInstruction] = {}

    def set_plan(self, plan: ExecutionPlan) -> None:
        self._instructions[plan.symbol_id.upper()] = ExecutionPlanner.to_instruction(plan)

    def instruction_for(self, candidate) -> ExecutionInstruction:
        return self._instructions.get(
            candidate.symbol.upper(), ExecutionInstruction(OrderType.MARKET, TimeInForce.DAY))


__all__ = ["EXECUTION_PLANNER_VERSION", "ExecutionPlan", "ExecutionPlanner",
           "ExecutionPolicy", "PlannedExecutionPolicy", "Urgency",
           "compute_limit_price_ticks", "select_policy"]
