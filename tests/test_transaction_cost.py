"""Transaction-cost estimation + analysis tests (Milestone 4, §7/§18).

Expected (pre-trade) and realized (post-trade) costs are kept separate; realized
cost is side-aware; conservative fallbacks are deterministic.
"""

from __future__ import annotations

from cengine.microstructure import MicrostructureFeatures
from cengine.transaction_cost import (
    CostEstimate,
    ExecutionOutcome,
    TransactionCostEstimator,
    analyze_execution,
)
from order_manager import ManagedOrder, OrderIntent, OrderType, Side, TimeInForce


def _features(**over):
    base = dict(symbol="AAA", spread_bps=4.0, mid=100.0, kyle_lambda=1e-9,
                trade_arrival_intensity=2.0, mean_trade_size=300.0, is_stale=False)
    base.update(over)
    return MicrostructureFeatures(**base)


def _order(*, side=Side.BUY, qty=100, avg_fill=1_000_200.0, filled=100, approved=100):
    intent = OrderIntent(symbol="AAA", side=side, quantity=qty, order_type=OrderType.MARKET,
                         time_in_force=TimeInForce.DAY, strategy_id="x",
                         client_order_id="c1", created_ns=1)
    return ManagedOrder(intent=intent, client_order_id="c1",
                        average_fill_price_ticks=avg_fill,
                        cumulative_filled_quantity=filled, approved_quantity=approved)


# ---------------------------------------------------------------- expected cost
def test_microstructure_based_estimate_components():
    tce = TransactionCostEstimator()
    est = tce.estimate(side=Side.BUY, quantity=100, features=_features(), crossing=True)
    assert isinstance(est, CostEstimate)
    assert est.basis == "microstructure" and est.is_conservative_fallback is False
    # components separated and summing to the total
    assert abs(est.total_cost_bps
               - (est.spread_cost_bps + est.market_impact_bps + est.fee_bps)) < 1e-6
    # crossing pays half the spread
    assert abs(est.spread_cost_bps - 4.0 * 0.5) < 1e-6


def test_conservative_fallback_when_no_features():
    tce = TransactionCostEstimator()
    est = tce.estimate(side=Side.BUY, quantity=100, features=None, crossing=True)
    assert est.basis == "conservative_fallback" and est.is_conservative_fallback is True
    assert est.total_cost_bps == tce.conservative_spread_bps + tce.conservative_impact_bps
    # deterministic
    est2 = tce.estimate(side=Side.BUY, quantity=100, features=None, crossing=True)
    assert est2.total_cost_bps == est.total_cost_bps


def test_stale_features_use_conservative_fallback():
    tce = TransactionCostEstimator()
    est = tce.estimate(side=Side.BUY, quantity=10, features=_features(is_stale=True))
    assert est.is_conservative_fallback is True


def test_crossing_vs_passive_spread_cost():
    tce = TransactionCostEstimator()
    cross = tce.estimate(side=Side.BUY, quantity=100, features=_features(), crossing=True)
    passive = tce.estimate(side=Side.BUY, quantity=100, features=_features(), crossing=False)
    assert cross.spread_cost_bps > passive.spread_cost_bps


def test_total_cost_ticks():
    est = TransactionCostEstimator().estimate(side=Side.BUY, quantity=100,
                                              features=_features(), crossing=True)
    # 1_000_000 notional ticks * total_bps/10000
    assert est.total_cost_ticks(1_000_000) == int(round(1_000_000 * est.total_cost_bps / 10_000))


# ---------------------------------------------------------------- realized cost
def test_buy_implementation_shortfall_positive_when_paid_up():
    o = _order(side=Side.BUY, avg_fill=1_000_200.0)  # paid above decision
    out = analyze_execution(o, side=Side.BUY, decision_timestamp_ns=1000,
                            decision_price_ticks=1_000_000, arrival_price_ticks=1_000_100,
                            submission_timestamp_ns=1100, acknowledgement_timestamp_ns=1200,
                            final_fill_timestamp_ns=1500, quoted_spread_bps=2.0,
                            expected_slippage_bps=1.0)
    assert isinstance(out, ExecutionOutcome)
    assert out.implementation_shortfall_bps is not None and out.implementation_shortfall_bps > 0
    assert out.realized_slippage_bps is not None and out.realized_slippage_bps > 0
    assert out.adverse_price_movement_bps is not None and out.adverse_price_movement_bps > 0
    assert out.acknowledgement_latency_ns == 100 and out.fill_latency_ns == 300
    assert out.unfilled_quantity == 0 and out.filled_quantity == 100
    # expected is passed through, not recomputed from realized
    assert out.expected_slippage_bps == 1.0


def test_sell_cost_is_side_aware():
    # SELL filled BELOW decision -> positive cost (received less)
    o = _order(side=Side.SELL, avg_fill=999_800.0)
    out = analyze_execution(o, side=Side.SELL, decision_timestamp_ns=1000,
                            decision_price_ticks=1_000_000, arrival_price_ticks=999_900)
    assert out.implementation_shortfall_bps > 0
    assert out.realized_slippage_bps > 0


def test_partial_fill_unfilled_quantity():
    o = _order(side=Side.BUY, avg_fill=1_000_000.0, filled=40, approved=100)
    out = analyze_execution(o, side=Side.BUY, decision_timestamp_ns=1,
                            decision_price_ticks=1_000_000)
    assert out.filled_quantity == 40 and out.unfilled_quantity == 60


def test_unknown_prices_stay_none():
    o = _order(side=Side.BUY, avg_fill=1_000_200.0)
    out = analyze_execution(o, side=Side.BUY, decision_timestamp_ns=1,
                            decision_price_ticks=None)
    assert out.implementation_shortfall_bps is None  # no decision price -> unknown
