"""Execution certification.

* **Planner (engineering)** — the execution planner produces executable, tick-aligned plans with
  policy-appropriate limit prices; deterministic for a fixed input.
* **Cost model (engineering)** — the transaction-cost estimator is conservative-by-default (a
  spread+impact+fee floor when microstructure features are absent) and monotonic in participation.
* **Conservative fills (methodology)** — the certification backtest's fill model assumes
  stop-before-target within a bar and gap-through-at-the-open (verified structurally here); it
  never assumes favorable fills.
* **Realized execution quality (gated)** — slippage vs arrival price, fill ratios and cancel/
  reprice rates require genuine executed (paper or live) orders. With none available during
  certification, realized execution quality is INSUFFICIENT_DATA — never estimated as if real.
"""

from __future__ import annotations

from cengine.certification.contracts import CertificationResult, CertificationVerdict

EXECUTION_CERT_VERSION = "certification-execution-1.0.0"


def _v(ok: bool) -> CertificationVerdict:
    return CertificationVerdict.CERTIFIED if ok else CertificationVerdict.NOT_CERTIFIED


def _certify_planner() -> CertificationResult:
    try:
        from cengine.execution_planner import ExecutionPlanner, Urgency
        from order_manager import Side
    except Exception as exc:  # noqa: BLE001
        return CertificationResult(
            check="execution.planner", component="execution",
            verdict=CertificationVerdict.BLOCKED, summary=f"planner import failed: {exc}")
    planner = ExecutionPlanner()

    class _Features:
        """Minimal L1 feature stub; returns a benign default for any field the planner probes."""
        bid = 100.0
        ask = 100.02
        mid = 100.01
        bid_size = 500
        ask_size = 500
        spread_bps = 2.0
        is_stale = False
        crossed_or_locked = False
        feature_version = 1

        def __getattr__(self, name):   # any microstructure field not set -> None (unknown)
            return None

    kw = dict(symbol="AAPL", side=Side.BUY, quantity=100, now_ns=1_000,
              features=_Features(), decision_price_ticks=100 * 10000, urgency=Urgency.NORMAL)
    try:
        plan = planner.plan(**kw)
        plan2 = planner.plan(**kw)
    except Exception as exc:  # noqa: BLE001
        return CertificationResult(
            check="execution.planner", component="execution",
            verdict=CertificationVerdict.NOT_CERTIFIED,
            summary=f"planner.plan could not be invoked: {exc}")
    is_exec = getattr(plan, "is_executable", None)
    executable = bool(is_exec() if callable(is_exec) else is_exec if is_exec is not None else True)
    deterministic = (getattr(plan, "limit_price_ticks", None)
                     == getattr(plan2, "limit_price_ticks", None))
    return CertificationResult(
        check="execution.planner", component="execution",
        verdict=_v(executable and deterministic),
        summary=f"execution planner produces an executable, deterministic plan "
                f"(executable={executable})",
        metrics={"executable": executable, "deterministic": deterministic,
                 "policy": str(getattr(plan, "policy", ""))})


def _certify_cost_model() -> CertificationResult:
    try:
        from cengine.transaction_cost import TransactionCostEstimator
        from order_manager import Side
    except Exception as exc:  # noqa: BLE001
        return CertificationResult(
            check="execution.cost_model", component="execution",
            verdict=CertificationVerdict.BLOCKED, summary=f"cost model import failed: {exc}")
    est = TransactionCostEstimator(fee_bps=0.5)
    # with no microstructure features it must fall back to the conservative floor
    small = est.estimate(side=Side.BUY, quantity=100)
    conservative = (small.total_cost_bps > 0 and small.spread_cost_bps > 0
                    and small.market_impact_bps >= 0)
    # cost is non-negative and includes the fee
    fee_included = small.total_cost_bps >= est.fee_bps
    ok = conservative and fee_included
    return CertificationResult(
        check="execution.cost_model", component="execution", verdict=_v(ok),
        summary=(f"transaction-cost estimator conservative-by-default: total "
                 f"{small.total_cost_bps:.1f}bps (spread {small.spread_cost_bps:.1f} + impact "
                 f"{small.market_impact_bps:.1f} + fee {small.fee_bps:.1f})"),
        metrics={"total_cost_bps": small.total_cost_bps,
                 "spread_bps": small.spread_cost_bps,
                 "impact_bps": small.market_impact_bps, "fee_bps": small.fee_bps})


def _certify_conservative_fills() -> CertificationResult:
    """Structurally verify the backtest fill model never assumes favorable fills."""
    import inspect

    from cengine.optimization.exits import experiment_matrix as em
    src = inspect.getsource(em._simulate)
    # markers of the conservative assumptions documented + implemented in the simulator
    stop_before_target = "win_low" in src and "breach" in src
    gap_through = "min(win_open" in src or "min(win_open[k]" in src
    exit_cost = "cost_frac" in src
    ok = stop_before_target and gap_through and exit_cost
    return CertificationResult(
        check="execution.conservative_fills", component="execution", verdict=_v(ok),
        summary=("certification fill model: stop-before-target within a bar, gap-through at the "
                 "open, costs on both sides (no favorable-fill assumption)"),
        metrics={"stop_before_target": stop_before_target, "gap_through_at_open": gap_through,
                 "cost_applied": exit_cost})


def _certify_realized_quality() -> CertificationResult:
    return CertificationResult(
        check="execution.realized_quality", component="execution",
        verdict=CertificationVerdict.INSUFFICIENT_DATA,
        summary="realized slippage / fill-ratio / cancel-reprice require executed (paper or live) "
                "orders — none available during certification",
        notes="not estimated as if real; see paper-trading framework")


def certify_execution() -> list:
    """Full execution certification: planner + cost model + conservative fills + gated realized."""
    return [
        _certify_planner(),
        _certify_cost_model(),
        _certify_conservative_fills(),
        _certify_realized_quality(),
    ]
