"""Intelligent portfolio replacement workflow (Milestone 3, §13)."""

from __future__ import annotations

from cengine.portfolio_replacement import ReplacementCoordinator, ReplacementStage
from market_scan.replacement import ReplacementEligibility


class _Clock:
    def __init__(self, t=1_000):
        self.t = t

    def __call__(self):
        return self.t


def _eligible() -> ReplacementEligibility:
    return ReplacementEligibility(
        seller_exhaustion=True, positive_momentum=True, di_plus_gt_minus=True,
        above_bb_mid=True, above_vwap=True, liquidity_ok=True, regime_aligned=True,
        risk_ok=True, require_seller_exhaustion=True)


def _coordinator(clock=None, cooldown=10_000):
    return ReplacementCoordinator(no_reentry_cooldown_ns=cooldown,
                                  now_fn=clock or _Clock())


def test_full_workflow_stages():
    clk = _Clock(1_000)
    c = _coordinator(clk)
    c.on_mandatory_exit("NVDA")
    assert c.record("NVDA").stage is ReplacementStage.MANDATORY_EXIT_TRIGGERED
    c.on_exit_submitted("NVDA")
    assert c.record("NVDA").stage is ReplacementStage.EXIT_ORDER_SUBMITTED
    c.on_exit_partial("NVDA")
    assert c.record("NVDA").stage is ReplacementStage.EXIT_PARTIALLY_FILLED
    c.on_exit_filled("NVDA")
    assert c.record("NVDA").stage is ReplacementStage.EXIT_FULLY_FILLED
    assert not c.ready_for_replacement("NVDA")   # not until reconciled
    c.on_broker_reconciled("NVDA")
    assert c.record("NVDA").stage is ReplacementStage.BROKER_RECONCILED
    assert c.ready_for_replacement("NVDA")


def test_evaluate_before_reconciliation_rejected():
    c = _coordinator()
    c.on_mandatory_exit("NVDA")
    c.on_exit_submitted("NVDA")
    d = c.evaluate_replacement("NVDA", ranked_universe=["AAPL"],
                               evaluate=lambda s: _eligible(), held=set(), pending=set(),
                               capital_available_ticks=1_000_000)
    assert d.authorized is False
    assert d.reason == "exit_not_reconciled"


def test_reconciled_selects_eligible_replacement():
    clk = _Clock(1_000)
    c = _coordinator(clk)
    c.on_mandatory_exit("NVDA")
    c.on_exit_filled("NVDA")
    c.on_broker_reconciled("NVDA")

    def evaluate(sym):
        return _eligible() if sym == "AAPL" else ReplacementEligibility()

    d = c.evaluate_replacement("NVDA", ranked_universe=["MSFT", "AAPL"],
                               evaluate=evaluate, held=set(), pending=set(),
                               capital_available_ticks=5_000_000)
    assert d.authorized is True
    assert d.replacement_symbol == "AAPL"
    assert c.record("NVDA").stage is ReplacementStage.REPLACEMENT_AUTHORIZED


def test_no_eligible_candidate_holds_cash():
    clk = _Clock(1_000)
    c = _coordinator(clk)
    c.on_mandatory_exit("NVDA")
    c.on_exit_filled("NVDA")
    c.on_broker_reconciled("NVDA")
    d = c.evaluate_replacement("NVDA", ranked_universe=["AAPL", "MSFT"],
                               evaluate=lambda s: ReplacementEligibility(),  # none eligible
                               held=set(), pending=set(), capital_available_ticks=5_000_000)
    assert d.authorized is False
    assert c.record("NVDA").stage is ReplacementStage.HELD_CASH


def test_no_capital_holds_cash():
    clk = _Clock(1_000)
    c = _coordinator(clk)
    c.on_mandatory_exit("NVDA")
    c.on_exit_filled("NVDA")
    c.on_broker_reconciled("NVDA")
    d = c.evaluate_replacement("NVDA", ranked_universe=["AAPL"],
                               evaluate=lambda s: _eligible(), held=set(), pending=set(),
                               capital_available_ticks=0)
    assert d.authorized is False
    assert d.reason == "no_capital_available"
    assert c.record("NVDA").stage is ReplacementStage.HELD_CASH


def test_no_reentry_cooldown_excludes_just_exited_and_cooling_symbols():
    clk = _Clock(1_000)
    c = _coordinator(clk, cooldown=10_000)
    # NVDA just stopped out -> cooldown starts
    c.on_mandatory_exit("NVDA")
    c.on_exit_filled("NVDA")
    assert c.in_cooldown("NVDA")
    # A different exited name wants a replacement; NVDA (still cooling) must be excluded
    c.on_mandatory_exit("TSLA")
    c.on_exit_filled("TSLA", now_ns=1_001)
    c.on_broker_reconciled("TSLA", now_ns=1_002)
    d = c.evaluate_replacement("TSLA", ranked_universe=["NVDA", "AAPL"],
                               evaluate=lambda s: _eligible(),  # both would be eligible
                               held=set(), pending=set(), capital_available_ticks=5_000_000,
                               now_ns=1_003)
    # NVDA is in cooldown -> skipped; AAPL chosen instead (never re-enter the stopped name)
    assert d.replacement_symbol == "AAPL"


def test_cooldown_expires():
    clk = _Clock(1_000)
    c = _coordinator(clk, cooldown=10_000)
    c.on_mandatory_exit("NVDA")
    c.on_exit_filled("NVDA")
    assert c.in_cooldown("NVDA", now_ns=5_000)
    assert not c.in_cooldown("NVDA", now_ns=99_999)


def test_on_replacement_submitted_advances_stage():
    clk = _Clock(1_000)
    c = _coordinator(clk)
    c.on_mandatory_exit("NVDA")
    c.on_exit_filled("NVDA")
    c.on_broker_reconciled("NVDA")
    c.evaluate_replacement("NVDA", ranked_universe=["AAPL"],
                           evaluate=lambda s: _eligible(), held=set(), pending=set(),
                           capital_available_ticks=5_000_000)
    c.on_replacement_submitted("NVDA")
    assert c.record("NVDA").stage is ReplacementStage.REPLACEMENT_ORDER_SUBMITTED
