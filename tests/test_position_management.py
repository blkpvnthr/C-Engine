"""Adaptive position management + exit arbitration (Milestone 4, §12/§13/§14)."""

from __future__ import annotations

from cengine.position_management import (
    AdaptivePositionManager,
    AtrTrailingProtection,
    BreakEvenProtection,
    ExitArbiter,
    ExitPriority,
    MomentumDeteriorationProtection,
    PositionAction,
    PositionActionType,
    PositionView,
    ProfitProtectionPolicy,
    TechnicalInvalidationExit,
    TimeBasedExit,
)
from cengine.strategy_maturity import ExecutionAuthorization, MaturityState

NONE = ExecutionAuthorization.NONE
PAPER = ExecutionAuthorization.PAPER


# ------------------------------------------------- individual policy firing
def test_atr_trailing_fires_on_giveback():
    v = PositionView("AAA", 100, unrealized_return=0.006, max_favorable_excursion=0.02,
                     atr_pct=0.005)
    a = AtrTrailingProtection(atr_multiple=2.0).evaluate(v, authorization=NONE)
    assert a is not None and a.policy == "atr_trailing"
    assert a.priority is ExitPriority.PROFIT_PROTECTION


def test_atr_trailing_silent_without_giveback():
    v = PositionView("AAA", 100, unrealized_return=0.019, max_favorable_excursion=0.02,
                     atr_pct=0.005)
    assert AtrTrailingProtection().evaluate(v, authorization=NONE) is None


def test_break_even_fires_after_arm():
    v = PositionView("AAA", 100, unrealized_return=0.0, max_favorable_excursion=0.01)
    a = BreakEvenProtection(arm_profit=0.004, cost_buffer=0.0005).evaluate(v, authorization=NONE)
    assert a is not None and a.policy == "break_even"


def test_momentum_deterioration_fires():
    v = PositionView("AAA", 100, unrealized_return=0.006, momentum_accel=-0.001,
                     buyer_exhaustion=0.6)
    a = MomentumDeteriorationProtection().evaluate(v, authorization=NONE)
    assert a is not None and a.policy == "momentum_deterioration"


def test_momentum_deterioration_silent_when_flat():
    v = PositionView("AAA", 100, unrealized_return=-0.001, momentum_accel=-0.001,
                     buyer_exhaustion=0.9)
    assert MomentumDeteriorationProtection().evaluate(v, authorization=NONE) is None


def test_time_based_exit_fires_when_stagnant():
    v = PositionView("AAA", 100, unrealized_return=0.0005, holding_duration_s=7 * 3600)
    a = TimeBasedExit(max_holding_s=6 * 3600, min_move=0.002).evaluate(v, authorization=NONE)
    assert a is not None and a.priority is ExitPriority.TECHNICAL_EXIT


def test_technical_invalidation_fires():
    v = PositionView("AAA", 100, strategy_invalidated=True)
    a = TechnicalInvalidationExit().evaluate(v, authorization=NONE)
    assert a is not None and a.reason == "strategy_invalidated"


# ------------------------------------------------- shadow vs authorized
def test_default_authorization_is_shadow_only():
    v = PositionView("AAA", 100, unrealized_return=0.006, max_favorable_excursion=0.02,
                     atr_pct=0.005)
    mgr = AdaptivePositionManager()  # authorization NONE
    acts = mgr.evaluate(v)
    assert acts and all(a.shadow_only for a in acts)


def test_authorized_policy_is_not_shadow():
    v = PositionView("AAA", 100, unrealized_return=0.006, max_favorable_excursion=0.02,
                     atr_pct=0.005)
    mgr = AdaptivePositionManager(
        policies=[AtrTrailingProtection(maturity=MaturityState.PAPER_ELIGIBLE)],
        authorization=PAPER)
    acts = mgr.evaluate(v)
    assert acts and acts[0].shadow_only is False


# ------------------------------------------------- exit arbitration (§14)
def test_mandatory_outranks_discretionary():
    arb = ExitArbiter()
    disc = PositionAction("AAA", ExitPriority.PROFIT_PROTECTION,
                          PositionActionType.EXIT_FULL, "atr", shadow_only=True,
                          policy="atr_trailing")
    mand = PositionAction("AAA", ExitPriority.MANDATORY_PROTECTIVE, PositionActionType.EXIT_FULL,
                          "mandatory_-0.5pct", shadow_only=False)
    chosen = arb.arbitrate([disc, mand])
    assert chosen.priority is ExitPriority.MANDATORY_PROTECTIVE


def test_exit_outranks_entry_same_priority():
    arb = ExitArbiter()
    exit_a = PositionAction("AAA", ExitPriority.POSITION_ADJUST,
                            PositionActionType.EXIT_PARTIAL, "x")
    entry = PositionAction("AAA", ExitPriority.POSITION_ADJUST,
                           PositionActionType.ENTRY, "y")
    assert arb.arbitrate([entry, exit_a]).action_type in (
        PositionActionType.EXIT_PARTIAL, PositionActionType.EXIT_FULL)


def test_empty_arbitration_is_none():
    assert ExitArbiter().arbitrate([]) is None


def test_conflict_detection():
    arb = ExitArbiter()
    actions = [
        PositionAction("AAA", ExitPriority.MANDATORY_PROTECTIVE, PositionActionType.EXIT_FULL, "m"),
        PositionAction("AAA", ExitPriority.NEW_ENTRY, PositionActionType.ENTRY, "e"),
    ]
    assert arb.conflicts(actions) is True
    assert arb.conflicts([actions[0]]) is False


# ------------------------------------------------- robustness
def test_raising_policy_is_swallowed():
    class _Boom(ProfitProtectionPolicy):
        name = "boom"

        def evaluate(self, view, *, authorization):
            raise RuntimeError("boom")

    v = PositionView("AAA", 100, unrealized_return=0.006, max_favorable_excursion=0.02,
                     atr_pct=0.005)
    mgr = AdaptivePositionManager(policies=[_Boom(), AtrTrailingProtection()])
    acts = mgr.evaluate(v)  # must not raise
    assert any(a.policy == "atr_trailing" for a in acts)
