"""Maturity + execution-authorization gate (directive §4, §16).

Verifies the safety-critical invariant: isolated MOCK authorization never implies
paper/live routing, BLOCKED is a hard stop, and promotion is explicit.
"""

from __future__ import annotations

from cengine.strategy_families import StrategyFamily, StrategyFamilyRegistry
from cengine.strategy_maturity import (
    ExecutionAuthorization,
    MaturityState,
    authorization_permits,
)

A = ExecutionAuthorization
M = MaturityState


def test_none_authorization_routes_nothing():
    for m in MaturityState:
        assert authorization_permits(m, A.NONE) is False


def test_mock_authorization_permits_only_mock_and_above():
    assert authorization_permits(M.MOCK_EXECUTION, A.MOCK) is True
    assert authorization_permits(M.PAPER_ELIGIBLE, A.MOCK) is True
    assert authorization_permits(M.LIVE_ELIGIBLE, A.MOCK) is True
    for m in (M.RESEARCH, M.SHADOW, M.REPLAY_VALIDATED, M.DISABLED):
        assert authorization_permits(m, A.MOCK) is False


def test_mock_only_family_is_never_paper_or_live_routable():
    # The crux: exercising a family in the mock harness must not make it paper/live.
    assert authorization_permits(M.MOCK_EXECUTION, A.PAPER) is False
    assert authorization_permits(M.MOCK_EXECUTION, A.LIVE) is False


def test_paper_and_live_gates():
    assert authorization_permits(M.PAPER_ELIGIBLE, A.PAPER) is True
    assert authorization_permits(M.LIVE_ELIGIBLE, A.PAPER) is True
    assert authorization_permits(M.PAPER_ELIGIBLE, A.LIVE) is False  # paper != live
    assert authorization_permits(M.LIVE_ELIGIBLE, A.LIVE) is True


def test_blocked_overrides_every_authorization():
    for a in ExecutionAuthorization:
        assert authorization_permits(M.BLOCKED, a) is False


def test_registry_defaults_all_research_maturity():
    reg = StrategyFamilyRegistry()
    for fam in reg.families():
        assert reg.maturity(fam) is M.RESEARCH


def test_mark_mock_requires_tests_and_explicit_authorization():
    reg = StrategyFamilyRegistry()
    fam = StrategyFamily.MOMENTUM_CONTINUATION
    # no authorization -> unchanged
    assert reg.mark_mock_execution_eligible(fam, test_authorization=False) is M.RESEARCH
    # explicit authorization -> mock
    assert reg.mark_mock_execution_eligible(fam, test_authorization=True) is M.MOCK_EXECUTION
    # and it is NOT advisory execution-eligible (live gate untouched)
    assert reg.is_execution_eligible(fam) is False


def test_mock_promotion_does_not_grant_paper_or_live():
    reg = StrategyFamilyRegistry()
    fam = StrategyFamily.VOLATILITY_BREAKOUT
    reg.mark_mock_execution_eligible(fam, test_authorization=True)
    assert authorization_permits(reg.maturity(fam), A.PAPER) is False
    assert authorization_permits(reg.maturity(fam), A.LIVE) is False
    assert authorization_permits(reg.maturity(fam), A.MOCK) is True


def test_blocked_family_cannot_be_promoted():
    reg = StrategyFamilyRegistry()
    fam = StrategyFamily.MEAN_REVERSION
    reg.block(fam)
    assert reg.maturity(fam) is M.BLOCKED
    assert reg.mark_mock_execution_eligible(fam, test_authorization=True) is M.BLOCKED
    assert reg.mark_live_eligible(fam, validated=True) is M.BLOCKED


def test_paper_and_live_promotion_require_validation():
    reg = StrategyFamilyRegistry()
    fam = StrategyFamily.TREND_PULLBACK
    assert reg.mark_paper_eligible(fam, validated=False) is M.RESEARCH
    assert reg.mark_paper_eligible(fam, validated=True) is M.PAPER_ELIGIBLE
    assert reg.mark_live_eligible(fam, validated=True) is M.LIVE_ELIGIBLE


def test_registry_maturity_is_instance_isolated():
    reg_a = StrategyFamilyRegistry()
    reg_b = StrategyFamilyRegistry()
    reg_a.mark_mock_execution_eligible(StrategyFamily.MOMENTUM_CONTINUATION,
                                       test_authorization=True)
    assert reg_b.maturity(StrategyFamily.MOMENTUM_CONTINUATION) is M.RESEARCH
