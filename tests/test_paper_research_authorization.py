"""P0.3 regression: the PAPER_RESEARCH authorization tier.

Root cause of `recs=20 selected=0 research=20` -> zero paper orders: every
strategy family defaults to MaturityState.RESEARCH and nothing promotes it at
runtime, but no ExecutionAuthorization permits a RESEARCH-maturity family to
route. The enum was {NONE, MOCK, PAPER, LIVE}; PAPER only permits PAPER_ELIGIBLE/
LIVE_ELIGIBLE (certified). So an authorized-but-uncertified strategy had no legal
path to the paper broker -- contradicting the directive's requirement that
expressly-authorized PAPER_RESEARCH execution must work WITHOUT economic
certification.

PAPER_RESEARCH permits uncertified RESEARCH-grade families to route to the PAPER
venue while preserving every existing safety contract:
  * BLOCKED stays a hard stop.
  * DISABLED stays excluded (administratively off).
  * SHADOW stays observe-only ("recommendations surfaced, never routed").
  * MOCK_EXECUTION stays mock-isolated (a mock-only family is never paper-routable).
  * LIVE stays {LIVE_ELIGIBLE} -- live is NOT widened.
"""

from __future__ import annotations

from cengine.strategy_maturity import (
    ExecutionAuthorization,
    MaturityState,
    authorization_from_string,
    authorization_permits,
    describe_authorization,
)


def test_paper_research_is_a_distinct_authorization():
    assert ExecutionAuthorization.PAPER_RESEARCH.value == "paper_research"
    assert ExecutionAuthorization.PAPER_RESEARCH is not ExecutionAuthorization.PAPER


def test_paper_research_permits_uncertified_research_family():
    # THE core fix: a default RESEARCH-maturity family routes under PAPER_RESEARCH
    # without any economic certification.
    assert authorization_permits(MaturityState.RESEARCH,
                                 ExecutionAuthorization.PAPER_RESEARCH) is True


def test_paper_research_also_permits_more_validated_rungs():
    for maturity in (
        MaturityState.REPLAY_VALIDATED,
        MaturityState.PAPER_ELIGIBLE,
        MaturityState.LIVE_ELIGIBLE,
    ):
        assert authorization_permits(maturity,
                                     ExecutionAuthorization.PAPER_RESEARCH) is True


def test_paper_research_preserves_hard_contracts():
    # BLOCKED hard stop, DISABLED off, SHADOW observe-only, MOCK isolated.
    for maturity in (
        MaturityState.BLOCKED,
        MaturityState.DISABLED,
        MaturityState.SHADOW,
        MaturityState.MOCK_EXECUTION,
    ):
        assert authorization_permits(maturity,
                                     ExecutionAuthorization.PAPER_RESEARCH) is False


def test_plain_paper_still_requires_certification():
    # Unchanged: PAPER (not PAPER_RESEARCH) still rejects uncertified RESEARCH.
    assert authorization_permits(MaturityState.RESEARCH,
                                 ExecutionAuthorization.PAPER) is False


def test_live_authorization_is_not_widened():
    # LIVE must still permit ONLY LIVE_ELIGIBLE -- the new tier must not touch live.
    assert authorization_permits(MaturityState.RESEARCH,
                                 ExecutionAuthorization.LIVE) is False
    assert authorization_permits(MaturityState.PAPER_ELIGIBLE,
                                 ExecutionAuthorization.LIVE) is False
    assert authorization_permits(MaturityState.LIVE_ELIGIBLE,
                                 ExecutionAuthorization.LIVE) is True


def test_describe_authorization_covers_paper_research():
    text = describe_authorization(ExecutionAuthorization.PAPER_RESEARCH)
    assert "paper_research" in text
    assert "research" in text


def test_authorization_from_string_resolves_all_tiers():
    assert authorization_from_string("none") is ExecutionAuthorization.NONE
    assert authorization_from_string("mock") is ExecutionAuthorization.MOCK
    assert authorization_from_string("paper_research") is ExecutionAuthorization.PAPER_RESEARCH
    assert authorization_from_string("paper") is ExecutionAuthorization.PAPER
    assert authorization_from_string("live") is ExecutionAuthorization.LIVE


def test_authorization_from_string_is_case_insensitive_and_trims():
    assert authorization_from_string("  PAPER_RESEARCH  ") is ExecutionAuthorization.PAPER_RESEARCH


def test_authorization_from_string_is_fail_safe_to_none():
    # A typo / empty / None must never silently widen execution.
    for raw in ("", None, "bogus", "paperresearch", "all"):
        assert authorization_from_string(raw) is ExecutionAuthorization.NONE
