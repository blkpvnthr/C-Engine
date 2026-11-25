"""P0.9 regression: the dominant-execution-blocker classifier for the dashboard.

Surfaces WHY orders do/don't flow on the primary (strategy-bridge) order path so
the operator sees the dominant blocker prominently instead of inferring it from
scattered counters.
"""

from __future__ import annotations

from trade import _dominant_execution_blocker


def test_no_authorization_is_the_dominant_blocker_first():
    assert _dominant_execution_blocker(
        bridge_active=False, candidates=42, submitted=0
    ) == "no_strategy_execution_authorization"


def test_no_candidates_when_authorized_but_nothing_eligible():
    assert _dominant_execution_blocker(
        bridge_active=True, candidates=0, submitted=0
    ) == "no_strategy_candidates"


def test_candidates_but_none_submitted_points_at_entry_contract_or_risk():
    assert _dominant_execution_blocker(
        bridge_active=True, candidates=8, submitted=0
    ) == "candidates_blocked_by_entry_contract_or_risk"


def test_orders_flowing_when_submissions_exist():
    assert _dominant_execution_blocker(
        bridge_active=True, candidates=8, submitted=2
    ) == "orders_flowing"
