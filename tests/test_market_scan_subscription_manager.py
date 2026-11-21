"""Tests for market_scan.subscription_manager.SubscriptionManager.

Allocates a bounded IEX subscription budget across priority groups
(positions > pending > references > candidates). Positions/pending are MANDATORY
and never rotated out. Candidates beyond the cap are rotated across calls (fair
coverage) and reported as `dropped` — never silently truncated.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from market_scan.subscription_manager import SubscriptionManager


def test_mandatory_kept_by_priority_but_cap_never_silently_exceeded():
    m = SubscriptionManager(max_subscriptions=3, mandatory_groups=2)
    plan = m.plan([["AAA", "BBB"], ["CCC", "DDD"], ["EEE"], ["FFF", "GGG"]])
    # positions(AAA,BBB)+pending(CCC,DDD) = 4 mandatory > cap 3: STRICT cap, the
    # lowest-priority mandatory (DDD) is dropped and reported (not silent).
    assert len(plan.target) == 3
    assert {"AAA", "BBB", "CCC"} <= set(plan.target)   # highest priority kept
    assert "DDD" in plan.mandatory_dropped
    assert plan.overflow is True


def test_cap_respected_for_non_mandatory():
    m = SubscriptionManager(max_subscriptions=5, mandatory_groups=2)
    plan = m.plan([["AAA"], ["BBB"], ["R1", "R2"], ["C1", "C2", "C3", "C4"]])
    assert len(plan.target) == 5
    assert "AAA" in plan.target and "BBB" in plan.target


def test_candidates_rotate_across_calls_and_report_dropped():
    m = SubscriptionManager(max_subscriptions=3, mandatory_groups=2)
    groups = [["AAA"], ["BBB"], [], ["C1", "C2", "C3", "C4"]]
    p1 = m.plan(groups)
    # 1 free slot after AAA,BBB -> exactly one candidate chosen, 3 dropped
    chosen1 = [s for s in p1.target if s.startswith("C")]
    assert len(chosen1) == 1
    assert len(p1.dropped) == 3
    p2 = m.plan(groups)
    chosen2 = [s for s in p2.target if s.startswith("C")]
    assert len(chosen2) == 1
    assert chosen2 != chosen1  # rotation advanced


def test_add_remove_diffs_against_current():
    m = SubscriptionManager(max_subscriptions=10, mandatory_groups=2)
    p1 = m.plan([["AAA", "BBB"], [], [], ["C1"]])
    m.commit(p1)
    assert set(p1.to_add) == {"AAA", "BBB", "C1"}
    p2 = m.plan([["AAA"], [], [], ["C2"]])  # BBB gone, C1->C2
    m.commit(p2)
    assert "C2" in p2.to_add and "AAA" not in p2.to_add
    assert set(p2.to_remove) == {"BBB", "C1"}


def test_dedupes_across_groups_preserving_priority():
    m = SubscriptionManager(max_subscriptions=10, mandatory_groups=2)
    plan = m.plan([["AAA"], ["AAA"], ["AAA"], ["AAA", "BBB"]])
    assert plan.target.count("AAA") == 1
    assert "BBB" in plan.target


def test_available_capacity_reported():
    m = SubscriptionManager(max_subscriptions=5, mandatory_groups=2)
    plan = m.plan([["AAA"], ["BBB"], [], ["C1"]])
    assert plan.active == 3
    assert plan.available == 2
