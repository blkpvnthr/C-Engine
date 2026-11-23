"""Tests for learning + autonomous-optimization certification.

These exercise the real learning stack, eligibility gate, economic provenance discipline,
recommendation-only scan and optimization-config safety guards — all fast.
"""

from __future__ import annotations

import pytest

from cengine.certification import learning_cert as lc
from cengine.certification.contracts import CertificationVerdict


def test_learning_infrastructure_imports():
    r = lc._certify_infrastructure()
    assert r.verdict == CertificationVerdict.CERTIFIED
    assert r.metrics["missing"] == []


def test_provenance_discipline_blocks_synthetic_claims():
    r = lc._certify_provenance_discipline()
    assert r.verdict == CertificationVerdict.CERTIFIED
    assert r.metrics["synthetic_claim_blocked"] is True
    assert r.metrics["historical_replay_claim_blocked"] is True
    assert r.metrics["real_claim_allowed"] is True


def test_learning_is_recommendation_only():
    r = lc._certify_recommendation_only()
    assert r.verdict == CertificationVerdict.CERTIFIED, r.metrics
    assert r.metrics["offenders"] == []


def test_autonomous_opt_safety_guards():
    r = lc._certify_autonomous_opt_safety()
    assert r.verdict == CertificationVerdict.CERTIFIED
    m = r.metrics
    assert m["default_valid"] is True
    assert m["risk_expansion_blocked"] is True
    assert m["production_exit_locked"] is True
    assert m["emergency_loosen_blocked"] is True       # more-negative (looser) rejected
    assert m["emergency_tighten_allowed"] is True      # less-negative (tighter) allowed


@pytest.mark.slow
def test_data_eligibility_reports_states():
    r = lc._certify_data_eligibility()
    # whatever the data, the verdict is an honest certified/insufficient and reports a breakdown
    assert r.verdict in (CertificationVerdict.CERTIFIED, CertificationVerdict.INSUFFICIENT_DATA)
    assert "by_state" in r.metrics
    assert isinstance(r.metrics["tasks"], list)


def test_bandit_rl_insufficient_data():
    r = lc._certify_bandit_rl()
    assert r.verdict == CertificationVerdict.INSUFFICIENT_DATA


@pytest.mark.slow
def test_full_learning_certification_runs():
    results = lc.certify_learning()
    assert len(results) == 6
    assert all(r.summary for r in results)
