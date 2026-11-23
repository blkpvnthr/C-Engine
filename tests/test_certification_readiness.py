"""Tests for readiness classification (§32), the paper framework and the staged live framework."""

from __future__ import annotations

import pytest

from cengine.certification.contracts import CertificationVerdict, ReadinessClass
from cengine.certification.live_framework import (
    LiveAuthorizationError,
    LiveStage,
    StagedLiveDeployment,
    live_status,
)
from cengine.certification.paper_framework import (
    PaperSession,
    PaperTradingFramework,
    paper_status,
)
from cengine.certification.readiness import classify_readiness

_INFRA_OK = {d: "certified" for d in
             ("engineering", "data_integrity", "native_metal", "failure_injection",
              "protective_exit", "entry_contract", "execution", "learning_safety")}
_ECON = ("strategy_economics", "capital_sensitivity", "stress", "portfolio_economics")


# --------------------------------------------------------------------------- readiness


def test_research_validated_when_infra_sound_but_no_edge():
    dv = {**_INFRA_OK, **{d: "not_certified" for d in _ECON}}
    a = classify_readiness(dv)
    assert a.readiness == ReadinessClass.RESEARCH_VALIDATED
    assert a.infrastructure_sound is True
    assert a.economic_edge_certified is False
    assert a.live_eligible is False


def test_not_ready_when_infra_fails():
    dv = {**_INFRA_OK, "data_integrity": "not_certified",
          **{d: "certified" for d in _ECON}}
    a = classify_readiness(dv)
    assert a.readiness == ReadinessClass.NOT_READY


def test_paper_pending_when_edge_certified_no_paper():
    dv = {**_INFRA_OK, **{d: "certified" for d in _ECON}}
    a = classify_readiness(dv, paper_certified=False)
    assert a.readiness == ReadinessClass.PAPER_PENDING


def test_live_candidate_requires_full_chain_and_authorization():
    dv = {**_INFRA_OK, **{d: "certified" for d in _ECON}}
    paper = classify_readiness(dv, paper_certified=True, human_live_authorization=False)
    assert paper.readiness == ReadinessClass.PAPER_CERTIFIED
    live = classify_readiness(dv, paper_certified=True, human_live_authorization=True)
    assert live.readiness == ReadinessClass.LIVE_CANDIDATE
    assert live.live_eligible is True


# --------------------------------------------------------------------------- paper framework


def test_paper_not_certified_without_research_edge():
    r = paper_status(research_validated=False)
    assert r.verdict == CertificationVerdict.NOT_CERTIFIED


def test_paper_insufficient_until_window_elapsed():
    fw = PaperTradingFramework(research_validated=True)
    for i in range(5):
        fw.record_session(PaperSession(session_date=f"2026-11-{i+1:02d}",
                                       net_return=0.001, trades=3))
    assert fw.status().verdict == CertificationVerdict.INSUFFICIENT_DATA


def test_paper_rejects_fabricated_session():
    fw = PaperTradingFramework(research_validated=True)
    with pytest.raises(ValueError):
        fw.record_session(PaperSession(session_date="2026-11-01", net_return=0.01,
                                       trades=3, data_label="historical_replay"))


def test_paper_certifies_on_genuine_positive_window():
    fw = PaperTradingFramework(research_validated=True)
    for i in range(30):
        fw.record_session(PaperSession(session_date=f"2026-{(i//28)+1:02d}-{(i%28)+1:02d}",
                                       net_return=0.002, trades=4))
    r = fw.status()
    assert r.verdict == CertificationVerdict.CERTIFIED
    assert r.metrics["expectancy"] > 0


# --------------------------------------------------------------------------- live framework


def test_live_inactive_by_default():
    r = live_status(infrastructure_sound=True)
    assert r.verdict == CertificationVerdict.NOT_APPLICABLE
    assert r.metrics["active"] is False


def test_live_pilot_refused_without_edge_paper_token():
    d = StagedLiveDeployment(infrastructure_sound=True)
    with pytest.raises(LiveAuthorizationError):
        d.authorize(LiveStage.PILOT_TINY, human_token="t")   # no edge/paper
    assert d.is_active() is False


def test_live_pilot_refused_without_human_token_even_when_gated_green():
    d = StagedLiveDeployment(infrastructure_sound=True, economic_edge_certified=True,
                             paper_certified=True)
    with pytest.raises(LiveAuthorizationError):
        d.authorize(LiveStage.PILOT_TINY, human_token="")    # gates green, no human token
    # with a token it may enter and becomes active (order-submitting)
    d.authorize(LiveStage.PILOT_TINY, human_token="human-signed")
    assert d.stage == LiveStage.PILOT_TINY
    assert d.is_active() is True


def test_shadow_stage_never_submits_orders():
    d = StagedLiveDeployment(infrastructure_sound=True)
    d.authorize(LiveStage.SHADOW)
    assert d.stage == LiveStage.SHADOW
    assert d.is_active() is False   # shadow is log-only; never order-submitting
