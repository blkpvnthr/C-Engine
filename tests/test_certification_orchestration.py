"""Tests for certification orchestration + §36 report rendering.

The heavy component certifications are stubbed so the orchestration wiring, roll-ups, readiness
mapping and report rendering are tested fast and hermetically.
"""

from __future__ import annotations

import json

from cengine.certification import orchestration as orch
from cengine.certification import report as rpt
from cengine.certification.contracts import (
    CertificationReport,
    CertificationResult,
    CertificationVerdict,
)


def _res(check, component, verdict):
    return CertificationResult(check=check, component=component,
                               verdict=CertificationVerdict(verdict), summary=f"{check} ok")


def _stub_orchestration(monkeypatch):
    monkeypatch.setattr(orch, "run_engineering_audit",
                        lambda: ([], [_res("engineering_audit.platform", "platform", "certified")]))
    monkeypatch.setattr(orch, "run_data_integrity_certification",
                        lambda: [_res("data_integrity.inventory", "lakehouse", "certified")])
    monkeypatch.setattr(orch, "certify_native_metal",
                        lambda: [_res("native.module", "risk_native", "certified")])
    monkeypatch.setattr(orch, "certify_failure_injection",
                        lambda: [_res("failure_injection.worker_isolation", "runtime",
                                      "certified")])
    monkeypatch.setattr(orch, "certify_mandatory_exit",
                        lambda: [_res("protective_exit.contract_math", "protective_exit",
                                      "certified")])
    monkeypatch.setattr(orch, "certify_entry_contract",
                        lambda symbols, config=None: [_res("entry_contract.AAPL", "strategy",
                                                           "certified")])
    monkeypatch.setattr(orch, "certify_execution",
                        lambda: [_res("execution.planner", "execution", "certified"),
                                 _res("execution.realized_quality", "execution",
                                      "insufficient_data")])
    monkeypatch.setattr(orch, "certify_learning",
                        lambda: [_res("learning.provenance_discipline", "learning_ml", "certified"),
                                 _res("learning.recommendation_only", "learning_ml", "certified"),
                                 _res("learning.autonomous_opt_safety", "autonomous_opt",
                                      "certified"),
                                 _res("learning.infrastructure", "learning_ml", "certified"),
                                 _res("learning.bandit_rl", "learning_bandit_rl",
                                      "insufficient_data")])


def test_quick_orchestration_wires_domains(monkeypatch):
    _stub_orchestration(monkeypatch)
    suite = orch.run_full_certification(["AAPL"], quick=True)
    # fast domains present
    for d in ("engineering", "data_integrity", "native_metal", "failure_injection",
              "protective_exit", "entry_contract", "execution", "learning"):
        assert d in suite.domains
    # economic domains NOT run in quick mode
    assert "strategy_economics" not in suite.domains
    # readiness: infra sound, no economic edge established (quick skips economics)
    assert suite.readiness["infrastructure_sound"] is True
    assert suite.live["metrics"]["active"] is False


def test_learning_safety_verdict_isolates_safety_subset():
    learning = [
        CertificationResult("learning.provenance_discipline", "x",
                            CertificationVerdict.CERTIFIED),
        CertificationResult("learning.recommendation_only", "x", CertificationVerdict.CERTIFIED),
        CertificationResult("learning.autonomous_opt_safety", "x", CertificationVerdict.CERTIFIED),
        CertificationResult("learning.infrastructure", "x", CertificationVerdict.CERTIFIED),
        CertificationResult("learning.bandit_rl", "x", CertificationVerdict.INSUFFICIENT_DATA),
    ]
    # the safety subset is all-certified even though bandit_rl is insufficient
    assert orch._learning_safety_verdict(learning) == "certified"


def test_rollup_is_conservative():
    results = [_res("a", "x", "certified"), _res("b", "x", "not_certified")]
    assert orch._rollup(results) == CertificationVerdict.NOT_CERTIFIED


# --------------------------------------------------------------------------- report rendering


def _synthetic_suite():
    report = CertificationReport(title="t", metadata={"symbols": ["AAPL"], "quick": False})
    report.add(_res("x.y", "x", "certified"))
    domains = {"engineering": "certified", "data_integrity": "certified",
               "native_metal": "certified", "failure_injection": "certified",
               "protective_exit": "certified", "entry_contract": "certified",
               "execution": "insufficient_data", "learning": "insufficient_data",
               "strategy_economics": "not_certified", "recovery": "certified",
               "capital_sensitivity": "not_certified", "stress": "not_certified",
               "portfolio": "not_certified"}
    readiness = {"readiness": "research_validated", "infrastructure_sound": True,
                 "economic_edge_certified": False, "live_eligible": False}
    paper = {"verdict": "not_certified"}
    live = {"metrics": {"active": False}}
    return orch.CertificationSuite(report=report, domains=domains, readiness=readiness,
                                   paper=paper, live=live, timings={"engineering": 2.2})


def test_report_has_twenty_questions_and_honest_conclusion():
    suite = _synthetic_suite()
    qs = rpt._twenty_questions(suite)
    assert len(qs) == 20
    md = rpt.generate_markdown_report(suite)
    assert "20 certification questions" in md
    assert "no certified economic edge" in md.lower()
    # Q5 (economic edge) answers NO
    assert any("NO" in a for (q, a, _) in qs if q.startswith("5."))


def test_dashboard_and_json_render():
    suite = _synthetic_suite()
    dash = rpt.generate_dashboard(suite)
    assert "Certification Dashboard" in dash
    assert "RESEARCH_VALIDATED" in dash
    payload = json.loads(rpt.generate_json(suite))
    assert payload["domains"]["strategy_economics"] == "not_certified"
