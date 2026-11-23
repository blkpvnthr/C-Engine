"""Certification orchestration — run every domain, roll up, classify readiness.

Assembles the full M7 certification from its components into one :class:`CertificationReport`,
computes a conservative per-domain roll-up, and feeds those into the §32 readiness classifier +
the paper/live framework status. A ``quick`` mode runs only the fast engineering/operational/
contract/safety domains (no multi-minute economic sweeps) so the suite is usable interactively;
the full mode adds the genuine economic, sensitivity, stress and recovery passes on the sealed
holdout / historical bars.

Nothing here changes a verdict — it only aggregates what the component certifications honestly
return. The economic domains are expected to come back NOT_CERTIFIED on the current incumbent;
that result flows through to a RESEARCH_VALIDATED / NOT_READY-for-capital classification.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from cengine.certification.audit import run_engineering_audit
from cengine.certification.backtest import certify_entry_contract, certify_strategy_economics
from cengine.certification.contracts import (
    CertificationReport,
    CertificationResult,
    CertificationVerdict,
)
from cengine.certification.data_integrity import run_data_integrity_certification
from cengine.certification.execution_cert import certify_execution
from cengine.certification.failure_injection import certify_failure_injection
from cengine.certification.holdout import HoldoutSeal, build_holdout_manifest
from cengine.certification.learning_cert import certify_learning
from cengine.certification.live_framework import live_status
from cengine.certification.native_cert import certify_native_metal
from cengine.certification.paper_framework import paper_status
from cengine.certification.portfolio_cert import certify_portfolio
from cengine.certification.protective_exit_cert import (
    certify_mandatory_exit,
    certify_recovery_reproduction,
)
from cengine.certification.readiness import classify_readiness
from cengine.certification.sensitivity import certify_capital_sensitivity, certify_stress

ORCHESTRATION_VERSION = "certification-orchestration-1.0.0"
CANONICAL_SYMBOLS = ["AAPL", "MSFT", "NVDA", "SPY", "QQQ"]


def _rollup(results: list) -> CertificationVerdict:
    rep = CertificationReport(title="_")
    rep.results = list(results)
    return rep.rollup()


@dataclass
class CertificationSuite:
    report: CertificationReport
    domains: dict = field(default_factory=dict)          # domain -> verdict str
    readiness: dict = field(default_factory=dict)
    paper: dict = field(default_factory=dict)
    live: dict = field(default_factory=dict)
    timings: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = self.report.to_dict()
        d.update({"domains": self.domains, "readiness": self.readiness,
                  "paper": self.paper, "live": self.live, "timings": self.timings,
                  "orchestration_version": ORCHESTRATION_VERSION})
        return d


def run_full_certification(symbols: list | None = None, *, quick: bool = False,
                           config=None) -> CertificationSuite:
    """Run the certification suite. ``quick`` skips the heavy economic/sensitivity/stress passes."""
    symbols = symbols or CANONICAL_SYMBOLS
    report = CertificationReport(title="C-Engine M7 Certification",
                                 metadata={"symbols": symbols, "quick": quick})
    timings: dict = {}
    domain_results: dict = {}

    def _run(domain: str, fn):
        t0 = time.time()
        try:
            res = fn()
        except Exception as exc:  # noqa: BLE001 - a domain failure is a BLOCKED result, not a crash
            res = [CertificationResult(check=f"{domain}.error", component=domain,
                                       verdict=CertificationVerdict.BLOCKED,
                                       summary=f"domain raised: {exc}")]
        timings[domain] = round(time.time() - t0, 1)
        domain_results[domain] = res
        for r in res:
            report.add(r)
        return res

    # audit produces (audits, results)
    t0 = time.time()
    audits, audit_results = run_engineering_audit()
    for a in audits:
        report.add_audit(a)
    for r in audit_results:
        report.add(r)
    domain_results["engineering"] = audit_results
    timings["engineering"] = round(time.time() - t0, 1)

    # fast domains (always)
    _run("data_integrity", run_data_integrity_certification)
    _run("native_metal", certify_native_metal)
    _run("failure_injection", certify_failure_injection)
    _run("protective_exit", certify_mandatory_exit)
    _run("entry_contract", lambda: certify_entry_contract(symbols, config=config))
    _run("execution", certify_execution)
    _run("learning", lambda: certify_learning())

    # sealed-holdout manifest (always recorded)
    manifest = build_holdout_manifest(symbols, config=config)
    report.metadata["holdout_manifest"] = manifest.to_dict()

    if not quick:
        seal = HoldoutSeal()
        _run("strategy_economics",
             lambda: certify_strategy_economics(symbols, config=config, seal=seal))
        report.metadata["holdout_access_audit"] = seal.audit()
        _run("recovery", lambda: certify_recovery_reproduction(symbols, config=config))
        _run("capital_sensitivity",
             lambda: certify_capital_sensitivity(symbols, config=config))
        _run("stress", lambda: certify_stress(symbols, config=config))
        _run("portfolio", lambda: certify_portfolio(symbols, strategy_certified=False))

    # per-domain roll-ups
    domains = {d: _rollup(res).value for d, res in domain_results.items()}

    # map component domains -> readiness domains
    def dv(name: str) -> str:
        return domains.get(name, CertificationVerdict.INSUFFICIENT_DATA.value)

    readiness_domains = {
        "engineering": dv("engineering"),
        "data_integrity": dv("data_integrity"),
        "native_metal": dv("native_metal"),
        "failure_injection": dv("failure_injection"),
        "protective_exit": dv("protective_exit"),
        "entry_contract": dv("entry_contract"),
        "execution": dv("execution"),
        "learning_safety": _learning_safety_verdict(domain_results.get("learning", [])),
    }
    if not quick:
        readiness_domains.update({
            "strategy_economics": dv("strategy_economics"),
            "capital_sensitivity": dv("capital_sensitivity"),
            "stress": dv("stress"),
            "portfolio_economics": _portfolio_econ_verdict(domain_results.get("portfolio", [])),
        })

    assessment = classify_readiness(readiness_domains)
    paper = paper_status(research_validated=assessment.economic_edge_certified)
    live = live_status(infrastructure_sound=assessment.infrastructure_sound,
                       economic_edge_certified=assessment.economic_edge_certified,
                       paper_certified=(paper.verdict == CertificationVerdict.CERTIFIED))

    return CertificationSuite(
        report=report, domains=domains, readiness=assessment.to_dict(),
        paper=paper.to_dict(), live=live.to_dict(), timings=timings)


def _learning_safety_verdict(learning_results: list) -> str:
    """Learning SAFETY (not economic value): provenance + recommendation-only + opt safety."""
    safety_checks = {"learning.provenance_discipline", "learning.recommendation_only",
                     "learning.autonomous_opt_safety", "learning.infrastructure"}
    relevant = [r for r in learning_results if r.check in safety_checks]
    if not relevant:
        return CertificationVerdict.INSUFFICIENT_DATA.value
    return _rollup(relevant).value


def _portfolio_econ_verdict(portfolio_results: list) -> str:
    econ = [r for r in portfolio_results if r.check == "portfolio.economics"]
    return econ[0].verdict.value if econ else CertificationVerdict.INSUFFICIENT_DATA.value
