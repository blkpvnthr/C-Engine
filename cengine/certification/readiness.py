"""Deployment-readiness classification (§32).

Maps the full body of certification evidence onto an honest :class:`ReadinessClass`, keeping two
questions separate so they are never conflated:

* **Infrastructure readiness** — is the platform engineering-sound and operationally robust?
  (engineering audit, data integrity, native/Metal, failure injection, protective-exit + entry
  contracts, learning recommendation-only + safety.)
* **Strategy/economic readiness** — is there a certified economic edge, net of costs, out of
  sample, surviving stress? (strategy economics, capital sensitivity, stress, portfolio economics.)

A platform can be infrastructure-sound (RESEARCH_VALIDATED) while having NO deployable edge
(strategy NOT_READY). Capital readiness requires BOTH. The classifier is deliberately
conservative: a single failed mandatory gate caps the whole platform at NOT_READY for capital,
and paper/live readiness can never be reached without the prior rung's genuine evidence. It can
only ever return a class the evidence supports — it cannot promote on optimism.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from cengine.certification.contracts import (
    CertificationVerdict,
    ReadinessClass,
)

READINESS_VERSION = "certification-readiness-1.0.0"

# which certification domains gate which readiness question
_INFRA_DOMAINS = ("engineering", "data_integrity", "native_metal", "failure_injection",
                  "protective_exit", "entry_contract", "execution", "learning_safety")
_ECON_DOMAINS = ("strategy_economics", "capital_sensitivity", "stress", "portfolio_economics")


@dataclass
class ReadinessAssessment:
    readiness: ReadinessClass
    infrastructure_sound: bool
    economic_edge_certified: bool
    paper_certified: bool
    live_eligible: bool
    rationale: list = field(default_factory=list)
    domain_verdicts: dict = field(default_factory=dict)
    version: str = READINESS_VERSION

    def to_dict(self) -> dict:
        return {"readiness": self.readiness.value,
                "infrastructure_sound": self.infrastructure_sound,
                "economic_edge_certified": self.economic_edge_certified,
                "paper_certified": self.paper_certified,
                "live_eligible": self.live_eligible,
                "rationale": self.rationale, "domain_verdicts": self.domain_verdicts,
                "version": self.version}


def _is(v) -> CertificationVerdict:
    return v if isinstance(v, CertificationVerdict) else CertificationVerdict(v)


def classify_readiness(domain_verdicts: dict, *, paper_certified: bool = False,
                       human_live_authorization: bool = False) -> ReadinessAssessment:
    """Classify readiness from per-domain rollup verdicts.

    ``domain_verdicts`` maps a domain name (see _INFRA_DOMAINS / _ECON_DOMAINS) to a
    CertificationVerdict (or its string). Missing domains are treated as INSUFFICIENT_DATA.
    """
    dv = {k: _is(v) for k, v in domain_verdicts.items()}
    rationale: list = []

    def verdict(domain: str) -> CertificationVerdict:
        return dv.get(domain, CertificationVerdict.INSUFFICIENT_DATA)

    # infrastructure: every present infra domain must be CERTIFIED (none NOT_CERTIFIED/BLOCKED)
    infra_present = [d for d in _INFRA_DOMAINS if d in dv]
    infra_failed = [d for d in infra_present
                    if verdict(d) in (CertificationVerdict.NOT_CERTIFIED,
                                      CertificationVerdict.BLOCKED,
                                      CertificationVerdict.QUARANTINE)]
    infrastructure_sound = bool(infra_present) and not infra_failed
    if infra_failed:
        rationale.append(f"infrastructure NOT sound — failed: {infra_failed}")
    else:
        rationale.append(f"infrastructure sound across {len(infra_present)} domains")

    # economic edge: every present economic domain must be CERTIFIED
    econ_present = [d for d in _ECON_DOMAINS if d in dv]
    econ_certified = bool(econ_present) and all(
        verdict(d) == CertificationVerdict.CERTIFIED for d in econ_present)
    econ_failed = [d for d in econ_present if verdict(d) == CertificationVerdict.NOT_CERTIFIED]
    if econ_failed:
        rationale.append(f"no certified economic edge — failed: {econ_failed}")
    elif econ_certified:
        rationale.append("economic edge certified on the sealed holdout")
    else:
        rationale.append("economic edge not established (insufficient data)")

    # live eligibility requires the full chain + explicit human authorization
    live_eligible = (infrastructure_sound and econ_certified and paper_certified
                     and human_live_authorization)

    # classify (most-conservative wins)
    if infra_failed:
        readiness = ReadinessClass.NOT_READY
        rationale.append("=> NOT_READY: a mandatory infrastructure/safety gate failed")
    elif not econ_certified and econ_failed:
        # sound infra, but the strategy was measured and has no edge
        readiness = (ReadinessClass.RESEARCH_VALIDATED if infrastructure_sound
                     else ReadinessClass.NOT_READY)
        rationale.append("=> infrastructure RESEARCH_VALIDATED; strategy has NO certified edge "
                         "(NOT_READY for capital)")
    elif not econ_certified:
        readiness = (ReadinessClass.PROTOTYPE if not infrastructure_sound
                     else ReadinessClass.RESEARCH_VALIDATED)
        rationale.append("=> economic edge not yet established")
    elif not paper_certified:
        readiness = ReadinessClass.PAPER_PENDING
        rationale.append("=> economic edge certified; eligible for forward paper (PAPER_PENDING)")
    elif not human_live_authorization:
        readiness = ReadinessClass.PAPER_CERTIFIED
        rationale.append("=> paper-certified; LIVE requires explicit human authorization")
    else:
        readiness = ReadinessClass.LIVE_CANDIDATE
        rationale.append("=> all gates green + human authorization present: LIVE_CANDIDATE")

    return ReadinessAssessment(
        readiness=readiness, infrastructure_sound=infrastructure_sound,
        economic_edge_certified=econ_certified, paper_certified=paper_certified,
        live_eligible=live_eligible, rationale=rationale,
        domain_verdicts={k: v.value for k, v in dv.items()})
