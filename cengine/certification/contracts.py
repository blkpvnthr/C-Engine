"""Shared contracts for the M7 certification platform.

Versioned, serializable vocabulary threaded through every certification component:
engineering audit -> data integrity -> holdout/gates -> backtest -> strategy/portfolio/
execution/protective-exit/capital/stress/learning/native/failure-injection certs ->
readiness classification -> final report.

Nothing here executes orders, grants financial authority, or fabricates evidence. These are
**assessment records only**. Two honesty invariants are enforced structurally (not by
convention), consistent with the rest of the platform:

* **Provenance is explicit.** Every quantitative certification result carries a ``data_label``
  (synthetic / real / historical_replay). Only ``real`` realized outcomes may back a
  profitability claim; ``historical_replay`` (simulated fills over GENUINE historical bars) and
  ``synthetic`` (engineering fixtures) may NOT. We re-export the single source of truth from
  ``cengine.learning.evaluation.economic`` so the labels can never drift apart.

* **Insufficiency is a first-class verdict.** A certification with too little evidence returns
  ``INSUFFICIENT_DATA`` — never a silent pass. Absence of disproof is not certification.

The certification burden-of-proof mirrors CLAUDE.md "falsification first": a component is
NOT_CERTIFIED until genuine evidence certifies it, and we would rather under-claim than
over-claim readiness.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Optional

# Single source of truth for provenance labels — re-exported so certification and the learning
# economic engine can never diverge on what "real" vs "simulated" means.
from cengine.learning.evaluation.economic import (
    DATA_LABEL_HISTORICAL_REPLAY,
    DATA_LABEL_REAL,
    DATA_LABEL_SYNTHETIC,
    validate_data_label,
)

CERTIFICATION_CONTRACTS_VERSION = "certification-contracts-1.0.0"

__all__ = [
    "CERTIFICATION_CONTRACTS_VERSION",
    "DATA_LABEL_HISTORICAL_REPLAY",
    "DATA_LABEL_REAL",
    "DATA_LABEL_SYNTHETIC",
    "CertificationVerdict",
    "ComponentStatus",
    "ReadinessClass",
    "CertificationResult",
    "ComponentAudit",
    "CertificationReport",
    "safe_ticker",
    "stable_id",
    "config_fingerprint",
    "validate_data_label",
]


class CertificationVerdict(str, Enum):
    """Outcome of a single certification check.

    ``CERTIFIED`` requires genuine passing evidence. ``INSUFFICIENT_DATA`` is returned whenever
    the evidence floor is not met — it is never collapsed into a pass OR a fail, because "we
    could not measure this" is distinct from "we measured it and it failed".
    """

    CERTIFIED = "certified"
    NOT_CERTIFIED = "not_certified"
    INSUFFICIENT_DATA = "insufficient_data"
    QUARANTINE = "quarantine"            # anomalous — withhold from any promotion pending review
    NOT_APPLICABLE = "not_applicable"    # check does not apply to this platform configuration
    BLOCKED = "blocked"                  # could not run (missing dependency / safety stop)


class ComponentStatus(str, Enum):
    """M7A engineering-audit lifecycle status for a platform component.

    Deliberately ordered from least to most evidence. A component may only claim a higher
    status when the evidence for that rung genuinely exists; e.g. ``LIVE_APPROVED`` is
    unreachable during a certification exercise with no live executions.
    """

    SPECIFIED = "specified"                      # designed/documented, code may be absent
    IMPLEMENTED = "implemented"                  # code exists and imports
    UNIT_TESTED = "unit_tested"                  # covered by passing unit tests
    OPERATIONALLY_CONSUMED = "operationally_consumed"   # wired into a real runtime path
    MOCK_CERTIFIED = "mock_certified"            # validated on synthetic/fixture data only
    HISTORICALLY_VALIDATED = "historically_validated"   # validated on GENUINE historical bars
    PAPER_CERTIFIED = "paper_certified"          # validated in forward paper trading
    LIVE_APPROVED = "live_approved"              # approved for live capital (never auto-granted)
    BLOCKED = "blocked"                          # broken / missing dependency
    DEPRECATED = "deprecated"


class ReadinessClass(str, Enum):
    """§32 overall deployment-readiness classification for the platform or a strategy family."""

    PROTOTYPE = "prototype"                       # implemented, not yet validated on real data
    RESEARCH_VALIDATED = "research_validated"     # survives historical OOS/holdout; no live edge
    PAPER_PENDING = "paper_pending"               # eligible for paper; window not yet elapsed
    PAPER_CERTIFIED = "paper_certified"           # positive forward paper evidence
    LIVE_CANDIDATE = "live_candidate"             # meets all gates for staged live (sign-off req.)
    NOT_READY = "not_ready"                       # fails one or more mandatory gates
    QUARANTINED = "quarantined"


# ordering for "highest status wins" roll-ups (index = strength of evidence)
_STATUS_RANK = {
    ComponentStatus.BLOCKED: -1,
    ComponentStatus.DEPRECATED: -1,
    ComponentStatus.SPECIFIED: 0,
    ComponentStatus.IMPLEMENTED: 1,
    ComponentStatus.UNIT_TESTED: 2,
    ComponentStatus.OPERATIONALLY_CONSUMED: 3,
    ComponentStatus.MOCK_CERTIFIED: 4,
    ComponentStatus.HISTORICALLY_VALIDATED: 5,
    ComponentStatus.PAPER_CERTIFIED: 6,
    ComponentStatus.LIVE_APPROVED: 7,
}


def status_rank(status: ComponentStatus) -> int:
    return _STATUS_RANK.get(status, 0)


# --------------------------------------------------------------------------- identity helpers

_TICKER_RE = re.compile(r"^[A-Z0-9.\-]{1,12}$")


def safe_ticker(ticker: str) -> str:
    """Validate a ticker against a strict allow-list before it is ever interpolated into SQL.

    Mirrors the optimization-platform validator so certification SQL inherits the same
    injection defense. Raises ``ValueError`` on anything outside ``^[A-Z0-9.\\-]{1,12}$``.
    """
    t = str(ticker).strip().upper()
    if not _TICKER_RE.match(t):
        raise ValueError(f"unsafe ticker {ticker!r}: must match {_TICKER_RE.pattern}")
    return t


def stable_id(*parts: object) -> str:
    """Deterministic short id from arbitrary parts (for reproducible record keys)."""
    raw = "|".join(str(p) for p in parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def config_fingerprint(config: dict) -> str:
    """Order-independent fingerprint of a config dict (frozen-gate / reproducibility checks)."""
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def _now_ns() -> int:
    return time.time_ns()


# --------------------------------------------------------------------------- result records


@dataclass
class CertificationResult:
    """One certification check's outcome. Serializable; the atom of every M7 report.

    ``data_label`` is validated on construction so a quantitative result can never silently
    omit provenance. ``profitability_claim_allowed`` is derived, never trusted from the caller:
    only genuine ``real`` outcomes may support a profitability assertion.
    """

    check: str                                   # dotted id, e.g. "data_integrity.bar_count"
    component: str                               # subsystem, e.g. "lakehouse" | "protective_exit"
    verdict: CertificationVerdict
    summary: str = ""
    metrics: dict = field(default_factory=dict)
    sample_size: int = 0
    effective_sample_size: Optional[float] = None
    data_label: Optional[str] = None             # None for non-quantitative/engineering checks
    evidence_refs: list = field(default_factory=list)
    gate_refs: list = field(default_factory=list)   # frozen gates this check was judged against
    notes: str = ""
    as_of_ns: int = field(default_factory=_now_ns)

    def __post_init__(self) -> None:
        if isinstance(self.verdict, str):
            self.verdict = CertificationVerdict(self.verdict)
        if self.data_label is not None:
            self.data_label = validate_data_label(self.data_label)

    @property
    def profitability_claim_allowed(self) -> bool:
        """Only genuine realized ('real') outcomes may back a profitability claim."""
        return self.data_label == DATA_LABEL_REAL

    @property
    def passed(self) -> bool:
        return self.verdict == CertificationVerdict.CERTIFIED

    def to_dict(self) -> dict:
        d = asdict(self)
        d["verdict"] = self.verdict.value
        d["profitability_claim_allowed"] = self.profitability_claim_allowed
        return d


@dataclass
class ComponentAudit:
    """M7A audit row for a single platform component."""

    component: str
    status: ComponentStatus
    module: str = ""                             # import path / file
    evidence: str = ""                           # why this status (imports? tests? consumed where?)
    consumed_by: list = field(default_factory=list)
    gaps: list = field(default_factory=list)     # what is missing to reach the next rung
    notes: str = ""

    def __post_init__(self) -> None:
        if isinstance(self.status, str):
            self.status = ComponentStatus(self.status)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["status"] = self.status.value
        d["status_rank"] = status_rank(self.status)
        return d


@dataclass
class CertificationReport:
    """A named collection of certification results + optional audit rows + a roll-up verdict."""

    title: str
    results: list = field(default_factory=list)          # list[CertificationResult]
    audits: list = field(default_factory=list)           # list[ComponentAudit]
    metadata: dict = field(default_factory=dict)
    as_of_ns: int = field(default_factory=_now_ns)
    version: str = CERTIFICATION_CONTRACTS_VERSION

    def add(self, result: CertificationResult) -> CertificationResult:
        self.results.append(result)
        return result

    def add_audit(self, audit: ComponentAudit) -> ComponentAudit:
        self.audits.append(audit)
        return audit

    def counts(self) -> dict:
        c: dict = {v.value: 0 for v in CertificationVerdict}
        for r in self.results:
            c[r.verdict.value] += 1
        return c

    def rollup(self) -> CertificationVerdict:
        """Conservative roll-up: any NOT_CERTIFIED -> NOT_CERTIFIED; else any INSUFFICIENT_DATA
        or QUARANTINE -> that; else if there is at least one CERTIFIED -> CERTIFIED; else
        INSUFFICIENT_DATA. Absence of results is never a pass."""
        verdicts = {r.verdict for r in self.results}
        if CertificationVerdict.NOT_CERTIFIED in verdicts:
            return CertificationVerdict.NOT_CERTIFIED
        if CertificationVerdict.QUARANTINE in verdicts:
            return CertificationVerdict.QUARANTINE
        if CertificationVerdict.BLOCKED in verdicts:
            return CertificationVerdict.BLOCKED
        if CertificationVerdict.INSUFFICIENT_DATA in verdicts:
            return CertificationVerdict.INSUFFICIENT_DATA
        if CertificationVerdict.CERTIFIED in verdicts:
            return CertificationVerdict.CERTIFIED
        return CertificationVerdict.INSUFFICIENT_DATA

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "version": self.version,
            "as_of_ns": self.as_of_ns,
            "rollup_verdict": self.rollup().value,
            "counts": self.counts(),
            "metadata": self.metadata,
            "audits": [a.to_dict() for a in self.audits],
            "results": [r.to_dict() for r in self.results],
        }

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)
