"""M7 — Comprehensive economic, operational and trading certification.

An *independent certification* layer over the existing C-Engine platform (M1–M6B). It does not
rebuild infrastructure and it grants no new financial authority: it measures what is built,
against frozen gates, on genuine historical data, and renders honest verdicts — including
INSUFFICIENT_DATA and NOT_CERTIFIED where the evidence does not support a pass.

Guiding invariants (mirroring CLAUDE.md):

* Falsification first — every component is NOT_CERTIFIED until genuine evidence certifies it.
* Net performance only — economic certification scores realistic, cost-inclusive outcomes.
* Out-of-sample is king — a sealed certification holdout is never touched for tuning.
* No fabrication — synthetic/historical-replay evidence may never masquerade as a live
  profitability claim; no live orders are placed during certification.
"""

from cengine.certification.contracts import (
    CERTIFICATION_CONTRACTS_VERSION,
    CertificationReport,
    CertificationResult,
    CertificationVerdict,
    ComponentAudit,
    ComponentStatus,
    ReadinessClass,
)

__all__ = [
    "CERTIFICATION_CONTRACTS_VERSION",
    "CertificationReport",
    "CertificationResult",
    "CertificationVerdict",
    "ComponentAudit",
    "ComponentStatus",
    "ReadinessClass",
]
