"""Strategy maturity + execution authorization (directive §4, §16).

The merged advisory layer uses a 3-state :class:`EligibilityState`
(``DISABLED``/``RESEARCH``/``EXECUTION``) that governs the ADVISORY arbitration
path, where ``EXECUTION`` means *authorized live elective entry* and stays
default-off.

This module adds the richer, INDEPENDENT maturity ladder the directive requires
and — critically — keeps *mock execution* strictly separate from paper/live
eligibility. Exercising a family through the isolated mock broker must never, as a
side effect, make it paper- or live-routable.

Safety contract
---------------
* ``authorization_permits(maturity, authorization)`` is the single gate the
  execution bridge consults. It is deliberately conservative:
    - ``NONE``  → nothing routes (default; advisory only).
    - ``MOCK``  → only families explicitly at ``MOCK_EXECUTION`` or beyond.
    - ``PAPER`` → only ``PAPER_ELIGIBLE`` or ``LIVE_ELIGIBLE`` (a *mock-only*
      family is NOT paper-routable).
    - ``LIVE``  → only ``LIVE_ELIGIBLE``.
* ``BLOCKED`` is a hard stop that overrides every authorization.
* Nothing in this module promotes a family automatically. Promotion is an
  explicit, separately-validated operator decision.
"""

from __future__ import annotations

from enum import Enum


class MaturityState(str, Enum):
    """Independently-verified maturity of a strategy family.

    Ordering (low → high capability): BLOCKED < DISABLED < RESEARCH < SHADOW <
    REPLAY_VALIDATED < MOCK_EXECUTION < PAPER_ELIGIBLE < LIVE_ELIGIBLE. Only the
    top three rungs are ever routable, and only under a matching authorization.
    """

    BLOCKED = "blocked"                 # hard stop; never routes, never recommends
    DISABLED = "disabled"              # administratively off
    RESEARCH = "research"             # evaluated + logged only (default)
    SHADOW = "shadow"                 # recommendations surfaced, never routed
    REPLAY_VALIDATED = "replay_validated"   # validated on historical replay only
    MOCK_EXECUTION = "mock_execution"       # isolated test authorization (mock broker)
    PAPER_ELIGIBLE = "paper_eligible"       # eligible for paper execution
    LIVE_ELIGIBLE = "live_eligible"         # eligible for authorized live execution


class ExecutionAuthorization(str, Enum):
    """The authorization context a run is operating under."""

    NONE = "none"      # default — advisory only, routes nothing
    MOCK = "mock"      # isolated mock-broker certification
    PAPER = "paper"    # paper execution
    LIVE = "live"      # authorized live execution


# Explicit permit sets — intentionally NOT derived from enum ordering, so a future
# reordering cannot silently widen what each authorization permits.
_PERMITTED: dict[ExecutionAuthorization, frozenset[MaturityState]] = {
    ExecutionAuthorization.NONE: frozenset(),
    ExecutionAuthorization.MOCK: frozenset(
        {MaturityState.MOCK_EXECUTION, MaturityState.PAPER_ELIGIBLE,
         MaturityState.LIVE_ELIGIBLE}
    ),
    ExecutionAuthorization.PAPER: frozenset(
        {MaturityState.PAPER_ELIGIBLE, MaturityState.LIVE_ELIGIBLE}
    ),
    ExecutionAuthorization.LIVE: frozenset({MaturityState.LIVE_ELIGIBLE}),
}


def authorization_permits(maturity: MaturityState,
                          authorization: ExecutionAuthorization) -> bool:
    """Return True iff a family at ``maturity`` may be routed under ``authorization``.

    ``BLOCKED`` is a hard stop regardless of authorization. ``NONE`` routes nothing.
    A ``MOCK_EXECUTION`` family is routable ONLY under ``MOCK`` — never paper/live.
    """
    if maturity is MaturityState.BLOCKED:
        return False
    return maturity in _PERMITTED.get(authorization, frozenset())


def describe_authorization(authorization: ExecutionAuthorization) -> str:
    """Human-readable summary of what an authorization permits (for logs/evidence)."""
    permitted = sorted(m.value for m in _PERMITTED[authorization])
    return f"{authorization.value} -> {permitted or 'nothing'}"


__all__ = [
    "ExecutionAuthorization",
    "MaturityState",
    "authorization_permits",
    "describe_authorization",
]
