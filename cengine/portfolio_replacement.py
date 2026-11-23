"""Intelligent portfolio replacement workflow (Milestone 3, §13).

A state machine that, ONLY after a mandatory protective exit has fully completed and
broker state is reconciled, evaluates new qualified opportunities for the freed
capital. It reuses the existing replacement decision logic
(``market_scan.replacement.select_replacement`` + ``ReplacementEligibility``) and
places NO orders itself — a replacement decision is handed to the existing bridge /
router / native authority.

Guarantees:
  * Replacement NEVER gates or delays the mandatory exit (the exit runs on its own
    independent path; this coordinator only reacts AFTER it has reconciled).
  * No forced reinvestment — if no candidate qualifies, hold cash.
  * No immediate uncontrolled re-entry into a just-stopped-out symbol (cooldown).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional

from market_scan.replacement import ReplacementEligibility, select_replacement

REPLACEMENT_VERSION = "portfolio-replacement-1.0.0"


class ReplacementStage(str, Enum):
    MANDATORY_EXIT_TRIGGERED = "mandatory_exit_triggered"
    EXIT_ORDER_SUBMITTED = "exit_order_submitted"
    EXIT_PARTIALLY_FILLED = "exit_partially_filled"
    EXIT_FULLY_FILLED = "exit_fully_filled"
    BROKER_RECONCILED = "broker_reconciled"
    CAPITAL_AVAILABLE = "capital_available"
    REPLACEMENT_EVALUATION = "replacement_evaluation"
    REPLACEMENT_AUTHORIZED = "replacement_authorized"
    REPLACEMENT_ORDER_SUBMITTED = "replacement_order_submitted"
    HELD_CASH = "held_cash"


@dataclass
class ReplacementRecord:
    exited_symbol: str
    stage: ReplacementStage
    triggered_ns: int
    reconciled_ns: Optional[int] = None
    replacement_symbol: Optional[str] = None
    reason: str = ""


@dataclass
class ReplacementDecision:
    exited_symbol: str
    replacement_symbol: Optional[str]
    authorized: bool
    reason: str
    considered: list = field(default_factory=list)
    version: str = REPLACEMENT_VERSION


class ReplacementCoordinator:
    """Drives the exit -> reconcile -> capital-available -> replacement workflow.

    ``no_reentry_cooldown_ns`` prevents re-entering a just-stopped symbol for a
    window. ``now_fn`` is injectable for deterministic tests.
    """

    def __init__(self, *, no_reentry_cooldown_ns: int = 15 * 60 * 1_000_000_000,
                 now_fn: Callable[[], int] = time.time_ns) -> None:
        self._records: dict[str, ReplacementRecord] = {}
        self._cooldown: dict[str, int] = {}          # symbol -> cooldown-until ns
        self._cooldown_ns = int(no_reentry_cooldown_ns)
        self._now = now_fn

    # -- exit lifecycle ------------------------------------------------------
    def on_mandatory_exit(self, symbol: str, now_ns: Optional[int] = None) -> None:
        now_ns = now_ns if now_ns is not None else self._now()
        self._records[symbol] = ReplacementRecord(
            symbol, ReplacementStage.MANDATORY_EXIT_TRIGGERED, now_ns)

    def on_exit_submitted(self, symbol: str) -> None:
        self._advance(symbol, ReplacementStage.EXIT_ORDER_SUBMITTED)

    def on_exit_partial(self, symbol: str) -> None:
        self._advance(symbol, ReplacementStage.EXIT_PARTIALLY_FILLED)

    def on_exit_filled(self, symbol: str, now_ns: Optional[int] = None) -> None:
        now_ns = now_ns if now_ns is not None else self._now()
        self._advance(symbol, ReplacementStage.EXIT_FULLY_FILLED)
        # start the no-reentry cooldown the moment the stop-out completes
        self._cooldown[symbol] = now_ns + self._cooldown_ns

    def on_broker_reconciled(self, symbol: str, now_ns: Optional[int] = None) -> None:
        now_ns = now_ns if now_ns is not None else self._now()
        rec = self._records.get(symbol)
        if rec is None or rec.stage is not ReplacementStage.EXIT_FULLY_FILLED:
            return  # only reconcile a fully-filled exit (do not shortcut the workflow)
        rec.stage = ReplacementStage.BROKER_RECONCILED
        rec.reconciled_ns = now_ns

    def ready_for_replacement(self, symbol: str) -> bool:
        rec = self._records.get(symbol)
        return rec is not None and rec.stage in (
            ReplacementStage.BROKER_RECONCILED, ReplacementStage.CAPITAL_AVAILABLE)

    def in_cooldown(self, symbol: str, now_ns: Optional[int] = None) -> bool:
        now_ns = now_ns if now_ns is not None else self._now()
        return now_ns < self._cooldown.get(symbol.upper(), 0)

    # -- replacement evaluation ----------------------------------------------
    def evaluate_replacement(
        self, exited_symbol: str, *, ranked_universe,
        evaluate: Callable[[str], Optional[ReplacementEligibility]],
        held: set, pending: set, capital_available_ticks: int,
        leverage_guard: Optional[Callable[[str], bool]] = None,
        now_ns: Optional[int] = None,
    ) -> ReplacementDecision:
        """Evaluate a replacement for freed capital AFTER reconciliation.

        Returns an authorized candidate or a hold-cash decision. Excludes held,
        pending, the just-exited symbol, and any symbol still in its no-reentry
        cooldown. No forced reinvestment.
        """
        now_ns = now_ns if now_ns is not None else self._now()
        rec = self._records.get(exited_symbol)
        if rec is None or not self.ready_for_replacement(exited_symbol):
            return ReplacementDecision(exited_symbol, None, False,
                                       "exit_not_reconciled")
        rec.stage = ReplacementStage.REPLACEMENT_EVALUATION

        if capital_available_ticks <= 0:
            rec.stage = ReplacementStage.HELD_CASH
            return ReplacementDecision(exited_symbol, None, False, "no_capital_available")

        exclude = {s.upper() for s in held} | {s.upper() for s in pending}
        exclude.add(exited_symbol.upper())
        # exclude everything still inside a no-reentry cooldown window
        exclude |= {s for s, until in self._cooldown.items() if now_ns < until}

        chosen, info = select_replacement(
            ranked_universe, evaluate=evaluate, exclude=exclude,
            leverage_guard=leverage_guard)

        if chosen is None:
            rec.stage = ReplacementStage.HELD_CASH
            return ReplacementDecision(exited_symbol, None, False,
                                       info.get("reason", "no_eligible_candidate_hold_cash"),
                                       considered=info.get("considered", []))
        rec.stage = ReplacementStage.REPLACEMENT_AUTHORIZED
        rec.replacement_symbol = chosen
        return ReplacementDecision(exited_symbol, chosen, True, "selected",
                                   considered=info.get("considered", []))

    def on_replacement_submitted(self, exited_symbol: str) -> None:
        self._advance(exited_symbol, ReplacementStage.REPLACEMENT_ORDER_SUBMITTED)

    def record(self, symbol: str) -> Optional[ReplacementRecord]:
        return self._records.get(symbol)

    def _advance(self, symbol: str, stage: ReplacementStage) -> None:
        rec = self._records.get(symbol)
        if rec is not None:
            rec.stage = stage
