"""Adaptive position management + exit arbitration (Milestone 4, §12/§13/§14).

A unified position-management layer over ACTUAL positions and validated
observations. It distinguishes MANDATORY protection (the independent -0.5% exit in
``protective_exit.py``, never produced here) from DISCRETIONARY exits (trailing,
break-even, momentum-deterioration, time-based, concentration) which are research/
shadow by default and never auto-enabled for live trading.

The :class:`ExitArbiter` enforces a deterministic priority hierarchy (§14) and
prevents simultaneous incompatible orders for the same position. The mandatory
protective exit always outranks every discretionary portfolio/strategy decision.

Discretionary profit-protection policies are gated by
:class:`cengine.strategy_maturity.MaturityState`/``ExecutionAuthorization`` (reused):
a policy emits a SHADOW action unless it has been explicitly promoted + authorized.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, IntEnum
from typing import Optional

from cengine.strategy_maturity import (
    ExecutionAuthorization,
    MaturityState,
    authorization_permits,
)

POSITION_MGMT_VERSION = "position-management-1.0.0"


class ExitPriority(IntEnum):
    """Lower value = higher priority (§14)."""
    EMERGENCY = 1              # emergency financial protection
    MANDATORY_PROTECTIVE = 2   # mandatory -0.5% protective exit
    PORTFOLIO_RISK = 3         # portfolio risk reduction
    PROFIT_PROTECTION = 4      # authorized profit protection
    TECHNICAL_EXIT = 5         # authorized technical exit
    POSITION_ADJUST = 6        # elective position adjustment
    NEW_ENTRY = 7              # new entry
    REALLOCATION = 8           # elective reallocation


class PositionActionType(str, Enum):
    EXIT_FULL = "exit_full"
    EXIT_PARTIAL = "exit_partial"
    ADJUST = "adjust"
    ENTRY = "entry"
    HOLD = "hold"


@dataclass(frozen=True)
class PositionAction:
    symbol: str
    priority: ExitPriority
    action_type: PositionActionType
    reason: str
    quantity: Optional[int] = None
    maturity: MaturityState = MaturityState.RESEARCH
    shadow_only: bool = True        # True unless the policy is authorized for the run
    policy: str = ""
    version: str = POSITION_MGMT_VERSION


@dataclass
class PositionView:
    """Validated inputs for position evaluation (no fabricated values)."""
    symbol: str
    quantity: int
    unrealized_return: Optional[float] = None
    max_favorable_excursion: Optional[float] = None
    max_adverse_excursion: Optional[float] = None
    atr_pct: Optional[float] = None
    realized_vol: Optional[float] = None
    momentum: Optional[float] = None
    momentum_accel: Optional[float] = None
    buyer_exhaustion: Optional[float] = None
    seller_exhaustion: Optional[float] = None
    above_vwap: Optional[bool] = None
    above_bb_mid: Optional[bool] = None
    holding_duration_s: Optional[float] = None
    strategy_invalidated: bool = False
    effective_exposure_fraction: Optional[float] = None


# --- discretionary profit-protection policies (§13) -------------------------
class ProfitProtectionPolicy:
    """Base class. Each policy is independently validated; default maturity RESEARCH
    (shadow). Emits at most one :class:`PositionAction` per evaluation, or None."""

    name = "base"
    priority = ExitPriority.PROFIT_PROTECTION

    def __init__(self, *, maturity: MaturityState = MaturityState.RESEARCH) -> None:
        self.maturity = maturity

    def _action(self, view: PositionView, reason: str, authorization: ExecutionAuthorization,
                action_type: PositionActionType = PositionActionType.EXIT_FULL,
                quantity: Optional[int] = None) -> PositionAction:
        shadow = not authorization_permits(self.maturity, authorization)
        return PositionAction(
            symbol=view.symbol, priority=self.priority, action_type=action_type,
            reason=reason, quantity=quantity, maturity=self.maturity,
            shadow_only=shadow, policy=self.name)

    def evaluate(self, view: PositionView, *,
                 authorization: ExecutionAuthorization) -> Optional[PositionAction]:
        raise NotImplementedError


class AtrTrailingProtection(ProfitProtectionPolicy):
    """Volatility-adjusted trailing: give back no more than k*ATR from the peak."""
    name = "atr_trailing"

    def __init__(self, *, atr_multiple: float = 2.0, **kw) -> None:
        super().__init__(**kw)
        self.atr_multiple = float(atr_multiple)

    def evaluate(self, view, *, authorization):
        mfe, atr = view.max_favorable_excursion, view.atr_pct
        if mfe is None or atr is None or view.unrealized_return is None:
            return None
        giveback = mfe - view.unrealized_return
        if mfe > 0 and giveback >= self.atr_multiple * atr:
            return self._action(view, f"atr_trailing_giveback>={self.atr_multiple}*ATR",
                                authorization)
        return None


class BreakEvenProtection(ProfitProtectionPolicy):
    """Cost-adjusted break-even: once in sufficient profit, exit if it falls back to
    the cost-adjusted break-even level."""
    name = "break_even"

    def __init__(self, *, arm_profit: float = 0.004, cost_buffer: float = 0.0005, **kw) -> None:
        super().__init__(**kw)
        self.arm_profit = float(arm_profit)
        self.cost_buffer = float(cost_buffer)

    def evaluate(self, view, *, authorization):
        mfe, ur = view.max_favorable_excursion, view.unrealized_return
        if mfe is None or ur is None:
            return None
        if mfe >= self.arm_profit and ur <= self.cost_buffer:
            return self._action(view, "break_even_after_arm", authorization)
        return None


class MomentumDeteriorationProtection(ProfitProtectionPolicy):
    """Exit a profitable long when momentum turns down + buyer exhaustion appears."""
    name = "momentum_deterioration"

    def evaluate(self, view, *, authorization):
        ur = view.unrealized_return
        if ur is None or ur <= 0:
            return None
        decel = (view.momentum_accel is not None and view.momentum_accel < 0)
        exhausted = (view.buyer_exhaustion is not None and view.buyer_exhaustion >= 0.5)
        if decel and exhausted:
            return self._action(view, "momentum_down+buyer_exhaustion", authorization)
        return None


class TimeBasedExit(ProfitProtectionPolicy):
    """Exit a stagnant position after a maximum holding duration."""
    name = "time_based"
    priority = ExitPriority.TECHNICAL_EXIT

    def __init__(self, *, max_holding_s: float = 6 * 3600, min_move: float = 0.002, **kw) -> None:
        super().__init__(**kw)
        self.max_holding_s = float(max_holding_s)
        self.min_move = float(min_move)

    def evaluate(self, view, *, authorization):
        dur, ur = view.holding_duration_s, view.unrealized_return
        if dur is None or ur is None:
            return None
        if dur >= self.max_holding_s and abs(ur) < self.min_move:
            return self._action(view, "time_based_stagnant", authorization)
        return None


class TechnicalInvalidationExit(ProfitProtectionPolicy):
    """Exit when the strategy thesis is invalidated (lost all references)."""
    name = "technical_invalidation"
    priority = ExitPriority.TECHNICAL_EXIT

    def evaluate(self, view, *, authorization):
        if view.strategy_invalidated:
            return self._action(view, "strategy_invalidated", authorization)
        return None


class AdaptivePositionManager:
    """Runs enabled discretionary policies over a position view (§12/§13).

    Mandatory protection is NOT handled here — it stays in protective_exit.py. All
    actions here are discretionary and default to shadow-only until authorized.
    """

    def __init__(self, *, policies: Optional[list] = None,
                 authorization: ExecutionAuthorization = ExecutionAuthorization.NONE) -> None:
        self.policies = policies if policies is not None else [
            AtrTrailingProtection(), BreakEvenProtection(),
            MomentumDeteriorationProtection(), TimeBasedExit(),
            TechnicalInvalidationExit()]
        self.authorization = authorization

    def evaluate(self, view: PositionView) -> list[PositionAction]:
        actions: list[PositionAction] = []
        for policy in self.policies:
            try:
                action = policy.evaluate(view, authorization=self.authorization)
            except Exception:  # noqa: BLE001 — a policy must never crash the loop
                action = None
            if action is not None:
                actions.append(action)
        return actions


class ExitArbiter:
    """Deterministic priority arbitration (§14). Mandatory exit always wins; never
    emits simultaneous incompatible orders for the same position."""

    def arbitrate(self, actions: list[PositionAction]) -> Optional[PositionAction]:
        """Return the single highest-priority action (lowest ExitPriority value).
        Among equal priority, a full exit outranks partial/adjust/entry. Shadow-only
        actions are returned too (the caller decides whether to route) but a routable
        (non-shadow) action of the same priority is preferred."""
        if not actions:
            return None
        def key(a: PositionAction):
            exit_rank = 0 if a.action_type in (PositionActionType.EXIT_FULL,
                                               PositionActionType.EXIT_PARTIAL) else 1
            return (int(a.priority), a.shadow_only, exit_rank)
        return sorted(actions, key=key)[0]

    def conflicts(self, actions: list[PositionAction]) -> bool:
        """True if there are both an exit and an entry for the same symbol (an
        incompatible simultaneous pair the arbiter must collapse)."""
        has_exit = any(a.action_type in (PositionActionType.EXIT_FULL,
                                         PositionActionType.EXIT_PARTIAL) for a in actions)
        has_entry = any(a.action_type is PositionActionType.ENTRY for a in actions)
        return has_exit and has_entry


__all__ = ["AdaptivePositionManager", "AtrTrailingProtection", "BreakEvenProtection",
           "ExitArbiter", "ExitPriority", "MomentumDeteriorationProtection",
           "PositionAction", "PositionActionType", "PositionView",
           "ProfitProtectionPolicy", "TechnicalInvalidationExit", "TimeBasedExit"]
