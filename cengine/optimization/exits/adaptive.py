"""Adaptive protective-exit candidate policies (Milestone 6B, §14).

The incumbent production rule (unconditional -0.5% liquidation) is PRESERVED untouched as the
control / fallback / rollback. This module implements CANDIDATE policies in which -0.5%
becomes a mandatory reassessment threshold, with an adaptive stop (−1.0% default, −2.5% max),
HOLD/REDUCE/EXIT/REPLACE decisions driven by independently-validated recovery evidence, and an
INDEPENDENT emergency trigger (−3.0%) that is never bypassed. All signed-P/L thresholds are
decimal fractions (−0.005 = −0.5%); stop DISTANCES are positive fractions.

Safety rules enforced here:
  * The policy never authorizes holding beyond the active adaptive stop.
  * It never widens an active stop after entry unless explicitly in a fixed-at-entry or
    tightening-only mode (prohibit_unapproved_stop_widening).
  * A wider stop never increases capital at risk (risk-normalized sizing).
  * On invalid/unavailable intelligence it FALLS BACK to the incumbent -0.5% behavior.
  * These are recommendations only — native C++ retains execution authority; nothing here
    submits orders.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

ADAPTIVE_EXIT_VERSION = "adaptive-exit-1.0.0"

# the preserved production incumbent (§14.1) — a signed P/L fraction.
PRODUCTION_EXIT_PCT = -0.005

# independent emergency trigger (§14.7) — a liquidation level, not a guaranteed max loss.
EMERGENCY_EXIT_PCT = -0.03


class ExitAction(str, Enum):
    HOLD = "hold"
    REDUCE_POSITION = "reduce_position"
    EXIT_POSITION = "exit_position"
    EXIT_AND_REPLACE = "exit_and_replace"


# §14.5 candidate protective-distance ranges by asset class (positive fractions).
_ASSET_CLASS_RANGES = {
    "broad_etf": (0.0050, 0.0100),
    "large_cap": (0.0075, 0.0150),
    "high_vol": (0.0100, 0.0200),
    "leveraged_etf": (0.0125, 0.0250),
}


def asset_class_stop_range(asset_class: str) -> tuple:
    return _ASSET_CLASS_RANGES.get(asset_class, _ASSET_CLASS_RANGES["large_cap"])


def volatility_adjusted_stop(atr: float, reference_price: float, *, asset_class: str,
                             atr_multiplier: float = 1.5,
                             min_stop: Optional[float] = None,
                             max_stop: Optional[float] = None) -> float:
    """Candidate stop DISTANCE (positive fraction) = clip(atr_mult*ATR/ref, [min,max]).

    ATR and reference_price must share units. Returns a positive fraction of price.
    """
    lo, hi = asset_class_stop_range(asset_class)
    lo = lo if min_stop is None else min_stop
    hi = hi if max_stop is None else max_stop
    if reference_price <= 0 or atr <= 0:
        return lo
    raw = atr_multiplier * atr / reference_price
    return float(min(hi, max(lo, raw)))


def risk_normalized_notional(allowable_position_risk: float,
                             stop_distance_fraction: float) -> float:
    """Position notional so that a stop-out loses ~allowable_position_risk (§14.6).

    A WIDER stop therefore yields a SMALLER position — risk, not notional, is held constant.
    Callers must still clamp by liquidity/buying-power/leverage/exposure/concentration.
    """
    if stop_distance_fraction <= 0:
        return 0.0
    return float(allowable_position_risk / stop_distance_fraction)


@dataclass
class AdaptiveExitConfig:
    warning_pct: float = PRODUCTION_EXIT_PCT          # −0.5% reassessment trigger
    initial_stop_pct: float = -0.010                  # −1.0% default adaptive stop
    max_stop_pct: float = -0.025                      # −2.5% widest adaptive stop
    emergency_pct: float = EMERGENCY_EXIT_PCT          # −3.0% independent final trigger
    atr_multiplier: float = 1.5
    asset_class: str = "large_cap"
    tightening_only: bool = False                     # allow tightening but never widening
    fixed_at_entry: bool = False                      # stop fixed when the position opened
    min_recovery_probability: float = 0.6
    max_acceptable_downside: float = -0.02
    allow_replace: bool = True


@dataclass
class ExitDecision:
    action: str
    reason: str
    active_stop_pct: float
    emergency: bool = False
    fallback: bool = False
    evidence: dict = field(default_factory=dict)
    version: str = ADAPTIVE_EXIT_VERSION

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


class AdaptiveExitPolicy:
    """Candidate adaptive policy. ``decide`` consumes plain dicts so it is testable without
    the live engine and never touches orders."""

    def __init__(self, config: Optional[AdaptiveExitConfig] = None) -> None:
        self.cfg = config or AdaptiveExitConfig()

    def initial_stop(self, *, atr: Optional[float] = None,
                     reference_price: Optional[float] = None) -> float:
        """The active adaptive stop at entry (signed fraction), bounded to [max, warning)."""
        if atr and reference_price:
            dist = volatility_adjusted_stop(atr, reference_price,
                                            asset_class=self.cfg.asset_class,
                                            atr_multiplier=self.cfg.atr_multiplier,
                                            max_stop=abs(self.cfg.max_stop_pct))
            stop = -abs(dist)
        else:
            stop = self.cfg.initial_stop_pct
        # never wider than max; never tighter than the warning threshold
        return max(self.cfg.max_stop_pct, min(stop, self.cfg.warning_pct))

    def decide(self, *, total_pl_frac: float, active_stop_pct: float,
               intelligence: Optional[dict] = None,
               replacement_available: bool = False) -> ExitDecision:
        """Decide an action at/under the warning threshold.

        total_pl_frac, active_stop_pct are signed fractions (negative = loss). intelligence
        carries recovery_probability, downside_prediction, seller_exhaustion,
        momentum_confirmation, order_flow_confirmation, valid.
        """
        cfg = self.cfg
        ev = dict(intelligence or {})
        # 1. INDEPENDENT emergency trigger — never bypassed (§14.7)
        if total_pl_frac <= cfg.emergency_pct:
            return ExitDecision(ExitAction.EXIT_POSITION.value,
                                "emergency_trigger", active_stop_pct, emergency=True,
                                evidence=ev)
        # 2. active adaptive stop hit — never hold beyond it (§14.2)
        if total_pl_frac <= active_stop_pct:
            return ExitDecision(ExitAction.EXIT_POSITION.value,
                                "adaptive_stop_hit", active_stop_pct, evidence=ev)
        # 3. above the warning threshold -> continue holding within the boundary
        if total_pl_frac > cfg.warning_pct:
            return ExitDecision(ExitAction.HOLD.value, "above_warning", active_stop_pct,
                                evidence=ev)
        # 4. at/under warning but above the adaptive stop -> reassess with intelligence
        if not ev or not ev.get("valid", False):
            # invalid/unavailable intelligence -> FALL BACK to the incumbent -0.5% behavior
            return ExitDecision(ExitAction.EXIT_POSITION.value,
                                "fallback_incumbent_exit", cfg.warning_pct,
                                fallback=True, evidence=ev)
        recovery = float(ev.get("recovery_probability", 0.0) or 0.0)
        downside = float(ev.get("downside_prediction", 0.0) or 0.0)
        exhaustion = bool(ev.get("seller_exhaustion", False))
        momentum = bool(ev.get("momentum_confirmation", False))
        flow = bool(ev.get("order_flow_confirmation", False))
        confirmations = sum((exhaustion, momentum, flow))
        strong_recovery = (recovery >= cfg.min_recovery_probability
                           and confirmations >= 2
                           and downside >= cfg.max_acceptable_downside)
        if strong_recovery:
            return ExitDecision(ExitAction.HOLD.value,
                                "validated_recovery_evidence", active_stop_pct,
                                evidence={"recovery_probability": recovery,
                                          "confirmations": confirmations,
                                          "downside_prediction": downside})
        # superior independently-eligible alternative -> exit and replace
        if (cfg.allow_replace and replacement_available
                and downside < cfg.max_acceptable_downside):
            return ExitDecision(ExitAction.EXIT_AND_REPLACE.value,
                                "deteriorating_with_superior_alternative", active_stop_pct,
                                evidence={"downside_prediction": downside})
        # elevated uncertainty -> reduce; otherwise thesis invalidated -> exit
        if recovery >= 0.4 and confirmations >= 1:
            return ExitDecision(ExitAction.REDUCE_POSITION.value,
                                "elevated_uncertainty", active_stop_pct,
                                evidence={"recovery_probability": recovery,
                                          "confirmations": confirmations})
        return ExitDecision(ExitAction.EXIT_POSITION.value, "thesis_invalidated",
                            active_stop_pct, evidence={"recovery_probability": recovery,
                                                       "downside_prediction": downside})
