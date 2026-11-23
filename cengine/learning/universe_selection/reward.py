"""Bounded reward for universe-selection policy learning (spec §9 part 3).

The reward measures a selection decision's CONTRIBUTION to downstream trading
performance, not price-prediction accuracy. It is:

  * bounded (always in [-1, 1]) so no single episode dominates;
  * attribution-windowed (the caller supplies the realized net-of-cost outcome
    for the window the symbol was active);
  * honest about observability — it returns ``None`` when the outcome was not
    observed (the symbol was selected but never traded, or the position is still
    open past the window). Callers MUST mask ``None`` rewards out of training; we
    never fabricate a reward for an unobserved / unmonitored security.

The composite blends realized net return with light shaping for capital
efficiency and churn, each bounded, so the learner is nudged toward selections
that actually convert to profitable, low-churn trades.
"""

from __future__ import annotations

from typing import Optional

UNIVERSE_SELECTION_REWARD_VERSION = "universe-selection-reward-1.0.0"


def _clip(x: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def selection_reward(
    *,
    observed: bool,
    net_return_after_cost: Optional[float] = None,
    traded: bool = False,
    return_scale: float = 0.02,
    churned_within_residence: bool = False,
    churn_penalty: float = 0.1,
) -> Optional[float]:
    """Return a bounded reward in [-1, 1], or ``None`` when unobserved.

    Parameters
    ----------
    observed:
        Whether a realized, attributable outcome exists for this selection. When
        ``False`` the reward is ``None`` (masked out of training — never
        fabricated).
    net_return_after_cost:
        Realized net return (fraction, e.g. 0.01 = +1%) over the attribution
        window, AFTER transaction costs/slippage. Required when ``observed``.
    traded:
        Whether the selection actually converted to a trade. A selected-but-never
        -traded symbol with an observed (flat) outcome earns ~0 (no credit, no
        strong penalty) — selection should favor convertible opportunities.
    return_scale:
        Net return magnitude mapped to the reward's unit; |return| >= scale
        saturates the return component at +/-1 before shaping.
    churned_within_residence:
        Whether this symbol was rotated out before its minimum residence (a churn
        signal); applies a small bounded penalty.
    churn_penalty:
        Magnitude of the churn penalty (bounded).
    """
    if not observed:
        return None
    base = 0.0
    if traded and net_return_after_cost is not None:
        base = _clip(net_return_after_cost / max(1e-9, return_scale))
    # A selected-but-untraded symbol contributes a neutral (zero) reward: the
    # policy earns neither credit nor a large penalty for a missed conversion.
    reward = base
    if churned_within_residence:
        reward -= churn_penalty
    return _clip(reward)
