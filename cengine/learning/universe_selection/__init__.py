"""Shadow-only learned universe-selection policy (spec §9, Option 2).

A contextual-bandit challenger to the deterministic rotation scheduler (the
production champion). Recommendation-only: it may emit a full alternative active
universe but never touches production subscriptions or orders. It reuses the
existing learning infrastructure wholesale — LinUCB, the model registry +
maturity gates (capped at SHADOW this milestone), the champion/challenger harness,
and the bronze→silver→gold→feature-store pipeline. If data-sufficiency gates fail,
training is DEFERRED and collection requirements are emitted — never fabricated.
"""

from .reward import UNIVERSE_SELECTION_REWARD_VERSION, selection_reward
from .policy import (
    UNIVERSE_SELECTION_TASK,
    UniverseSelectionBandit,
    build_shadow_card,
)

__all__ = [
    "selection_reward",
    "UNIVERSE_SELECTION_REWARD_VERSION",
    "UNIVERSE_SELECTION_TASK",
    "UniverseSelectionBandit",
    "build_shadow_card",
]
