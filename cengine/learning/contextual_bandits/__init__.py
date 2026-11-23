"""Offline contextual-bandit research over a FIXED set of AUTHORIZED actions (Milestone 6).

Two reward models over caller-supplied authorized actions (e.g. execution policies /
strategy families — never invented here):

  * :class:`LinUCB` — numpy/CPU batched per-action ridge reward model with UCB scoring and
    (optional, research-only) offline Thompson sampling; the independent baseline.
  * :class:`NeuralBanditRewardModel` — MLX GPU-first neural per-action reward model
    (import-guarded, with a documented/observable CPU fallback via :class:`DevicePolicy`).

Training data comes from the lakehouse gold ``offline_rl`` dataset via
:func:`build_bandit_dataset` (context features + ``selected_action`` + reward, reward used
only where present).

NON-NEGOTIABLE: behavior-policy action probabilities are UNKNOWN (always NULL, never
fabricated), so NO IPS / propensity / doubly-robust off-policy estimator is implemented or
exposed — only direct reward-model scoring and clearly-labeled offline Thompson sampling.
These are recommendation-only research tools: no orders, no live exploration, and no contact
with the mandatory -0.5% protective-exit path.
"""

from cengine.learning.contextual_bandits.dataset import (
    CONTEXTUAL_BANDIT_VERSION,
    DEFAULT_CONTEXT_COLUMNS,
    OFF_POLICY_CORRECTION_SUPPORTED,
    BanditDataset,
    build_bandit_dataset,
    off_policy_correction_reason,
)
from cengine.learning.contextual_bandits.linucb import LinUCB
from cengine.learning.contextual_bandits.neural_bandit import (
    NeuralBanditRewardModel,
    neural_bandit_available,
)

__all__ = [
    "CONTEXTUAL_BANDIT_VERSION",
    "DEFAULT_CONTEXT_COLUMNS",
    "OFF_POLICY_CORRECTION_SUPPORTED",
    "BanditDataset",
    "LinUCB",
    "NeuralBanditRewardModel",
    "build_bandit_dataset",
    "neural_bandit_available",
    "off_policy_correction_reason",
]
