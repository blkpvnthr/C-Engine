"""Contextual-bandit universe-selection policy (shadow-only challenger).

Framing: a per-candidate binary contextual bandit over two arms
``["exclude", "include"]``. For each candidate symbol the context is its feature
row; the challenger recommends include/exclude, and the "include" set (with
protected symbols always kept and the active cap enforced) is its ALTERNATIVE
active universe. This maps cleanly onto the existing disjoint ``LinUCB`` and is
recommendation-only — it never touches production subscriptions or orders.

Governance: the model is registered in the existing ``LearningModelRegistry``
and advanced only as far as ``SHADOW`` (``UNTRAINED→TRAINED→VALIDATED→SHADOW``,
none of which require operator approval). The deterministic rotation scheduler
remains the production champion.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from cengine.learning.contextual_bandits.linucb import LinUCB
from cengine.learning.registry.maturity import ModelMaturity
from cengine.learning.registry.model_registry import ModelCard

UNIVERSE_SELECTION_TASK = "universe_selection"
ACTION_NAMES = ("exclude", "include")
_INCLUDE = 1


class UniverseSelectionBandit:
    """Per-candidate include/exclude LinUCB wrapper (shadow challenger)."""

    model_type = LinUCB.model_type  # "linucb_contextual_bandit" (registry key)

    def __init__(self, n_features: int, *, alpha: float = 1.0,
                 feature_names: Optional[Sequence[str]] = None) -> None:
        self._model = LinUCB(
            n_actions=2, n_features=int(n_features), alpha=float(alpha),
            action_names=list(ACTION_NAMES),
        )
        self.feature_names = list(feature_names or [f"f{i}" for i in range(n_features)])
        self.n_features = int(n_features)

    def fit(self, contexts: np.ndarray, actions: np.ndarray, rewards: np.ndarray,
            *, reward_mask: Optional[np.ndarray] = None):
        """Online ridge update on observed (masked) rows only."""
        return self._model.fit(contexts, actions, rewards, reward_mask=reward_mask)

    def recommend_active(
        self,
        contexts: np.ndarray,
        symbols: Sequence[str],
        *,
        cap: int,
        protected: Sequence[str] = (),
        use_ucb: bool = False,
    ) -> list[str]:
        """Return the challenger's alternative active universe (<= cap).

        Protected symbols are always retained (shadow recommendation still honors
        the production safety invariant). Remaining slots are filled by the
        bandit's "include" recommendations, highest include-score first.
        """
        protected_norm = [s for s in protected if s]
        active: list[str] = []
        seen: set[str] = set()
        for s in protected_norm:
            if s not in seen and len(active) < cap:
                seen.add(s)
                active.append(s)
        if len(contexts) == 0 or len(active) >= cap:
            return active[:cap]
        scores = self._model.ucb_scores(contexts) if use_ucb else self._model.predict_reward(contexts)
        include_score = scores[:, _INCLUDE]
        order = np.argsort(-include_score)  # highest include-score first
        for idx in order:
            if len(active) >= cap:
                break
            sym = symbols[idx]
            if sym in seen:
                continue
            # Recommend include only when the include arm outscores exclude.
            if scores[idx, _INCLUDE] >= scores[idx, 0]:
                seen.add(sym)
                active.append(sym)
        return active[:cap]

    def params(self) -> dict:
        return self._model.params()

    def load_params(self, params: dict) -> None:
        self._model.load_params(params)


def shadow_compare(champion_active: Sequence[str],
                   challenger_active: Sequence[str]) -> dict:
    """Set-overlap metrics between the champion and the shadow challenger.

    Shadow-only: this NEVER changes production membership — it only quantifies how
    the learned policy WOULD differ, for independent review.
    """
    champ = set(champion_active)
    chall = set(challenger_active)
    union = champ | chall
    overlap = len(champ & chall)
    denom = len(union) or 1
    return {
        "champion_size": len(champ),
        "challenger_size": len(chall),
        "overlap": overlap,
        "jaccard": overlap / denom,
        "challenger_only": sorted(chall - champ),
        "champion_only": sorted(champ - chall),
        "disagreement": len(union) - overlap,
    }


def build_shadow_card(*, model_id: str, model_version: str, n_features: int,
                      feature_schema_version: str = "",
                      dataset_version: str = "", alpha: float = 1.0) -> ModelCard:
    """Build a ModelCard for registering the shadow challenger (maturity TRAINED).

    Promotion to VALIDATED→SHADOW is done via the registry's transition() calls
    (none require approval); the model never advances past SHADOW this milestone.
    """
    return ModelCard(
        model_id=model_id,
        model_version=model_version,
        model_type=UniverseSelectionBandit.model_type,
        model_architecture="disjoint_linucb_include_exclude",
        framework="numpy",
        training_device="cpu",
        inference_device="cpu",
        dataset_version=dataset_version,
        feature_schema_version=feature_schema_version,
        hyperparameters={"alpha": alpha, "n_features": n_features,
                         "action_names": list(ACTION_NAMES)},
        applicable_strategy_families=("universe_selection",),
        execution_environment="research",
        maturity=ModelMaturity.TRAINED.value,
        transition_reason="universe-selection shadow challenger trained",
    )
