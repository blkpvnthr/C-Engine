"""Disjoint LinUCB offline contextual-bandit reward model (Milestone 6, numpy/CPU).

A batched, per-action ridge-regression reward model with an upper-confidence-bound (UCB)
exploration bonus, after Li et al. 2010 ("A contextual-bandit approach to personalized
news article recommendation"). Each AUTHORIZED action ``a`` keeps a ridge system
``A_a = lambda*I + sum x x^T`` and ``b_a = sum r x``; its reward estimate for context
``x`` is ``theta_a . x`` with ``theta_a = A_a^{-1} b_a`` and an optimism bonus
``alpha * sqrt(x^T A_a^{-1} x)``.

Deliberately numpy/CPU: this is the independent baseline the GPU neural bandit must beat,
and the per-action ridge solve is tiny linear algebra dominated by host/device sync if
pushed to the GPU. Every ``fit`` still records a :class:`PlacementDecision` via the shared
:class:`DevicePolicy` so the device is OBSERVABLE (``gpu_supported=False`` → CPU, recorded).

Invariants:
  * Actions are FIXED and authorized by the caller (``n_actions`` / ``action_names``); the
    model never creates new actions.
  * Only DIRECT reward-model scoring and (optional, clearly-labeled) OFFLINE Thompson
    sampling are exposed. NO IPS / propensity / doubly-robust path exists — behavior-policy
    probabilities are unknown and never fabricated.
  * Recommendation-only: scoring/ranking offline contexts emits NO orders and performs NO
    live/uncontrolled exploration. It never touches the mandatory protective-exit path.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from cengine.learning.hardware import DevicePolicy
from cengine.learning.supervised.base import FitResult, regression_metrics


class LinUCB:
    """Batched per-action ridge reward model with UCB scoring (numpy, CPU)."""

    framework = "numpy"
    model_type = "linucb_contextual_bandit"

    def __init__(self, n_actions: int, n_features: int, *, alpha: float = 1.0,
                 ridge_lambda: float = 1.0, reward_noise: float = 1.0,
                 action_names: Optional[list[str]] = None,
                 device_policy: Optional[DevicePolicy] = None) -> None:
        if n_actions < 1 or n_features < 1:
            raise ValueError("n_actions and n_features must be >= 1")
        self.n_actions = int(n_actions)
        self.n_features = int(n_features)
        self.alpha = float(alpha)
        self.ridge_lambda = float(ridge_lambda)
        self.reward_noise = float(reward_noise)
        self.action_names = (list(action_names) if action_names is not None
                             else [f"action_{i}" for i in range(self.n_actions)])
        if len(self.action_names) != self.n_actions:
            raise ValueError("action_names length must equal n_actions")
        self.policy = device_policy or DevicePolicy()
        self.device = "cpu"
        eye = np.eye(self.n_features, dtype=np.float64)
        self._A = np.stack([self.ridge_lambda * eye for _ in range(self.n_actions)])
        self._b = np.zeros((self.n_actions, self.n_features), dtype=np.float64)
        self._theta = np.zeros((self.n_actions, self.n_features), dtype=np.float64)
        self._Ainv = np.stack([eye / self.ridge_lambda for _ in range(self.n_actions)])
        self._fitted = False

    # -- fitting -------------------------------------------------------------
    def _refresh(self) -> None:
        self._Ainv = np.linalg.inv(self._A)
        self._theta = np.linalg.solve(self._A, self._b[..., None])[..., 0]

    def fit(self, contexts: np.ndarray, actions: np.ndarray, rewards: np.ndarray,
            *, reward_mask: Optional[np.ndarray] = None) -> FitResult:
        """Accumulate ridge statistics from logged (context, action, reward) rows.

        Only rows with an observed reward are used (``reward_mask``); rows without a reward
        are ignored (never fabricated). Repeated calls accumulate (online ridge update).
        """
        X = np.asarray(contexts, dtype=np.float64)
        a = np.asarray(actions, dtype=np.int64)
        r = np.asarray(rewards, dtype=np.float64)
        if reward_mask is not None:
            m = np.asarray(reward_mask, dtype=bool)
            X, a, r = X[m], a[m], r[m]
        if X.ndim != 2 or X.shape[1] != self.n_features:
            raise ValueError(f"contexts must be (n, {self.n_features})")

        # Device decision is RECORDED even though this numpy op runs on CPU (observable).
        decision = self.policy.choose(f"fit_{self.model_type}",
                                      n_elements=max(1, X.shape[0]) * self.n_features,
                                      gpu_supported=False)
        self.device = decision.device

        for idx in range(self.n_actions):
            sel = a == idx
            if not np.any(sel):
                continue
            Xa = X[sel]
            ra = r[sel]
            self._A[idx] += Xa.T @ Xa
            self._b[idx] += Xa.T @ ra
        self._refresh()
        self._fitted = True

        if X.shape[0] > 0:
            pred_taken = self.predict_reward(X)[np.arange(X.shape[0]), a]
            metrics = regression_metrics(r, pred_taken)
            train_loss = metrics["mse"]
        else:
            train_loss = 0.0
        return FitResult(train_loss=float(train_loss), epochs_run=1, device=self.device)

    # -- scoring -------------------------------------------------------------
    def predict_reward(self, contexts: np.ndarray) -> np.ndarray:
        """Direct expected-reward estimate per action: ``(n, n_actions)``."""
        X = np.asarray(contexts, dtype=np.float64)
        if X.ndim != 2 or X.shape[1] != self.n_features:
            raise ValueError(f"contexts must be (n, {self.n_features})")
        return (X @ self._theta.T).astype(np.float64)

    def ucb_bonus(self, contexts: np.ndarray) -> np.ndarray:
        """Per-action UCB width ``sqrt(x^T A_a^{-1} x)``: ``(n, n_actions)``."""
        X = np.asarray(contexts, dtype=np.float64)
        quad = np.einsum("nd,ade,ne->na", X, self._Ainv, X)
        return np.sqrt(np.clip(quad, 0.0, None))

    def ucb_scores(self, contexts: np.ndarray, *,
                   alpha: Optional[float] = None) -> np.ndarray:
        """Optimistic score ``theta_a . x + alpha * bonus`` per action: ``(n, n_actions)``."""
        alpha = self.alpha if alpha is None else float(alpha)
        return self.predict_reward(contexts) + alpha * self.ucb_bonus(contexts)

    def _masked(self, scores: np.ndarray,
                eligibility_mask: Optional[np.ndarray]) -> np.ndarray:
        if eligibility_mask is None:
            return scores
        mask = np.asarray(eligibility_mask, dtype=bool)
        if mask.shape != scores.shape:
            raise ValueError("eligibility_mask must match scores shape (n, n_actions)")
        out = scores.copy()
        out[~mask] = -np.inf
        return out

    def recommend(self, contexts: np.ndarray, *,
                  eligibility_mask: Optional[np.ndarray] = None,
                  use_ucb: bool = False) -> np.ndarray:
        """Return the argmax action index per context (RESEARCH recommendation only).

        ``eligibility_mask`` (``(n, n_actions)`` bool) restricts the choice to currently
        authorized/eligible actions. ``use_ucb`` scores with the optimism bonus; by default
        greedy direct-reward scoring is used (pure offline recommendation, no exploration).
        """
        scores = self.ucb_scores(contexts) if use_ucb else self.predict_reward(contexts)
        return np.argmax(self._masked(scores, eligibility_mask), axis=1)

    def thompson_sample(self, contexts: np.ndarray, *,
                        rng: Optional[np.random.Generator] = None,
                        eligibility_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """OFFLINE Thompson-sampling recommendation (RESEARCH-ONLY).

        Draws one ``theta_a ~ N(theta_hat_a, reward_noise^2 * A_a^{-1})`` per action from
        the Bayesian-linear-regression posterior and returns the argmax sampled reward.
        This is a research ranking over offline-provided contexts; it is NOT live or
        uncontrolled exploration and emits no orders. Uses the posterior ONLY — never any
        behavior-policy probability (none exists).
        """
        rng = rng or np.random.default_rng()
        X = np.asarray(contexts, dtype=np.float64)
        var = self.reward_noise ** 2
        sampled = np.empty((X.shape[0], self.n_actions), dtype=np.float64)
        for idx in range(self.n_actions):
            cov = var * self._Ainv[idx]
            cov = 0.5 * (cov + cov.T)  # symmetrize for numerical stability
            theta_s = rng.multivariate_normal(self._theta[idx], cov)
            sampled[:, idx] = X @ theta_s
        return np.argmax(self._masked(sampled, eligibility_mask), axis=1)

    # -- serialization -------------------------------------------------------
    def params(self) -> dict:
        return {
            "A": self._A.astype(np.float64),
            "b": self._b.astype(np.float64),
            "config": np.frombuffer(
                __import__("json").dumps({
                    "n_actions": self.n_actions, "n_features": self.n_features,
                    "alpha": self.alpha, "ridge_lambda": self.ridge_lambda,
                    "reward_noise": self.reward_noise,
                    "action_names": self.action_names}).encode(), dtype=np.uint8),
        }

    def load_params(self, params: dict) -> None:
        self._A = np.asarray(params["A"], dtype=np.float64)
        self._b = np.asarray(params["b"], dtype=np.float64)
        self._refresh()
        self._fitted = True
