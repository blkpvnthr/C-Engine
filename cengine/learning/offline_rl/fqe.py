"""Fitted Q Evaluation (FQE) for constrained offline RL (Milestone 6, §6/§7).

FQE is an off-policy VALUE-ESTIMATION method: it estimates the value of a FIXED target
policy ``pi`` from logged transitions by iteratively regressing a Q-function toward the
bootstrap target ``r + gamma * (1 - terminal) * Q(s', pi(s'))``.

Why FQE here (and NOT importance sampling): FQE is model-based and needs rewards +
transitions but does NOT need the behavior-policy probabilities, which are unknown and are
never fabricated (``action_probability_if_known`` is always ``None``). IPS / weighted-IS /
doubly-robust estimators that DO need them are therefore deliberately not provided. The
mandatory protective exit is environment dynamics and is never in the action space, so it is
never evaluated as a policy action.

Default target policy is the LOGGED policy itself (``pi(s') = `` the logged next action), i.e.
fitted-SARSA evaluation of the behavior policy — a conservative baseline value. A different
discrete policy (e.g. a trained ``BehaviorCloner`` or any callable mapping states to action
indices) can be supplied. GPU-first (MLX) with a documented numpy CPU fallback; the device
used is recorded on every call.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Union

import numpy as np

from cengine.learning.hardware import DevicePolicy
from cengine.learning.offline_rl._nets import (
    _MLX,
    MLXMLP,
    _RLBase,
    tree_flatten,
    tree_unflatten,
)
from cengine.learning.offline_rl.transitions import TrainingGuard, TransitionDataset

if _MLX:  # pragma: no cover - import guard
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim

FQE_VERSION = "offline-rl-fqe-1.0.0"

# A target policy may be: None (evaluate the logged policy), a callable states->action idx,
# or any object exposing ``predict_action`` (e.g. a BehaviorCloner).
TargetPolicy = Union[None, Callable[[np.ndarray], np.ndarray], object]


@dataclass
class FQEConfig:
    discount: float = 0.99
    fitted_iterations: int = 8
    inner_epochs: int = 40
    batch_size: int = 256
    learning_rate: float = 1e-3
    hidden: tuple = (64, 32)
    weight_decay: float = 1e-4
    standardize: bool = True
    seed: int = 7
    ridge_l2: float = 1e-2  # CPU fallback regulariser


@dataclass
class FQEResult:
    trained: bool
    device: str
    backend: str
    mode: str
    promotable: bool
    value_estimate: float
    discount: float
    iterations: int
    bellman_mae: float
    n_reward_known: int
    reasons: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


def _resolve_target_next_actions(dataset: TransitionDataset,
                                 target_policy: TargetPolicy) -> np.ndarray:
    """Action indices a' = pi(s') for every transition's next state."""
    n = dataset.n_transitions
    if target_policy is None:
        a_next = dataset.next_actions.copy()
        # where no logged next action exists, fall back to the logged current action
        a_next[~dataset.next_action_valid] = dataset.actions[~dataset.next_action_valid]
        return a_next.astype(np.int64)
    if hasattr(target_policy, "predict_action"):
        return np.asarray(target_policy.predict_action(dataset.next_states)).astype(np.int64)
    if callable(target_policy):
        return np.asarray(target_policy(dataset.next_states)).astype(np.int64)
    return np.full(n, int(target_policy), dtype=np.int64)  # constant action


def _resolve_initial_actions(dataset: TransitionDataset,
                             target_policy: TargetPolicy) -> np.ndarray:
    """Action indices a = pi(s) for initial states (for the value estimate)."""
    if target_policy is None:
        return dataset.actions.astype(np.int64)
    if hasattr(target_policy, "predict_action"):
        return np.asarray(target_policy.predict_action(dataset.states)).astype(np.int64)
    if callable(target_policy):
        return np.asarray(target_policy(dataset.states)).astype(np.int64)
    return np.full(dataset.n_transitions, int(target_policy), dtype=np.int64)


class FittedQEvaluator(_RLBase):
    """Estimates the value of a fixed target policy from logged transitions (no IPS)."""

    model_type = "offline_rl_fqe"

    def __init__(self, *, config: Optional[FQEConfig] = None,
                 device_policy: Optional[DevicePolicy] = None,
                 guard: Optional[TrainingGuard] = None) -> None:
        self.cfg = config or FQEConfig()
        super().__init__(standardize=self.cfg.standardize, device_policy=device_policy,
                         seed=self.cfg.seed)
        self.guard = guard or TrainingGuard()
        self.n_features = 0
        self.n_actions = 0

    def evaluate(self, dataset: TransitionDataset, *,
                 target_policy: TargetPolicy = None) -> FQEResult:
        decision = self.guard.check(dataset, method="fqe")
        self.n_features = dataset.n_features
        self.n_actions = dataset.n_actions
        if not decision.allowed_to_train:
            return FQEResult(trained=False, device="cpu", backend=self.backend,
                             mode=decision.mode, promotable=False, value_estimate=float("nan"),
                             discount=self.cfg.discount, iterations=0, bellman_mae=float("nan"),
                             n_reward_known=dataset.n_reward_known,
                             reasons=decision.reasons + ["untrained_research_only"])

        # FQE learns from transitions with a realized reward only.
        rk = dataset.reward_known
        sub = dataset.subset(rk)
        S = self._fit_standardizer(sub.states)
        Sn = self._apply_standardizer(sub.next_states)
        a = sub.actions.astype(np.int64)
        a_next = _resolve_target_next_actions(sub, target_policy)
        r = sub.rewards.astype(np.float64)
        not_term = (~sub.terminal).astype(np.float64)

        n_elements = max(1, sub.n_transitions * dataset.n_features *
                         self.cfg.fitted_iterations * self.cfg.inner_epochs)
        _, mlx_dev = self._choose_device("fqe_value_eval", n_elements=n_elements)

        if self.backend == "mlx":
            q_fn, mae = self._fit_mlx(S, a, r, Sn, a_next, not_term, mlx_dev)
        else:
            q_fn, mae = self._fit_numpy(S, a, r, Sn, a_next, not_term)

        # value of the target policy = mean over INITIAL states of Q(s0, pi(s0))
        init = dataset.is_initial & rk
        if not np.any(init):
            init = rk
        S0 = self._apply_standardizer(dataset.states[init])
        a0 = _resolve_initial_actions(dataset.subset(init), target_policy)
        q0 = q_fn(S0)
        value = float(np.mean(q0[np.arange(len(a0)), a0])) if len(a0) else float("nan")

        return FQEResult(trained=True, device=self.device, backend=self.backend,
                         mode=decision.mode, promotable=decision.promotable,
                         value_estimate=value, discount=self.cfg.discount,
                         iterations=self.cfg.fitted_iterations, bellman_mae=float(mae),
                         n_reward_known=dataset.n_reward_known, reasons=decision.reasons)

    # -- GPU (MLX) fitted iteration ------------------------------------------
    def _fit_mlx(self, S, a, r, Sn, a_next, not_term, dev):  # pragma: no cover non-MLX
        mx.random.seed(self.seed)
        net = MLXMLP(self.n_features, self.cfg.hidden, self.n_actions)
        opt = optim.AdamW(learning_rate=self.cfg.learning_rate,
                          weight_decay=self.cfg.weight_decay)
        Sd = mx.array(np.asarray(S, dtype=np.float32))
        Snd = mx.array(np.asarray(Sn, dtype=np.float32))
        aidx = mx.array(a.reshape(-1, 1).astype(np.int32))
        anext = mx.array(a_next.reshape(-1, 1).astype(np.int32))
        rd = mx.array(r.astype(np.float32))
        ntd = mx.array(not_term.astype(np.float32))
        gamma = float(self.cfg.discount)
        n = Sd.shape[0]
        bs = min(self.cfg.batch_size, n)
        mae = float("nan")
        with mx.stream(dev):
            for _ in range(self.cfg.fitted_iterations):
                qn = net(Snd)
                qn_a = mx.take_along_axis(qn, anext, axis=1).squeeze(-1)
                targets = mx.stop_gradient(rd + gamma * ntd * qn_a)

                def loss_fn(model, s_b, a_b, t_b):
                    q = model(s_b)
                    qa = mx.take_along_axis(q, a_b, axis=1).squeeze(-1)
                    return mx.mean((qa - t_b) ** 2)

                lvg = nn.value_and_grad(net, loss_fn)
                for b in range(0, n, bs):
                    sl = slice(b, b + bs)
                    loss, grads = lvg(net, Sd[sl], aidx[sl], targets[sl])
                    opt.update(net, grads)
                    mx.eval(net.parameters(), opt.state)
                q = net(Sd)
                qa = mx.take_along_axis(q, aidx, axis=1).squeeze(-1)
                mae = float(mx.mean(mx.abs(qa - targets)))
        params = [(k, np.array(v)) for k, v in tree_flatten(net.parameters())]

        def q_fn(states):
            nn2 = MLXMLP(self.n_features, self.cfg.hidden, self.n_actions)
            nn2.update(tree_unflatten([(k, mx.array(v)) for k, v in params]))
            with mx.stream(dev):
                out = np.array(nn2(mx.array(np.asarray(states, dtype=np.float32))))
            return out

        return q_fn, mae

    # -- CPU (numpy) fitted iteration ----------------------------------------
    def _fit_numpy(self, S, a, r, Sn, a_next, not_term):
        S = np.asarray(S, dtype=np.float64)
        Sn = np.asarray(Sn, dtype=np.float64)
        n, d = S.shape
        A = self.n_actions
        W = np.zeros((d, A), dtype=np.float64)
        b = np.zeros(A, dtype=np.float64)
        gamma = float(self.cfg.discount)
        lr = 0.1
        mae = float("nan")
        for _ in range(self.cfg.fitted_iterations):
            qn = Sn @ W + b
            qn_a = qn[np.arange(n), a_next]
            targets = r + gamma * not_term * qn_a
            for _e in range(self.cfg.inner_epochs):
                q = S @ W + b
                qa = q[np.arange(n), a]
                resid = qa - targets
                dq = np.zeros((n, A), dtype=np.float64)
                dq[np.arange(n), a] = 2.0 * resid / n
                W -= lr * (S.T @ dq + self.cfg.ridge_l2 * W)
                b -= lr * dq.sum(axis=0)
            q = S @ W + b
            mae = float(np.mean(np.abs(q[np.arange(n), a] - targets)))
        self._np_W, self._np_b = W, b

        def q_fn(states):
            return np.asarray(states, dtype=np.float64) @ W + b

        return q_fn, mae
