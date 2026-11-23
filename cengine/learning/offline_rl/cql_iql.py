"""Conservative offline-RL policy/value learners: CQL + IQL (Milestone 6, §6/§7).

Two conservative, offline-safe learners over the authorized discrete action space (allocation
policies; the mandatory protective exit is excluded upstream as environment dynamics and is
NEVER a learnable/recommendable action):

* **CQL (Conservative Q-Learning)** — adds a conservative regulariser that pushes Q DOWN on
  out-of-distribution actions and UP on the logged action, counteracting the overestimation
  that sinks naive offline Q-learning. Loss = Bellman-MSE + ``alpha * (logsumexp_a Q(s,a) -
  Q(s, a_logged))``.
* **IQL (Implicit Q-Learning)** — never queries Q at out-of-distribution actions: it fits a
  state value ``V`` with expectile regression toward ``Q(s, a_logged)``, bootstraps ``Q``
  from ``V(s')``, and extracts a policy by advantage-weighted regression on the logged
  actions. This is the most data-conservative of the two.

Neither uses behavior-policy probabilities (unknown; never fabricated) so no importance
sampling is involved. Both are GPU-first (MLX on the Metal GPU) with a documented, observable
numpy CPU fallback and record the device actually used. All outputs are RECOMMENDATION-ONLY
research artifacts: ``recommend`` returns an advisory action, never an order, and never the
protective exit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from cengine.learning.hardware import DevicePolicy
from cengine.learning.offline_rl._nets import (
    _MLX,
    MLXMLP,
    _RLBase,
    np_logsumexp,
    np_softmax,
    np_softmax_regression,
    tree_flatten,
    tree_unflatten,
)
from cengine.learning.offline_rl.transitions import TrainingGuard, TransitionDataset

if _MLX:  # pragma: no cover - import guard
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim

CQL_IQL_VERSION = "offline-rl-cql-iql-1.0.0"


# ===========================================================================
# CQL
# ===========================================================================

@dataclass
class CQLConfig:
    discount: float = 0.99
    alpha: float = 1.0            # conservative penalty weight
    iterations: int = 8
    inner_epochs: int = 20
    batch_size: int = 256
    learning_rate: float = 1e-3
    hidden: tuple = (64, 32)
    weight_decay: float = 1e-4
    standardize: bool = True
    seed: int = 7
    ridge_l2: float = 1e-2


@dataclass
class CQLResult:
    trained: bool
    device: str
    backend: str
    mode: str
    promotable: bool
    iterations: int
    td_loss: float
    conservative_penalty: float
    value_estimate: float
    n_actions: int
    reasons: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


class ConservativeQLearner(_RLBase):
    """CQL over the authorized discrete action space (reward required; no IPS)."""

    model_type = "offline_rl_cql"

    def __init__(self, *, config: Optional[CQLConfig] = None,
                 device_policy: Optional[DevicePolicy] = None,
                 guard: Optional[TrainingGuard] = None) -> None:
        self.cfg = config or CQLConfig()
        super().__init__(standardize=self.cfg.standardize, device_policy=device_policy,
                         seed=self.cfg.seed)
        self.guard = guard or TrainingGuard()
        self.action_space = None
        self.n_features = 0
        self.n_actions = 0
        self._q_fn = None

    def fit(self, dataset: TransitionDataset) -> CQLResult:
        decision = self.guard.check(dataset, method="cql")
        self.action_space = dataset.action_space
        self.n_features = dataset.n_features
        self.n_actions = dataset.n_actions
        if not decision.allowed_to_train:
            return CQLResult(trained=False, device="cpu", backend=self.backend,
                             mode=decision.mode, promotable=False,
                             iterations=0, td_loss=float("nan"),
                             conservative_penalty=float("nan"), value_estimate=float("nan"),
                             n_actions=dataset.n_actions,
                             reasons=decision.reasons + ["untrained_research_only"])
        sub = dataset.subset(dataset.reward_known)
        S = self._fit_standardizer(sub.states)
        Sn = self._apply_standardizer(sub.next_states)
        a = sub.actions.astype(np.int64)
        r = sub.rewards.astype(np.float64)
        not_term = (~sub.terminal).astype(np.float64)

        n_elements = max(1, sub.n_transitions * dataset.n_features *
                         self.cfg.iterations * self.cfg.inner_epochs)
        _, mlx_dev = self._choose_device("cql_train", n_elements=n_elements)

        if self.backend == "mlx":
            td, pen = self._fit_mlx(S, a, r, Sn, not_term, mlx_dev)
        else:
            td, pen = self._fit_numpy(S, a, r, Sn, not_term)

        init = dataset.is_initial & dataset.reward_known
        if not np.any(init):
            init = dataset.reward_known
        q0 = self.q_values(dataset.states[init])
        value = float(np.mean(np.max(q0, axis=1))) if len(q0) else float("nan")
        return CQLResult(trained=True, device=self.device, backend=self.backend,
                         mode=decision.mode, promotable=decision.promotable,
                         iterations=self.cfg.iterations, td_loss=float(td),
                         conservative_penalty=float(pen), value_estimate=value,
                         n_actions=self.n_actions, reasons=decision.reasons)

    def _fit_mlx(self, S, a, r, Sn, not_term, dev):  # pragma: no cover on non-MLX hosts
        mx.random.seed(self.seed)
        net = MLXMLP(self.n_features, self.cfg.hidden, self.n_actions)
        opt = optim.AdamW(learning_rate=self.cfg.learning_rate,
                          weight_decay=self.cfg.weight_decay)
        Sd = mx.array(np.asarray(S, dtype=np.float32))
        Snd = mx.array(np.asarray(Sn, dtype=np.float32))
        aidx = mx.array(a.reshape(-1, 1).astype(np.int32))
        rd = mx.array(r.astype(np.float32))
        ntd = mx.array(not_term.astype(np.float32))
        gamma, alpha = float(self.cfg.discount), float(self.cfg.alpha)
        n = Sd.shape[0]
        bs = min(self.cfg.batch_size, n)
        td_loss = pen = float("nan")
        with mx.stream(dev):
            for _ in range(self.cfg.iterations):
                qn = net(Snd)
                maxn = mx.max(qn, axis=1)
                targets = mx.stop_gradient(rd + gamma * ntd * maxn)

                def loss_fn(model, s_b, a_b, t_b):
                    q = model(s_b)
                    qa = mx.take_along_axis(q, a_b, axis=1).squeeze(-1)
                    bellman = mx.mean((qa - t_b) ** 2)
                    conservative = mx.mean(mx.logsumexp(q, axis=1) - qa)
                    return 0.5 * bellman + alpha * conservative

                lvg = nn.value_and_grad(net, loss_fn)
                for b in range(0, n, bs):
                    sl = slice(b, b + bs)
                    _loss, grads = lvg(net, Sd[sl], aidx[sl], targets[sl])
                    opt.update(net, grads)
                    mx.eval(net.parameters(), opt.state)
                q = net(Sd)
                qa = mx.take_along_axis(q, aidx, axis=1).squeeze(-1)
                td_loss = float(mx.mean((qa - targets) ** 2))
                pen = float(mx.mean(mx.logsumexp(q, axis=1) - qa))
        params = [(k, np.array(v)) for k, v in tree_flatten(net.parameters())]
        self._q_fn = self._make_mlx_q_fn(params, dev)
        return td_loss, pen

    def _make_mlx_q_fn(self, params, dev):  # pragma: no cover on non-MLX hosts
        def q_fn(std_states):
            net = MLXMLP(self.n_features, self.cfg.hidden, self.n_actions)
            net.update(tree_unflatten([(k, mx.array(v)) for k, v in params]))
            with mx.stream(dev):
                return np.array(net(mx.array(np.asarray(std_states, dtype=np.float32))))
        return q_fn

    def _fit_numpy(self, S, a, r, Sn, not_term):
        S = np.asarray(S, dtype=np.float64)
        Sn = np.asarray(Sn, dtype=np.float64)
        n, d = S.shape
        A = self.n_actions
        rng = np.random.default_rng(self.seed)
        W = rng.normal(0.0, 0.01, size=(d, A))
        b = np.zeros(A, dtype=np.float64)
        gamma, alpha, lr = float(self.cfg.discount), float(self.cfg.alpha), 0.1
        oh = np.zeros((n, A))
        oh[np.arange(n), a] = 1.0
        td_loss = pen = float("nan")
        for _ in range(self.cfg.iterations):
            maxn = np.max(Sn @ W + b, axis=1)
            targets = r + gamma * not_term * maxn
            for _e in range(self.cfg.inner_epochs):
                q = S @ W + b
                qa = q[np.arange(n), a]
                td = qa - targets
                P = np_softmax(q, axis=1)
                dq = alpha * (P - oh) / n
                dq[np.arange(n), a] += td / n
                W -= lr * (S.T @ dq + self.cfg.ridge_l2 * W)
                b -= lr * dq.sum(axis=0)
            q = S @ W + b
            qa = q[np.arange(n), a]
            td_loss = float(np.mean((qa - targets) ** 2))
            pen = float(np.mean(np_logsumexp(q, axis=1) - qa))
        self._np_W, self._np_b = W, b

        def q_fn(std_states):
            return np.asarray(std_states, dtype=np.float64) @ W + b

        self._q_fn = q_fn
        return td_loss, pen

    # -- inference (advisory only) -------------------------------------------
    def q_values(self, states: np.ndarray) -> np.ndarray:
        if self._q_fn is None:
            raise RuntimeError("ConservativeQLearner is untrained; call fit() first")
        return np.asarray(self._q_fn(self._apply_standardizer(states)), dtype=np.float32)

    def greedy_action(self, states: np.ndarray) -> np.ndarray:
        return np.argmax(self.q_values(states), axis=1).astype(np.int32)

    def recommend(self, state, action_mask=None) -> dict:
        q = self.q_values(np.asarray(state, dtype=np.float32).reshape(1, -1))[0]
        masked = q.copy()
        if action_mask is not None:
            m = np.asarray(action_mask, dtype=np.float32).ravel().astype(bool)
            masked = np.where(m, q, -np.inf)
        idx = int(np.argmax(masked))
        return {"action_index": idx,
                "action": self.action_space.name(idx) if self.action_space else None,
                "q_values": q.tolist(), "advisory_only": True,
                "note": "recommendation-only; not an order; protective exit excluded"}


# ===========================================================================
# IQL
# ===========================================================================

@dataclass
class IQLConfig:
    discount: float = 0.99
    expectile: float = 0.7        # tau in (0.5, 1); higher = more optimistic value
    beta: float = 3.0             # AWR temperature
    weight_clip: float = 100.0
    iterations: int = 8
    inner_epochs: int = 20
    batch_size: int = 256
    learning_rate: float = 1e-3
    hidden: tuple = (64, 32)
    weight_decay: float = 1e-4
    standardize: bool = True
    seed: int = 7
    ridge_l2: float = 1e-2


@dataclass
class IQLResult:
    trained: bool
    device: str
    backend: str
    mode: str
    promotable: bool
    iterations: int
    q_loss: float
    v_loss: float
    policy_loss: float
    value_estimate: float
    n_actions: int
    reasons: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


class ImplicitQLearner(_RLBase):
    """IQL: expectile value, implicit Q bootstrap, advantage-weighted policy (no IPS)."""

    model_type = "offline_rl_iql"

    def __init__(self, *, config: Optional[IQLConfig] = None,
                 device_policy: Optional[DevicePolicy] = None,
                 guard: Optional[TrainingGuard] = None) -> None:
        self.cfg = config or IQLConfig()
        super().__init__(standardize=self.cfg.standardize, device_policy=device_policy,
                         seed=self.cfg.seed)
        self.guard = guard or TrainingGuard()
        self.action_space = None
        self.n_features = 0
        self.n_actions = 0
        self._v_fn = None
        self._policy_fn = None

    def fit(self, dataset: TransitionDataset) -> IQLResult:
        decision = self.guard.check(dataset, method="iql")
        self.action_space = dataset.action_space
        self.n_features = dataset.n_features
        self.n_actions = dataset.n_actions
        if not decision.allowed_to_train:
            return IQLResult(trained=False, device="cpu", backend=self.backend,
                             mode=decision.mode, promotable=False, iterations=0,
                             q_loss=float("nan"), v_loss=float("nan"),
                             policy_loss=float("nan"), value_estimate=float("nan"),
                             n_actions=dataset.n_actions,
                             reasons=decision.reasons + ["untrained_research_only"])
        sub = dataset.subset(dataset.reward_known)
        S = self._fit_standardizer(sub.states)
        Sn = self._apply_standardizer(sub.next_states)
        a = sub.actions.astype(np.int64)
        r = sub.rewards.astype(np.float64)
        not_term = (~sub.terminal).astype(np.float64)

        n_elements = max(1, sub.n_transitions * dataset.n_features *
                         self.cfg.iterations * self.cfg.inner_epochs)
        _, mlx_dev = self._choose_device("iql_train", n_elements=n_elements)

        if self.backend == "mlx":
            ql, vl, pl = self._fit_mlx(S, a, r, Sn, not_term, mlx_dev)
        else:
            ql, vl, pl = self._fit_numpy(S, a, r, Sn, not_term)

        init = dataset.is_initial & dataset.reward_known
        if not np.any(init):
            init = dataset.reward_known
        v0 = self.state_value(dataset.states[init])
        value = float(np.mean(v0)) if len(v0) else float("nan")
        return IQLResult(trained=True, device=self.device, backend=self.backend,
                         mode=decision.mode, promotable=decision.promotable,
                         iterations=self.cfg.iterations, q_loss=float(ql), v_loss=float(vl),
                         policy_loss=float(pl), value_estimate=value,
                         n_actions=self.n_actions, reasons=decision.reasons)

    def _fit_mlx(self, S, a, r, Sn, not_term, dev):  # pragma: no cover on non-MLX hosts
        mx.random.seed(self.seed)
        qnet = MLXMLP(self.n_features, self.cfg.hidden, self.n_actions)
        vnet = MLXMLP(self.n_features, self.cfg.hidden, 1)
        pnet = MLXMLP(self.n_features, self.cfg.hidden, self.n_actions)
        q_opt = optim.AdamW(learning_rate=self.cfg.learning_rate,
                            weight_decay=self.cfg.weight_decay)
        v_opt = optim.AdamW(learning_rate=self.cfg.learning_rate,
                            weight_decay=self.cfg.weight_decay)
        p_opt = optim.AdamW(learning_rate=self.cfg.learning_rate,
                            weight_decay=self.cfg.weight_decay)
        Sd = mx.array(np.asarray(S, dtype=np.float32))
        Snd = mx.array(np.asarray(Sn, dtype=np.float32))
        aidx = mx.array(a.reshape(-1, 1).astype(np.int32))
        a1 = mx.array(a.astype(np.int32))
        rd = mx.array(r.astype(np.float32))
        ntd = mx.array(not_term.astype(np.float32))
        gamma, tau, beta = (float(self.cfg.discount), float(self.cfg.expectile),
                            float(self.cfg.beta))
        wclip = float(self.cfg.weight_clip)
        ql = vl = pl = float("nan")

        def q_loss(model, s_b, a_b, t_b):
            q = model(s_b)
            qa = mx.take_along_axis(q, a_b, axis=1).squeeze(-1)
            return mx.mean((qa - t_b) ** 2)

        def v_loss(model, s_b, qa_b):
            v = model(s_b).squeeze(-1)
            u = qa_b - v
            w = mx.where(u < 0, 1.0 - tau, tau)
            return mx.mean(w * u ** 2)

        def p_loss(model, s_b, a_b, w_b):
            ce = nn.losses.cross_entropy(model(s_b), a_b)
            return mx.mean(w_b * ce)

        q_vg, v_vg, p_vg = (nn.value_and_grad(qnet, q_loss),
                            nn.value_and_grad(vnet, v_loss),
                            nn.value_and_grad(pnet, p_loss))
        with mx.stream(dev):
            for _ in range(self.cfg.iterations):
                vn = vnet(Snd).squeeze(-1)
                q_target = mx.stop_gradient(rd + gamma * ntd * vn)
                for _e in range(self.cfg.inner_epochs):
                    loss, grads = q_vg(qnet, Sd, aidx, q_target)
                    q_opt.update(qnet, grads)
                    mx.eval(qnet.parameters(), q_opt.state)
                    ql = float(loss)
                qa = mx.stop_gradient(
                    mx.take_along_axis(qnet(Sd), aidx, axis=1).squeeze(-1))
                for _e in range(self.cfg.inner_epochs):
                    loss, grads = v_vg(vnet, Sd, qa)
                    v_opt.update(vnet, grads)
                    mx.eval(vnet.parameters(), v_opt.state)
                    vl = float(loss)
                adv = qa - vnet(Sd).squeeze(-1)
                weights = mx.stop_gradient(
                    mx.minimum(mx.exp(mx.minimum(beta * adv, 30.0)), wclip))
                for _e in range(self.cfg.inner_epochs):
                    loss, grads = p_vg(pnet, Sd, a1, weights)
                    p_opt.update(pnet, grads)
                    mx.eval(pnet.parameters(), p_opt.state)
                    pl = float(loss)
        v_params = [(k, np.array(v)) for k, v in tree_flatten(vnet.parameters())]
        p_params = [(k, np.array(v)) for k, v in tree_flatten(pnet.parameters())]
        self._v_fn = self._make_mlx_fn(v_params, 1, dev)
        self._policy_fn = self._make_mlx_fn(p_params, self.n_actions, dev)
        return ql, vl, pl

    def _make_mlx_fn(self, params, n_out, dev):  # pragma: no cover on non-MLX hosts
        def fn(std_states):
            net = MLXMLP(self.n_features, self.cfg.hidden, n_out)
            net.update(tree_unflatten([(k, mx.array(v)) for k, v in params]))
            with mx.stream(dev):
                return np.array(net(mx.array(np.asarray(std_states, dtype=np.float32))))
        return fn

    def _fit_numpy(self, S, a, r, Sn, not_term):
        S = np.asarray(S, dtype=np.float64)
        Sn = np.asarray(Sn, dtype=np.float64)
        n, d = S.shape
        A = self.n_actions
        rng = np.random.default_rng(self.seed)
        Wq = rng.normal(0.0, 0.01, size=(d, A))
        bq = np.zeros(A)
        wv = np.zeros(d)
        bv = 0.0
        gamma, tau, beta = (float(self.cfg.discount), float(self.cfg.expectile),
                            float(self.cfg.beta))
        lr, wclip = 0.1, float(self.cfg.weight_clip)
        ql = vl = pl = float("nan")
        Wp = bp = None
        for _ in range(self.cfg.iterations):
            vn = Sn @ wv + bv
            q_target = r + gamma * not_term * vn
            for _e in range(self.cfg.inner_epochs):
                q = S @ Wq + bq
                qa = q[np.arange(n), a]
                td = qa - q_target
                dq = np.zeros((n, A))
                dq[np.arange(n), a] = 2.0 * td / n
                Wq -= lr * (S.T @ dq + self.cfg.ridge_l2 * Wq)
                bq -= lr * dq.sum(axis=0)
                ql = float(np.mean(td ** 2))
            qa = (S @ Wq + bq)[np.arange(n), a]
            for _e in range(self.cfg.inner_epochs):
                v = S @ wv + bv
                u = qa - v
                w = np.where(u < 0, 1.0 - tau, tau)
                dv = -2.0 * w * u / n
                wv -= lr * (S.T @ dv + self.cfg.ridge_l2 * wv)
                bv -= lr * dv.sum()
                vl = float(np.mean(w * u ** 2))
            adv = qa - (S @ wv + bv)
            weights = np.minimum(np.exp(np.minimum(beta * adv, 30.0)), wclip)
            Wp, bp, hist = np_softmax_regression(S, a, A, sample_weight=weights,
                                                 epochs=self.cfg.inner_epochs * 3,
                                                 lr=0.3, l2=self.cfg.weight_decay,
                                                 seed=self.seed)
            pl = float(hist[-1]) if hist else float("nan")

        def v_fn(std_states):
            return (np.asarray(std_states, dtype=np.float64) @ wv + bv).reshape(-1, 1)

        def p_fn(std_states):
            return np.asarray(std_states, dtype=np.float64) @ Wp + bp

        self._v_fn, self._policy_fn = v_fn, p_fn
        return ql, vl, pl

    # -- inference (advisory only) -------------------------------------------
    def state_value(self, states: np.ndarray) -> np.ndarray:
        if self._v_fn is None:
            raise RuntimeError("ImplicitQLearner is untrained; call fit() first")
        return np.asarray(self._v_fn(self._apply_standardizer(states)),
                          dtype=np.float32).ravel()

    def policy_action_probs(self, states: np.ndarray) -> np.ndarray:
        if self._policy_fn is None:
            raise RuntimeError("ImplicitQLearner is untrained; call fit() first")
        logits = self._policy_fn(self._apply_standardizer(states))
        return np_softmax(np.asarray(logits, dtype=np.float64), axis=1).astype(np.float32)

    def greedy_action(self, states: np.ndarray) -> np.ndarray:
        return np.argmax(self.policy_action_probs(states), axis=1).astype(np.int32)

    def predict_action(self, states: np.ndarray) -> np.ndarray:
        """Alias so an IQL policy can be used directly as an FQE target policy."""
        return self.greedy_action(states)

    def recommend(self, state, action_mask=None) -> dict:
        probs = self.policy_action_probs(np.asarray(state, dtype=np.float32).reshape(1, -1))[0]
        if action_mask is not None:
            m = np.asarray(action_mask, dtype=np.float32).ravel()
            probs = probs * m
            if probs.sum() <= 0:
                probs = np.asarray(action_mask, dtype=np.float32).ravel()
        idx = int(np.argmax(probs))
        return {"action_index": idx,
                "action": self.action_space.name(idx) if self.action_space else None,
                "probabilities": probs.tolist(), "advisory_only": True,
                "note": "recommendation-only; not an order; protective exit excluded"}
