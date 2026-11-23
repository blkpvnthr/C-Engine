"""Behavior Cloning for the constrained offline-RL framework (Milestone 6, §6/§7).

BC is the simplest conservative offline learner: it imitates the LOGGED policy by fitting a
discrete classifier ``pi(a | s)`` to the ``(state, logged_action)`` pairs. It needs neither
rewards nor behavior-policy probabilities, so it is always safe to run as a reference /
support model and as FQE's evaluation target.

GPU-first (MLX MLP + cross-entropy on the Metal GPU) with a documented, observable numpy
CPU fallback (multinomial logistic regression) selected via ``DevicePolicy`` and recorded
on every fit. Everything here is RECOMMENDATION-ONLY research infrastructure: ``recommend``
returns an advisory action, never an order, and the mandatory protective exit is never in
the action space (it is excluded upstream as environment dynamics), so BC can never imitate
or recommend it.
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

BC_VERSION = "offline-rl-bc-1.0.0"


@dataclass
class BCConfig:
    epochs: int = 120
    batch_size: int = 256
    learning_rate: float = 1e-3
    hidden: tuple = (64, 32)
    weight_decay: float = 1e-4
    patience: int = 12
    standardize: bool = True
    seed: int = 7


@dataclass
class BCResult:
    trained: bool
    device: str
    backend: str
    mode: str
    promotable: bool
    n_actions: int
    n_features: int
    train_loss: float = 0.0
    val_loss: Optional[float] = None
    train_accuracy: float = 0.0
    val_accuracy: Optional[float] = None
    epochs_run: int = 0
    reasons: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


class BehaviorCloner(_RLBase):
    """Imitation-learning policy ``pi(a|s)`` over the authorized (non-exit) action space."""

    model_type = "offline_rl_bc"

    def __init__(self, *, config: Optional[BCConfig] = None,
                 device_policy: Optional[DevicePolicy] = None,
                 guard: Optional[TrainingGuard] = None) -> None:
        self.cfg = config or BCConfig()
        super().__init__(standardize=self.cfg.standardize, device_policy=device_policy,
                         seed=self.cfg.seed)
        self.guard = guard or TrainingGuard()
        self.action_space = None
        self.n_features = 0
        self.n_actions = 0
        self._mlx_params = None          # tree_flatten list (GPU path)
        self._np_W = None                # (d, A) (CPU path)
        self._np_b = None                # (A,)

    # -- fit -----------------------------------------------------------------
    def fit(self, dataset: TransitionDataset, *,
            val_dataset: Optional[TransitionDataset] = None) -> BCResult:
        decision = self.guard.check(dataset, method="bc")
        self.action_space = dataset.action_space
        self.n_features = dataset.n_features
        self.n_actions = dataset.n_actions
        if not decision.allowed_to_train:
            return BCResult(trained=False, device="cpu", backend=self.backend,
                            mode=decision.mode, promotable=False,
                            n_actions=dataset.n_actions, n_features=dataset.n_features,
                            reasons=decision.reasons + ["untrained_research_only"])

        X = self._fit_standardizer(dataset.states)
        y = dataset.actions.astype(np.int64)
        Xv = yv = None
        if val_dataset is not None and val_dataset.n_transitions:
            Xv = self._apply_standardizer(val_dataset.states)
            yv = val_dataset.actions.astype(np.int64)

        n_elements = max(1, dataset.n_transitions * dataset.n_features * self.cfg.epochs)
        _, mlx_dev = self._choose_device("train_bc", n_elements=n_elements)

        if self.backend == "mlx":
            res = self._fit_mlx(X, y, Xv, yv, mlx_dev)
        else:
            res = self._fit_numpy(X, y, Xv, yv)

        res.mode = decision.mode
        res.promotable = decision.promotable
        res.reasons = decision.reasons
        return res

    def _fit_mlx(self, X, y, Xv, yv, dev) -> BCResult:  # pragma: no cover on non-MLX hosts
        mx.random.seed(self.seed)
        net = MLXMLP(self.n_features, self.cfg.hidden, self.n_actions)
        opt = optim.AdamW(learning_rate=self.cfg.learning_rate,
                          weight_decay=self.cfg.weight_decay)
        Xd = mx.array(np.asarray(X, dtype=np.float32))
        yd = mx.array(np.asarray(y, dtype=np.int32))

        def loss_fn(model, xb, yb):
            return mx.mean(nn.losses.cross_entropy(model(xb), yb))

        lvg = nn.value_and_grad(net, loss_fn)
        n = Xd.shape[0]
        bs = min(self.cfg.batch_size, n)
        best_val = float("inf")
        best_params = None
        bad = 0
        epochs_run = 0
        ep_loss = 0.0
        with mx.stream(dev):
            for epoch in range(self.cfg.epochs):
                perm = np.random.default_rng(self.seed + epoch).permutation(n)
                idx = mx.array(perm)
                Xs, ys = Xd[idx], yd[idx]
                ep_loss, nb = 0.0, 0
                for b in range(0, n, bs):
                    loss, grads = lvg(net, Xs[b:b + bs], ys[b:b + bs])
                    opt.update(net, grads)
                    mx.eval(net.parameters(), opt.state)
                    ep_loss += float(loss)
                    nb += 1
                ep_loss /= max(1, nb)
                epochs_run = epoch + 1
                if Xv is not None:
                    vl = float(loss_fn(net, mx.array(np.asarray(Xv, dtype=np.float32)),
                                       mx.array(np.asarray(yv, dtype=np.int32))))
                    if vl < best_val - 1e-6:
                        best_val, best_params, bad = vl, tree_flatten(net.parameters()), 0
                    else:
                        bad += 1
                        if bad >= self.cfg.patience:
                            break
        if best_params is not None:
            net.update(tree_unflatten(best_params))
            mx.eval(net.parameters())
        self._mlx_params = tree_flatten(net.parameters())
        train_acc = self._accuracy_mlx(net, X, y, dev)
        val_acc = self._accuracy_mlx(net, Xv, yv, dev) if Xv is not None else None
        return BCResult(trained=True, device=self.device, backend="mlx", mode="",
                        promotable=False, n_actions=self.n_actions,
                        n_features=self.n_features, train_loss=ep_loss,
                        val_loss=(best_val if best_params is not None else None),
                        train_accuracy=train_acc, val_accuracy=val_acc,
                        epochs_run=epochs_run)

    def _accuracy_mlx(self, net, X, y, dev) -> float:  # pragma: no cover on non-MLX hosts
        if X is None or len(X) == 0:
            return 0.0
        with mx.stream(dev):
            logits = net(mx.array(np.asarray(X, dtype=np.float32)))
            pred = np.array(mx.argmax(logits, axis=1))
        return float(np.mean(pred == np.asarray(y)))

    def _fit_numpy(self, X, y, Xv, yv) -> BCResult:
        W, b, hist = np_softmax_regression(
            np.asarray(X, dtype=np.float64), np.asarray(y), self.n_actions,
            epochs=self.cfg.epochs, lr=0.3, l2=self.cfg.weight_decay, seed=self.seed)
        self._np_W, self._np_b = W, b
        train_acc = float(np.mean(np.argmax(X @ W + b, axis=1) == y))
        val_acc = (float(np.mean(np.argmax(Xv @ W + b, axis=1) == yv))
                   if Xv is not None else None)
        return BCResult(trained=True, device=self.device, backend="numpy", mode="",
                        promotable=False, n_actions=self.n_actions,
                        n_features=self.n_features,
                        train_loss=float(hist[-1]) if hist else 0.0,
                        train_accuracy=train_acc, val_accuracy=val_acc,
                        epochs_run=self.cfg.epochs)

    # -- inference (advisory only) -------------------------------------------
    def predict_action_probs(self, states: np.ndarray) -> np.ndarray:
        """Return ``pi(a|s)`` as an ``(N, n_actions)`` probability matrix."""
        X = self._apply_standardizer(np.asarray(states, dtype=np.float32))
        if self.backend == "mlx" and self._mlx_params is not None:  # pragma: no cover non-MLX
            net = MLXMLP(self.n_features, self.cfg.hidden, self.n_actions)
            net.update(tree_unflatten(self._mlx_params))
            logits = np.array(net(mx.array(X)))
        elif self._np_W is not None:
            logits = X @ self._np_W + self._np_b
        else:
            raise RuntimeError("BehaviorCloner is untrained; call fit() first")
        return np_softmax(logits, axis=1).astype(np.float32)

    def predict_action(self, states: np.ndarray) -> np.ndarray:
        """Greedy action index under the cloned policy (``argmax`` of ``pi``)."""
        return np.argmax(self.predict_action_probs(states), axis=1).astype(np.int32)

    def recommend(self, state: np.ndarray,
                  action_mask: Optional[np.ndarray] = None) -> dict:
        """ADVISORY-ONLY recommendation for one state. Never an order; the protective exit
        is not in the action space and can never be recommended here."""
        state = np.asarray(state, dtype=np.float32).reshape(1, -1)
        probs = self.predict_action_probs(state)[0]
        if action_mask is not None:
            m = np.asarray(action_mask, dtype=np.float32).ravel()
            probs = probs * m
            if probs.sum() <= 0:
                probs = np.asarray(action_mask, dtype=np.float32).ravel()
        idx = int(np.argmax(probs))
        return {"action_index": idx,
                "action": self.action_space.name(idx) if self.action_space else None,
                "probabilities": probs.tolist(),
                "advisory_only": True,
                "note": "recommendation-only; not an order; protective exit excluded"}

    # -- serialization -------------------------------------------------------
    def params(self) -> dict:
        out: dict = {"meta::version": np.frombuffer(BC_VERSION.encode(), dtype=np.uint8),
                     "meta::n_features": np.array([self.n_features], dtype=np.int64),
                     "meta::n_actions": np.array([self.n_actions], dtype=np.int64),
                     "meta::backend": np.frombuffer(self.backend.encode(), dtype=np.uint8)}
        if self._std is not None and self._std.mean is not None:
            out["std::mean"] = self._std.mean
            out["std::std"] = self._std.std
        if self._mlx_params is not None:  # pragma: no cover on non-MLX hosts
            for k, v in self._mlx_params:
                out[f"w::{k}"] = np.array(v)
        if self._np_W is not None:
            out["np::W"] = self._np_W.astype(np.float32)
            out["np::b"] = self._np_b.astype(np.float32)
        return out

    def load_params(self, params: dict) -> None:
        self.n_features = int(np.asarray(params["meta::n_features"]).ravel()[0])
        self.n_actions = int(np.asarray(params["meta::n_actions"]).ravel()[0])
        if "std::mean" in params and self._std is not None:
            self._std.mean = np.asarray(params["std::mean"], dtype=np.float32)
            self._std.std = np.asarray(params["std::std"], dtype=np.float32)
        if "np::W" in params:
            self._np_W = np.asarray(params["np::W"], dtype=np.float64)
            self._np_b = np.asarray(params["np::b"], dtype=np.float64)
            self.backend = "numpy"
        mlx_w = [(k[3:], mx.array(v)) for k, v in params.items()
                 if k.startswith("w::")] if _MLX else []
        if mlx_w:  # pragma: no cover on non-MLX hosts
            self._mlx_params = mlx_w
            self.backend = "mlx"
