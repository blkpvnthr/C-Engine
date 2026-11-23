"""MLX (Apple Metal GPU) neural contextual-bandit reward model (Milestone 6, §6/§7).

A GPU-first neural DIRECT reward model for offline contextual bandits over a FIXED set of
AUTHORIZED actions: a shared MLP trunk with one reward head per action, trained on logged
``(context, selected_action, reward)`` rows with a per-sample MASKED loss (only the head of
the action actually taken receives gradient for that row). It is the learned counterpart to
the numpy :class:`LinUCB` baseline and must beat that baseline on held-out data before any
promotion — never preferred merely for running on the GPU.

Mirrors :mod:`cengine.learning.supervised.mlx_models`: MLX is import-guarded (the module
imports on a machine without MLX and construction raises a clear error), training uses
``nn.value_and_grad`` / ``opt.update`` / ``mx.eval`` under ``mx.stream(policy.mlx_device)``,
and weights serialize to portable numpy. Device placement is chosen by the shared
:class:`DevicePolicy` (GPU-first with a DOCUMENTED, OBSERVABLE, MEASURABLE CPU fallback) and
is RECORDED on every fit — the platform never silently claims GPU acceleration.

Invariants:
  * Actions are FIXED/authorized by the caller; the model never invents actions.
  * Only DIRECT reward-model scoring is exposed. NO IPS / propensity / doubly-robust path
    exists — behavior-policy probabilities are unknown and never fabricated.
  * Recommendation-only: scoring offline contexts emits NO orders and runs NO live
    exploration; it never touches the mandatory protective-exit path.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from cengine.learning.hardware import DevicePolicy, mlx_available
from cengine.learning.supervised.base import FitResult, Standardizer, TrainConfig

try:  # pragma: no cover - import guard
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx.utils import tree_flatten, tree_unflatten

    _MLX = True
except Exception:  # noqa: BLE001
    mx = None  # type: ignore
    nn = object  # type: ignore
    _MLX = False


def _require_mlx() -> None:
    if not _MLX:
        raise RuntimeError("MLX is not available; use the LinUCB numpy baseline instead")


def neural_bandit_available() -> bool:
    """True iff MLX imported AND a Metal GPU is actually usable."""
    return _MLX and mlx_available()


if _MLX:
    class _BanditNet(nn.Module):
        """Shared ReLU MLP trunk with a linear reward head per authorized action."""

        def __init__(self, n_in: int, hidden: tuple, n_actions: int) -> None:
            super().__init__()
            dims = [n_in, *hidden, n_actions]
            self.layers = [nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)]

        def __call__(self, x):
            for i, layer in enumerate(self.layers):
                x = layer(x)
                if i < len(self.layers) - 1:
                    x = nn.relu(x)
            return x
else:  # pragma: no cover
    class _BanditNet:  # type: ignore
        def __init__(self, *a, **k):
            _require_mlx()


class NeuralBanditRewardModel:
    """GPU-first neural per-action reward model for offline contextual bandits."""

    framework = "mlx"
    model_type = "neural_contextual_bandit"

    def __init__(self, n_features: int, n_actions: int, *,
                 config: Optional[TrainConfig] = None,
                 action_names: Optional[list[str]] = None,
                 device_policy: Optional[DevicePolicy] = None) -> None:
        _require_mlx()
        if n_features < 1 or n_actions < 1:
            raise ValueError("n_features and n_actions must be >= 1")
        self.cfg = config or TrainConfig()
        self.n_features = int(n_features)
        self.n_actions = int(n_actions)
        self.action_names = (list(action_names) if action_names is not None
                             else [f"action_{i}" for i in range(self.n_actions)])
        if len(self.action_names) != self.n_actions:
            raise ValueError("action_names length must equal n_actions")
        self.policy = device_policy or DevicePolicy()
        self._std = Standardizer() if self.cfg.standardize else None
        self._net = _BanditNet(self.n_features, self.cfg.hidden, self.n_actions)
        self.device = "cpu"

    def _mx_device(self, n_rows: int):
        decision = self.policy.choose(
            f"train_{self.model_type}",
            n_elements=max(1, n_rows) * max(1, self.n_features))
        self.device = decision.device
        return self.policy.mlx_device(decision)

    def fit(self, contexts: np.ndarray, actions: np.ndarray, rewards: np.ndarray,
            *, reward_mask: Optional[np.ndarray] = None,
            X_val: Optional[np.ndarray] = None, a_val: Optional[np.ndarray] = None,
            r_val: Optional[np.ndarray] = None) -> FitResult:
        """Train the per-action reward heads on labeled (reward-present) rows only."""
        X = np.asarray(contexts, dtype=np.float32)
        a = np.asarray(actions, dtype=np.int64)
        r = np.asarray(rewards, dtype=np.float32)
        if reward_mask is not None:
            m = np.asarray(reward_mask, dtype=bool)
            X, a, r = X[m], a[m], r[m]
        if X.shape[0] == 0:
            # No labeled evidence -> untrained, research-only result (never promoted).
            return FitResult(train_loss=float("nan"), epochs_run=0, device=self.device)

        mx.random.seed(self.cfg.seed)
        dev = self._mx_device(X.shape[0])
        if self._std is not None:
            X = self._std.fit(X).transform(X)
            if X_val is not None:
                X_val = self._std.transform(np.asarray(X_val, dtype=np.float32))
        Xd = mx.array(X)
        ad = mx.array(a.astype(np.int32)).reshape((-1, 1))
        rd = mx.array(r)
        net = self._net
        opt = optim.AdamW(learning_rate=self.cfg.learning_rate,
                          weight_decay=self.cfg.weight_decay)

        def loss_fn(model, xb, ab, rb):
            pred = model(xb)  # (B, n_actions)
            taken = mx.take_along_axis(pred, ab, axis=1).squeeze(-1)
            return mx.mean((taken - rb) ** 2)

        lvg = nn.value_and_grad(net, loss_fn)
        n = Xd.shape[0]
        bs = min(self.cfg.batch_size, n)
        best_val = float("inf")
        best_params = None
        bad = 0
        history = []
        ep_loss = 0.0
        epochs_run = 0
        with mx.stream(dev):
            for epoch in range(self.cfg.epochs):
                perm = np.random.default_rng(self.cfg.seed + epoch).permutation(n)
                idx = mx.array(perm)
                Xs, as_, rs = Xd[idx], ad[idx], rd[idx]
                ep_loss = 0.0
                nb = 0
                for b in range(0, n, bs):
                    xb, ab, rb = Xs[b:b + bs], as_[b:b + bs], rs[b:b + bs]
                    loss, grads = lvg(net, xb, ab, rb)
                    opt.update(net, grads)
                    mx.eval(net.parameters(), opt.state)
                    ep_loss += float(loss)
                    nb += 1
                ep_loss /= max(1, nb)
                epochs_run = epoch + 1
                vl = None
                if (X_val is not None and a_val is not None and r_val is not None
                        and len(X_val)):
                    vout = net(mx.array(np.asarray(X_val, dtype=np.float32)))
                    vtaken = mx.take_along_axis(
                        vout, mx.array(np.asarray(a_val, dtype=np.int32)).reshape((-1, 1)),
                        axis=1).squeeze(-1)
                    vl = float(mx.mean((vtaken - mx.array(
                        np.asarray(r_val, dtype=np.float32))) ** 2))
                    mx.eval(vout)
                history.append({"epoch": epoch, "train_loss": ep_loss, "val_loss": vl})
                if vl is not None:
                    if vl < best_val - 1e-6:
                        best_val = vl
                        best_params = tree_flatten(net.parameters())
                        bad = 0
                    else:
                        bad += 1
                        if bad >= self.cfg.patience:
                            break
        if best_params is not None:
            net.update(tree_unflatten(best_params))
            mx.eval(net.parameters())
        return FitResult(train_loss=float(ep_loss),
                         val_loss=(best_val if best_params else None),
                         epochs_run=epochs_run, device=self.device, history=history)

    # -- scoring -------------------------------------------------------------
    def predict_reward(self, contexts: np.ndarray) -> np.ndarray:
        """Direct expected-reward estimate per action: ``(n, n_actions)``."""
        X = np.asarray(contexts, dtype=np.float32)
        if X.ndim != 2 or X.shape[1] != self.n_features:
            raise ValueError(f"contexts must be (n, {self.n_features})")
        if self._std is not None:
            X = self._std.transform(X)
        decision = self.policy.choose(f"infer_{self.model_type}",
                                      n_elements=max(1, len(X)) * self.n_features)
        with mx.stream(self.policy.mlx_device(decision)):
            out = self._net(mx.array(X))
            mx.eval(out)
        return np.array(out)

    def recommend(self, contexts: np.ndarray, *,
                  eligibility_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Return the argmax action index per context (RESEARCH recommendation only).

        ``eligibility_mask`` (``(n, n_actions)`` bool) restricts to authorized/eligible
        actions. Greedy direct-reward scoring only — no exploration, no orders.
        """
        scores = self.predict_reward(contexts)
        if eligibility_mask is not None:
            mask = np.asarray(eligibility_mask, dtype=bool)
            if mask.shape != scores.shape:
                raise ValueError("eligibility_mask must match scores shape (n, n_actions)")
            scores = scores.copy()
            scores[~mask] = -np.inf
        return np.argmax(scores, axis=1)

    # -- serialization -------------------------------------------------------
    def params(self) -> dict:
        flat = dict(tree_flatten(self._net.parameters()))
        out = {f"w::{k}": np.array(v) for k, v in flat.items()}
        if self._std is not None:
            out["std::mean"] = (self._std.mean if self._std.mean is not None
                                else np.zeros(self.n_features, dtype=np.float32))
            out["std::std"] = (self._std.std if self._std.std is not None
                               else np.ones(self.n_features, dtype=np.float32))
        out["meta::config"] = np.frombuffer(
            __import__("json").dumps(
                {"hidden": list(self.cfg.hidden), "n_features": self.n_features,
                 "n_actions": self.n_actions, "action_names": self.action_names}
            ).encode(), dtype=np.uint8)
        return out

    def load_params(self, params: dict) -> None:
        flat = [(k[3:], mx.array(v)) for k, v in params.items() if k.startswith("w::")]
        if flat:
            self._net.update(tree_unflatten(flat))
            mx.eval(self._net.parameters())
        if "std::mean" in params and self._std is not None:
            self._std.mean = np.asarray(params["std::mean"], dtype=np.float32)
            self._std.std = np.asarray(params["std::std"], dtype=np.float32)
