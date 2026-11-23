"""MLX (Apple Metal GPU) neural supervised models (Milestone 6 addendum §6/§7).

GPU-first neural regression / classification / quantile regression trained with MLX
autodiff on the Metal GPU, with a documented CPU fallback (via DevicePolicy) and an
explicit device record on every fit. Weights serialize to portable numpy (npz) for the
checksummed artifact store. These are compared against the independent sklearn/numpy
baselines (baselines.py) — a neural model is never preferred merely for running on the GPU.

Import is guarded: on a machine without MLX the module still imports, and constructing a
model raises a clear error so the trainer degrades to baselines/CPU instead of crashing.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from cengine.learning.hardware import DevicePolicy, mlx_available
from cengine.learning.supervised.base import FitResult, LearningModel, Standardizer, TrainConfig

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
        raise RuntimeError("MLX is not available; use a CPU baseline instead")


if _MLX:
    class _MLP(nn.Module):
        """Configurable MLP with ReLU hidden layers and a linear output head."""

        def __init__(self, n_in: int, hidden: tuple, n_out: int) -> None:
            super().__init__()
            dims = [n_in, *hidden, n_out]
            self.layers = [nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)]

        def __call__(self, x):
            for i, layer in enumerate(self.layers):
                x = layer(x)
                if i < len(self.layers) - 1:
                    x = nn.relu(x)
            return x
else:  # pragma: no cover
    class _MLP:  # type: ignore
        def __init__(self, *a, **k):
            _require_mlx()


class _MLXBase(LearningModel):
    framework = "mlx"

    def __init__(self, n_features: int, *, n_out: int = 1,
                 config: Optional[TrainConfig] = None,
                 device_policy: Optional[DevicePolicy] = None) -> None:
        _require_mlx()
        self.cfg = config or TrainConfig()
        self.n_features = int(n_features)
        self.n_out = int(n_out)
        self.policy = device_policy or DevicePolicy()
        self._std = Standardizer() if self.cfg.standardize else None
        self._net = _MLP(self.n_features, self.cfg.hidden, self.n_out)
        self.device = "cpu"

    # subclasses implement the per-example loss on raw model output
    def _loss(self, out, y):  # out: (B, n_out) mx.array, y: (B,) mx.array
        raise NotImplementedError

    def _mx_device(self):
        d = self.policy.choose(f"train_{self.model_type}",
                               n_elements=max(1, self.n_features) * 20_000)
        self.device = d.device
        return self.policy.mlx_device(d)

    def fit(self, X, y, *, X_val=None, y_val=None) -> FitResult:
        mx.random.seed(self.cfg.seed)
        dev = self._mx_device()
        if self._std is not None:
            X = self._std.fit(X).transform(X)
            if X_val is not None:
                X_val = self._std.transform(X_val)
        Xd = mx.array(np.asarray(X, dtype=np.float32))
        yd = mx.array(np.asarray(y, dtype=np.float32))
        net = self._net
        opt = optim.AdamW(learning_rate=self.cfg.learning_rate,
                          weight_decay=self.cfg.weight_decay)

        def loss_fn(model, xb, yb):
            return self._loss(model(xb), yb)

        lvg = nn.value_and_grad(net, loss_fn)
        n = Xd.shape[0]
        bs = min(self.cfg.batch_size, n)
        best_val = float("inf")
        best_params = None
        patience = self.cfg.patience
        bad = 0
        history = []
        epochs_run = 0
        with mx.stream(dev):
            for epoch in range(self.cfg.epochs):
                perm = np.random.default_rng(self.cfg.seed + epoch).permutation(n)
                idx = mx.array(perm)
                Xs, ys = Xd[idx], yd[idx]
                ep_loss = 0.0
                nb = 0
                for b in range(0, n, bs):
                    xb, yb = Xs[b:b + bs], ys[b:b + bs]
                    loss, grads = lvg(net, xb, yb)
                    opt.update(net, grads)
                    mx.eval(net.parameters(), opt.state)
                    ep_loss += float(loss)
                    nb += 1
                ep_loss /= max(1, nb)
                epochs_run = epoch + 1
                vl = None
                if X_val is not None and y_val is not None and len(X_val):
                    vout = net(mx.array(np.asarray(X_val, dtype=np.float32)))
                    vl = float(self._loss(vout, mx.array(np.asarray(y_val, dtype=np.float32))))
                    mx.eval(vout)
                history.append({"epoch": epoch, "train_loss": ep_loss, "val_loss": vl})
                if vl is not None:
                    if vl < best_val - 1e-6:
                        best_val = vl
                        best_params = tree_flatten(net.parameters())
                        bad = 0
                    else:
                        bad += 1
                        if bad >= patience:
                            break
        if best_params is not None:
            net.update(tree_unflatten(best_params))
            mx.eval(net.parameters())
        return FitResult(train_loss=ep_loss, val_loss=(best_val if best_params else None),
                         epochs_run=epochs_run, device=self.device, history=history)

    def _forward_np(self, X) -> np.ndarray:
        if self._std is not None:
            X = self._std.transform(X)
        decision = self.policy.choose(f"infer_{self.model_type}",
                                      n_elements=len(X) * self.n_features)
        with mx.stream(self.policy.mlx_device(decision)):
            out = self._net(mx.array(np.asarray(X, dtype=np.float32)))
            mx.eval(out)
        return np.array(out)

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
                 "n_out": self.n_out, "quantiles": list(self.cfg.quantiles)}
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


class MLXRegressor(_MLXBase):
    model_type = "mlx_regressor"

    def __init__(self, n_features, **kw) -> None:
        super().__init__(n_features, n_out=1, **kw)

    def _loss(self, out, y):
        return mx.mean((out.squeeze(-1) - y) ** 2)

    def predict(self, X) -> np.ndarray:
        return self._forward_np(X).squeeze(-1).astype(np.float32)


class MLXBinaryClassifier(_MLXBase):
    model_type = "mlx_classifier"

    def __init__(self, n_features, **kw) -> None:
        super().__init__(n_features, n_out=1, **kw)

    def _loss(self, out, y):
        logits = out.squeeze(-1)
        # numerically-stable BCE-with-logits
        return mx.mean(mx.maximum(logits, 0) - logits * y + mx.log1p(mx.exp(-mx.abs(logits))))

    def predict(self, X) -> np.ndarray:
        """Return P(y=1)."""
        logits = self._forward_np(X).squeeze(-1)
        return (1.0 / (1.0 + np.exp(-logits))).astype(np.float32)


class MLXQuantileRegressor(_MLXBase):
    """Predicts multiple quantiles with pinball loss → point + uncertainty + downside."""

    model_type = "mlx_quantile_regressor"

    def __init__(self, n_features, *, config: Optional[TrainConfig] = None, **kw) -> None:
        cfg = config or TrainConfig()
        super().__init__(n_features, n_out=len(cfg.quantiles), config=cfg, **kw)
        self._q = mx.array(np.asarray(cfg.quantiles, dtype=np.float32)) if _MLX else None

    def _loss(self, out, y):
        # out: (B, Q), y: (B,) -> pinball loss averaged over quantiles
        yb = y.reshape((-1, 1))
        err = yb - out
        return mx.mean(mx.maximum(self._q * err, (self._q - 1) * err))

    def predict(self, X) -> np.ndarray:
        out = self._forward_np(X)  # (B, Q)
        qs = list(self.cfg.quantiles)
        mid = qs.index(0.5) if 0.5 in qs else len(qs) // 2
        return out[:, mid].astype(np.float32)

    def predict_with_uncertainty(self, X) -> tuple:
        out = self._forward_np(X)  # (B, Q)
        qs = list(self.cfg.quantiles)
        mid = qs.index(0.5) if 0.5 in qs else len(qs) // 2
        lo, hi = 0, len(qs) - 1
        pred = out[:, mid].astype(np.float32)
        unc = ((out[:, hi] - out[:, lo]) / 2.0).astype(np.float32)
        return pred, np.abs(unc)

    def predict_downside(self, X) -> np.ndarray:
        """Lowest-quantile prediction (expected downside, §7)."""
        return self._forward_np(X)[:, 0].astype(np.float32)


def mlx_is_available() -> bool:
    return _MLX and mlx_available()
