"""Shared network / tensor primitives for the constrained offline-RL framework (M6, §6/§7).

This private module factors out what BC / FQE / CQL / IQL all need so the four public
algorithm files do not duplicate it:

* an import-guarded MLX (Apple Metal GPU) MLP following the reference pattern in
  ``cengine/learning/supervised/mlx_models.py`` (``try import mlx.* ; _MLX`` flag;
  ``_require_mlx`` raises only when a GPU path is explicitly requested on a host without
  MLX), so non-MLX machines still import this package and degrade to the numpy CPU path;
* small, bullet-proof numpy helpers (softmax / one-hot / logsumexp) for the documented,
  observable CPU fallback — never a silent GPU claim; and
* ``_RLBase``, which holds the standardizer (fit on TRAIN states only — no leakage),
  the ``DevicePolicy`` bookkeeping, and the GPU-first device choice that records the
  actual device used on every fit.

Nothing here fabricates data: it is pure numerics over whatever transitions it is given.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from cengine.learning.hardware import DevicePolicy
from cengine.learning.supervised.base import Standardizer

try:  # pragma: no cover - import guard (absence must degrade, not crash)
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx.utils import tree_flatten, tree_unflatten

    _MLX = True
except Exception:  # noqa: BLE001
    mx = None  # type: ignore
    nn = object  # type: ignore
    optim = None  # type: ignore
    tree_flatten = tree_unflatten = None  # type: ignore
    _MLX = False


def mlx_backend_available() -> bool:
    """True iff the MLX python package imported (Metal usability is checked separately)."""
    return _MLX


def _require_mlx() -> None:
    if not _MLX:
        raise RuntimeError("MLX is not available; the numpy CPU fallback must be used instead")


if _MLX:
    class MLXMLP(nn.Module):
        """Configurable ReLU MLP: ``(B, n_in) -> (B, n_out)`` (matches mlx_models._MLP)."""

        def __init__(self, n_in: int, hidden: tuple, n_out: int) -> None:
            super().__init__()
            dims = [int(n_in), *[int(h) for h in hidden], int(n_out)]
            self.layers = [nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)]

        def __call__(self, x):
            for i, layer in enumerate(self.layers):
                x = layer(x)
                if i < len(self.layers) - 1:
                    x = nn.relu(x)
            return x
else:  # pragma: no cover - exercised only on non-MLX hosts
    class MLXMLP:  # type: ignore
        def __init__(self, *a, **k):
            _require_mlx()


# -- numpy helpers (CPU fallback) --------------------------------------------

def np_softmax(z: np.ndarray, axis: int = -1) -> np.ndarray:
    z = z - np.max(z, axis=axis, keepdims=True)
    e = np.exp(z)
    return e / (np.sum(e, axis=axis, keepdims=True) + 1e-12)


def np_logsumexp(z: np.ndarray, axis: int = -1) -> np.ndarray:
    m = np.max(z, axis=axis, keepdims=True)
    return (m + np.log(np.sum(np.exp(z - m), axis=axis, keepdims=True) + 1e-12)).squeeze(axis)


def np_one_hot(idx: np.ndarray, n: int) -> np.ndarray:
    out = np.zeros((len(idx), n), dtype=np.float64)
    out[np.arange(len(idx)), idx.astype(int)] = 1.0
    return out


def np_softmax_regression(
    X: np.ndarray, y: np.ndarray, n_classes: int, *,
    sample_weight: Optional[np.ndarray] = None,
    epochs: int = 300, lr: float = 0.3, l2: float = 1e-4, seed: int = 7,
) -> tuple:
    """Multinomial logistic regression by full-batch GD — the CPU behavior-cloning learner."""
    rng = np.random.default_rng(seed)
    n, d = X.shape
    W = rng.normal(0.0, 0.01, size=(d, n_classes))
    b = np.zeros(n_classes, dtype=np.float64)
    Y = np_one_hot(y, n_classes)
    sw = np.ones(n) if sample_weight is None else np.asarray(sample_weight, dtype=np.float64)
    sw = sw / (sw.mean() + 1e-12)
    history: list[float] = []
    for e in range(epochs):
        logits = X @ W + b
        P = np_softmax(logits, axis=1)
        G = (P - Y) * sw[:, None]
        W -= lr * (X.T @ G / n + l2 * W)
        b -= lr * (G.sum(axis=0) / n)
        if e % 25 == 0:
            loss = float(-np.mean(sw * np.log((P * Y).sum(axis=1) + 1e-12)))
            history.append(loss)
    return W, b, history


# -- shared RL learner base --------------------------------------------------

class _RLBase:
    """Common standardizer + GPU-first device bookkeeping for the offline-RL learners."""

    model_type = "rl_base"

    def __init__(self, *, standardize: bool = True,
                 device_policy: Optional[DevicePolicy] = None, seed: int = 7) -> None:
        self.policy = device_policy or DevicePolicy()
        self.seed = int(seed)
        self._std = Standardizer() if standardize else None
        self.device = "cpu"
        self.backend = "mlx" if _MLX else "numpy"

    def _fit_standardizer(self, states: np.ndarray) -> np.ndarray:
        if self._std is None:
            return np.asarray(states, dtype=np.float32)
        return self._std.fit(np.asarray(states, dtype=np.float32)).transform(states)

    def _apply_standardizer(self, states: np.ndarray) -> np.ndarray:
        if self._std is None:
            return np.asarray(states, dtype=np.float32)
        return self._std.transform(np.asarray(states, dtype=np.float32))

    def _choose_device(self, workload: str, *, n_elements: int):
        """Pick GPU-first (when MLX present + large enough) else CPU, and RECORD it."""
        decision = self.policy.choose(workload, n_elements=int(n_elements),
                                      gpu_supported=_MLX)
        self.device = decision.device
        self.backend = "mlx" if (decision.device == "gpu" and _MLX) else "numpy"
        mlx_dev = self.policy.mlx_device(decision) if _MLX else None
        return decision, mlx_dev
