"""Covariance bridge connecting the Metal backend to the portfolio optimizer.

This is the consumer seam for the covariance kernels: it computes an asset
covariance matrix from a returns matrix using the Metal ``covariance_matrix``
kernel when a backend is available, otherwise numpy. The result is symmetrized
and lightly ridge-regularized (so it is a valid PSD input to the CVXPY allocator)
and returned as a plain ``np.ndarray`` for :class:`RegimeMarkowitzAllocator`.

GPU is never a prerequisite: a missing/failed backend transparently falls back to
numpy. GPU-derived covariance is an *input* to the optimizer; it never authorizes
trades (native risk + OrderManager remain authoritative).
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np


def covariance_from_returns(
    returns: Sequence[Sequence[float]] | np.ndarray,
    *,
    backend: Any = None,
    ridge: float = 1e-10,
) -> tuple[np.ndarray, str]:
    """Return (covariance, source) where source is 'metal' or 'cpu'.

    ``returns`` is observations x assets. The covariance is sample covariance
    (ddof=1), symmetrized and ridge-regularized for PSD stability.
    """
    arr = np.asarray(returns, dtype=float)
    if arr.ndim != 2 or arr.shape[0] < 2 or arr.shape[1] < 1:
        raise ValueError("returns must be observations x assets with >=2 observations")

    source = "cpu"
    cov: np.ndarray | None = None
    if backend is not None and getattr(backend, "available", False):
        try:
            from cengine.metal_covariance import MetalCovariance

            cov = np.asarray(MetalCovariance(backend).covariance_matrix(arr.tolist()), dtype=float)
            source = "metal"
        except Exception:  # noqa: BLE001 - GPU failure must never block allocation
            cov = None
            source = "cpu"
    if cov is None:
        cov = np.atleast_2d(np.cov(arr, rowvar=False, ddof=1))

    cov = np.atleast_2d(cov)
    cov = 0.5 * (cov + cov.T)  # enforce symmetry
    cov = cov + ridge * np.eye(cov.shape[0])  # tiny ridge -> numerically PSD
    return cov, source
