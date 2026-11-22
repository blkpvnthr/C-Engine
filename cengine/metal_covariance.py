"""Metal-accelerated covariance analytics (host interface for covariance.metal).

Wires the ``column_means`` + ``covariance_matrix`` kernels (and
``ewma_covariance_matrix``) through the shared :class:`MetalBackend`. These are
inputs to the portfolio optimizer / risk manager; they never authorize trades.
The ABI mirrors the shader's ``CovarianceParams`` {observations, assets,
row_stride, reserved} and ``EwmaParams`` (adds float lambda, minimum_variance,
pad0, pad1). All buffers are float32 row-major: ``values[row*row_stride + asset]``.
"""

from __future__ import annotations

import struct
from typing import Sequence

from cengine.metal_backend import MetalBackend


class MetalCovariance:
    def __init__(self, backend: MetalBackend) -> None:
        if not backend.available:
            raise RuntimeError("MetalCovariance requires an available Metal backend")
        self.backend = backend
        self.buffers = backend.buffers

    def _flatten(self, values: Sequence[Sequence[float]]) -> tuple[list[float], int, int]:
        obs = len(values)
        if obs == 0:
            raise ValueError("values cannot be empty")
        assets = len(values[0])
        if assets == 0:
            raise ValueError("values rows cannot be empty")
        flat: list[float] = []
        for row in values:
            if len(row) != assets:
                raise ValueError("all rows must have equal length")
            flat.extend(float(x) for x in row)
        return flat, obs, assets

    def covariance_matrix(self, values: Sequence[Sequence[float]]) -> list[list[float]]:
        """Sample covariance (assets x assets, divisor obs-1) via two GPU kernels."""
        flat, obs, assets = self._flatten(values)
        values_buf = self.buffers.float32(flat)
        params_buf = self.buffers.uint32_struct(obs, assets, assets, 0)
        means_buf = self.buffers.zero_float32(assets)
        self.backend.dispatch("column_means", [values_buf, params_buf, means_buf], assets)

        cov_buf = self.buffers.zero_float32(assets * assets)
        self.backend.dispatch(
            "covariance_matrix",
            [values_buf, means_buf, params_buf, cov_buf],
            assets * assets,
        )
        flat_cov = self.buffers.read_float32(cov_buf, assets * assets)
        return [flat_cov[i * assets : (i + 1) * assets] for i in range(assets)]

    def ewma_covariance_matrix(
        self, values: Sequence[Sequence[float]], lam: float, minimum_variance: float = 0.0
    ) -> list[list[float]]:
        flat, obs, assets = self._flatten(values)
        if not (0.0 < lam <= 1.0):
            raise ValueError("lambda must be in (0, 1]")
        values_buf = self.buffers.float32(flat)
        params = struct.pack(
            "<4I4f", obs, assets, assets, 0, float(lam), float(minimum_variance), 0.0, 0.0
        )
        params_buf = self.buffers.from_bytes(params)
        cov_buf = self.buffers.zero_float32(assets * assets)
        self.backend.dispatch(
            "ewma_covariance_matrix", [values_buf, params_buf, cov_buf], assets * assets
        )
        flat_cov = self.buffers.read_float32(cov_buf, assets * assets)
        return [flat_cov[i * assets : (i + 1) * assets] for i in range(assets)]
