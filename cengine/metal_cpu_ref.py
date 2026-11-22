"""CPU reference implementations (oracles) for Metal kernels.

These mirror the exact kernel semantics so GPU output can be validated for
numerical parity. Integer-tick indicators use C-style truncating division to
match the Metal ``long`` ``/`` operator; float covariance uses the same
accumulation order as the kernels (parity compared within float32 tolerance).
"""

from __future__ import annotations

from typing import Sequence


def _trunc_div(a: int, b: int) -> int:
    """Integer division truncating toward zero (matches Metal/C `long` /)."""
    q = abs(a) // abs(b)
    return -q if (a < 0) != (b < 0) else q


# ------------------------------------------------------------------ indicators
def sma_ticks(prices: Sequence[int], window: int) -> list[int]:
    """Mirror of rolling_sma_ticks (0 for the first window-1 positions)."""
    n = len(prices)
    out = [0] * n
    if window <= 0:
        return out
    for i in range(n):
        if i + 1 < window:
            continue
        out[i] = _trunc_div(sum(prices[i + 1 - window : i + 1]), window)
    return out


def ema_ticks(prices: Sequence[int], alpha_numerator: int, alpha_denominator: int) -> list[int]:
    n = len(prices)
    if n == 0 or alpha_denominator == 0 or alpha_numerator > alpha_denominator:
        return [0] * n
    out = [0] * n
    out[0] = prices[0]
    for i in range(1, n):
        out[i] = _trunc_div(
            alpha_numerator * prices[i] + (alpha_denominator - alpha_numerator) * out[i - 1],
            alpha_denominator,
        )
    return out


def apo_ticks(fast_ema: Sequence[int], slow_ema: Sequence[int]) -> list[int]:
    return [int(f) - int(s) for f, s in zip(fast_ema, slow_ema, strict=True)]


def true_range_ticks(
    highs: Sequence[int], lows: Sequence[int], prev_closes: Sequence[int]
) -> list[int]:
    out = []
    for h, low, pc in zip(highs, lows, prev_closes, strict=True):
        out.append(max(h - low, abs(h - pc), abs(low - pc)))
    return out


# ------------------------------------------------------------------ covariance
def column_means(values: Sequence[Sequence[float]]) -> list[float]:
    """values: observations x assets row-major (list of rows)."""
    if not values:
        return []
    obs = len(values)
    assets = len(values[0])
    return [sum(row[a] for row in values) / obs for a in range(assets)]


def covariance_matrix(values: Sequence[Sequence[float]]) -> list[list[float]]:
    """Sample covariance (assets x assets), divisor (obs-1). Mirrors the kernel."""
    obs = len(values)
    assets = len(values[0]) if obs else 0
    means = column_means(values)
    cov = [[0.0] * assets for _ in range(assets)]
    if obs < 2:
        return [[float("nan")] * assets for _ in range(assets)]
    for i in range(assets):
        for j in range(assets):
            acc = 0.0
            for row in values:
                acc += (row[i] - means[i]) * (row[j] - means[j])
            cov[i][j] = acc / (obs - 1)
    return cov


def ewma_covariance_matrix(values: Sequence[Sequence[float]], lam: float) -> list[list[float]]:
    """EWMA covariance: newest weight 1, older weighted by lam, lam^2, ...
    normalized by weight sum. Mirrors ewma_covariance_matrix."""
    obs = len(values)
    assets = len(values[0]) if obs else 0
    if obs == 0 or not (0.0 < lam <= 1.0):
        return [[float("nan")] * assets for _ in range(assets)]
    # weighted means
    wsum = 0.0
    wmean = [0.0] * assets
    w = 1.0
    for reverse in range(obs):
        row = values[obs - 1 - reverse]
        for a in range(assets):
            wmean[a] += w * row[a]
        wsum += w
        w *= lam
    wmean = [m / wsum for m in wmean]
    cov = [[0.0] * assets for _ in range(assets)]
    for i in range(assets):
        for j in range(assets):
            acc = 0.0
            w = 1.0
            for reverse in range(obs):
                row = values[obs - 1 - reverse]
                acc += w * (row[i] - wmean[i]) * (row[j] - wmean[j])
                w *= lam
            cov[i][j] = acc / wsum
    return cov
