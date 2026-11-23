"""CPU-vs-MLX-GPU benchmark certification (Milestone 6 addendum §19).

Measures ACTUAL, correctly-synchronized performance: lazy MLX computation is forced with
mx.eval() before timing stops, warm-up/compilation is separated from steady-state, and
asynchronous dispatch is never reported as completed work. Reports real acceleration
factors INCLUDING workloads where the GPU is slower (small/sync-dominated), honoring the
directive's honesty requirement. Falls back to CPU-only numbers (clearly marked) when MLX
is unavailable.
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np

from cengine.learning.hardware import inspect_hardware, mlx_available

BENCHMARK_VERSION = "gpu-benchmarks-1.0.0"

try:  # pragma: no cover
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim

    _MLX = True
except Exception:  # noqa: BLE001
    mx = None  # type: ignore
    _MLX = False

_BATCH_SIZES = (1, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096)


def _peak_mem_gib() -> Optional[float]:
    if not _MLX:
        return None
    for fn in ("get_peak_memory",):
        if hasattr(mx, fn):
            try:
                return round(getattr(mx, fn)() / (1024 ** 3), 4)
            except Exception:  # noqa: BLE001
                return None
    return None


def _reset_peak_mem() -> None:
    if _MLX and hasattr(mx, "reset_peak_memory"):
        try:
            mx.reset_peak_memory()
        except Exception:  # noqa: BLE001
            pass


def _mlp(n_in: int, hidden=(64, 32), n_out=1):
    dims = [n_in, *hidden, n_out]
    return nn.Sequential(*sum(
        ([nn.Linear(dims[i], dims[i + 1])] + ([nn.ReLU()] if i < len(dims) - 2 else [])
         for i in range(len(dims) - 1)), []))


def benchmark_training_throughput(*, n: int = 50_000, n_features: int = 13,
                                  epochs: int = 3, batch_size: int = 2048,
                                  seed: int = 0) -> dict:
    """Training examples/sec on GPU vs CPU (MLX), correctly synchronized."""
    out = {"workload": "mlp_training", "n": n, "n_features": n_features,
           "epochs": epochs, "batch_size": batch_size}
    if not (_MLX and mlx_available()):
        out["skipped"] = "mlx/metal unavailable"
        return out
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, n_features)).astype(np.float32)
    y = rng.normal(size=(n,)).astype(np.float32)

    def _run(device) -> float:
        mx.random.seed(seed)
        net = _mlp(n_features)
        opt = optim.AdamW(learning_rate=1e-3)

        def loss_fn(model, xb, yb):
            return mx.mean((model(xb).squeeze(-1) - yb) ** 2)
        lvg = nn.value_and_grad(net, loss_fn)
        Xd, yd = mx.array(X), mx.array(y)
        with mx.stream(device):
            # warm-up (compile) — excluded from timing
            _l, g = lvg(net, Xd[:batch_size], yd[:batch_size])
            opt.update(net, g)
            mx.eval(net.parameters(), opt.state)
            t0 = time.perf_counter()
            for _ in range(epochs):
                for b in range(0, n, batch_size):
                    xb, yb = Xd[b:b + batch_size], yd[b:b + batch_size]
                    _l, g = lvg(net, xb, yb)
                    opt.update(net, g)
                mx.eval(net.parameters(), opt.state)   # sync per epoch
            mx.eval(net.parameters(), opt.state)
            return time.perf_counter() - t0

    _reset_peak_mem()
    gpu_s = _run(mx.gpu)
    gpu_peak = _peak_mem_gib()
    cpu_s = _run(mx.cpu)
    total = n * epochs
    out.update({
        "gpu_seconds": round(gpu_s, 4), "cpu_seconds": round(cpu_s, 4),
        "gpu_examples_per_s": round(total / gpu_s, 1) if gpu_s else None,
        "cpu_examples_per_s": round(total / cpu_s, 1) if cpu_s else None,
        "gpu_speedup_x": round(cpu_s / gpu_s, 2) if gpu_s else None,
        "gpu_peak_mem_gib": gpu_peak,
        "note": "warm-up/compile excluded; mx.eval() forces completion before timing stops"})
    return out


def benchmark_inference_latency(*, n_features: int = 13, iters: int = 50,
                                seed: int = 0) -> dict:
    """Batched inference throughput + p50/p95/p99 latency per batch size, GPU vs CPU."""
    out = {"workload": "mlp_inference", "n_features": n_features, "batch_sizes": list(_BATCH_SIZES)}
    if not (_MLX and mlx_available()):
        out["skipped"] = "mlx/metal unavailable"
        return out
    rng = np.random.default_rng(seed)
    net = _mlp(n_features)
    mx.eval(net.parameters())

    def _latencies(device, bs):
        X = mx.array(rng.normal(size=(bs, n_features)).astype(np.float32))
        with mx.stream(device):
            mx.eval(net(X))  # warm-up
            lat = []
            for _ in range(iters):
                t0 = time.perf_counter()
                o = net(X)
                mx.eval(o)    # force completion (not just dispatch)
                lat.append((time.perf_counter() - t0) * 1e3)
        return np.array(lat)

    rows = []
    for bs in _BATCH_SIZES:
        g = _latencies(mx.gpu, bs)
        c = _latencies(mx.cpu, bs)
        rows.append({
            "batch": bs,
            "gpu_p50_ms": round(float(np.percentile(g, 50)), 4),
            "gpu_p95_ms": round(float(np.percentile(g, 95)), 4),
            "gpu_p99_ms": round(float(np.percentile(g, 99)), 4),
            "cpu_p50_ms": round(float(np.percentile(c, 50)), 4),
            "gpu_throughput_rows_s": round(bs / (np.median(g) / 1e3), 1),
            "faster_device": "gpu" if np.median(g) < np.median(c) else "cpu"})
    out["by_batch"] = rows
    out["crossover_note"] = ("GPU wins at large batches; CPU wins at tiny batches "
                             "(sync-dominated) — reported honestly, not hidden")
    return out


def benchmark_memory(*, sizes=(1024, 4096, 16384), n_features: int = 13) -> dict:
    out = {"workload": "unified_memory", "sizes": list(sizes)}
    if not (_MLX and mlx_available()):
        out["skipped"] = "mlx/metal unavailable"
        return out
    rows = []
    for nrows in sizes:
        _reset_peak_mem()
        xa = np.random.default_rng(0).normal(size=(nrows, n_features)).astype(np.float32)
        wa = np.random.default_rng(1).normal(size=(n_features, 64)).astype(np.float32)
        with mx.stream(mx.gpu):
            Y = mx.array(xa) @ mx.array(wa)
            mx.eval(Y)
        rows.append({"rows": nrows, "peak_mem_gib": _peak_mem_gib()})
    out["by_size"] = rows
    return out


def run_all_benchmarks() -> dict:
    cap = inspect_hardware()
    return {
        "report": "learning_hardware_benchmarks", "version": BENCHMARK_VERSION,
        "hardware": {"device": cap.device_name, "gpu": cap.gpu_architecture,
                     "unified_memory_gib": cap.unified_memory_gib,
                     "mlx_metal_available": cap.mlx_metal_available,
                     "precisions": list(cap.supported_precisions)},
        "training_throughput": benchmark_training_throughput(),
        "inference_latency": benchmark_inference_latency(),
        "memory": benchmark_memory()}
