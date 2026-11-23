"""Metal GPU research benchmarks (M8 §18).

Measures **completed** GPU computation (via ``mx.eval``) against CPU for workloads used in alpha
discovery, so acceleration claims are honest (never asynchronous-dispatch time). Where the CPU is
competitive or faster at realistic sizes, that is reported — the goal is measured truth, not a GPU
win. Research workloads never touch the financial-protection path.
"""

from __future__ import annotations

import time

import numpy as np

BENCHMARKS_VERSION = "alpha-metal-benchmarks-1.0.0"


def _time(fn, *, repeat: int = 3) -> float:
    best = float("inf")
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def benchmark_batched_inference(n_rows: int = 200_000, n_feat: int = 16) -> dict:
    """Batched linear inference: GPU (MLX, completed via mx.eval) vs CPU (numpy)."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((n_rows, n_feat)).astype(np.float32)
    W = rng.standard_normal((n_feat, 1)).astype(np.float32)

    def cpu():
        return X @ W

    cpu_s = _time(cpu)
    gpu_s = None
    device = "cpu_only"
    try:
        import mlx.core as mx
        Xm = mx.array(X)
        Wm = mx.array(W)

        def gpu():
            out = Xm @ Wm
            mx.eval(out)        # force completion, not just dispatch
            return out

        gpu()  # warmup (compile)
        gpu_s = _time(gpu)
        device = str(mx.default_device())
    except Exception as exc:  # noqa: BLE001
        device = f"mlx_unavailable: {exc}"
    speedup = round(cpu_s / gpu_s, 2) if gpu_s else None
    return {"workload": "batched_linear_inference", "n_rows": n_rows, "n_features": n_feat,
            "cpu_seconds": round(cpu_s, 5), "gpu_seconds": round(gpu_s, 5) if gpu_s else None,
            "gpu_speedup_x": speedup, "device": device,
            "faster": ("gpu" if speedup and speedup > 1.1 else "cpu" if speedup else "cpu")}


def benchmark_feature_matrix(n_rows: int = 500_000) -> dict:
    """Column standardization over a large feature matrix: GPU vs CPU (completed)."""
    rng = np.random.default_rng(1)
    X = rng.standard_normal((n_rows, 16)).astype(np.float32)

    def cpu():
        mu = X.mean(axis=0)
        sd = X.std(axis=0) + 1e-8
        return (X - mu) / sd

    cpu_s = _time(cpu)
    gpu_s = None
    device = "cpu_only"
    try:
        import mlx.core as mx
        Xm = mx.array(X)

        def gpu():
            mu = Xm.mean(axis=0)
            sd = Xm.std(axis=0) + 1e-8
            out = (Xm - mu) / sd
            mx.eval(out)
            return out

        gpu()
        gpu_s = _time(gpu)
        device = str(mx.default_device())
    except Exception as exc:  # noqa: BLE001
        device = f"mlx_unavailable: {exc}"
    speedup = round(cpu_s / gpu_s, 2) if gpu_s else None
    return {"workload": "feature_standardization", "n_rows": n_rows,
            "cpu_seconds": round(cpu_s, 5), "gpu_seconds": round(gpu_s, 5) if gpu_s else None,
            "gpu_speedup_x": speedup, "device": device,
            "faster": ("gpu" if speedup and speedup > 1.1 else "cpu")}


def run_metal_benchmarks() -> dict:
    results = [benchmark_batched_inference(), benchmark_feature_matrix()]
    gpu_available = any("gpu" in str(r.get("device", "")).lower() for r in results)
    return {
        "version": BENCHMARKS_VERSION, "gpu_available": gpu_available, "benchmarks": results,
        "honesty_note": ("GPU timings force completion with mx.eval (not dispatch time); CPU wins "
                         "are reported where measured — small/medium workloads often favor the "
                         "CPU; the GPU pays off at large batch sizes. Research workloads never "
                         "interfere with financial protection."),
    }
