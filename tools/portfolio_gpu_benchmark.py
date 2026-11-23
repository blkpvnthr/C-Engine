#!/usr/bin/env python3
"""GPU portfolio-analytics performance certification (Milestone 3, §20).

Benchmarks the covariance workload (the operationally-consumed GPU portfolio kernel)
at representative portfolio/candidate-set sizes, measuring CPU vs GPU time,
numerical parity, latency percentiles, and the measured CPU/GPU crossover. Writes
reports/gpu_portfolio_benchmarks.json with ACTUAL on-device measurements.

Covariance matrices are only allocated over instruments that could enter the
portfolio (candidate-set dimensions), separate from the full discovery universe,
per §20 (do not allocate huge covariance matrices over non-portfolio instruments).
"""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

import _cengine_native as native
import numpy as np

REPORTS = Path(__file__).resolve().parents[1] / "reports"
OBSERVATIONS = 120          # ~ a trading-session's worth of return bars
CANDIDATE_SIZES = [1, 8, 16, 30, 64, 128, 256, 500, 1000, 2000, 4096]
REPEATS = 20


def _metallib() -> str | None:
    for c in (Path(native.__file__).parent / "cengine.metallib",
              Path("build-metal/cengine.metallib")):
        if c.exists():
            return str(c)
    return None


def _percentiles(samples_ms: list[float]) -> dict:
    s = sorted(samples_ms)
    return {
        "p50_ms": round(statistics.median(s), 4),
        "p95_ms": round(s[min(len(s) - 1, int(0.95 * len(s)))], 4),
        "p99_ms": round(s[min(len(s) - 1, int(0.99 * len(s)))], 4),
        "min_ms": round(s[0], 4),
        "max_ms": round(s[-1], 4),
    }


def _bench_size(runtime, assets: int) -> dict:
    rng = np.random.default_rng(1234 + assets)
    returns = rng.normal(0.0, 0.01, size=(OBSERVATIONS, assets)).astype(np.float32)
    flat = returns.flatten().tolist()

    # CPU (numpy) timing
    cpu_samples = []
    cpu_cov = None
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        cpu_cov = np.cov(returns, rowvar=False, ddof=1)
        cpu_samples.append((time.perf_counter() - t0) * 1e3)

    # GPU timing (covariance_matrix kernel)
    gpu_samples = []
    gpu_cov = None
    gpu_available = runtime is not None and runtime.available
    if gpu_available:
        runtime.ensure_pipeline("covariance_matrix")
        runtime.ensure_pipeline("column_means")
        for _ in range(REPEATS):
            t0 = time.perf_counter()
            raw = runtime.covariance_matrix(flat, OBSERVATIONS, assets)
            gpu_samples.append((time.perf_counter() - t0) * 1e3)
        gpu_cov = np.asarray(raw, dtype=float).reshape(assets, assets)

    parity = None
    if gpu_cov is not None and assets >= 1:
        parity = float(np.max(np.abs(gpu_cov - np.atleast_2d(cpu_cov))))

    cpu_p = _percentiles(cpu_samples)
    out = {"assets": assets, "observations": OBSERVATIONS, "cpu": cpu_p,
           "max_abs_parity_diff": parity, "gpu_available": gpu_available}
    if gpu_samples:
        gpu_p = _percentiles(gpu_samples)
        out["gpu"] = gpu_p
        out["gpu_faster"] = gpu_p["p50_ms"] < cpu_p["p50_ms"]
    return out


def main() -> None:
    lib = _metallib()
    runtime = native.MetalRuntime(lib) if lib else None
    device = runtime.device_name if (runtime and runtime.available) else "cpu-only"

    results = [_bench_size(runtime, n) for n in CANDIDATE_SIZES]

    crossover = None
    for r in results:
        if r.get("gpu") and r.get("gpu_faster"):
            crossover = r["assets"]
            break

    report = {
        "report": "gpu_portfolio_benchmarks",
        "milestone": "M3 §20 — GPU portfolio performance certification",
        "device": device,
        "metallib": lib,
        "observations_per_matrix": OBSERVATIONS,
        "repeats": REPEATS,
        "candidate_sizes": CANDIDATE_SIZES,
        "kernel": "covariance_matrix (+ column_means) — operationally consumed",
        "results": results,
        "cpu_gpu_crossover_assets": crossover,
        "crossover_policy": (
            f"GPU covariance becomes faster at >= {crossover} candidate instruments; "
            "below that, CPU/numpy is used (small assets x assets matrices are not "
            "dispatched to the GPU)." if crossover else
            "GPU did not beat CPU at the tested sizes for this deterministic (non-"
            "parallel-reduction) kernel; covariance stays on CPU for typical candidate "
            "sets. The kernel remains available + parity-validated; crossover may shift "
            "with a parallel-reduction covariance implementation (future increment)."),
        "scope_note": "Covariance is sized to the CANDIDATE set (instruments that can "
                      "enter the portfolio), never the full discovery universe.",
        "max_parity_diff_overall": max((r["max_abs_parity_diff"] or 0.0)
                                       for r in results),
    }
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "gpu_portfolio_benchmarks.json").write_text(json.dumps(report, indent=2))
    print(f"device={device} crossover={crossover} "
          f"max_parity={report['max_parity_diff_overall']:.2e}")
    print(f"-> {REPORTS}/gpu_portfolio_benchmarks.json")


if __name__ == "__main__":
    main()
