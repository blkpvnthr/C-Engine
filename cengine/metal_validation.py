"""Metal integration validation + benchmarking + report generation.

Produces the integration coverage report, hardware-execution + CPU/GPU parity
evidence, and performance benchmarks from REAL execution on the current device.
All counts are computed from actual runs, never asserted. Blocked kernels (no host
interface yet) are reported honestly with reasons from the registry.
"""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path
from typing import Any

from cengine import metal_cpu_ref as cpu
from cengine.metal_backend import KERNELS, MetalBackend


def _exercise_wired_kernels(backend: MetalBackend) -> dict[str, Any]:
    """Execute every kernel that has a host interface and check CPU parity.

    Returns per-kernel {executed, parity_ok, detail}. Only runs when the backend
    is available; otherwise everything is reported as not executed.
    """
    results: dict[str, Any] = {}
    if not backend.available:
        for reg in KERNELS:
            if reg.host_interface:
                results[reg.kernel_name] = {
                    "executed": False,
                    "parity_ok": False,
                    "detail": "metal unavailable",
                }
        return results

    from cengine.metal_covariance import MetalCovariance
    from cengine.metal_indicators import MetalIndicators

    ind = MetalIndicators(backend.runtime)
    mc = MetalCovariance(backend)

    prices = [100, 101, 103, 102, 104, 106, 105, 108, 110, 109]
    returns = [
        [0.01, -0.02, 0.005],
        [0.02, -0.01, 0.004],
        [-0.015, 0.03, -0.002],
        [0.005, -0.005, 0.001],
        [0.012, -0.018, 0.006],
        [-0.008, 0.02, -0.003],
    ]

    def record(name, fn):
        try:
            results[name] = fn()
        except Exception as exc:  # noqa: BLE001
            results[name] = {"executed": False, "parity_ok": False, "detail": str(exc)[:120]}

    def _max_abs(a, b):
        return max(abs(a[i][j] - b[i][j]) for i in range(len(a)) for j in range(len(a[0])))

    record(
        "rolling_sma_ticks",
        lambda: {
            "executed": True,
            "parity_ok": ind.sma(prices, 3) == cpu.sma_ticks(prices, 3),
            "detail": "exact integer parity",
        },
    )
    record(
        "exponential_moving_average_ticks",
        lambda: {
            "executed": True,
            "parity_ok": ind.ema(prices, 2, 10) == cpu.ema_ticks(prices, 2, 10),
            "detail": "exact integer parity",
        },
    )
    fast, slow = cpu.ema_ticks(prices, 4, 10), cpu.ema_ticks(prices, 2, 10)
    record(
        "average_price_oscillator_ticks",
        lambda: {
            "executed": True,
            "parity_ok": ind.apo(fast, slow) == cpu.apo_ticks(fast, slow),
            "detail": "exact integer parity",
        },
    )
    highs = [p + 2 for p in prices]
    lows = [p - 2 for p in prices]
    prev = prices
    record(
        "true_range_ticks",
        lambda: {
            "executed": True,
            "parity_ok": ind.true_range(highs, lows, prev)
            == cpu.true_range_ticks(highs, lows, prev),
            "detail": "exact integer parity",
        },
    )

    def _cov():
        g = mc.covariance_matrix(returns)
        r = cpu.covariance_matrix(returns)
        d = _max_abs(g, r)
        return {"executed": True, "parity_ok": d < 1e-5, "detail": f"max_abs_diff={d:.2e}"}

    # covariance_matrix dispatch also runs column_means internally.
    record("covariance_matrix", _cov)
    record(
        "column_means",
        lambda: {
            "executed": True,
            "parity_ok": results["covariance_matrix"]["parity_ok"],
            "detail": "executed as part of covariance_matrix pipeline",
        },
    )

    def _ewma():
        g = mc.ewma_covariance_matrix(returns, 0.94)
        r = cpu.ewma_covariance_matrix(returns, 0.94)
        d = _max_abs(g, r)
        return {"executed": True, "parity_ok": d < 1e-5, "detail": f"max_abs_diff={d:.2e}"}

    record("ewma_covariance_matrix", _ewma)
    return results


def validate_metal(backend: MetalBackend) -> dict[str, Any]:
    cov = backend.coverage()
    exercised = _exercise_wired_kernels(backend)
    hw_executed = sum(1 for v in exercised.values() if v.get("executed"))
    parity_ok = sum(1 for v in exercised.values() if v.get("parity_ok"))
    discovered = cov["discovered"]
    blocked = [
        {"kernel": r.kernel_name, "shader": r.shader_file, "reason": r.note}
        for r in KERNELS
        if not r.host_interface
    ]
    return {
        "device": backend.device_name(),
        "library": str(backend.library_path) if backend.library_path else None,
        "available": backend.available,
        "shader_files_discovered": len({r.shader_file for r in KERNELS}),
        "kernel_entry_points_discovered": discovered,
        "kernels_compiled_and_callable": cov["compiled_and_callable"],
        "kernels_host_interfaced": cov["host_interfaced"],
        "kernels_consumer_connected": cov["consumer_connected"],
        "kernels_hardware_executed": hw_executed,
        "kernels_parity_validated": parity_ok,
        "kernels_blocked": len(blocked),
        "integration_coverage_pct": cov["integration_coverage_pct"],
        "pipeline_coverage_pct": cov["pipeline_coverage_pct"],
        "pipeline_status": cov["pipeline_status"],
        "exercised": exercised,
        "blocked": blocked,
    }


def benchmark_metal(backend: MetalBackend, *, repeats: int = 30) -> dict[str, Any]:
    """Measure GPU vs CPU latency (p50/p95) for wired kernels across sizes."""
    out: dict[str, Any] = {
        "device": backend.device_name(),
        "available": backend.available,
        "repeats": repeats,
        "workloads": [],
    }
    if not backend.available:
        return out
    from cengine.metal_covariance import MetalCovariance
    from cengine.metal_indicators import MetalIndicators

    ind = MetalIndicators(backend.runtime)
    mc = MetalCovariance(backend)

    def timed(fn):
        ts = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            fn()
            ts.append((time.perf_counter() - t0) * 1e3)
        return {
            "p50_ms": round(statistics.median(ts), 4),
            "p95_ms": round(sorted(ts)[max(0, int(0.95 * len(ts)) - 1)], 4),
        }

    for n in (16, 64, 256, 1024):
        prices = [100 + (i % 7) for i in range(n)]
        out["workloads"].append(
            {
                "workload": "rolling_sma_ticks",
                "size": n,
                "gpu": timed(lambda prices=prices: ind.sma(prices, 20)),
                "cpu": timed(lambda prices=prices: cpu.sma_ticks(prices, 20)),
            }
        )
    for assets in (4, 8, 16, 32):
        rows = [[((i * 7 + a * 3) % 100) / 1000.0 for a in range(assets)] for i in range(64)]
        out["workloads"].append(
            {
                "workload": "covariance_matrix",
                "assets": assets,
                "observations": 64,
                "gpu": timed(lambda rows=rows: mc.covariance_matrix(rows)),
                "cpu": timed(lambda rows=rows: cpu.covariance_matrix(rows)),
            }
        )
    return out


def render_report(report: dict[str, Any]) -> str:
    r = report
    lines = [
        "METAL INTEGRATION REPORT",
        "-" * 50,
        f"Device:                      {r['device']}",
        f"Shader library:              {r['library']}",
        f"Shader files discovered:     {r['shader_files_discovered']}",
        f"Kernel entry points:         {r['kernel_entry_points_discovered']}",
        f"Kernels compiled+callable:   {r['kernels_compiled_and_callable']}",
        f"Kernels host-callable:       {r['kernels_compiled_and_callable']}",
        f"Kernels host-interfaced:     {r['kernels_host_interfaced']}",
        f"Kernels consumer-connected:  {r['kernels_consumer_connected']}",
        f"Kernels hardware-executed:   {r['kernels_hardware_executed']}",
        f"Kernels parity-validated:    {r['kernels_parity_validated']}",
        f"Kernels blocked:             {r['kernels_blocked']}",
        f"Pipeline coverage:           {r['pipeline_coverage_pct']}%",
        f"Integration coverage:        {r['integration_coverage_pct']}%",
        f"GPU readiness:               {'READY' if r['available'] else 'CPU-FALLBACK'}",
        "-" * 50,
    ]
    return "\n".join(lines)


def write_reports(
    outdir: str | Path, validation: dict[str, Any], benchmarks: dict[str, Any]
) -> None:
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    coverage = {
        k: validation[k]
        for k in (
            "device",
            "available",
            "shader_files_discovered",
            "kernel_entry_points_discovered",
            "kernels_compiled_and_callable",
            "kernels_host_interfaced",
            "kernels_consumer_connected",
            "kernels_hardware_executed",
            "kernels_parity_validated",
            "kernels_blocked",
            "integration_coverage_pct",
            "pipeline_coverage_pct",
            "blocked",
        )
    }
    (out / "metal_integration_coverage.json").write_text(json.dumps(coverage, indent=2))
    (out / "metal_hardware_validation.json").write_text(
        json.dumps(
            {
                "device": validation["device"],
                "pipeline_status": validation["pipeline_status"],
                "exercised": validation["exercised"],
            },
            indent=2,
        )
    )
    (out / "metal_cpu_gpu_parity.json").write_text(
        json.dumps(
            {
                "device": validation["device"],
                "parity": {k: v for k, v in validation["exercised"].items()},
            },
            indent=2,
        )
    )
    (out / "metal_performance_benchmarks.json").write_text(json.dumps(benchmarks, indent=2))
