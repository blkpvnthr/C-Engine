"""Hardware capability inspection + GPU-first device policy (Milestone 6 addendum §2/§3).

Inspects the ACTUAL Apple-silicon / Metal / MLX capabilities at runtime (never hardcodes
specs) and provides an explicit device-selection policy: default to the Metal GPU for
sufficiently large learning workloads, with a DOCUMENTED, OBSERVABLE, MEASURABLE CPU
fallback for baselines, small latency-sensitive workloads, unsupported ops, and degraded
recovery. The platform never silently claims GPU acceleration when work runs on the CPU —
every placement decision is recorded with a reason and surfaced to callers/telemetry.

MLX is optional at import time: if it is unavailable (non-arm64/non-Darwin, or not
installed) the module degrades to CPU-only and reports ``mlx_available = False`` rather
than raising, so the rest of the engine keeps working.
"""

from __future__ import annotations

import os
import platform
from dataclasses import dataclass, field
from typing import Optional

HARDWARE_REPORT_VERSION = "learning-hardware-1.0.0"

# -- optional MLX import (never hard-fails) -----------------------------------
try:  # pragma: no cover - import guard
    import mlx.core as mx

    _MLX_IMPORT_ERROR: Optional[str] = None
except Exception as exc:  # noqa: BLE001 - absence must degrade, not crash
    mx = None  # type: ignore
    _MLX_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"


def mlx_available() -> bool:
    """True iff MLX imported AND a Metal GPU is actually usable."""
    if mx is None:
        return False
    try:
        return bool(mx.metal.is_available())
    except Exception:  # noqa: BLE001
        try:  # newer MLX: metal availability via device_info presence
            return mx.default_device().type == mx.gpu.type
        except Exception:  # noqa: BLE001
            return False


def _device_info() -> dict:
    if mx is None:
        return {}
    for getter in ("device_info", None):
        try:
            if getter == "device_info" and hasattr(mx, "device_info"):
                return dict(mx.device_info())
            if getter is None and hasattr(mx, "metal"):
                return dict(mx.metal.device_info())  # deprecated but a fallback
        except Exception:  # noqa: BLE001
            continue
    return {}


@dataclass
class HardwareCapabilities:
    report_version: str = HARDWARE_REPORT_VERSION
    platform: str = ""
    machine: str = ""
    is_apple_silicon: bool = False
    cpu_brand: str = ""
    unified_memory_gib: float = 0.0
    mlx_installed: bool = False
    mlx_metal_available: bool = False
    mlx_import_error: Optional[str] = None
    device_name: str = ""
    gpu_architecture: str = ""
    max_recommended_working_set_gib: float = 0.0
    max_buffer_length_gib: float = 0.0
    memory_size_gib: float = 0.0
    supported_precisions: tuple = ()
    native_metallib_present: bool = False
    raw_device_info: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items()}
        d["supported_precisions"] = list(self.supported_precisions)
        return d


def _sysctl(name: str) -> str:
    try:
        import subprocess

        return subprocess.check_output(["sysctl", "-n", name], text=True).strip()
    except Exception:  # noqa: BLE001
        return ""


def _validate_precisions() -> tuple:
    """Probe which float precisions actually round-trip through an MLX GPU op."""
    if not mlx_available():
        return ("float32 (cpu/numpy)",)
    ok = []
    for name, dt in (("float32", getattr(mx, "float32", None)),
                     ("float16", getattr(mx, "float16", None)),
                     ("bfloat16", getattr(mx, "bfloat16", None))):
        if dt is None:
            continue
        try:
            a = mx.ones((8, 8), dtype=dt)
            b = a @ a
            mx.eval(b)
            ok.append(name)
        except Exception:  # noqa: BLE001
            pass
    return tuple(ok)


def inspect_hardware(*, native_metallib_path: Optional[str] = None) -> HardwareCapabilities:
    """Inspect ACTUAL runtime hardware/MLX capabilities (no hardcoded specs)."""
    cap = HardwareCapabilities()
    cap.platform = platform.platform()
    cap.machine = platform.machine()
    cap.is_apple_silicon = (platform.system() == "Darwin" and cap.machine == "arm64")
    cap.cpu_brand = _sysctl("machdep.cpu.brand_string")
    membytes = _sysctl("hw.memsize")
    if membytes.isdigit():
        cap.unified_memory_gib = round(int(membytes) / (1024 ** 3), 2)
    cap.mlx_installed = mx is not None
    cap.mlx_import_error = _MLX_IMPORT_ERROR
    cap.mlx_metal_available = mlx_available()
    info = _device_info()
    cap.raw_device_info = {k: info[k] for k in info}
    if info:
        cap.device_name = str(info.get("device_name", ""))
        cap.gpu_architecture = str(info.get("architecture", ""))
        gib = 1024 ** 3
        for src, dst in (("max_recommended_working_set_size", "max_recommended_working_set_gib"),
                         ("max_buffer_length", "max_buffer_length_gib"),
                         ("memory_size", "memory_size_gib")):
            if src in info:
                try:
                    setattr(cap, dst, round(float(info[src]) / gib, 2))
                except (TypeError, ValueError):
                    pass
    cap.supported_precisions = _validate_precisions()
    # native Metal runtime coexistence signal (metallib compiled by M1 Phase A)
    path = native_metallib_path or os.path.join(os.getcwd(), "build-metal", "cengine.metallib")
    cap.native_metallib_present = os.path.exists(path)
    return cap


# -- device policy -----------------------------------------------------------

DEVICE_GPU = "gpu"
DEVICE_CPU = "cpu"


@dataclass
class PlacementDecision:
    workload: str
    device: str
    reason: str
    gpu_eligible: bool
    fell_back: bool = False


class DevicePolicy:
    """GPU-first placement with a documented, measurable CPU fallback.

    ``min_gpu_elements`` is the workload size (e.g. rows*features, or batch*params)
    below which CPU is preferred — small workloads are latency-dominated by host/device
    sync, so GPU is counter-productive. Every decision is recorded in ``fallbacks`` /
    ``decisions`` so telemetry can prove where computation actually ran.
    """

    def __init__(self, *, prefer_gpu: bool = True, min_gpu_elements: int = 50_000,
                 force_cpu: Optional[bool] = None) -> None:
        env_cpu = os.environ.get("CENGINE_LEARNING_FORCE_CPU", "")
        self.force_cpu = force_cpu if force_cpu is not None else (env_cpu not in ("", "0"))
        self.prefer_gpu = prefer_gpu
        self.min_gpu_elements = max(1, int(min_gpu_elements))
        self.gpu_available = mlx_available() and not self.force_cpu
        self.decisions: list[PlacementDecision] = []
        self.fallback_count = 0

    def choose(self, workload: str, *, n_elements: int = 10 ** 9,
               gpu_supported: bool = True) -> PlacementDecision:
        if self.force_cpu:
            d = PlacementDecision(workload, DEVICE_CPU, "forced_cpu (config/env)",
                                  gpu_eligible=False)
        elif not mlx_available():
            d = PlacementDecision(workload, DEVICE_CPU,
                                  f"mlx/metal unavailable: {_MLX_IMPORT_ERROR or 'no metal'}",
                                  gpu_eligible=False, fell_back=True)
        elif not gpu_supported:
            d = PlacementDecision(workload, DEVICE_CPU, "op unsupported on gpu",
                                  gpu_eligible=False, fell_back=True)
        elif not self.prefer_gpu:
            d = PlacementDecision(workload, DEVICE_CPU, "policy prefers cpu",
                                  gpu_eligible=True)
        elif n_elements < self.min_gpu_elements:
            d = PlacementDecision(
                workload, DEVICE_CPU,
                f"small workload {n_elements} < {self.min_gpu_elements} (sync-dominated)",
                gpu_eligible=True)
        else:
            d = PlacementDecision(workload, DEVICE_GPU,
                                  f"gpu-first: {n_elements} elements", gpu_eligible=True)
        if d.fell_back:
            self.fallback_count += 1
        self.decisions.append(d)
        return d

    def mlx_device(self, decision: PlacementDecision):
        """Return the MLX device object for a decision (None if MLX absent)."""
        if mx is None:
            return None
        return mx.gpu if decision.device == DEVICE_GPU else mx.cpu

    def summary(self) -> dict:
        by_device: dict = {}
        for d in self.decisions:
            by_device[d.device] = by_device.get(d.device, 0) + 1
        return {"gpu_available": self.gpu_available, "force_cpu": self.force_cpu,
                "min_gpu_elements": self.min_gpu_elements,
                "placements": by_device, "fallback_count": self.fallback_count}
