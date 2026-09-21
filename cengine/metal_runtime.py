"""Explicit Metal library loading and compute dispatch for supported macOS hosts."""

from __future__ import annotations

import platform
from pathlib import Path
from typing import Any


class MetalUnavailableError(RuntimeError):
    pass


class MetalRuntime:
    def __init__(self, library: str | Path) -> None:
        if platform.system() != "Darwin" or platform.machine() != "arm64":
            raise MetalUnavailableError("Metal analytics require macOS on Apple Silicon")
        try:
            import Metal  # type: ignore[import-not-found]
        except ImportError as exc:
            raise MetalUnavailableError("PyObjC Metal bindings are not installed") from exc
        self._Metal = Metal
        self.device = Metal.MTLCreateSystemDefaultDevice()
        if self.device is None:
            raise MetalUnavailableError("no Metal device is available")
        self.queue = self.device.newCommandQueue()
        path = str(Path(library).resolve())
        loaded = self.device.newLibraryWithFile_error_(path, None)
        self.library = loaded[0] if isinstance(loaded, tuple) else loaded
        if self.library is None:
            raise MetalUnavailableError(f"cannot load Metal library {path}")

    def pipeline(self, function_name: str) -> Any:
        function = self.library.newFunctionWithName_(function_name)
        if function is None:
            raise KeyError(f"Metal function {function_name!r} not found")
        result = self.device.newComputePipelineStateWithFunction_error_(function, None)
        pipeline = result[0] if isinstance(result, tuple) else result
        if pipeline is None:
            raise MetalUnavailableError(f"cannot create pipeline {function_name!r}")
        return pipeline

    def dispatch(self, function_name: str, buffers: list[Any], element_count: int) -> None:
        if element_count <= 0:
            raise ValueError("element_count must be positive")
        pipeline = self.pipeline(function_name)
        command = self.queue.commandBuffer()
        encoder = command.computeCommandEncoder()
        encoder.setComputePipelineState_(pipeline)
        for index, buffer in enumerate(buffers):
            encoder.setBuffer_offset_atIndex_(buffer, 0, index)
        width = min(int(pipeline.maxTotalThreadsPerThreadgroup()), element_count)
        grid = self._Metal.MTLSizeMake(element_count, 1, 1)
        group = self._Metal.MTLSizeMake(width, 1, 1)
        encoder.dispatchThreads_threadsPerThreadgroup_(grid, group)
        encoder.endEncoding()
        command.commit()
        command.waitUntilCompleted()
        if command.error() is not None:
            raise RuntimeError(str(command.error()))
