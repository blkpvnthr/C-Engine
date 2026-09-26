"""Typed shared-buffer helpers for the C-Engine Metal backend."""

from __future__ import annotations

import struct
from typing import Any


class MetalBuffers:
    def __init__(self, device: Any) -> None:
        self.device = device

    def from_bytes(self, data: bytes) -> Any:
        if not data:
            raise ValueError("Metal buffer cannot be empty")

        buf = self.device.newBufferWithLength_options_(len(data), 0)

        if buf is None:
            raise RuntimeError("Metal buffer allocation failed")

        view = memoryview(buf.contents().as_buffer(len(data))).cast("B")
        view[:] = data

        return buf

    def int64(self, values: list[int]) -> Any:
        if not values:
            raise ValueError("values cannot be empty")

        return self.from_bytes(
            struct.pack(f"<{len(values)}q", *values)
        )

    def uint32(self, value: int) -> Any:
        if not 0 <= value <= 0xFFFFFFFF:
            raise ValueError("uint32 value out of range")

        return self.from_bytes(
            struct.pack("<I", value)
        )

    def zero_int64(self, count: int) -> Any:
        if count <= 0:
            raise ValueError("count must be positive")

        return self.from_bytes(
            b"\x00" * (count * 8)
        )

    def read_int64(self, buf: Any, count: int) -> list[int]:
        if count <= 0:
            raise ValueError("count must be positive")

        size = count * 8

        view = memoryview(
            buf.contents().as_buffer(size)
        ).cast("B")

        return list(
            struct.unpack(
                f"<{count}q",
                view.tobytes(),
            )
        )
