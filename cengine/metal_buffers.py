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

    # ------------------------------------------------------------------ float32
    def float32(self, values: list[float]) -> Any:
        if not values:
            raise ValueError("values cannot be empty")
        return self.from_bytes(struct.pack(f"<{len(values)}f", *values))

    def zero_float32(self, count: int) -> Any:
        if count <= 0:
            raise ValueError("count must be positive")
        return self.from_bytes(b"\x00" * (count * 4))

    def read_float32(self, buf: Any, count: int) -> list[float]:
        if count <= 0:
            raise ValueError("count must be positive")
        size = count * 4
        view = memoryview(buf.contents().as_buffer(size)).cast("B")
        return list(struct.unpack(f"<{count}f", view.tobytes()))

    # ------------------------------------------------- packed uint32 param struct
    def uint32_struct(self, *values: int) -> Any:
        """Pack a sequence of uint32 fields (e.g. a *Params ABI struct)."""
        if not values:
            raise ValueError("struct needs at least one field")
        for v in values:
            if not 0 <= v <= 0xFFFFFFFF:
                raise ValueError("uint32 field out of range")
        return self.from_bytes(struct.pack(f"<{len(values)}I", *values))
