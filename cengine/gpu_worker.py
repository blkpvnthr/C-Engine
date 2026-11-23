"""Local-only HTTP worker exposing allowlisted Metal analytics."""

from __future__ import annotations

import json
import os
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

from cengine.metal_backend import discover_metallib
from cengine.metal_indicators import MetalIndicators
from cengine.metal_runtime import MetalRuntime

HOST = "127.0.0.1"
PORT = 8770
MAX_BODY_BYTES = 2 * 1024 * 1024

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LIBRARY = PROJECT_ROOT / "build" / "cengine.metallib"

# Metal runtime is created lazily (on first request / server start), NOT at import time:
# importing this worker must never crash in a CPU-only or un-built environment, and the library
# path is resolved through the authoritative `discover_metallib()` (metallib installed beside
# the native extension, then build*/), falling back to the explicit env/DEFAULT only if
# discovery finds nothing. This fixes a hard-coded `build/cengine.metallib` import-time load.
_runtime: MetalRuntime | None = None
_gpu: MetalIndicators | None = None
_library: Path | None = None


def resolve_library() -> Path:
    """Resolve the metallib path (env override > discovery > DEFAULT)."""
    env = os.environ.get("CENGINE_METAL_LIBRARY", "").strip()
    if env:
        return Path(env).resolve()
    found = discover_metallib()
    return Path(found).resolve() if found else DEFAULT_LIBRARY.resolve()


def _ensure() -> None:
    global _runtime, _gpu, _library
    if _runtime is None:
        _library = resolve_library()
        _runtime = MetalRuntime(_library)
        _gpu = MetalIndicators(_runtime)


def get_runtime() -> MetalRuntime:
    _ensure()
    assert _runtime is not None
    return _runtime


def get_gpu() -> MetalIndicators:
    _ensure()
    assert _gpu is not None
    return _gpu


def get_library() -> Path:
    _ensure()
    assert _library is not None
    return _library


def require_int_list(value: Any, name: str) -> list[int]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a non-empty list")

    if not all(isinstance(item, int) and not isinstance(item, bool) for item in value):
        raise ValueError(f"{name} must contain only integers")

    return value


def require_int(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")

    return value


class Handler(BaseHTTPRequestHandler):
    server_version = "CEngineGPUWorker/1.0"

    def send_json(self, status: int, payload: dict[str, Any]) -> None:
        encoded = json.dumps(payload, separators=(",", ":")).encode("utf-8")

        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()

        self.wfile.write(encoded)

    def read_json(self) -> dict[str, Any]:
        raw_length = self.headers.get("Content-Length")

        if raw_length is None:
            raise ValueError("Content-Length is required")

        try:
            length = int(raw_length)
        except ValueError as exc:
            raise ValueError("invalid Content-Length") from exc

        if length <= 0:
            raise ValueError("request body cannot be empty")

        if length > MAX_BODY_BYTES:
            raise ValueError("request body too large")

        raw = self.rfile.read(length)

        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError("invalid JSON") from exc

        if not isinstance(payload, dict):
            raise ValueError("JSON body must be an object")

        return payload

    def do_GET(self) -> None:
        if self.path != "/health":
            self.send_json(404, {"error": "not_found"})
            return

        self.send_json(
            200,
            {
                "status": "ok",
                "backend": "metal",
                "device": get_runtime().device.name(),
                "library": get_library().name,
            },
        )

    def do_POST(self) -> None:
        try:
            payload = self.read_json()

            if self.path == "/v1/indicators/sma":
                prices = require_int_list(payload.get("prices"), "prices")
                window = require_int(payload.get("window"), "window")

                result = get_gpu().sma(prices, window)

            elif self.path == "/v1/indicators/ema":
                prices = require_int_list(payload.get("prices"), "prices")
                numerator = require_int(
                    payload.get("alpha_numerator"),
                    "alpha_numerator",
                )
                denominator = require_int(
                    payload.get("alpha_denominator"),
                    "alpha_denominator",
                )

                result = get_gpu().ema(
                    prices,
                    numerator,
                    denominator,
                )

            elif self.path == "/v1/indicators/apo":
                fast = require_int_list(payload.get("fast_ema"), "fast_ema")
                slow = require_int_list(payload.get("slow_ema"), "slow_ema")

                result = get_gpu().apo(fast, slow)

            elif self.path == "/v1/indicators/true-range":
                highs = require_int_list(payload.get("highs"), "highs")
                lows = require_int_list(payload.get("lows"), "lows")
                previous = require_int_list(
                    payload.get("previous_closes"),
                    "previous_closes",
                )

                result = get_gpu().true_range(
                    highs,
                    lows,
                    previous,
                )

            else:
                self.send_json(404, {"error": "not_found"})
                return

            self.send_json(
                200,
                {
                    "status": "ok",
                    "result": result,
                },
            )

        except ValueError as exc:
            self.send_json(
                400,
                {
                    "error": "invalid_request",
                    "detail": str(exc),
                },
            )

        except Exception as exc:
            self.send_json(
                500,
                {
                    "error": "compute_failed",
                    "detail": type(exc).__name__,
                },
            )

    def log_message(self, format: str, *args: Any) -> None:
        print(
            f"{self.client_address[0]} "
            f"{self.command} {self.path} "
            f"{format % args}"
        )


def main() -> None:
    print(f"Metal device: {get_runtime().device.name()}")
    print(f"Metal library: {get_library()}")
    print(f"GPU worker: http://{HOST}:{PORT}")
    print("Localhost only")

    server = HTTPServer((HOST, PORT), Handler)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
