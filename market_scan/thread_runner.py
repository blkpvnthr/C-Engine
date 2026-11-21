"""Daemon-thread lifecycle + single-instance lock for the scanner.

The scanner runs in-process as a ``daemon=True`` thread (mirroring trade.py's
existing dataset-builder thread), owning its own asyncio event loop. A
file-based single-instance lock (pid + timestamp) prevents a second scanner —
in this process or another — from opening a conflicting second IEX websocket or
double-writing universe.txt. A disabled config is a no-op (returns None).
"""

from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path
from typing import Any, Callable, Optional

from market_scan.config import MarketScanConfig

logger = logging.getLogger(__name__)

DEFAULT_LOCK_PATH = Path(__file__).resolve().parent / "scanner.lock"


def _pid_alive(pid: int) -> bool:
    if not isinstance(pid, int) or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists but owned by another user
    except (OverflowError, OSError, ValueError):
        return False
    return True


class SingleInstanceLock:
    """pid-based single-instance guard. The holding pid is authoritative: the
    lock is considered held iff the file exists and its pid is alive."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._held = False

    def acquire(self) -> bool:
        """Atomically acquire the lock.

        Uses an exclusive create (``O_CREAT|O_EXCL``) so two processes starting
        in the same window cannot both win (the TOCTOU a plain exists()+write
        would allow). If a lock file already exists, it is taken over only when
        its recorded pid is not alive. (A SIGKILL'd holder whose pid is later
        reused by an unrelated process is a documented residual limitation.)
        """
        payload = json.dumps({"pid": os.getpid(), "ts": 0.0})
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            fd = os.open(str(self.path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            try:
                os.write(fd, payload.encode("utf-8"))
            finally:
                os.close(fd)
            self._held = True
            return True
        except FileExistsError:
            # A lock file exists: take it over only if the holder is dead.
            try:
                data = json.loads(self.path.read_text(encoding="utf-8"))
                pid = int(data.get("pid", -1))
            except (ValueError, OSError, TypeError):
                pid = -1
            if _pid_alive(pid):
                return False
            try:
                self._write()  # holder is dead -> reclaim the stale lock
                self._held = True
                return True
            except OSError as exc:
                logger.warning("scanner lock reclaim failed: %s", exc)
                return False
        except OSError as exc:
            logger.warning("scanner lock acquire failed: %s", exc)
            return False

    def heartbeat(self, now: Optional[float] = None) -> None:
        if self._held:
            self._write(now)

    def _write(self, now: Optional[float] = None) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"pid": os.getpid(), "ts": now if now is not None else 0.0}
        self.path.write_text(json.dumps(payload), encoding="utf-8")

    def release(self) -> None:
        if not self._held:
            return
        try:
            if self.path.exists():
                data = json.loads(self.path.read_text(encoding="utf-8"))
                if int(data.get("pid", -1)) == os.getpid():
                    self.path.unlink()
        except (OSError, ValueError, TypeError):
            pass
        self._held = False


class MarketScanThread:
    """Owns the scanner daemon thread + its lock."""

    def __init__(self, scanner: Any, lock: SingleInstanceLock) -> None:
        self.scanner = scanner
        self.lock = lock
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._run, name="market-scan", daemon=True
        )
        self._thread.start()

    def _run(self) -> None:
        import asyncio
        try:
            asyncio.run(self.scanner.run())
        except Exception as exc:  # noqa: BLE001 - isolate scanner faults from the engine
            logger.error("market scanner crashed: %s", exc, exc_info=True)
        finally:
            self.lock.release()

    def stop(self, timeout: float = 5.0) -> None:
        try:
            self.scanner.stop()
        except Exception as exc:  # noqa: BLE001
            logger.warning("scanner stop error: %s", exc)
        if self._thread is not None:
            self._thread.join(timeout=timeout)
        self.lock.release()

    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()


def _default_scanner_factory(executor: Any, config: MarketScanConfig):
    from market_scan.scanner import MarketScanner
    return MarketScanner(executor, config)


def start_market_scanner(
    executor: Any,
    config: Optional[MarketScanConfig] = None,
    *,
    scanner_factory: Optional[Callable[[Any, MarketScanConfig], Any]] = None,
    lock_path: str | Path = DEFAULT_LOCK_PATH,
) -> Optional[MarketScanThread]:
    """Start the scanner daemon thread. Returns the handle, or None if disabled
    or if another live instance already holds the lock."""
    cfg = config or MarketScanConfig.from_env()
    if not cfg.enabled:
        logger.info("market scanner disabled (MARKET_SCAN_ENABLED=0); not starting")
        return None

    lock = SingleInstanceLock(lock_path)
    if not lock.acquire():
        logger.warning("market scanner already running (lock held); not starting a second")
        return None

    factory = scanner_factory or _default_scanner_factory
    try:
        scanner = factory(executor, cfg)
    except Exception as exc:  # noqa: BLE001 - never block the engine on scanner init
        logger.error("market scanner init failed: %s", exc, exc_info=True)
        lock.release()
        return None

    handle = MarketScanThread(scanner, lock)
    handle.start()
    return handle
