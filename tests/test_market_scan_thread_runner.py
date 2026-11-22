"""Tests for market_scan.thread_runner — daemon lifecycle + single-instance lock.

No real scanner/socket: a fake scanner (async run loop) is injected. Pins:
single-instance lock (duplicate prevention + stale takeover), start/stop
lifecycle, and disabled-config no-op.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from market_scan.config import MarketScanConfig
from market_scan.thread_runner import (
    SingleInstanceLock,
    start_market_scanner,
)


# ---- single-instance lock ----
def test_lock_acquire_and_release(tmp_path):
    lk = SingleInstanceLock(tmp_path / "scanner.lock")
    assert lk.acquire() is True
    assert (tmp_path / "scanner.lock").exists()
    lk.release()
    assert not (tmp_path / "scanner.lock").exists()


def test_second_lock_same_process_is_refused(tmp_path):
    lk1 = SingleInstanceLock(tmp_path / "scanner.lock")
    lk2 = SingleInstanceLock(tmp_path / "scanner.lock")
    assert lk1.acquire() is True
    assert lk2.acquire() is False  # our own pid is alive -> duplicate refused
    lk1.release()
    assert lk2.acquire() is True   # freed -> now acquirable


def test_stale_lock_from_dead_pid_is_taken_over(tmp_path):
    p = tmp_path / "scanner.lock"
    p.write_text('{"pid": 2147480000, "ts": 1.0}', encoding="utf-8")  # dead pid
    lk = SingleInstanceLock(p)
    assert lk.acquire() is True


# ---- lifecycle ----
class _FakeScanner:
    def __init__(self, *a, **k):
        self.ran = False
        self.stopped = False
        self._stop = False

    async def run(self):
        import asyncio
        self.ran = True
        while not self._stop:
            await asyncio.sleep(0.005)

    def stop(self):
        self._stop = True
        self.stopped = True


def test_start_runs_and_stop_joins(tmp_path):
    fake = _FakeScanner()
    handle = start_market_scanner(
        executor=object(),
        config=MarketScanConfig(enabled=True),
        scanner_factory=lambda executor, config: fake,
        lock_path=tmp_path / "scanner.lock",
    )
    assert handle is not None
    # give the daemon thread a moment to enter run()
    for _ in range(200):
        if fake.ran:
            break
        time.sleep(0.005)
    assert fake.ran is True
    assert handle.is_alive() is True
    handle.stop(timeout=2.0)
    assert fake.stopped is True
    assert handle.is_alive() is False
    assert not (tmp_path / "scanner.lock").exists()  # lock released on stop


def test_disabled_config_does_not_start(tmp_path):
    handle = start_market_scanner(
        executor=object(),
        config=MarketScanConfig(enabled=False),
        scanner_factory=lambda executor, config: _FakeScanner(),
        lock_path=tmp_path / "scanner.lock",
    )
    assert handle is None
    assert not (tmp_path / "scanner.lock").exists()


def test_duplicate_start_refused_while_running(tmp_path):
    f1 = _FakeScanner()
    h1 = start_market_scanner(
        executor=object(), config=MarketScanConfig(enabled=True),
        scanner_factory=lambda executor, config: f1,
        lock_path=tmp_path / "scanner.lock",
    )
    h2 = start_market_scanner(
        executor=object(), config=MarketScanConfig(enabled=True),
        scanner_factory=lambda executor, config: _FakeScanner(),
        lock_path=tmp_path / "scanner.lock",
    )
    assert h1 is not None
    assert h2 is None  # lock held -> duplicate refused
    h1.stop(timeout=2.0)
