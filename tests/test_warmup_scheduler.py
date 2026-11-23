"""Continuous, bounded, persistent warm-up scheduler (directive item 4). Real
GpuDiscoveryProvider + HistoricalWarmup with a mock fetcher; no network."""

from __future__ import annotations

from cengine.gpu_discovery import GpuDiscoveryProvider
from cengine.historical_warmup import HistoricalWarmup
from cengine.warmup_scheduler import ContinuousWarmupScheduler


def _bars(n, start=100.0):
    out = []
    px = start
    for i in range(n):
        o = px
        px *= 1.004
        out.append({"t": f"2026-11-{(i % 27) + 1:02d}T15:00:00Z",
                    "o": o, "h": max(o, px) * 1.001, "l": min(o, px) * 0.999,
                    "c": px, "v": 1000 + i})
    return out


def _fetch_full(depth=40):
    def fetch(symbols, timeframe, start, limit=10000):
        n = depth if timeframe == "5Min" else 6
        return {s: _bars(n) for s in symbols}
    return fetch


def _scheduler(tmp_path, syms, *, batch_size=10, fetch=None, priority=None,
               max_attempts=3):
    prov = GpuDiscoveryProvider(bar_capacity=128, window=64)
    prov.set_universe(syms)
    wu = HistoricalWarmup(prov, fetch or _fetch_full(), batch_size=batch_size,
                          min_5m_bars=30, min_1m_bars=2)
    sch = ContinuousWarmupScheduler(
        wu, universe_fn=lambda: list(syms), priority_fn=priority or (lambda: {}),
        batch_size=batch_size, state_path=str(tmp_path / "progress.json"),
        max_attempts=max_attempts)
    return sch, prov


def test_covers_full_universe_across_batches_no_ceiling(tmp_path):
    syms = [f"S{i:03d}" for i in range(25)]
    sch, _ = _scheduler(tmp_path, syms, batch_size=10)
    seen_batches = []
    for _ in range(10):
        out = sch.run_once(now=0.0)
        seen_batches.append(out["batch"])
        if out["complete"]:
            break
    # 25 symbols / 10 per batch -> 3 batches (10, 10, 5); NOT capped at one 500 shot.
    assert seen_batches[:3] == [10, 10, 5]
    cov = sch.coverage()
    assert cov["warmed"] == 25 and cov["pending"] == 0
    assert sch.is_complete()


def test_never_exceeds_batch_size(tmp_path):
    syms = [f"S{i:03d}" for i in range(50)]
    sch, _ = _scheduler(tmp_path, syms, batch_size=16)
    assert len(sch.next_batch()) == 16


def test_prioritizes_positions_first(tmp_path):
    syms = [f"S{i:03d}" for i in range(30)]
    sch, _ = _scheduler(tmp_path, syms, batch_size=5,
                        priority=lambda: {"positions": ["S029"], "pending": ["S028"]})
    first = sch.next_batch()
    assert first[0] == "S029" and first[1] == "S028"  # protected acquired first


def test_persistent_progress_resumes_after_restart(tmp_path):
    syms = [f"S{i:03d}" for i in range(20)]
    sch, _ = _scheduler(tmp_path, syms, batch_size=10)
    sch.run_once(now=0.0)  # warms first 10
    assert sch.coverage()["warmed"] == 10
    # A fresh scheduler (process restart) loads progress and does NOT re-warm.
    sch2, _ = _scheduler(tmp_path, syms, batch_size=10)
    nxt = sch2.next_batch()
    assert len(nxt) == 10 and all(s not in {f"S{i:03d}" for i in range(10)} for s in nxt)
    sch2.run_once(now=0.0)
    assert sch2.coverage()["warmed"] == 20 and sch2.is_complete()


def test_bounded_retry_then_terminal_failure(tmp_path):
    syms = ["AAA", "BBB"]

    def broken(symbols, timeframe, start, limit=10000):
        return {"__error__": "HTTP 429"}

    sch, _ = _scheduler(tmp_path, syms, batch_size=10, fetch=broken, max_attempts=2)
    sch.run_once(now=0.0)
    assert sch.coverage()["failed"] == 2
    assert sch.next_batch()  # still eligible for a retry (attempt 1 of 2)
    sch.run_once(now=0.0)
    # attempt cap reached -> no longer eligible (terminal), loop converges
    assert sch.next_batch() == []
    assert sch.is_complete()


def test_membership_growth_schedules_new_symbols(tmp_path):
    syms = ["AAA", "BBB"]
    prov = GpuDiscoveryProvider(bar_capacity=128, window=64)
    prov.set_universe(["AAA", "BBB", "CCC"])
    wu = HistoricalWarmup(prov, _fetch_full(), batch_size=10, min_5m_bars=30)
    sch = ContinuousWarmupScheduler(
        wu, universe_fn=lambda: list(syms), priority_fn=lambda: {},
        batch_size=10, state_path=str(tmp_path / "p.json"))
    sch.run_once(now=0.0)
    assert sch.is_complete()  # AAA, BBB covered
    syms.append("CCC")        # membership grows
    assert sch.next_batch() == ["CCC"]  # new member scheduled, no ceiling


def test_membership_version_change_reopens_failed(tmp_path):
    syms = ["AAA"]

    def broken(symbols, timeframe, start, limit=10000):
        return {"__error__": "HTTP 500"}

    sch, _ = _scheduler(tmp_path, syms, batch_size=10, fetch=broken, max_attempts=1)
    sch.set_membership_version(1)
    sch.run_once(now=0.0)
    assert sch.next_batch() == []          # terminal after 1 attempt
    sch.set_membership_version(2)          # new snapshot re-opens failed symbols
    assert sch.next_batch() == ["AAA"]


def test_coverage_counts_are_not_conflated(tmp_path):
    # AAA full, BBB partial (incomplete), CCC none (failed); DDD never processed.
    syms = ["AAA", "BBB", "CCC", "DDD"]
    prov = GpuDiscoveryProvider(bar_capacity=128, window=64)
    prov.set_universe(syms)

    def fetch(symbols, timeframe, start, limit=10000):
        depth = {"AAA": 40, "BBB": 1, "CCC": 0, "DDD": 40}
        n5 = 40 if timeframe == "5Min" else 6
        return {s: _bars(min(depth.get(s, 0), n5 if timeframe == "5Min" else 6))
                for s in symbols}

    wu = HistoricalWarmup(prov, fetch, batch_size=3, min_5m_bars=30, min_1m_bars=2)
    sch = ContinuousWarmupScheduler(
        wu, universe_fn=lambda: syms, priority_fn=lambda: {"positions": ["AAA", "BBB", "CCC"]},
        batch_size=3, state_path=str(tmp_path / "p.json"))
    sch.run_once(now=0.0)  # processes AAA, BBB, CCC (DDD still pending)
    cov = sch.coverage()
    assert cov["warmed"] == 1 and cov["incomplete"] == 1 and cov["failed"] == 1
    assert cov["pending"] == 1 and cov["processed"] == 3 and cov["universe"] == 4
