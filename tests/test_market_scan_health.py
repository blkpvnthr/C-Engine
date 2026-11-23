"""Tests for market_scan.health — operational states + resilient entry gate.

load_dynamic_universe is the fail-closed gate trade.py consults before new
entries. It must return [] (block entries) in every unhealthy/disabled/stale/
inconsistent case, and NEVER fall back to a static universe. It must only
return a non-empty entry universe when the scanner is enabled AND healthy AND
the file+metadata are fresh and mutually consistent.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import universe
from market_scan.health import (
    STATUS_DISABLED,
    STATUS_HEALTHY,
    STATUS_UNHEALTHY,
    load_dynamic_universe,
    read_live_context,
    write_live_context,
)

NOW = 1_000_000.0


def _publish(tmp_path, symbols, *, status=STATUS_HEALTHY, enabled=True,
             updated_at=NOW, market_data_fresh=True, ctx_universe=None):
    upath = tmp_path / "universe.txt"
    cpath = tmp_path / "live_context.json"
    universe.write_canonical_universe(symbols, upath)
    os.utime(upath, (updated_at, updated_at))
    payload = {
        "scanner_status": status,
        "enabled": enabled,
        "updated_at": updated_at,
        "universe_published_at": updated_at,
        "universe": list(symbols if ctx_universe is None else ctx_universe),
        "market_data_fresh": market_data_fresh,
    }
    write_live_context(cpath, payload)
    return upath, cpath


def _load(upath, cpath, *, enabled=True, now=NOW, max_age=15.0):
    return load_dynamic_universe(
        universe_path=upath, context_path=cpath,
        enabled=enabled, now=now, max_age_seconds=max_age,
    )


def test_healthy_returns_ranked_universe(tmp_path):
    upath, cpath = _publish(tmp_path, ["NVDA", "AMD", "GPN"])
    res = _load(upath, cpath)
    assert res.state == STATUS_HEALTHY
    assert res.symbols == ["NVDA", "AMD", "GPN"]  # descending rank preserved


def test_disabled_blocks_entries_and_no_static_fallback(tmp_path):
    upath, cpath = _publish(tmp_path, ["NVDA", "AMD"])  # file looks healthy
    res = _load(upath, cpath, enabled=False)
    assert res.state == STATUS_DISABLED
    assert res.symbols == []


def test_missing_context_is_unhealthy(tmp_path):
    upath = tmp_path / "universe.txt"
    universe.write_canonical_universe(["NVDA"], upath)
    res = _load(upath, tmp_path / "nope.json")
    assert res.state == STATUS_UNHEALTHY
    assert res.symbols == []


def test_status_unhealthy_blocks(tmp_path):
    upath, cpath = _publish(tmp_path, ["NVDA", "AMD"], status=STATUS_UNHEALTHY)
    assert _load(upath, cpath).symbols == []


def test_stale_context_metadata_blocks(tmp_path):
    upath, cpath = _publish(tmp_path, ["NVDA", "AMD"], updated_at=NOW - 60)
    res = _load(upath, cpath, now=NOW, max_age=15.0)
    assert res.symbols == []
    assert res.state == STATUS_UNHEALTHY


def test_stale_universe_file_mtime_blocks(tmp_path):
    upath, cpath = _publish(tmp_path, ["NVDA", "AMD"])
    os.utime(upath, (NOW - 120, NOW - 120))  # file old, context fresh
    assert _load(upath, cpath, now=NOW, max_age=15.0).symbols == []


def test_inconsistent_file_vs_metadata_blocks(tmp_path):
    # context advertises a different membership than the file actually holds
    upath, cpath = _publish(tmp_path, ["NVDA", "AMD"], ctx_universe=["TSLA", "F"])
    assert _load(upath, cpath).symbols == []


def test_market_data_not_fresh_blocks(tmp_path):
    upath, cpath = _publish(tmp_path, ["NVDA", "AMD"], market_data_fresh=False)
    assert _load(upath, cpath).symbols == []


def test_empty_universe_file_does_not_crash_blocks(tmp_path):
    upath = tmp_path / "universe.txt"
    upath.write_text("", encoding="utf-8")  # fail-closed loader would raise
    cpath = tmp_path / "live_context.json"
    write_live_context(cpath, {"scanner_status": STATUS_HEALTHY, "enabled": True,
                               "updated_at": NOW, "universe": [],
                               "market_data_fresh": True})
    res = _load(upath, cpath)  # must swallow RuntimeError, not propagate
    assert res.symbols == []
    assert res.state == STATUS_UNHEALTHY


def test_live_context_write_is_atomic_and_round_trips(tmp_path):
    cpath = tmp_path / "live_context.json"
    payload = {"scanner_status": STATUS_HEALTHY, "enabled": True, "updated_at": NOW}
    write_live_context(cpath, payload)
    assert read_live_context(cpath)["scanner_status"] == STATUS_HEALTHY
    leftovers = [p.name for p in tmp_path.iterdir() if p.name != "live_context.json"]
    assert leftovers == []


def test_zero_eligible_candidates_blocks_entries_even_if_healthy(tmp_path):
    upath, cpath = _publish(tmp_path, ["NVDA", "AMD"])
    ctx = read_live_context(cpath)
    ctx["entry_eligible_count"] = 0  # scanner healthy but nothing qualifies
    write_live_context(cpath, ctx)
    res = _load(upath, cpath)
    assert res.symbols == []
    assert res.state == STATUS_HEALTHY  # not a fault — just nothing to enter


def test_read_live_context_malformed_returns_none(tmp_path):
    cpath = tmp_path / "live_context.json"
    cpath.write_text("{not json", encoding="utf-8")
    assert read_live_context(cpath) is None
