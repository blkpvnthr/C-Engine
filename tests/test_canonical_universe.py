"""Tests for the canonical universe manager.

The manager is the single in-memory source of new-entry eligibility for all
engine components. It loads the canonical ``universe.txt`` (which the migrated
market scanner is the sole *file* publisher of), exposes immutable versioned
snapshots, and bumps the version only on a genuine membership change -- never on
an unchanged republish (the scanner heartbeats the file without rewriting it).
"""

from __future__ import annotations

import threading

import pytest

from canonical_universe import CanonicalUniverseManager, UniverseSnapshot


def _write(path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def test_loads_repr_quoted_csv_format(tmp_path):
    # The source universe.txt is a single line of repr-quoted, comma-separated
    # tickers. The manager must parse it via the canonical loader.
    uni = tmp_path / "universe.txt"
    _write(uni, "'NVDA', 'AMD', 'AAPL', 'SPY'")
    mgr = CanonicalUniverseManager(uni, source_feed="iex")
    snap = mgr.current()
    assert snap.symbols == ("NVDA", "AMD", "AAPL", "SPY")
    assert snap.version == 1
    assert snap.source_feed == "iex"
    assert snap.published_timestamp_ns > 0


def test_fail_closed_on_missing_file(tmp_path):
    with pytest.raises(RuntimeError):
        CanonicalUniverseManager(tmp_path / "does_not_exist.txt")


def test_fail_closed_on_empty_file(tmp_path):
    uni = tmp_path / "universe.txt"
    _write(uni, "   \n  # only a comment\n")
    with pytest.raises(RuntimeError):
        CanonicalUniverseManager(uni)


def test_snapshot_is_immutable(tmp_path):
    uni = tmp_path / "universe.txt"
    _write(uni, "NVDA AMD")
    mgr = CanonicalUniverseManager(uni)
    snap = mgr.current()
    assert isinstance(snap, UniverseSnapshot)
    assert isinstance(snap.symbols, tuple)
    with pytest.raises((AttributeError, TypeError, Exception)):
        snap.symbols = ("X",)  # type: ignore[misc]


def test_refresh_bumps_version_on_membership_change(tmp_path):
    uni = tmp_path / "universe.txt"
    _write(uni, "'NVDA', 'AMD'")
    mgr = CanonicalUniverseManager(uni)
    assert mgr.version() == 1

    _write(uni, "'NVDA', 'AMD', 'TSLA'")
    new = mgr.refresh_from_file()
    assert new is not None
    assert new.version == 2
    assert new.symbols == ("NVDA", "AMD", "TSLA")
    assert mgr.version() == 2


def test_refresh_no_bump_when_membership_unchanged(tmp_path):
    # The scanner heartbeats universe.txt (mtime) without rewriting content when
    # membership is unchanged; the manager must NOT bump the version for that.
    uni = tmp_path / "universe.txt"
    _write(uni, "'NVDA', 'AMD'")
    mgr = CanonicalUniverseManager(uni)
    v0 = mgr.version()

    # Rewrite identical membership (different ordering of the *same* set still
    # counts as a change in order; identical content must not bump).
    _write(uni, "'NVDA', 'AMD'")
    result = mgr.refresh_from_file()
    assert result is None  # no change surfaced
    assert mgr.version() == v0


def test_reordering_is_a_membership_version_change(tmp_path):
    # Descending-rank order is meaningful; a reorder is a new ranking/version.
    uni = tmp_path / "universe.txt"
    _write(uni, "'NVDA', 'AMD'")
    mgr = CanonicalUniverseManager(uni)
    _write(uni, "'AMD', 'NVDA'")
    new = mgr.refresh_from_file()
    assert new is not None
    assert new.symbols == ("AMD", "NVDA")
    assert new.version == 2


def test_refresh_keeps_last_good_on_read_error(tmp_path):
    uni = tmp_path / "universe.txt"
    _write(uni, "'NVDA', 'AMD'")
    mgr = CanonicalUniverseManager(uni)
    good = mgr.current()

    # Corrupt to empty -> loader would fail closed. refresh must NOT crash and
    # must keep the last good snapshot (protective behaviour: a bad republish
    # never blanks the live universe).
    _write(uni, "")
    result = mgr.refresh_from_file()
    assert result is None
    assert mgr.current() is good
    assert mgr.current().symbols == ("NVDA", "AMD")


def test_contains_and_symbols_helpers(tmp_path):
    uni = tmp_path / "universe.txt"
    _write(uni, "'NVDA', 'amd'")  # lower-case normalised to upper
    mgr = CanonicalUniverseManager(uni)
    assert mgr.contains("NVDA")
    assert mgr.contains("amd")  # normalised on query
    assert not mgr.contains("TSLA")
    assert set(mgr.symbols()) == {"NVDA", "AMD"}


def test_thread_safe_concurrent_read_refresh(tmp_path):
    uni = tmp_path / "universe.txt"
    _write(uni, "'A', 'B', 'C'")
    mgr = CanonicalUniverseManager(uni)
    stop = threading.Event()
    errors: list[Exception] = []

    def reader():
        try:
            while not stop.is_set():
                snap = mgr.current()
                # Snapshot must always be internally consistent.
                assert isinstance(snap.symbols, tuple)
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=reader) for _ in range(4)]
    for t in threads:
        t.start()
    for i in range(50):
        _write(uni, f"'A', 'B', 'C', 'S{i}'")
        mgr.refresh_from_file()
    stop.set()
    for t in threads:
        t.join(timeout=2)
    assert not errors
