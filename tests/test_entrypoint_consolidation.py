"""Milestone 6 acceptance: trade.py is the single entry point; run.py and
start_engine.py are removed and imported by nothing."""

from __future__ import annotations

import re
from pathlib import Path

import trade

REPO = Path(__file__).resolve().parents[1]

_LEGACY_IMPORT = re.compile(
    r"^\s*(?:import\s+run\b|from\s+run\s+import\b"
    r"|import\s+start_engine\b|from\s+start_engine\s+import\b)"
)


def test_legacy_entry_points_removed():
    assert not (REPO / "run.py").exists(), "run.py must be removed"
    assert not (REPO / "start_engine.py").exists(), "start_engine.py must be removed"


def test_trade_owns_the_engine():
    # The engine implementation lives in trade.py (not a separate launcher).
    for symbol in (
        "MarketEngine",
        "EngineConfig",
        "CandidateRouter",
        "config_from_env",
        "load_runtime_env",
        "run_engine",
        "_engine_composition_root",
        "ConfiguredQuantityPolicy",  # migrated from start_engine.py
        "main",
    ):
        assert hasattr(trade, symbol), f"trade.{symbol} missing after consolidation"


def test_no_module_imports_legacy_entry_points():
    offenders = []
    for py in REPO.rglob("*.py"):
        parts = set(py.parts)
        if ".venv" in parts or "build" in parts:
            continue
        if py.name in ("test_entrypoint_consolidation.py", "test_integrity.py"):
            continue  # this file names them as literals; integrity test uses fixtures
        for i, line in enumerate(py.read_text(encoding="utf-8").splitlines(), 1):
            if _LEGACY_IMPORT.match(line):
                offenders.append((py.name, i, line.strip()))
    assert offenders == [], f"legacy entry-point imports remain: {offenders}"


def test_trade_main_dispatches_validate(monkeypatch):
    # The unified CLI still routes modes; --validate runs offline and returns int.
    monkeypatch.setenv("ALPACA_API_KEY", "k")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "s")
    rc = trade.main(["--validate"])
    assert rc in (0, 1)  # offline readiness result (never raises, never trades)
