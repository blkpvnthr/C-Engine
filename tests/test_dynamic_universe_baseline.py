from pathlib import Path

import pytest

from cengine.dynamic_universe.baseline import (
    BASELINE_UNIVERSE,
    BaselineChange,
    BaselineRegistry,
)

SPEC_30 = (
    "ANT", "GPN", "DVA", "HUM", "UNH", "HCA", "UHS", "GIS", "WMT", "COST",
    "NVDA", "AMD", "AVGO", "AAPL", "MU", "TSLA", "SPY", "QQQ", "DIA", "IWM",
    "XLP", "XLV", "GLD", "SQQQ", "SOXS", "SPXU", "SDOW", "TZA", "TQQQ", "SOXL",
)


def _seed(tmp_path: Path, symbols=SPEC_30) -> Path:
    p = tmp_path / "baseline_universe.txt"
    p.write_text(", ".join(f"'{s}'" for s in symbols) + "\n", encoding="utf-8")
    return p


def test_constant_is_exact_spec_30():
    assert BASELINE_UNIVERSE == SPEC_30
    assert len(BASELINE_UNIVERSE) == 30
    assert "UPRO" not in BASELINE_UNIVERSE


def test_loads_all_30(tmp_path):
    reg = BaselineRegistry(_seed(tmp_path))
    assert reg.symbols() == SPEC_30
    assert reg.contains("nvda") is True       # normalization
    assert reg.is_baseline("UPRO") is False


def test_operator_update_adds_and_audits(tmp_path):
    reg = BaselineRegistry(_seed(tmp_path))
    change = reg.apply_operator_update(SPEC_30 + ("JEPQ",), actor="alice", now_s=100.0)
    assert isinstance(change, BaselineChange)
    assert change.added == ("JEPQ",)
    assert "JEPQ" in reg.symbols()
    assert reg.audit_log()[-1].actor == "alice"
    # persisted: a fresh registry over the same file sees JEPQ
    assert "JEPQ" in BaselineRegistry(reg.path).symbols()


def test_invalid_update_rejected_keeps_last_good(tmp_path):
    reg = BaselineRegistry(_seed(tmp_path))
    before = reg.symbols()
    with pytest.raises(ValueError):
        reg.apply_operator_update([], actor="bob")          # empty -> reject
    with pytest.raises(ValueError):
        reg.apply_operator_update(["!!!", "???"], actor="bob")  # all malformed
    assert reg.symbols() == before                          # unchanged
    assert BaselineRegistry(reg.path).symbols() == before


def test_opportunity_finder_cannot_mutate_via_reload(tmp_path):
    # Reload only re-reads; it never writes. Simulate an external bad write.
    p = _seed(tmp_path)
    reg = BaselineRegistry(p)
    p.write_text("", encoding="utf-8")       # external truncation (bad)
    assert reg.reload() is False             # keep last good
    assert reg.symbols() == SPEC_30


def test_first_run_seeds_from_constant(tmp_path):
    # Non-existent file is seeded with the spec-30 constant on construction.
    p = tmp_path / "baseline_universe.txt"
    assert not p.exists()
    reg = BaselineRegistry(p)
    assert reg.symbols() == SPEC_30
    assert p.exists()
