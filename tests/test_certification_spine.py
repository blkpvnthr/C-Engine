"""Tests for the M7 certification spine: contracts, engineering audit (M7A), data integrity (M7B).

These exercise the pure certification logic deterministically. The heavy real-data integrity
checks are smoke-tested only when a populated lakehouse is present, so the suite stays fast and
hermetic on a clean checkout.
"""

from __future__ import annotations

import datetime as _dt

import pytest

from cengine.certification import audit as audit_mod
from cengine.certification import data_integrity as di
from cengine.certification.contracts import (
    DATA_LABEL_HISTORICAL_REPLAY,
    DATA_LABEL_REAL,
    CertificationReport,
    CertificationResult,
    CertificationVerdict,
    ComponentAudit,
    ComponentStatus,
    config_fingerprint,
    safe_ticker,
    stable_id,
    status_rank,
)

# --------------------------------------------------------------------------- contracts


def test_safe_ticker_accepts_and_rejects():
    assert safe_ticker("spy") == "SPY"
    assert safe_ticker("BRK.B") == "BRK.B"
    for bad in ["", "a b", "DROP TABLE", "'; --", "TOOLONGSYMBOLXX"]:
        with pytest.raises(ValueError):
            safe_ticker(bad)


def test_stable_id_and_fingerprint_are_deterministic_and_order_independent():
    assert stable_id("a", 1, "b") == stable_id("a", 1, "b")
    assert stable_id("a", 1) != stable_id("a", 2)
    assert config_fingerprint({"x": 1, "y": 2}) == config_fingerprint({"y": 2, "x": 1})
    assert config_fingerprint({"x": 1}) != config_fingerprint({"x": 2})


def test_result_provenance_gates_profitability_claim():
    real = CertificationResult(check="c", component="x",
                               verdict=CertificationVerdict.CERTIFIED,
                               data_label=DATA_LABEL_REAL)
    replay = CertificationResult(check="c", component="x",
                                 verdict=CertificationVerdict.CERTIFIED,
                                 data_label=DATA_LABEL_HISTORICAL_REPLAY)
    assert real.profitability_claim_allowed is True
    assert replay.profitability_claim_allowed is False
    # serialization round-trips the derived flag and the enum value
    d = replay.to_dict()
    assert d["verdict"] == "certified"
    assert d["profitability_claim_allowed"] is False


def test_result_rejects_unknown_data_label():
    with pytest.raises(ValueError):
        CertificationResult(check="c", component="x",
                            verdict=CertificationVerdict.CERTIFIED, data_label="made_up")


def test_status_rank_orders_evidence():
    assert status_rank(ComponentStatus.IMPLEMENTED) < status_rank(ComponentStatus.UNIT_TESTED)
    assert (status_rank(ComponentStatus.HISTORICALLY_VALIDATED)
            < status_rank(ComponentStatus.PAPER_CERTIFIED)
            < status_rank(ComponentStatus.LIVE_APPROVED))
    assert status_rank(ComponentStatus.BLOCKED) < 0


def test_report_rollup_is_conservative():
    rep = CertificationReport(title="t")
    # empty report is never a pass
    assert rep.rollup() == CertificationVerdict.INSUFFICIENT_DATA
    rep.add(CertificationResult(check="a", component="x",
                                verdict=CertificationVerdict.CERTIFIED))
    assert rep.rollup() == CertificationVerdict.CERTIFIED
    rep.add(CertificationResult(check="b", component="x",
                                verdict=CertificationVerdict.INSUFFICIENT_DATA))
    assert rep.rollup() == CertificationVerdict.INSUFFICIENT_DATA
    rep.add(CertificationResult(check="c", component="x",
                                verdict=CertificationVerdict.NOT_CERTIFIED))
    # one failure dominates
    assert rep.rollup() == CertificationVerdict.NOT_CERTIFIED
    counts = rep.counts()
    assert counts["certified"] == 1 and counts["not_certified"] == 1


def test_component_audit_serializes_rank():
    a = ComponentAudit(component="m", status=ComponentStatus.UNIT_TESTED, module="m.py")
    d = a.to_dict()
    assert d["status"] == "unit_tested"
    assert d["status_rank"] == status_rank(ComponentStatus.UNIT_TESTED)


# --------------------------------------------------------------------------- M7A audit


def test_import_index_detects_consumption_and_tests():
    idx = {"cengine.foo", "cengine.foo.bar", "strategies"}
    assert audit_mod._consumed("cengine.foo", idx) is True       # exact
    assert audit_mod._consumed("cengine.foo.bar", idx) is True   # exact dotted
    assert audit_mod._consumed("strategies", idx) is True
    assert audit_mod._consumed("cengine.missing", idx) is False
    # tested: parent OR child import counts
    assert audit_mod._tested("cengine.foo", {"cengine.foo"}) is True
    assert audit_mod._tested("cengine.foo.bar", {"cengine.foo"}) is True
    assert audit_mod._tested("cengine.other", {"cengine.foo"}) is False


def test_try_import_distinguishes_good_and_bad_modules():
    ok, _ = audit_mod._try_import("cengine.certification.contracts")
    assert ok is True
    bad_ok, detail = audit_mod._try_import("cengine.__definitely_not_a_module__")
    assert bad_ok is False and detail


def test_audit_groups_are_well_formed():
    # every grouped module resolves to a repo-relative path candidate (file may or may not exist,
    # but the mapping must be sane) and groups are non-empty
    for group, mods in audit_mod.COMPONENT_GROUPS.items():
        assert mods, f"group {group} is empty"
        for m in mods:
            assert audit_mod._module_to_path(m).suffix == ".py"


# --------------------------------------------------------------------------- M7B data integrity


def test_expected_trading_days_known_window():
    # 2021-01-01 (Fri, New Year holiday) .. 2021-01-08 (Fri): trading days are
    # Jan 4,5,6,7,8 = 5 (Jan 1 holiday, 2/3 weekend)
    n = di._expected_trading_days(_dt.date(2021, 1, 1), _dt.date(2021, 1, 8))
    assert n == 5


def test_verdict_helper_and_date():
    assert di._v(True) == CertificationVerdict.CERTIFIED
    assert di._v(False) == CertificationVerdict.NOT_CERTIFIED
    # a known ns timestamp maps to the expected ET date
    ns = int(_dt.datetime(2022, 6, 15, 14, 30, tzinfo=_dt.timezone.utc).timestamp() * 1e9)
    assert di._date(ns) == _dt.date(2022, 6, 15)


def test_data_integrity_degrades_gracefully_without_lakehouse(monkeypatch):
    from cengine.certification.datasource import BarInventory
    monkeypatch.setattr(di, "bar_inventory",
                        lambda: BarInventory(available=False, glob="/nope/*.parquet"))
    results = di.run_data_integrity_certification()
    assert len(results) == 1
    assert results[0].verdict == CertificationVerdict.INSUFFICIENT_DATA


@pytest.mark.slow
def test_data_integrity_smoke_on_real_lakehouse_if_present():
    from cengine.certification.datasource import bar_inventory
    inv = bar_inventory()
    if not inv.available:
        pytest.skip("no populated lakehouse")
    results = di.run_data_integrity_certification()
    checks = {r.check for r in results}
    assert "data_integrity.inventory" in checks
    assert "data_integrity.duplicates" in checks
    # whatever the verdicts, each result must carry a non-empty summary
    assert all(r.summary for r in results)
