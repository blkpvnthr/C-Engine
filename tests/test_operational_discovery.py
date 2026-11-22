"""Membership-backed operational discovery (§4/§25.5): the master universe drives
the scanner's discovery_fn, replacing the legacy flat path."""

from __future__ import annotations

from cengine.exposure_registry import ExposureRegistry
from cengine.membership import MembershipRegistry
from cengine.operational_discovery import make_membership_discovery_fn


def _assets():
    # A master universe with equities + classified ETFs, incl. symbols well outside
    # any 250-symbol flat subset.
    rows = [{"symbol": f"EQ{i:04d}", "id": f"u{i}", "exchange": "NASDAQ",
             "class": "us_equity", "status": "active", "tradable": True,
             "fractionable": True} for i in range(600)]
    rows += [{"symbol": s, "id": f"e-{s}", "exchange": "ARCA", "class": "us_equity",
              "status": "active", "tradable": True, "fractionable": True}
             for s in ("SPY", "TQQQ", "SQQQ", "SH", "XLK")]
    # one non-tradable member (must be excluded from tradable discovery)
    rows.append({"symbol": "DEAD", "id": "u-dead", "exchange": "NASDAQ",
                 "class": "us_equity", "status": "active", "tradable": False,
                 "fractionable": False})
    return rows


def _registries():
    mem = MembershipRegistry()
    # ETFs are included in the master universe as configured members (as the real
    # build_membership does via data/configured_etfs.txt).
    mem.build(_assets(), configured=["SPY", "TQQQ", "SQQQ", "SH", "XLK"], now_iso="t")
    return mem, ExposureRegistry()


def test_discovery_returns_master_universe():
    mem, exp = _registries()
    fn = make_membership_discovery_fn(mem, exp)
    res = fn(extra=())
    # 600 equities + 5 tradable ETFs = 605 (DEAD excluded as non-tradable)
    assert len(res.symbols) == 605
    assert "EQ0599" in res.symbols      # a symbol far outside any flat-250 subset
    assert "DEAD" not in res.symbols
    assert res.error is None


def test_equities_and_etfs_split_via_exposure_registry():
    mem, exp = _registries()
    res = make_membership_discovery_fn(mem, exp)(extra=())
    assert set(res.etfs) == {"SPY", "TQQQ", "SQQQ", "SH", "XLK"}  # classified ETFs
    assert "EQ0001" in res.equities and "SPY" not in res.equities


def test_extra_symbols_included():
    mem, exp = _registries()
    res = make_membership_discovery_fn(mem, exp)(extra=["HELD1", "aapl"])
    assert "HELD1" in res.symbols and "AAPL" in res.symbols  # normalized + included


def test_max_symbols_cap_keeps_extra_first():
    mem, exp = _registries()
    fn = make_membership_discovery_fn(mem, exp, max_symbols=50)
    res = fn(extra=["HELD1", "HELD2"])
    assert len(res.symbols) <= 50
    assert "HELD1" in res.symbols and "HELD2" in res.symbols  # protected/extra kept


def test_failsafe_on_error():
    class _Broken(MembershipRegistry):
        def tradable_master_symbols(self):
            raise RuntimeError("boom")

    broken = _Broken()
    res = make_membership_discovery_fn(broken, ExposureRegistry())(extra=())
    assert res.symbols == [] and "membership_discovery" in (res.error or "")


def test_matches_scanner_discovery_contract():
    # The scanner calls discovery_fn(extra=...) and reads .symbols/.error.
    mem, exp = _registries()
    fn = make_membership_discovery_fn(mem, exp)
    res = fn(extra=sorted({"SPY"}))
    assert hasattr(res, "symbols") and hasattr(res, "error")
    assert hasattr(res, "equities") and hasattr(res, "etfs")
    assert res.source_counts["master"] >= 605
