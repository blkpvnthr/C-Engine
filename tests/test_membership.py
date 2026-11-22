"""Master-universe membership registry (§14-17). Synthetic assets, no network."""

from __future__ import annotations

from cengine.membership import MembershipRegistry


def _assets():
    return [
        {"symbol": "AAPL", "id": "u-aapl", "exchange": "NASDAQ", "class": "us_equity",
         "status": "active", "tradable": True, "fractionable": True},
        {"symbol": "MSFT", "id": "u-msft", "exchange": "NASDAQ", "class": "us_equity",
         "status": "active", "tradable": True, "fractionable": True},
        {"symbol": "JPM", "id": "u-jpm", "exchange": "NYSE", "class": "us_equity",
         "status": "active", "tradable": True, "fractionable": False},
        {"symbol": "ZZZ", "id": "u-zzz", "exchange": "NASDAQ", "class": "us_equity",
         "status": "active", "tradable": False, "fractionable": False},
    ]


def test_build_and_counts():
    r = MembershipRegistry()
    snap = r.build(_assets(), sp500=["AAPL", "MSFT", "JPM"], djia=["AAPL", "JPM"],
                   configured=["SPY"], now_iso="2026-11-22T00:00:00Z")
    c = snap.counts
    assert c["alpaca_supported"] == 5  # 4 assets + SPY (configured, not in assets)
    assert c["nasdaq_member"] == 3     # AAPL, MSFT, ZZZ by exchange
    assert c["sp500_member"] == 3
    assert c["djia_member"] == 2
    assert c["configured_member"] == 1
    # master universe deduplicated: AAPL/MSFT/JPM/ZZZ/SPY all qualify once each
    assert c["master_universe"] == 5
    assert c["tradable"] == 3


def test_nasdaq_from_exchange_not_index():
    r = MembershipRegistry()
    r.build(_assets(), now_iso="t")
    assert r.get("AAPL").nasdaq_member is True
    assert r.get("JPM").nasdaq_member is False  # NYSE
    # index membership is independent of exchange: none configured -> all False
    assert r.get("AAPL").sp500_member is False


def test_stable_ids_and_dedup():
    r = MembershipRegistry()
    r.build(_assets(), sp500=["AAPL"], now_iso="t")
    ids = sorted(r.get(s).symbol_id for s in ("AAPL", "MSFT", "JPM", "ZZZ"))
    assert ids == [0, 1, 2, 3]  # contiguous stable ids
    # AAPL in both nasdaq + sp500 counted once in master
    assert r.get("AAPL").in_master()


def test_index_member_absent_from_alpaca_is_recorded():
    r = MembershipRegistry()
    r.build(_assets(), sp500=["NOTLISTED"], now_iso="t")
    info = r.get("NOTLISTED")
    assert info is not None and info.sp500_member
    assert info.asset_status == "not_in_alpaca_assets" and info.tradable is False


def test_generation_bumps_on_status_change():
    r = MembershipRegistry()
    r.build(_assets(), now_iso="t1")
    assert r.get("AAPL").generation == 1
    # AAPL goes inactive -> generation bump; MSFT unchanged
    a2 = _assets()
    a2[0]["status"] = "inactive"
    snap = r.build(a2, now_iso="t2")
    assert r.get("AAPL").generation == 2
    assert r.get("MSFT").generation == 1
    assert snap.changes["changed_count"] >= 1


def test_persist_and_degraded_reload(tmp_path):
    r = MembershipRegistry()
    r.build(_assets(), sp500=["AAPL"], djia=["JPM"], now_iso="t")
    path = tmp_path / "membership.json"
    r.persist(path)
    r2 = MembershipRegistry()
    assert r2.load_cache(path) is True
    assert r2.snapshot_meta.degraded is True  # cache load is degraded recovery
    assert r2.get("AAPL").sp500_member and r2.get("JPM").djia_member
    assert len(r2) == len(r)


def test_group_filters_and_tradable_master():
    r = MembershipRegistry()
    r.build(_assets(), sp500=["AAPL", "JPM"], now_iso="t")
    assert set(r.group("nasdaq")) == {"AAPL", "MSFT", "ZZZ"}
    assert set(r.group("sp500")) == {"AAPL", "JPM"}
    # ZZZ is nasdaq but non-tradable -> excluded from tradable master
    assert "ZZZ" not in r.tradable_master_symbols()
    assert "AAPL" in r.tradable_master_symbols()


def test_index_source_provenance_parsing(tmp_path):
    """build_membership reads the provenance header (source/retrieved_at/as_of/count)
    written by fetch_index_constituents, so the snapshot records point-in-time
    source identity + effective date (directive item 5). Constituents are never
    fabricated: an absent file reports not_configured."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "build_membership", "tools/build_membership.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    f = tmp_path / "sp500.txt"
    f.write_text("# S&P 500 constituents — verifiable source, NEVER fabricated.\n"
                 "# source: https://example.test/constituents.csv\n"
                 "# retrieved_at: 2026-11-22T00:00:00Z\n"
                 "# as_of: 2026-11-20\n"
                 "# count: 2\n\nAAPL\nMSFT\n")
    prov = mod._read_index_provenance(str(f))
    assert prov["source"] == "https://example.test/constituents.csv"
    assert prov["as_of"] == "2026-11-20" and prov["count"] == "2"

    import os
    os.environ["CENGINE_SP500_FILE"] = str(f)
    try:
        syms, src = mod._index_source("CENGINE_SP500_FILE", "data/sp500.txt")
    finally:
        del os.environ["CENGINE_SP500_FILE"]
    assert syms == ["AAPL", "MSFT"]
    assert src["status"] == "configured" and src["effective_members"] == 2
    assert src["source"] == "https://example.test/constituents.csv"

    missing = mod._index_source("CENGINE_SP500_FILE_MISSING", "does/not/exist.txt")
    assert missing[1]["status"] == "not_configured"
