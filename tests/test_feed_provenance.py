"""Feed provenance in persisted datasets + feed/trading independence."""

from datetime import date

import h5py
import pytest

from market_data.alpaca_stream import BarEvent
from market_data.market_data_store import (
    DailyHDF5Writer,
    MarketDataStoreError,
    assert_feed_consistency,
    create_daily_file,
    read_provenance,
)


def _bar(ts_ns: int) -> BarEvent:
    return BarEvent(
        symbol="QQQ",
        timestamp_ns=ts_ns,
        open_ticks=4_000_000,
        high_ticks=4_010_000,
        low_ticks=3_995_000,
        close_ticks=4_005_000,
        volume=1000,
        trade_count=10,
        vwap_ticks=4_003_000,
        sequence=1,
        received_ns=ts_ns + 1,
    )


def test_create_daily_file_records_feed(tmp_path):
    path = create_daily_file(tmp_path, date(2026, 1, 2), provider="alpaca-iex", feed="iex")
    with h5py.File(path, "r") as handle:
        assert handle.attrs["provider"] == "alpaca-iex"
        assert handle.attrs["feed"] == "iex"


def test_writer_persists_provider_and_feed(tmp_path):
    writer = DailyHDF5Writer(root=tmp_path, provider="alpaca-iex", feed="iex")
    writer.start()
    # 2026-01-02 15:00 UTC is a normal NY trading session timestamp.
    writer.submit(_bar(1_767_366_000_000_000_000))
    writer.close(drain=True)

    meta = None
    for sub in tmp_path.rglob("market_*.h5"):
        meta = read_provenance(sub)
    assert meta is not None
    assert meta["provider"] == "alpaca-iex"
    assert meta["feed"] == "iex"
    assert meta["price_scale"] == 10_000


def test_assert_feed_consistency_allows_matching(tmp_path):
    a = create_daily_file(tmp_path, date(2026, 1, 2), provider="alpaca-iex", feed="iex")
    b = create_daily_file(tmp_path, date(2026, 1, 5), provider="alpaca-iex", feed="iex")
    shared = assert_feed_consistency([a, b])
    assert shared["feed"] == "iex"
    assert shared["provider"] == "alpaca-iex"


def test_assert_feed_consistency_rejects_mixed_feeds(tmp_path):
    iex = create_daily_file(tmp_path, date(2026, 1, 2), provider="alpaca-iex", feed="iex")
    sip = create_daily_file(tmp_path, date(2026, 1, 5), provider="alpaca-sip", feed="sip")
    with pytest.raises(MarketDataStoreError, match="different providers|different feeds"):
        assert_feed_consistency([iex, sip])


def test_feed_selection_does_not_change_trading_endpoint(monkeypatch):
    """Selecting a market-data feed must never change paper vs. live trading."""
    from cengine.execution import trading_base_url

    monkeypatch.delenv("ALPACA_TRADING_BASE_URL", raising=False)
    for feed in ("iex", "sip", "delayed_sip"):
        monkeypatch.setenv("CENGINE_ALPACA_FEED", feed)
        assert "paper-api" in trading_base_url()
