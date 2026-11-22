"""Phase B.2 — native market-state ingestion (trading::discovery::DiscoveryEngine).

Exercises the native ingestion/state/buffer/dirty-tracking mechanics through the
pybind11 surface. These are synthetic-batch unit tests of the ingestion
machinery; genuine real-time IEX observation flow is certified separately (and
remains pending a live session) per the directive's B.6 verification.
"""

from __future__ import annotations

import _cengine_native as native
import pytest

TF_1M = native.BarTimeframe.ONE_MINUTE
TF_5M = native.BarTimeframe.FIVE_MINUTE
SCALE = 10_000  # 1 tick == $0.0001


@pytest.fixture
def engine():
    de = native.DiscoveryEngine(bar_capacity=8)
    de.update_membership(["AAPL", "MSFT", "NVDA"])
    return de


# ----------------------------------------------------------------- membership
def test_membership_assigns_stable_ids_and_generation(engine):
    assert engine.symbol_count == 3
    assert engine.membership_version == 1
    ids = {s: engine.symbol_id(s) for s in ("AAPL", "MSFT", "NVDA")}
    assert sorted(ids.values()) == [0, 1, 2]  # stable, contiguous
    for s, i in ids.items():
        assert engine.symbol_name(i) == s
        assert engine.generation(i) == 1
    assert engine.symbol_id("ZZZZ") is None


def test_membership_version_only_bumps_on_change(engine):
    v = engine.membership_version
    engine.update_membership(["AAPL", "MSFT", "NVDA"])  # unchanged
    assert engine.membership_version == v
    engine.update_membership(["AAPL", "MSFT", "NVDA", "TSLA"])  # add
    assert engine.membership_version == v + 1


def test_reentry_bumps_generation_and_clears_state(engine):
    aid = engine.symbol_id("AAPL")
    engine.ingest_bars(
        [native.BarObs(symbol_id=aid, timestamp_ns=1, open_ticks=100 * SCALE,
                       high_ticks=101 * SCALE, low_ticks=99 * SCALE,
                       close_ticks=100 * SCALE, volume=10)], TF_5M)
    assert engine.snapshot(aid).bars_5m == 1
    engine.update_membership(["MSFT", "NVDA"])          # drop AAPL
    assert engine.snapshot(aid).active is False
    engine.update_membership(["AAPL", "MSFT", "NVDA"])  # re-enter
    snap = engine.snapshot(aid)
    assert engine.generation(aid) == 2  # generation bumped
    assert snap.active is True
    assert snap.bars_5m == 0  # stale buffers cleared on re-entry


# ----------------------------------------------------------------- ingestion
def test_ingest_quote_updates_authoritative_l1(engine):
    aid = engine.symbol_id("MSFT")
    n = engine.ingest_quotes([
        native.QuoteObs(symbol_id=aid, timestamp_ns=1_000, bid_ticks=400 * SCALE,
                        ask_ticks=400 * SCALE + 50, bid_size=3, ask_size=4,
                        sequence=1)])
    assert n == 1
    s = engine.snapshot(aid)
    assert s.has_quote and s.bid_ticks == 400 * SCALE and s.ask_ticks == 400 * SCALE + 50
    assert s.quote_updates == 1 and s.dirty


def test_ingest_trade_and_bars(engine):
    aid = engine.symbol_id("NVDA")
    assert engine.ingest_trades([
        native.TradeObs(symbol_id=aid, timestamp_ns=1_000,
                        price_ticks=900 * SCALE, size=5, sequence=1)]) == 1
    assert engine.ingest_bars([
        native.BarObs(symbol_id=aid, timestamp_ns=2_000, open_ticks=900 * SCALE,
                      high_ticks=905 * SCALE, low_ticks=898 * SCALE,
                      close_ticks=902 * SCALE, volume=1234)], TF_1M) == 1
    s = engine.snapshot(aid)
    assert s.has_trade and s.last_trade_ticks == 900 * SCALE
    assert s.bars_1m == 1
    # ticks -> dollars conversion at the configured scale
    assert engine.last_bar_close(aid, TF_1M) == pytest.approx(902.0, abs=1e-2)


def test_ingest_rejects_invalid_rows(engine):
    aid = engine.symbol_id("AAPL")
    applied = engine.ingest_quotes([
        native.QuoteObs(symbol_id=999, timestamp_ns=1, bid_ticks=1, ask_ticks=2),  # unknown id
        native.QuoteObs(symbol_id=aid, timestamp_ns=1, bid_ticks=200, ask_ticks=100),  # crossed
        native.QuoteObs(symbol_id=aid, timestamp_ns=1, bid_ticks=0, ask_ticks=1),  # non-positive
    ])
    assert applied == 0
    assert engine.coverage().rejected == 3
    assert not engine.snapshot(aid).has_quote


def test_bar_ring_buffer_bounded():
    de = native.DiscoveryEngine(bar_capacity=4)
    de.update_membership(["AAPL"])
    aid = de.symbol_id("AAPL")
    for i in range(10):
        de.ingest_bars([
            native.BarObs(symbol_id=aid, timestamp_ns=1_000 + i,
                          open_ticks=100 * SCALE, high_ticks=100 * SCALE,
                          low_ticks=100 * SCALE, close_ticks=(100 + i) * SCALE,
                          volume=1)], TF_1M)
    assert de.snapshot(aid).bars_1m == 4  # oldest dropped, capacity honored
    assert de.last_bar_close(aid, TF_1M) == pytest.approx(109.0, abs=1e-2)


# ----------------------------------------------------------------- dirty tracking
def test_dirty_tracking_and_clear(engine):
    a, n = engine.symbol_id("AAPL"), engine.symbol_id("NVDA")
    engine.ingest_trades([
        native.TradeObs(symbol_id=a, timestamp_ns=1, price_ticks=100 * SCALE, size=1),
        native.TradeObs(symbol_id=n, timestamp_ns=1, price_ticks=900 * SCALE, size=1)])
    assert engine.dirty_symbols() == sorted([a, n])
    engine.clear_dirty()
    assert engine.dirty_symbols() == []
    assert not engine.snapshot(a).dirty


# ----------------------------------------------------------------- coverage
def test_coverage_summary(engine):
    a = engine.symbol_id("AAPL")
    engine.ingest_quotes([
        native.QuoteObs(symbol_id=a, timestamp_ns=1, bid_ticks=100 * SCALE,
                        ask_ticks=100 * SCALE + 10, sequence=1)])
    engine.ingest_bars([
        native.BarObs(symbol_id=a, timestamp_ns=2, open_ticks=100 * SCALE,
                      high_ticks=101 * SCALE, low_ticks=99 * SCALE,
                      close_ticks=100 * SCALE, volume=1)], TF_5M)
    cov = engine.coverage()
    assert cov.symbols == 3
    assert cov.observed == 1 and cov.with_quote == 1
    assert cov.with_bars_5m == 1
    assert cov.total_quote_updates == 1 and cov.total_bar_updates == 1
    assert cov.membership_version == 1
