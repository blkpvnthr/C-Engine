"""P0.1 regression: analytical-persistence backpressure must NOT kill the feed.

Root cause (systematic-debugging Phase 1): ``MarketEngine._publish_market_event``
calls ``self.archive.submit(event)`` first and unguarded. ``DailyHDF5Writer.submit``
raises :class:`PersistenceQueueFull` (a ``MarketDataStoreError`` -> ``RuntimeError``)
when its bounded HDF5 queue saturates behind the slow one-row-at-a-time writer. That
exception is NOT in the ``(KeyError, TypeError, ValueError, InvalidOperation)`` tuple
``_dispatch_message`` treats as malformed, so it escapes the market-data dispatch
callback into ``AlpacaMarketDataStream.run_forever``'s broad ``except Exception``,
which treats it as a connection failure and reconnects -- the observed
``connecting feed=sip symbols=30 attempt=40`` loop.

The directive (P0.1) requires: separate critical market processing from asynchronous
historical persistence; the WebSocket must not disconnect because an analytical
queue saturated; do not silently discard observations -- expose the degraded state
and record coverage gaps.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from market_data.alpaca_stream import QuoteEvent
from market_data.market_data_store import MarketDataStoreError, PersistenceQueueFull
from trade import EngineHealth, MarketEngine


class _FakeGate:
    def __init__(self) -> None:
        self.observed: list[tuple[str, int, str]] = []

    def observe(self, symbol: str, timestamp_ns: int, *, channel: str) -> None:
        self.observed.append((symbol, timestamp_ns, channel))


class _FakeBus:
    def __init__(self) -> None:
        self.published: list[object] = []

    def publish(self, event: object) -> None:
        self.published.append(event)


class _FakeJournal:
    def __init__(self) -> None:
        self.appended: list[tuple] = []

    def append(self, *args) -> None:
        self.appended.append(args)


class _RaisingArchive:
    """Archive whose submit() always raises, simulating a saturated/failed writer."""

    def __init__(self, exc: BaseException) -> None:
        self._exc = exc
        self.calls = 0

    def submit(self, event: object) -> None:
        self.calls += 1
        raise self._exc


def _engine_with_archive(archive: object) -> MarketEngine:
    """Minimal MarketEngine exercising only the _publish_market_event quote path."""
    engine = MarketEngine.__new__(MarketEngine)
    engine.archive = archive
    engine.market_gate = _FakeGate()
    engine.journal = _FakeJournal()
    engine._adapter_module = SimpleNamespace(STATE=None)
    engine.event_bus = _FakeBus()
    engine._scanner_bridge = None
    engine._microstructure = None
    engine._lakehouse_market_buf = None
    engine._gpu_discovery = None
    engine._adaptive_controller = None
    engine.health = EngineHealth()
    return engine


def _quote(ts: int = 1_000) -> QuoteEvent:
    return QuoteEvent(
        symbol="QQQ",
        timestamp_ns=ts,
        bid_price_ticks=100,
        bid_size=5,
        bid_exchange="V",
        ask_price_ticks=101,
        ask_size=5,
        ask_exchange="V",
        conditions=(),
        tape="C",
        sequence=1,
        received_ns=ts,
    )


def test_persistence_queue_full_does_not_propagate_and_live_path_continues() -> None:
    archive = _RaisingArchive(
        PersistenceQueueFull(
            "HDF5 persistence queue is full; refusing to silently drop market data"
        )
    )
    engine = _engine_with_archive(archive)
    quote = _quote()

    # Must NOT raise: archival backpressure cannot be allowed to kill the feed.
    asyncio.run(engine._publish_market_event(quote))

    # The critical live path must still run after the archive failed.
    assert engine.market_gate.observed == [("QQQ", 1_000, "quote")]
    assert engine.event_bus.published == [quote]
    # The drop must be accounted (NOT silent) so coverage gaps are observable.
    assert engine.health.archive_persistence_dropped == 1


def test_disabled_archive_is_skipped_without_crashing() -> None:
    # When the redundant HDF5 archive is gated off (self.archive is None), ingestion
    # must proceed normally: no submit attempt, no crash, no false coverage-gap count.
    engine = _engine_with_archive(None)
    quote = _quote(3_000)

    asyncio.run(engine._publish_market_event(quote))

    assert engine.market_gate.observed == [("QQQ", 3_000, "quote")]
    assert engine.event_bus.published == [quote]
    assert engine.health.archive_persistence_dropped == 0  # nothing attempted -> no drop


def test_generic_archive_failure_is_nonfatal_to_feed() -> None:
    archive = _RaisingArchive(MarketDataStoreError("HDF5 writer failed: disk full"))
    engine = _engine_with_archive(archive)
    quote = _quote(2_000)

    asyncio.run(engine._publish_market_event(quote))

    assert engine.market_gate.observed == [("QQQ", 2_000, "quote")]
    assert engine.event_bus.published == [quote]
    assert engine.health.archive_persistence_dropped == 1
