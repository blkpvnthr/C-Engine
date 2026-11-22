"""Deterministic replay helpers for normalized event sequences.

Feed provenance: replay operates on already-normalized events decoded from the
daily HDF5 archive. Each daily file carries a single ``provider``/``feed`` label
(written once, never overwritten), and
:func:`market_data.market_data_store.assert_feed_consistency` must be used to
refuse combining files from different feeds (e.g. live IEX bars with historical
SIP bars) before feeding their events here. This keeps a replay session tied to
one market view and prevents accidental mixing of incompatible datasets.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

from .event_bus import MarketEventBus
from .journal import AuditJournal


def replay(events: Iterable[Any], bus: MarketEventBus[Any]) -> int:
    """Publish in supplied order without wall-clock sleeps or timestamp invention."""
    count = 0
    previous_sequence: dict[str, int] = {}
    for event in events:
        symbol = str(event.symbol)
        sequence = int(event.sequence)
        if sequence <= previous_sequence.get(symbol, -1):
            raise ValueError(f"non-monotonic replay sequence for {symbol}")
        previous_sequence[symbol] = sequence
        bus.publish(event)
        count += 1
    return count


def replay_journal(
    journal: AuditJournal,
    handlers: dict[str, Callable[[dict[str, Any]], None]],
) -> int:
    """Replay verified records in journal order without fabricating timing."""
    count = 0
    expected = 1
    for record in journal.records():
        if record["sequence"] != expected:
            raise ValueError(f"journal sequence gap: expected {expected}")
        handler = handlers.get(record["kind"])
        if handler is not None:
            handler(record)
        count += 1
        expected += 1
    return count
