"""Deterministic replay helpers for normalized event sequences."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from .event_bus import MarketEventBus


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
