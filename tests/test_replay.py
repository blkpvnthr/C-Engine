from dataclasses import dataclass

import pytest

from cengine.event_bus import MarketEventBus
from cengine.replay import replay


@dataclass
class Event:
    symbol: str
    sequence: int


def test_replay_is_ordered_and_rejects_duplicate_sequence():
    bus = MarketEventBus[Event]()
    sub = bus.subscribe("test")
    assert replay([Event("AAPL", 1), Event("AAPL", 2)], bus) == 2
    assert [x.sequence for x in sub.queue._queue] == [1, 2]
    with pytest.raises(ValueError):
        replay([Event("AAPL", 2), Event("AAPL", 1)], bus)
