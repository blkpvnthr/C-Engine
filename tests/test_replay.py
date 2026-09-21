from dataclasses import dataclass

import pytest

from cengine.event_bus import MarketEventBus
from cengine.journal import AuditJournal
from cengine.replay import replay, replay_journal


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


def test_verified_journal_replay_dispatches_in_sequence(tmp_path):
    journal = AuditJournal(tmp_path / "audit.jsonl")
    journal.append("market", 1, {"value": 1})
    journal.append("execution", 2, {"value": 2})
    observed = []
    count = replay_journal(
        journal,
        {
            "market": lambda record: observed.append(record["payload"]["value"]),
            "execution": lambda record: observed.append(record["payload"]["value"]),
        },
    )
    assert count == 2
    assert observed == [1, 2]
