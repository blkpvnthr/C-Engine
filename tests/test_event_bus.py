import pytest

from cengine.event_bus import MarketEventBus


@pytest.mark.asyncio
async def test_each_subscriber_receives_independent_copy_reference():
    bus = MarketEventBus[int]()
    strategy = bus.subscribe("strategy", maxsize=2)
    archive = bus.subscribe("archive", maxsize=2)
    bus.publish(7)
    assert await strategy.queue.get() == 7
    assert await archive.queue.get() == 7


def test_slow_subscriber_does_not_consume_another_queue():
    bus = MarketEventBus[int]()
    slow = bus.subscribe("slow", maxsize=1)
    fast = bus.subscribe("fast", maxsize=3)
    bus.publish(1)
    bus.publish(2)
    assert slow.dropped_events == 1
    assert list(fast.queue._queue) == [1, 2]
