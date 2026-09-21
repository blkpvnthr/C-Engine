"""Independent bounded streams for normalized market events."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Callable, Generic, TypeVar

T = TypeVar("T")


@dataclass(slots=True)
class Subscription(Generic[T]):
    name: str
    queue: asyncio.Queue[T]
    predicate: Callable[[T], bool]
    dropped_events: int = 0


class MarketEventBus(Generic[T]):
    def __init__(self) -> None:
        self._subscriptions: dict[str, Subscription[T]] = {}

    def subscribe(
        self, name: str, *, maxsize: int = 100_000, predicate: Callable[[T], bool] = lambda _: True
    ) -> Subscription[T]:
        if name in self._subscriptions:
            raise ValueError(f"duplicate subscriber {name!r}")
        if maxsize <= 0:
            raise ValueError("maxsize must be positive")
        sub: Subscription[T] = Subscription(name, asyncio.Queue(maxsize=maxsize), predicate)
        self._subscriptions[name] = sub
        return sub

    def publish(self, event: T) -> None:
        for sub in self._subscriptions.values():
            if not sub.predicate(event):
                continue
            if sub.queue.full():
                sub.queue.get_nowait()
                sub.queue.task_done()
                sub.dropped_events += 1
            sub.queue.put_nowait(event)

    def subscription(self, name: str) -> Subscription[T]:
        return self._subscriptions[name]
