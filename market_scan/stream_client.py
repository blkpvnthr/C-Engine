"""The scanner's single Alpaca IEX market-data websocket.

Wraps alpaca-py ``StockDataStream`` (feed=IEX). It is the ONLY websocket the
scanner owns; the live trade.py path opens none, and a single-instance lock
(see thread_runner) guards against a second IEX connection. Trade/quote handlers
parse the alpaca objects and forward to injected ``on_trade``/``on_quote``
callbacks (the OrderFlowEngine ingest). Subscriptions are applied dynamically.
The run loop reconnects with exponential backoff when ``reconnect`` is set.

The real stream class is created via ``stream_factory`` (default: the alpaca
class) so tests inject a fake and never open a socket.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional

from order_flow_engine import _epoch

logger = logging.getLogger(__name__)


def _default_stream_factory(api_key: str, secret_key: str, feed: str):
    from alpaca.data.live import StockDataStream
    try:
        from alpaca.data.enums import DataFeed
        feed_enum = DataFeed.IEX if str(feed).lower() == "iex" else DataFeed.SIP
    except Exception:  # noqa: BLE001
        feed_enum = None
    return StockDataStream(api_key, secret_key, feed=feed_enum)


class StreamClient:
    def __init__(
        self,
        *,
        api_key: str,
        secret_key: str,
        on_trade: Callable[..., None],
        on_quote: Callable[..., None],
        feed: str = "iex",
        stream_factory: Optional[Callable[..., Any]] = None,
        reconnect: bool = True,
        backoff_base: float = 1.0,
        backoff_max: float = 30.0,
        sleep_fn=None,
        log_fn: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.api_key = api_key
        self.secret_key = secret_key
        self.feed = feed
        self.on_trade = on_trade
        self.on_quote = on_quote
        self._factory = stream_factory or _default_stream_factory
        self.reconnect = reconnect
        self.backoff_base = backoff_base
        self.backoff_max = backoff_max
        self._sleep_fn = sleep_fn
        self._log = log_fn or (lambda m: logger.info(m))
        self._now_fn = None
        self._stream: Any = None
        self._subscribed: set[str] = set()
        self._last_message_ts: float | None = None
        self._messages: int = 0

    def _now(self) -> float:
        if self._now_fn is not None:
            return self._now_fn()
        import time
        return time.time()

    @property
    def subscribed(self) -> set[str]:
        return set(self._subscribed)

    @property
    def connected(self) -> bool:
        """A stream OBJECT exists (subscription may be registered). NOT proof of
        live data -- use :meth:`is_live` for that."""
        return self._stream is not None

    @property
    def received_any(self) -> bool:
        return self._messages > 0

    @property
    def last_message_ts(self) -> float | None:
        return self._last_message_ts

    def is_live(self, now: float | None = None, max_staleness_s: float = 20.0) -> bool:
        """True only if the websocket has actually DELIVERED a message recently.

        Successful subscription registration is NOT connectivity: a subscribe()
        call succeeds locally before (or even without) the socket delivering a
        single tick. Readiness requires a real inbound message within
        ``max_staleness_s``.
        """
        if self._last_message_ts is None:
            return False
        t = now if now is not None else self._now()
        return (t - self._last_message_ts) <= max_staleness_s

    def _mark_message(self) -> None:
        self._messages += 1
        self._last_message_ts = self._now()

    def connect(self) -> None:
        """Create the stream (idempotent) and (re)register handlers for the
        currently-subscribed symbols."""
        if self._stream is not None:
            return
        self._stream = self._factory(self.api_key, self.secret_key, self.feed)
        if self._subscribed:
            syms = sorted(self._subscribed)
            self._stream.subscribe_trades(self._handle_trade, *syms)
            self._stream.subscribe_quotes(self._handle_quote, *syms)

    async def apply_subscriptions(self, *, to_add, to_remove) -> None:
        """Add/remove symbol subscriptions on the live stream."""
        if self._stream is None:
            self.connect()
        add = [s.upper() for s in to_add if s.upper() not in self._subscribed]
        remove = [s.upper() for s in to_remove if s.upper() in self._subscribed]
        if add:
            self._stream.subscribe_trades(self._handle_trade, *add)
            self._stream.subscribe_quotes(self._handle_quote, *add)
            self._subscribed.update(add)
        if remove:
            self._stream.unsubscribe_trades(*remove)
            self._stream.unsubscribe_quotes(*remove)
            self._subscribed.difference_update(remove)

    async def _handle_trade(self, data: Any) -> None:
        try:
            self._mark_message()
            sym = str(getattr(data, "symbol", "")).upper()
            price = float(getattr(data, "price", 0.0) or 0.0)
            size = float(getattr(data, "size", 0.0) or 0.0)
            ts = _epoch(getattr(data, "timestamp", None))
            if sym and ts is not None:
                self.on_trade(sym, ts, price, size)
        except Exception as exc:  # noqa: BLE001 - never kill the stream on one bad tick
            self._log(f"trade handler error: {exc}")

    async def _handle_quote(self, data: Any) -> None:
        try:
            self._mark_message()
            sym = str(getattr(data, "symbol", "")).upper()
            bid = float(getattr(data, "bid_price", 0.0) or 0.0)
            ask = float(getattr(data, "ask_price", 0.0) or 0.0)
            bsz = float(getattr(data, "bid_size", 0.0) or 0.0)
            asz = float(getattr(data, "ask_size", 0.0) or 0.0)
            ts = _epoch(getattr(data, "timestamp", None))
            if sym and ts is not None:
                self.on_quote(sym, ts, bid, ask, bsz, asz)
        except Exception as exc:  # noqa: BLE001
            self._log(f"quote handler error: {exc}")

    async def run(self, max_reconnects: Optional[int] = None) -> None:
        """Run the stream, reconnecting with exponential backoff on drops.

        ``max_reconnects`` bounds the retry count (mainly for tests); None means
        retry indefinitely while ``reconnect`` is True.
        """
        import asyncio
        sleep = self._sleep_fn or asyncio.sleep
        attempt = 0
        while True:
            if self._stream is None:
                self.connect()
            try:
                await self._stream._run_forever()
                return
            except Exception as exc:  # noqa: BLE001 - connection drop
                if not self.reconnect:
                    raise
                if max_reconnects is not None and attempt >= max_reconnects:
                    raise
                delay = min(self.backoff_max, self.backoff_base * (2 ** attempt))
                self._log(f"stream dropped ({exc}); reconnecting in {delay:.1f}s")
                maybe = sleep(delay)
                if maybe is not None and hasattr(maybe, "__await__"):
                    await maybe
                attempt += 1
                # Force a fresh stream object on reconnect.
                self._stream = None

    def stop(self) -> None:
        stream = self._stream
        if stream is not None:
            try:
                close = getattr(stream, "stop", None) or getattr(stream, "close", None)
                if close:
                    close()
            except Exception as exc:  # noqa: BLE001
                self._log(f"stream stop error: {exc}")
