"""Single market-data pipeline bridge (Milestone 4).

The engine owns the one and only market-data websocket
(:class:`market_data.alpaca_stream.AlpacaMarketDataStream`). The migrated market
scanner must consume the *same* normalized observations rather than open a second
IEX socket. This module provides that convergence while keeping the scanner's
ranking/discovery compute off the latency-sensitive execution loop:

  * :class:`ScannerEventBridge` — a bounded, thread-safe conduit. The engine loop
    calls :meth:`offer` on every normalized event (non-blocking, drop-oldest, so
    a slow scanner can never stall the engine). The scanner thread drains it.
  * :class:`BridgeStreamClient` — a drop-in replacement for the scanner's own
    ``StreamClient``. It implements the exact interface the scanner expects
    (``run``/``stop``/``apply_subscriptions``/``is_live``) but sources ticks from
    the bridge instead of a socket, converting engine events (integer ticks,
    nanosecond timestamps) into the OrderFlowEngine's expected shape (dollar
    prices, epoch-second timestamps). The symbols the scanner asks to cover are
    recorded in :attr:`desired_symbols` so the engine's single
    ``UniverseSubscriptionSync`` can fold them into the one socket's subscription.

No second websocket is opened; there is exactly one logical pipeline.
"""

from __future__ import annotations

import collections
import logging
import threading
from typing import Any, Callable, Optional

logger = logging.getLogger("scanner_bridge")

DEFAULT_PRICE_SCALE = 10000


class ScannerEventBridge:
    """Bounded, thread-safe engine→scanner event conduit (drop-oldest)."""

    def __init__(self, maxlen: int = 20000) -> None:
        self._buf: collections.deque = collections.deque(maxlen=maxlen)
        self._lock = threading.Lock()
        self.offered = 0
        self.dropped = 0

    def offer(self, event: Any) -> None:
        """Called on the ENGINE loop; never blocks. Drops the oldest event when
        full so the engine's market-data fan-out is never back-pressured."""
        with self._lock:
            if len(self._buf) == self._buf.maxlen:
                self.dropped += 1  # deque auto-evicts oldest on append-when-full
            self._buf.append(event)
            self.offered += 1

    def drain(self, max_items: int = 10000) -> list:
        """Called on the SCANNER loop; returns up to ``max_items`` events FIFO."""
        out: list = []
        with self._lock:
            while self._buf and len(out) < max_items:
                out.append(self._buf.popleft())
        return out

    def depth(self) -> int:
        with self._lock:
            return len(self._buf)


class BridgeStreamClient:
    """Scanner stream adapter sourced from a :class:`ScannerEventBridge`."""

    def __init__(
        self,
        bridge: ScannerEventBridge,
        *,
        on_trade: Optional[Callable[..., None]] = None,
        on_quote: Optional[Callable[..., None]] = None,
        price_scale: int = DEFAULT_PRICE_SCALE,
        poll_interval_s: float = 0.05,
        now_fn: Optional[Callable[[], float]] = None,
        log_fn: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.bridge = bridge
        self.on_trade = on_trade
        self.on_quote = on_quote
        self._scale = price_scale if price_scale > 0 else DEFAULT_PRICE_SCALE
        self.poll_interval_s = poll_interval_s
        self._now_fn = now_fn
        self._log = log_fn or (lambda m: logger.info(m))

        self._desired: set[str] = set()
        self._desired_lock = threading.Lock()
        self._last_message_ts: float | None = None
        self._messages = 0
        self._stopped = False

    # ------------------------------------------------------------------ wiring
    def bind(self, *, on_trade: Callable[..., None], on_quote: Callable[..., None]) -> None:
        """Bind the OrderFlowEngine ingest callbacks (done after the scanner --
        and thus its OFE -- is constructed)."""
        self.on_trade = on_trade
        self.on_quote = on_quote

    def _now(self) -> float:
        if self._now_fn is not None:
            return self._now_fn()
        import time
        return time.time()

    # ------------------------------------------------- scanner stream interface
    async def apply_subscriptions(self, *, to_add, to_remove) -> None:
        """Record the symbols the scanner wants covered. The engine's single
        ``UniverseSubscriptionSync`` reads :attr:`desired_symbols` and applies the
        actual subscription on the one socket -- this adapter opens nothing."""
        with self._desired_lock:
            self._desired |= {str(s).upper() for s in to_add if s}
            self._desired -= {str(s).upper() for s in to_remove if s}

    @property
    def subscribed(self) -> set[str]:
        return self.desired_symbols  # optimistic: what we asked the engine to cover

    @property
    def desired_symbols(self) -> frozenset[str]:
        with self._desired_lock:
            return frozenset(self._desired)

    @property
    def received_any(self) -> bool:
        return self._messages > 0

    def is_live(self, now: float | None = None, max_staleness_s: float = 20.0) -> bool:
        if self._last_message_ts is None:
            return False
        t = now if now is not None else self._now()
        return (t - self._last_message_ts) <= max_staleness_s

    async def run(self, *_a, **_k) -> None:
        import asyncio
        while not self._stopped:
            self.drain_once()
            await asyncio.sleep(self.poll_interval_s)

    def stop(self) -> None:
        self._stopped = True

    # --------------------------------------------------------------- draining
    def drain_once(self) -> int:
        """Drain the bridge and forward to the OFE. Returns events forwarded."""
        forwarded = 0
        for ev in self.bridge.drain():
            try:
                if self._forward(ev):
                    forwarded += 1
            except Exception as exc:  # noqa: BLE001 - one bad tick never stops draining
                self._log(f"bridge forward error: {exc}")
        return forwarded

    def _forward(self, ev: Any) -> bool:
        # Duck-typed on the engine event shapes: a trade carries ``price_ticks``;
        # a quote carries ``bid_price_ticks``. Bars (and anything else) are ignored
        # (the OrderFlowEngine is trade/quote driven).
        ts_ns = getattr(ev, "timestamp_ns", None)
        sym = getattr(ev, "symbol", None)
        if ts_ns is None or not sym:
            return False
        ts = ts_ns / 1e9
        if hasattr(ev, "price_ticks"):
            price = ev.price_ticks / self._scale
            size = float(getattr(ev, "size", 0.0) or 0.0)
            if self.on_trade is not None:
                self.on_trade(str(sym).upper(), ts, price, size)
            self._mark()
            return True
        if hasattr(ev, "bid_price_ticks"):
            bid = ev.bid_price_ticks / self._scale
            ask = getattr(ev, "ask_price_ticks", 0) / self._scale
            bsz = float(getattr(ev, "bid_size", 0.0) or 0.0)
            asz = float(getattr(ev, "ask_size", 0.0) or 0.0)
            if self.on_quote is not None:
                self.on_quote(str(sym).upper(), ts, bid, ask, bsz, asz)
            self._mark()
            return True
        return False

    def _mark(self) -> None:
        self._messages += 1
        self._last_message_ts = self._now()
