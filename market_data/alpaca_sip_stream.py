"""
alpaca_sip_stream.py

Live U.S. equity market-data connection for Alpaca's SIP WebSocket feed.

Architecture
------------
Alpaca SIP WebSocket
        |
        v
AlpacaSIPStream
        |
        +--> QuoteEvent
        +--> TradeEvent
        +--> BarEvent
        |
        v
user callbacks / queue / future C++ bridge
        |
        v
MarketSample / DepthSample / FeatureEngine / Metal

Important:
- This module is MARKET DATA ONLY. It has no order-entry authority.
- Credentials are read from environment variables and are never logged.
- A single stream is used for all subscribed symbols.
- SIP provides consolidated trades and NBBO-style quote updates, not a
  full exchange-by-exchange Level-II depth book. Do not fabricate L2 depth
  from SIP quote messages.
- QQQ and SQQQ are subscribed through Alpaca SIP as equities.
- VIX and VXN are volatility-index factors and are intentionally NOT inserted
  into the Alpaca stock subscription. Supply them through IndexFactorBridge
  from a provider licensed to distribute those index levels.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import signal
import ssl
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from enum import Enum
from typing import Awaitable, Callable, Iterable, Optional, Sequence

from market_data_store import DailyHDF5Writer, MarketDataStoreError

try:
    import websockets
    from websockets.exceptions import ConnectionClosed
except ImportError as exc:  # pragma: no cover
    raise RuntimeError(
        "Missing dependency 'websockets'. Install with: "
        "python -m pip install websockets"
    ) from exc


LOGGER = logging.getLogger("alpaca_sip_stream")

SIP_URL = "wss://stream.data.alpaca.markets/v2/sip"
TEST_URL = "wss://stream.data.alpaca.markets/v2/test"

DEFAULT_PRICE_SCALE = 10_000  # $0.0001 == 1 tick
DEFAULT_AUTH_TIMEOUT = 9.0
DEFAULT_CONNECT_TIMEOUT = 10.0
DEFAULT_HEARTBEAT_TIMEOUT = 45.0
DEFAULT_BACKOFF_INITIAL = 0.5
DEFAULT_BACKOFF_MAX = 30.0


# =============================================================================
# TYPES
# =============================================================================


class ConnectionState(str, Enum):
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    AUTHENTICATED = "authenticated"
    SUBSCRIBED = "subscribed"
    STOPPING = "stopping"


@dataclass(frozen=True, slots=True)
class QuoteEvent:
    symbol: str
    timestamp_ns: int

    bid_price_ticks: int
    bid_size: int
    bid_exchange: str

    ask_price_ticks: int
    ask_size: int
    ask_exchange: str

    conditions: tuple[str, ...]
    tape: str

    sequence: int
    received_ns: int

    @property
    def latency_ns(self) -> int:
        return max(0, self.received_ns - self.timestamp_ns)


@dataclass(frozen=True, slots=True)
class TradeEvent:
    symbol: str
    timestamp_ns: int

    trade_id: int
    exchange: str
    price_ticks: int
    size: int

    conditions: tuple[str, ...]
    tape: str

    sequence: int
    received_ns: int

    @property
    def latency_ns(self) -> int:
        return max(0, self.received_ns - self.timestamp_ns)


@dataclass(frozen=True, slots=True)
class BarEvent:
    symbol: str
    timestamp_ns: int

    open_ticks: int
    high_ticks: int
    low_ticks: int
    close_ticks: int

    volume: int
    trade_count: int
    vwap_ticks: Optional[int]

    sequence: int
    received_ns: int


@dataclass(frozen=True, slots=True)
class IndexEvent:
    """Provider-neutral VIX/VXN observation for the stat-arb factor layer."""

    symbol: str
    timestamp_ns: int
    value_ticks: int
    received_ns: int
    provider: str = ""
    sequence: int = 0

    @property
    def latency_ns(self) -> int:
        return max(0, self.received_ns - self.timestamp_ns)


IndexHandler = Callable[[IndexEvent], Optional[Awaitable[None]]]


@dataclass(frozen=True, slots=True)
class VolatilityFactorState:
    symbol: str
    value_ticks: int
    timestamp_ns: int
    received_ns: int
    provider: str
    sequence: int

    @property
    def age_ns(self) -> int:
        return max(0, time.time_ns() - self.timestamp_ns)


class IndexFactorBridge:
    """
    Normalizes externally supplied VIX/VXN observations.

    The upstream source is deliberately abstract so a licensed real-time
    index feed or replay source can be attached without contaminating the
    Alpaca SIP equity connection.
    """

    SUPPORTED = frozenset({"VIX", "VXN"})

    def __init__(
        self,
        *,
        price_scale: int = DEFAULT_PRICE_SCALE,
        on_index: Optional[IndexHandler] = None,
    ) -> None:
        if price_scale <= 0:
            raise ValueError("price_scale must be positive")

        self._price_scale = price_scale
        self._on_index = on_index
        self._states: dict[str, VolatilityFactorState] = {}
        self._sequence = 0

    async def publish(
        self,
        symbol: str,
        value: object,
        *,
        timestamp_ns: Optional[int] = None,
        received_ns: Optional[int] = None,
        provider: str = "",
        sequence: int = 0,
    ) -> IndexEvent:
        normalized = symbol.strip().upper()

        if normalized not in self.SUPPORTED:
            raise ValueError(
                f"unsupported volatility index {normalized!r}; "
                f"expected one of {sorted(self.SUPPORTED)!r}"
            )

        ts = int(timestamp_ns) if timestamp_ns is not None else time.time_ns()
        recv = int(received_ns) if received_ns is not None else time.time_ns()

        if ts <= 0 or recv <= 0:
            raise ValueError("index timestamps must be positive")

        seq = int(sequence)
        if seq <= 0:
            self._sequence += 1
            seq = self._sequence

        event = IndexEvent(
            symbol=normalized,
            timestamp_ns=ts,
            value_ticks=_price_to_ticks(value, self._price_scale),
            received_ns=recv,
            provider=str(provider),
            sequence=seq,
        )

        previous = self._states.get(normalized)

        if previous is not None:
            if event.timestamp_ns < previous.timestamp_ns:
                raise ValueError(f"out-of-sequence {normalized} timestamp")
            if (
                event.sequence > 0
                and previous.sequence > 0
                and event.sequence <= previous.sequence
            ):
                raise ValueError(f"out-of-sequence {normalized} sequence")

        self._states[normalized] = VolatilityFactorState(
            symbol=event.symbol,
            value_ticks=event.value_ticks,
            timestamp_ns=event.timestamp_ns,
            received_ns=event.received_ns,
            provider=event.provider,
            sequence=event.sequence,
        )

        if self._on_index is not None:
            await _maybe_await(self._on_index(event))

        return event

    def latest(self, symbol: str) -> Optional[VolatilityFactorState]:
        return self._states.get(symbol.strip().upper())

    def snapshot(self) -> dict[str, VolatilityFactorState]:
        return dict(self._states)


@dataclass(frozen=True, slots=True)
class StreamHealth:
    state: ConnectionState
    connected: bool
    authenticated: bool
    subscribed: bool

    connection_attempts: int
    reconnects: int

    quotes_received: int
    trades_received: int
    bars_received: int
    malformed_messages: int

    last_message_monotonic_ns: int
    last_event_timestamp_ns: int


@dataclass(frozen=True, slots=True)
class Subscription:
    symbols: tuple[str, ...]
    quotes: bool = True
    trades: bool = True
    bars: bool = True

    def validate(self) -> None:
        if not self.symbols:
            raise ValueError("at least one symbol is required")

        if not (self.quotes or self.trades or self.bars):
            raise ValueError(
                "at least one channel must be enabled"
            )

        for symbol in self.symbols:
            _validate_symbol(symbol)


QuoteHandler = Callable[[QuoteEvent], Optional[Awaitable[None]]]
TradeHandler = Callable[[TradeEvent], Optional[Awaitable[None]]]
BarHandler = Callable[[BarEvent], Optional[Awaitable[None]]]
StateHandler = Callable[[ConnectionState], Optional[Awaitable[None]]]


# =============================================================================
# ERRORS
# =============================================================================


class AlpacaStreamError(RuntimeError):
    pass


class AlpacaAuthenticationError(AlpacaStreamError):
    pass


class AlpacaSubscriptionError(AlpacaStreamError):
    pass


class AlpacaEntitlementError(AlpacaStreamError):
    pass


class AlpacaConnectionLimitError(AlpacaStreamError):
    pass


# =============================================================================
# HELPERS
# =============================================================================


def _validate_symbol(symbol: str) -> None:
    if not symbol:
        raise ValueError("symbol cannot be empty")

    if len(symbol) > 32:
        raise ValueError(f"symbol is too long: {symbol!r}")

    allowed = set(
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "0123456789.-_/"
    )

    if any(ch not in allowed for ch in symbol):
        raise ValueError(
            f"unsupported symbol characters: {symbol!r}"
        )


def _normalize_symbols(
    symbols: Iterable[str],
) -> tuple[str, ...]:
    normalized: list[str] = []
    seen: set[str] = set()

    for raw in symbols:
        symbol = raw.strip().upper()
        _validate_symbol(symbol)

        if symbol not in seen:
            normalized.append(symbol)
            seen.add(symbol)

    if not normalized:
        raise ValueError("no valid symbols supplied")

    return tuple(normalized)


def _price_to_ticks(
    value: object,
    scale: int,
) -> int:
    try:
        price = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(
            f"invalid price value: {value!r}"
        ) from exc

    if not price.is_finite() or price <= 0:
        raise ValueError(
            f"price must be finite and positive: {value!r}"
        )

    ticks = (
        price * Decimal(scale)
    ).quantize(
        Decimal("1"),
        rounding=ROUND_HALF_UP,
    )

    result = int(ticks)

    if result <= 0:
        raise ValueError(
            f"price converted to invalid tick value: {value!r}"
        )

    return result


def _optional_price_to_ticks(
    value: object,
    scale: int,
) -> Optional[int]:
    if value is None:
        return None

    return _price_to_ticks(value, scale)


def _rfc3339_ns(value: str) -> int:
    """
    Convert Alpaca RFC-3339 timestamp, including nanosecond precision,
    into Unix epoch nanoseconds without truncating the fractional field.
    """
    if not value:
        raise ValueError("timestamp is empty")

    if not value.endswith("Z"):
        raise ValueError(
            f"expected UTC RFC-3339 timestamp: {value!r}"
        )

    body = value[:-1]

    if "." in body:
        whole, fraction = body.split(".", 1)
    else:
        whole, fraction = body, ""

    dt = datetime.strptime(
        whole,
        "%Y-%m-%dT%H:%M:%S",
    ).replace(tzinfo=timezone.utc)

    seconds = int(dt.timestamp())

    fraction_digits = "".join(
        ch for ch in fraction if ch.isdigit()
    )

    nanoseconds = int(
        (fraction_digits[:9]).ljust(9, "0")
        or "0"
    )

    return seconds * 1_000_000_000 + nanoseconds


async def _maybe_await(result: object) -> None:
    if result is not None and hasattr(result, "__await__"):
        await result  # type: ignore[misc]


# =============================================================================
# ALPACA SIP STREAM
# =============================================================================


class AlpacaSIPStream:
    """
    Resilient single-connection Alpaca SIP market-data consumer.

    Environment variables:
        APCA_API_KEY_ID
        APCA_API_SECRET_KEY

    The stream authenticates using Alpaca's WebSocket auth message and then
    subscribes to requested quote/trade/bar channels.

    It deliberately does not expose broker/order methods.
    """

    def __init__(
        self,
        symbols: Sequence[str],
        *,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        quotes: bool = True,
        trades: bool = True,
        bars: bool = True,
        price_scale: int = DEFAULT_PRICE_SCALE,
        url: str = SIP_URL,
        auth_timeout: float = DEFAULT_AUTH_TIMEOUT,
        connect_timeout: float = DEFAULT_CONNECT_TIMEOUT,
        heartbeat_timeout: float = DEFAULT_HEARTBEAT_TIMEOUT,
        backoff_initial: float = DEFAULT_BACKOFF_INITIAL,
        backoff_max: float = DEFAULT_BACKOFF_MAX,
        on_quote: Optional[QuoteHandler] = None,
        on_trade: Optional[TradeHandler] = None,
        on_bar: Optional[BarHandler] = None,
        on_state: Optional[StateHandler] = None,
    ) -> None:
        normalized = _normalize_symbols(symbols)

        self._subscription = Subscription(
            symbols=normalized,
            quotes=quotes,
            trades=trades,
            bars=bars,
        )
        self._subscription.validate()

        self._api_key = (
            api_key
            if api_key is not None
            else os.environ.get("APCA_API_KEY_ID", "")
        )

        self._api_secret = (
            api_secret
            if api_secret is not None
            else os.environ.get("APCA_API_SECRET_KEY", "")
        )

        if not self._api_key or not self._api_secret:
            raise ValueError(
                "Alpaca credentials are required. Set "
                "APCA_API_KEY_ID and APCA_API_SECRET_KEY."
            )

        if price_scale <= 0:
            raise ValueError("price_scale must be positive")

        if auth_timeout <= 0:
            raise ValueError("auth_timeout must be positive")

        if connect_timeout <= 0:
            raise ValueError("connect_timeout must be positive")

        if heartbeat_timeout <= 0:
            raise ValueError(
                "heartbeat_timeout must be positive"
            )

        if backoff_initial <= 0:
            raise ValueError(
                "backoff_initial must be positive"
            )

        if backoff_max < backoff_initial:
            raise ValueError(
                "backoff_max must be >= backoff_initial"
            )

        self._url = url
        self._price_scale = price_scale

        self._auth_timeout = auth_timeout
        self._connect_timeout = connect_timeout
        self._heartbeat_timeout = heartbeat_timeout

        self._backoff_initial = backoff_initial
        self._backoff_max = backoff_max

        self._on_quote = on_quote
        self._on_trade = on_trade
        self._on_bar = on_bar
        self._on_state = on_state

        self._state = ConnectionState.DISCONNECTED

        self._stop_event = asyncio.Event()
        self._socket = None

        self._connection_attempts = 0
        self._reconnects = 0

        self._quotes_received = 0
        self._trades_received = 0
        self._bars_received = 0
        self._malformed_messages = 0
        self._ingestion_sequence = 0

        self._last_message_monotonic_ns = 0
        self._last_event_timestamp_ns = 0

    # -------------------------------------------------------------------------
    # PUBLIC STATE
    # -------------------------------------------------------------------------

    @property
    def symbols(self) -> tuple[str, ...]:
        return self._subscription.symbols

    @property
    def state(self) -> ConnectionState:
        return self._state

    @property
    def connected(self) -> bool:
        return self._state in {
            ConnectionState.CONNECTED,
            ConnectionState.AUTHENTICATED,
            ConnectionState.SUBSCRIBED,
        }

    @property
    def authenticated(self) -> bool:
        return self._state in {
            ConnectionState.AUTHENTICATED,
            ConnectionState.SUBSCRIBED,
        }

    @property
    def subscribed(self) -> bool:
        return self._state == ConnectionState.SUBSCRIBED

    def health(self) -> StreamHealth:
        return StreamHealth(
            state=self._state,
            connected=self.connected,
            authenticated=self.authenticated,
            subscribed=self.subscribed,
            connection_attempts=self._connection_attempts,
            reconnects=self._reconnects,
            quotes_received=self._quotes_received,
            trades_received=self._trades_received,
            bars_received=self._bars_received,
            malformed_messages=self._malformed_messages,
            last_message_monotonic_ns=(
                self._last_message_monotonic_ns
            ),
            last_event_timestamp_ns=(
                self._last_event_timestamp_ns
            ),
        )

    # -------------------------------------------------------------------------
    # LIFECYCLE
    # -------------------------------------------------------------------------

    async def run_forever(self) -> None:
        """
        Connect, authenticate, subscribe, consume, and reconnect on transient
        failures until stop() is requested.
        """
        self._stop_event.clear()

        backoff = self._backoff_initial
        first_connection = True

        while not self._stop_event.is_set():
            try:
                await self._connect_and_consume()

                if self._stop_event.is_set():
                    break

                raise AlpacaStreamError(
                    "market-data stream ended unexpectedly"
                )

            except asyncio.CancelledError:
                raise

            except (
                AlpacaAuthenticationError,
                AlpacaEntitlementError,
                AlpacaConnectionLimitError,
            ):
                # Credential, entitlement, and connection-limit failures are
                # not transient network failures. Reconnecting in a tight loop
                # would only make the situation worse.
                await self._set_state(
                    ConnectionState.DISCONNECTED
                )
                raise

            except Exception as exc:
                if self._stop_event.is_set():
                    break

                if not first_connection:
                    self._reconnects += 1

                LOGGER.warning(
                    "Alpaca SIP stream disconnected: %s; "
                    "retrying in %.2fs",
                    type(exc).__name__,
                    backoff,
                )

                await self._set_state(
                    ConnectionState.DISCONNECTED
                )

                jitter = random.uniform(
                    0.0,
                    min(0.25, backoff * 0.10),
                )

                try:
                    await asyncio.wait_for(
                        self._stop_event.wait(),
                        timeout=backoff + jitter,
                    )
                except asyncio.TimeoutError:
                    pass

                backoff = min(
                    self._backoff_max,
                    backoff * 2.0,
                )

            else:
                backoff = self._backoff_initial

            first_connection = False

        await self._set_state(
            ConnectionState.DISCONNECTED
        )

    async def stop(self) -> None:
        await self._set_state(
            ConnectionState.STOPPING
        )

        self._stop_event.set()

        socket = self._socket

        if socket is not None:
            try:
                await socket.close(
                    code=1000,
                    reason="client shutdown",
                )
            except Exception:
                LOGGER.debug(
                    "error while closing Alpaca socket",
                    exc_info=True,
                )

    # -------------------------------------------------------------------------
    # CONNECTION
    # -------------------------------------------------------------------------

    async def _connect_and_consume(self) -> None:
        self._connection_attempts += 1

        await self._set_state(
            ConnectionState.CONNECTING
        )

        ssl_context = ssl.create_default_context()

        async with websockets.connect(
            self._url,
            ssl=ssl_context,
            open_timeout=self._connect_timeout,
            close_timeout=5.0,
            ping_interval=20.0,
            ping_timeout=20.0,
            max_size=8 * 1024 * 1024,
            max_queue=4096,
            compression=None,
        ) as socket:
            self._socket = socket

            await self._expect_connected(socket)

            await self._set_state(
                ConnectionState.CONNECTED
            )

            await self._authenticate(socket)

            await self._set_state(
                ConnectionState.AUTHENTICATED
            )

            await self._subscribe(socket)

            await self._set_state(
                ConnectionState.SUBSCRIBED
            )

            await self._consume(socket)

        self._socket = None

    async def _expect_connected(self, socket) -> None:
        raw = await asyncio.wait_for(
            socket.recv(),
            timeout=self._connect_timeout,
        )

        messages = self._decode_frame(raw)

        if not any(
            msg.get("T") == "success" and
            msg.get("msg") == "connected"
            for msg in messages
        ):
            raise AlpacaStreamError(
                f"unexpected Alpaca welcome message: "
                f"{self._redacted(messages)!r}"
            )

    # -------------------------------------------------------------------------
    # AUTHENTICATION
    # -------------------------------------------------------------------------

    async def _authenticate(self, socket) -> None:
        await socket.send(
            json.dumps(
                {
                    "action": "auth",
                    "key": self._api_key,
                    "secret": self._api_secret,
                },
                separators=(",", ":"),
            )
        )

        raw = await asyncio.wait_for(
            socket.recv(),
            timeout=self._auth_timeout,
        )

        messages = self._decode_frame(raw)

        for message in messages:
            if (
                message.get("T") == "success" and
                message.get("msg") == "authenticated"
            ):
                return

            if message.get("T") == "error":
                self._raise_server_error(message)

        raise AlpacaAuthenticationError(
            "Alpaca did not confirm authentication"
        )

    # -------------------------------------------------------------------------
    # SUBSCRIPTION
    # -------------------------------------------------------------------------

    async def _subscribe(self, socket) -> None:
        request: dict[str, object] = {
            "action": "subscribe"
        }

        symbols = list(self._subscription.symbols)

        if self._subscription.trades:
            request["trades"] = symbols

        if self._subscription.quotes:
            request["quotes"] = symbols

        if self._subscription.bars:
            request["bars"] = symbols

        await socket.send(
            json.dumps(
                request,
                separators=(",", ":"),
            )
        )

        while True:
            raw = await asyncio.wait_for(
                socket.recv(),
                timeout=self._connect_timeout,
            )

            messages = self._decode_frame(raw)

            for message in messages:
                msg_type = message.get("T")

                if msg_type == "error":
                    self._raise_server_error(message)

                if msg_type == "subscription":
                    self._verify_subscription(message)
                    return

                # Alpaca can theoretically deliver data adjacent to the
                # subscription acknowledgement. Do not discard it.
                await self._dispatch_message(message)

    def _verify_subscription(
        self,
        message: dict[str, object],
    ) -> None:
        expected = set(self._subscription.symbols)

        checks: list[tuple[str, bool]] = [
            ("quotes", self._subscription.quotes),
            ("trades", self._subscription.trades),
            ("bars", self._subscription.bars),
        ]

        for channel, enabled in checks:
            if not enabled:
                continue

            received = set(
                str(symbol)
                for symbol in message.get(channel, [])
            )

            missing = expected - received

            if missing:
                raise AlpacaSubscriptionError(
                    f"Alpaca did not subscribe {channel} "
                    f"for: {sorted(missing)!r}"
                )

    # -------------------------------------------------------------------------
    # RECEIVE LOOP
    # -------------------------------------------------------------------------

    async def _consume(self, socket) -> None:
        while not self._stop_event.is_set():
            try:
                raw = await asyncio.wait_for(
                    socket.recv(),
                    timeout=self._heartbeat_timeout,
                )
            except asyncio.TimeoutError as exc:
                raise AlpacaStreamError(
                    "market-data receive timeout"
                ) from exc
            except ConnectionClosed:
                raise

            self._last_message_monotonic_ns = (
                time.monotonic_ns()
            )

            messages = self._decode_frame(raw)

            for message in messages:
                await self._dispatch_message(message)

    async def _dispatch_message(
        self,
        message: dict[str, object],
    ) -> None:
        msg_type = message.get("T")

        try:
            if msg_type == "q":
                event = self._parse_quote(message)
                self._quotes_received += 1

                self._last_event_timestamp_ns = max(
                    self._last_event_timestamp_ns,
                    event.timestamp_ns,
                )

                if self._on_quote is not None:
                    await _maybe_await(
                        self._on_quote(event)
                    )

                return

            if msg_type == "t":
                event = self._parse_trade(message)
                self._trades_received += 1

                self._last_event_timestamp_ns = max(
                    self._last_event_timestamp_ns,
                    event.timestamp_ns,
                )

                if self._on_trade is not None:
                    await _maybe_await(
                        self._on_trade(event)
                    )

                return

            if msg_type == "b":
                event = self._parse_bar(message)
                self._bars_received += 1

                self._last_event_timestamp_ns = max(
                    self._last_event_timestamp_ns,
                    event.timestamp_ns,
                )

                if self._on_bar is not None:
                    await _maybe_await(
                        self._on_bar(event)
                    )

                return

            if msg_type == "error":
                self._raise_server_error(message)

            # subscription, success, status, correction, LULD, and other
            # message types can be added without contaminating downstream
            # market-data structures.

        except (
            KeyError,
            TypeError,
            ValueError,
            InvalidOperation,
        ):
            self._malformed_messages += 1

            LOGGER.warning(
                "discarding malformed Alpaca message type=%r",
                msg_type,
                exc_info=True,
            )

    def _next_ingestion_sequence(self) -> int:
        self._ingestion_sequence += 1
        return self._ingestion_sequence

    # -------------------------------------------------------------------------
    # NORMALIZATION
    # -------------------------------------------------------------------------

    def _parse_quote(
        self,
        message: dict[str, object],
    ) -> QuoteEvent:
        symbol = str(message["S"]).upper()
        _validate_symbol(symbol)

        timestamp_ns = _rfc3339_ns(
            str(message["t"])
        )

        return QuoteEvent(
            symbol=symbol,
            timestamp_ns=timestamp_ns,
            bid_price_ticks=_price_to_ticks(
                message["bp"],
                self._price_scale,
            ),
            bid_size=int(message["bs"]),
            bid_exchange=str(
                message.get("bx", "")
            ),
            ask_price_ticks=_price_to_ticks(
                message["ap"],
                self._price_scale,
            ),
            ask_size=int(message["as"]),
            ask_exchange=str(
                message.get("ax", "")
            ),
            conditions=tuple(
                str(value)
                for value in message.get("c", [])
            ),
            tape=str(message.get("z", "")),
            sequence=self._next_ingestion_sequence(),
            received_ns=time.time_ns(),
        )

    def _parse_trade(
        self,
        message: dict[str, object],
    ) -> TradeEvent:
        symbol = str(message["S"]).upper()
        _validate_symbol(symbol)

        timestamp_ns = _rfc3339_ns(
            str(message["t"])
        )

        size = int(message["s"])

        if size <= 0:
            raise ValueError(
                "trade size must be positive"
            )

        return TradeEvent(
            symbol=symbol,
            timestamp_ns=timestamp_ns,
            trade_id=int(message["i"]),
            exchange=str(message.get("x", "")),
            price_ticks=_price_to_ticks(
                message["p"],
                self._price_scale,
            ),
            size=size,
            conditions=tuple(
                str(value)
                for value in message.get("c", [])
            ),
            tape=str(message.get("z", "")),
            sequence=self._next_ingestion_sequence(),
            received_ns=time.time_ns(),
        )

    def _parse_bar(
        self,
        message: dict[str, object],
    ) -> BarEvent:
        symbol = str(message["S"]).upper()
        _validate_symbol(symbol)

        timestamp_ns = _rfc3339_ns(
            str(message["t"])
        )

        return BarEvent(
            symbol=symbol,
            timestamp_ns=timestamp_ns,
            open_ticks=_price_to_ticks(
                message["o"],
                self._price_scale,
            ),
            high_ticks=_price_to_ticks(
                message["h"],
                self._price_scale,
            ),
            low_ticks=_price_to_ticks(
                message["l"],
                self._price_scale,
            ),
            close_ticks=_price_to_ticks(
                message["c"],
                self._price_scale,
            ),
            volume=int(message["v"]),
            trade_count=int(message.get("n", 0)),
            vwap_ticks=_optional_price_to_ticks(
                message.get("vw"),
                self._price_scale,
            ),
            sequence=self._next_ingestion_sequence(),
            received_ns=time.time_ns(),
        )

    # -------------------------------------------------------------------------
    # SERVER ERRORS
    # -------------------------------------------------------------------------

    @staticmethod
    def _raise_server_error(
        message: dict[str, object],
    ) -> None:
        code = int(message.get("code", 0))
        text = str(
            message.get("msg", "unknown Alpaca error")
        )

        if code == 406:
            raise AlpacaConnectionLimitError(
                f"Alpaca connection limit exceeded: {text}"
            )

        if code in {401, 402, 403}:
            lowered = text.lower()

            if (
                "subscription" in lowered or
                "feed" in lowered or
                "permit" in lowered
            ):
                raise AlpacaEntitlementError(
                    f"Alpaca SIP entitlement rejected: {text}"
                )

            raise AlpacaAuthenticationError(
                f"Alpaca authentication rejected: {text}"
            )

        raise AlpacaStreamError(
            f"Alpaca stream error {code}: {text}"
        )

    # -------------------------------------------------------------------------
    # FRAME DECODING / REDACTION
    # -------------------------------------------------------------------------

    @staticmethod
    def _decode_frame(
        raw: object,
    ) -> list[dict[str, object]]:
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")

        if not isinstance(raw, str):
            raise ValueError(
                "unexpected WebSocket frame type"
            )

        decoded = json.loads(raw)

        if isinstance(decoded, dict):
            decoded = [decoded]

        if not isinstance(decoded, list):
            raise ValueError(
                "Alpaca frame must decode to object array"
            )

        output: list[dict[str, object]] = []

        for item in decoded:
            if not isinstance(item, dict):
                raise ValueError(
                    "Alpaca message must be an object"
                )

            output.append(item)

        return output

    @staticmethod
    def _redacted(
        value: object,
    ) -> object:
        # Frames received from Alpaca should not contain our credentials, but
        # keep this helper defensive for diagnostics.
        if isinstance(value, dict):
            return {
                key: (
                    "<redacted>"
                    if key.lower() in {
                        "key",
                        "secret",
                        "token",
                        "authorization",
                    }
                    else AlpacaSIPStream._redacted(item)
                )
                for key, item in value.items()
            }

        if isinstance(value, list):
            return [
                AlpacaSIPStream._redacted(item)
                for item in value
            ]

        return value

    # -------------------------------------------------------------------------
    # STATE CALLBACK
    # -------------------------------------------------------------------------

    async def _set_state(
        self,
        state: ConnectionState,
    ) -> None:
        if state == self._state:
            return

        self._state = state

        if self._on_state is not None:
            await _maybe_await(
                self._on_state(state)
            )


# =============================================================================
# QUEUE BRIDGE
# =============================================================================


class MarketDataQueue:
    """
    Bounded async bridge suitable for Python orchestration.

    A future pybind11/C++ bridge can consume the same normalized event types.
    The queue deliberately drops the oldest item when saturated so network
    ingestion does not block indefinitely behind slow analytics. The caller
    should monitor dropped_events and treat sustained drops as a health fault.
    """

    def __init__(
        self,
        maxsize: int = 100_000,
    ) -> None:
        if maxsize <= 0:
            raise ValueError("maxsize must be positive")

        self._queue: asyncio.Queue[
            QuoteEvent | TradeEvent | BarEvent | IndexEvent
        ] = asyncio.Queue(maxsize=maxsize)

        self.dropped_events = 0

    def put_nowait(
        self,
        event: QuoteEvent | TradeEvent | BarEvent | IndexEvent,
    ) -> None:
        if self._queue.full():
            try:
                self._queue.get_nowait()
                self._queue.task_done()
                self.dropped_events += 1
            except asyncio.QueueEmpty:
                pass

        self._queue.put_nowait(event)

    async def get(
        self,
    ) -> QuoteEvent | TradeEvent | BarEvent | IndexEvent:
        return await self._queue.get()

    def task_done(self) -> None:
        self._queue.task_done()

    @property
    def size(self) -> int:
        return self._queue.qsize()


# =============================================================================
# LIVE + ARCHIVE FAN-OUT
# =============================================================================


MarketEvent = QuoteEvent | TradeEvent | BarEvent | IndexEvent


class MarketDataFanout:
    """Send every normalized event to live analytics and archival storage.

    The live queue retains its latency-oriented bounded/drop-oldest behavior.
    HDF5 persistence uses its own queue inside DailyHDF5Writer; if that queue
    fills, submit() raises instead of silently losing archival observations.
    """

    def __init__(
        self,
        *,
        live_queue: MarketDataQueue,
        archive: DailyHDF5Writer,
    ) -> None:
        self._live_queue = live_queue
        self._archive = archive

    def publish(self, event: MarketEvent) -> None:
        # Archive first. If archival integrity is lost, surface the fault
        # immediately rather than pretending the historical file is complete.
        self._archive.submit(event)
        self._live_queue.put_nowait(event)


# =============================================================================
# EXAMPLE ENTRY POINT
# =============================================================================


PRIMARY_TARGET = "QQQ"
INVERSE_EQUITY = "SQQQ"

# Factor inputs; do not pass these to AlpacaSIPStream's stock subscription.
VOLATILITY_FACTORS = ("VIX", "VXN")

DEFAULT_UNIVERSE = (
    "QQQ",
    "SQQQ",
    "SPY",
    "IWM",
    "TLT",
    "IEF",
    "GLD",
    "DBC",
    "EFA",
    "EEM",
    "VNQ",
)


async def _example() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s %(levelname)s "
            "%(name)s %(message)s"
        ),
    )

    queue = MarketDataQueue()

    archive = DailyHDF5Writer(
        root="market_data",
        price_scale=DEFAULT_PRICE_SCALE,
        provider="alpaca_sip_equities+licensed_external_index_factors",
    )
    archive.start()

    fanout = MarketDataFanout(
        live_queue=queue,
        archive=archive,
    )

    # A licensed external index source can publish VIX/VXN here.
    volatility = IndexFactorBridge(
        on_index=lambda event: fanout.publish(event)
    )

    def enqueue(
        event: QuoteEvent | TradeEvent | BarEvent | IndexEvent,
    ) -> None:
        fanout.publish(event)

    async def state_changed(
        state: ConnectionState,
    ) -> None:
        LOGGER.info(
            "Alpaca SIP state=%s",
            state.value,
        )

    stream = AlpacaSIPStream(
        DEFAULT_UNIVERSE,
        quotes=True,
        trades=True,
        bars=True,
        on_quote=enqueue,
        on_trade=enqueue,
        on_bar=enqueue,
        on_state=state_changed,
    )

    loop = asyncio.get_running_loop()

    stop_requested = asyncio.Event()

    def request_stop() -> None:
        stop_requested.set()

    for sig in (
        signal.SIGINT,
        signal.SIGTERM,
    ):
        try:
            loop.add_signal_handler(
                sig,
                request_stop,
            )
        except NotImplementedError:
            pass

    stream_task = asyncio.create_task(
        stream.run_forever(),
        name="alpaca-sip-stream",
    )

    async def consumer() -> None:
        while not stop_requested.is_set():
            event = await queue.get()

            try:
                # Replace this with the bridge into the C++/Metal pipeline.
                if isinstance(event, QuoteEvent):
                    LOGGER.info(
                        "QUOTE %s bid=%d@%d ask=%d@%d latency_us=%.1f",
                        event.symbol,
                        event.bid_size,
                        event.bid_price_ticks,
                        event.ask_size,
                        event.ask_price_ticks,
                        event.latency_ns / 1_000.0,
                    )

                elif isinstance(event, TradeEvent):
                    LOGGER.info(
                        "TRADE %s %d@%d latency_us=%.1f",
                        event.symbol,
                        event.size,
                        event.price_ticks,
                        event.latency_ns / 1_000.0,
                    )

                elif isinstance(event, BarEvent):
                    LOGGER.info(
                        "BAR %s close=%d volume=%d",
                        event.symbol,
                        event.close_ticks,
                        event.volume,
                    )

                elif isinstance(event, IndexEvent):
                    LOGGER.info(
                        "INDEX %s value=%d provider=%s latency_us=%.1f",
                        event.symbol,
                        event.value_ticks,
                        event.provider,
                        event.latency_ns / 1_000.0,
                    )

            finally:
                queue.task_done()

    consumer_task = asyncio.create_task(
        consumer(),
        name="market-data-consumer",
    )

    await stop_requested.wait()

    await stream.stop()

    consumer_task.cancel()

    try:
        await consumer_task
    except asyncio.CancelledError:
        pass

    await stream_task

    try:
        archive.close(drain=True)
    except MarketDataStoreError:
        LOGGER.exception("market-data archive shutdown failed")
        raise


if __name__ == "__main__":
    asyncio.run(_example())
