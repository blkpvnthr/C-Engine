"""
liquidity.py

Python liquidity/orchestration layer for the hybrid trading engine.

Architecture
------------
AlpacaSIPStream
    |
    +-- QuoteEvent ------+
    +-- TradeEvent ------+----> NativeMarketBridge
                                  |
                                  v
                           C++ OrderBookEngine
                           +-------------------+
                           | ObservedMarketBook|
                           | LimitOrderBook    |
                           +-------------------+
                                  |
                                  v
                           Feature/Risk layers

This module deliberately does NOT implement exchange matching in Python.
The C++ LimitOrderBook is the simulated/internal matching authority.

Responsibilities
----------------
* Normalize Alpaca SIP quote/trade events into the C++ binding types.
* Route live market data into OrderBookRegistry/OrderBookEngine.
* Generate optional synthetic resting liquidity for simulations.
* Submit/amend/cancel synthetic liquidity through the native order book.
* Maintain Python-side metadata for synthetic orders.
* Expose a clean interface that can later sit behind RiskEngine bindings.
* Maintain read-only VIX/VXN factor state for QQQ-focused statistical arbitrage.
* Keep VIX/VXN outside OrderBookRegistry and synthetic-liquidity generation.

Non-responsibilities
--------------------
* No WebSocket authentication.
* No .env parsing.
* No Alpaca credentials.
* No live broker order routing.
* No Python-side fills.
* No fabricated Level-II depth from SIP NBBO quotes.

Expected native module
----------------------
The pybind11 module should expose the following C++ types from orderbook.hpp:

    Side
    MarketQuoteUpdate
    MarketTradeUpdate
    Execution
    Order
    ObservedMarketSnapshot
    BookSnapshot
    OrderBookEngine
    OrderBookRegistry

By default this file imports ``trading_native``. Override with
TRADING_NATIVE_MODULE if a different extension-module name is used.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from importlib import import_module
from itertools import count
import os
from random import Random
from threading import RLock
from time import time_ns
from types import ModuleType
from typing import Any, Iterable, Mapping, Protocol, Sequence


# ============================================================================
# CONSTANTS
# ============================================================================

DEFAULT_PRICE_SCALE = 10_000
DEFAULT_TICK_SIZE = Decimal("0.01")
DEFAULT_NATIVE_MODULE = "trading_native"

PRIMARY_TARGET = "QQQ"
INVERSE_EQUITY = "SQQQ"
VOLATILITY_FACTORS = frozenset({"VIX", "VXN"})
DEFAULT_EXECUTABLE_UNIVERSE = (
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


# ============================================================================
# ERRORS
# ============================================================================


class LiquidityError(RuntimeError):
    """Base class for liquidity-layer errors."""


class NativeBindingError(LiquidityError):
    """Native C++/pybind11 bridge is unavailable or incompatible."""


class InvalidLiquidityOrder(LiquidityError):
    """Synthetic-liquidity request is invalid."""


class LiquidityOrderNotFound(LiquidityError):
    """Requested synthetic order is unknown or no longer active."""


class MarketDataRejected(LiquidityError):
    """Native observed-market book rejected a market-data update."""


class NonExecutableFactorError(LiquidityError):
    """Attempted to treat a read-only statistical factor as an executable symbol."""


# ============================================================================
# ENUMS / PYTHON METADATA
# ============================================================================


class Side(str, Enum):
    BUY = "buy"
    SELL = "sell"


class LiquidityOrderStatus(str, Enum):
    NEW = "new"
    ACTIVE = "active"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    UNKNOWN = "unknown"


TERMINAL_LIQUIDITY_STATUSES = {
    LiquidityOrderStatus.FILLED,
    LiquidityOrderStatus.CANCELLED,
    LiquidityOrderStatus.REJECTED,
}


@dataclass(frozen=True, slots=True)
class LiveQuote:
    """
    Provider-neutral top-of-book quote.

    Prices are integer ticks because that is the execution-boundary format used
    by the C++ engine. SIP supplies consolidated best bid/ask; this is not L2.
    """

    symbol: str
    timestamp_ns: int
    received_ns: int

    bid_price_ticks: int
    bid_size: int
    bid_exchange: str

    ask_price_ticks: int
    ask_size: int
    ask_exchange: str

    sequence: int = 0

    @property
    def spread_ticks(self) -> int:
        return self.ask_price_ticks - self.bid_price_ticks

    @property
    def midpoint_ticks(self) -> int:
        return self.bid_price_ticks + self.spread_ticks // 2

    def validate(self) -> None:
        if not self.symbol:
            raise ValueError("quote symbol cannot be empty")
        if self.timestamp_ns <= 0:
            raise ValueError("quote timestamp_ns must be positive")
        if self.bid_price_ticks <= 0:
            raise ValueError("bid price must be positive")
        if self.ask_price_ticks <= 0:
            raise ValueError("ask price must be positive")
        if self.bid_price_ticks >= self.ask_price_ticks:
            raise ValueError("bid must be below ask")
        if self.bid_size < 0 or self.ask_size < 0:
            raise ValueError("quote sizes cannot be negative")


@dataclass(frozen=True, slots=True)
class LiveTrade:
    symbol: str
    timestamp_ns: int
    received_ns: int

    trade_id: int
    exchange: str
    price_ticks: int
    size: int

    sequence: int = 0

    def validate(self) -> None:
        if not self.symbol:
            raise ValueError("trade symbol cannot be empty")
        if self.timestamp_ns <= 0:
            raise ValueError("trade timestamp_ns must be positive")
        if self.price_ticks <= 0:
            raise ValueError("trade price must be positive")
        if self.size <= 0:
            raise ValueError("trade size must be positive")


@dataclass(frozen=True, slots=True)
class LiveIndex:
    """Provider-neutral VIX/VXN observation."""

    symbol: str
    timestamp_ns: int
    received_ns: int
    value_ticks: int
    provider: str = ""
    sequence: int = 0

    def validate(self) -> None:
        if self.symbol not in VOLATILITY_FACTORS:
            raise ValueError(
                f"unsupported volatility factor: {self.symbol!r}"
            )
        if self.timestamp_ns <= 0:
            raise ValueError("index timestamp_ns must be positive")
        if self.received_ns <= 0:
            raise ValueError("index received_ns must be positive")
        if self.value_ticks <= 0:
            raise ValueError("index value_ticks must be positive")

    @property
    def latency_ns(self) -> int:
        return max(0, self.received_ns - self.timestamp_ns)


@dataclass(frozen=True, slots=True)
class VolatilityFactorSnapshot:
    vix: LiveIndex | None
    vxn: LiveIndex | None
    updated_ns: int

    def get(self, symbol: str) -> LiveIndex | None:
        normalized = _normalize_symbol(symbol)
        if normalized == "VIX":
            return self.vix
        if normalized == "VXN":
            return self.vxn
        raise KeyError(f"not a volatility factor: {normalized}")


@dataclass(frozen=True, slots=True)
class LiquidityOrder:
    """
    Python metadata for an order whose execution state is owned by C++.

    Do not treat this object as authoritative for fill state. Call
    LiquidityProvider.refresh_order() or query the native book.
    """

    id: int
    symbol: str
    side: Side
    quantity: int
    price_ticks: int

    status: LiquidityOrderStatus = LiquidityOrderStatus.NEW
    filled_quantity: int = 0
    remaining_quantity: int = 0

    created_ns: int = 0
    updated_ns: int = 0

    @property
    def price(self) -> Decimal:
        return ticks_to_price(self.price_ticks)

    @property
    def is_terminal(self) -> bool:
        return self.status in TERMINAL_LIQUIDITY_STATUSES


@dataclass(frozen=True, slots=True)
class LiquidityConfig:
    """
    Synthetic-liquidity policy.

    ``levels`` is simulated local depth only. It is NOT derived from SIP depth.
    """

    tick_size: Decimal = DEFAULT_TICK_SIZE

    min_quantity: int = 1
    max_quantity: int = 100

    levels: int = 1

    # Distance from observed NBBO for our first synthetic level.
    inside_offset_ticks: int = 0

    # Additional spacing between synthetic levels.
    level_spacing_ticks: int = 1

    # Keep our synthetic quotes from crossing observed NBBO.
    prevent_crossing_observed_market: bool = True

    # Replace current synthetic orders whenever a new quote is processed.
    replace_on_quote: bool = True

    random_seed: int = 0

    def validate(self) -> None:
        if self.tick_size <= 0:
            raise ValueError("tick_size must be positive")
        if self.min_quantity <= 0:
            raise ValueError("min_quantity must be positive")
        if self.max_quantity < self.min_quantity:
            raise ValueError(
                "max_quantity must be >= min_quantity"
            )
        if self.levels <= 0:
            raise ValueError("levels must be positive")
        if self.inside_offset_ticks < 0:
            raise ValueError(
                "inside_offset_ticks cannot be negative"
            )
        if self.level_spacing_ticks <= 0:
            raise ValueError(
                "level_spacing_ticks must be positive"
            )


# ============================================================================
# STRUCTURAL PROTOCOLS
# ============================================================================


class AlpacaQuoteLike(Protocol):
    symbol: str
    timestamp_ns: int
    received_ns: int
    bid_price_ticks: int
    bid_size: int
    bid_exchange: str
    ask_price_ticks: int
    ask_size: int
    ask_exchange: str


class AlpacaTradeLike(Protocol):
    symbol: str
    timestamp_ns: int
    received_ns: int
    trade_id: int
    exchange: str
    price_ticks: int
    size: int



class IndexEventLike(Protocol):
    symbol: str
    timestamp_ns: int
    received_ns: int
    value_ticks: int
    provider: str
    sequence: int


# ============================================================================
# PRICE HELPERS
# ============================================================================


def price_to_ticks(
    price: Decimal | str | float | int,
    *,
    price_scale: int = DEFAULT_PRICE_SCALE,
) -> int:
    if price_scale <= 0:
        raise ValueError("price_scale must be positive")

    value = Decimal(str(price))

    if not value.is_finite() or value <= 0:
        raise ValueError("price must be finite and positive")

    result = int(
        (value * Decimal(price_scale)).quantize(
            Decimal("1"),
            rounding=ROUND_HALF_UP,
        )
    )

    if result <= 0:
        raise ValueError("price converted to invalid ticks")

    return result


def ticks_to_price(
    ticks: int,
    *,
    price_scale: int = DEFAULT_PRICE_SCALE,
) -> Decimal:
    if price_scale <= 0:
        raise ValueError("price_scale must be positive")
    if ticks <= 0:
        raise ValueError("ticks must be positive")

    return Decimal(ticks) / Decimal(price_scale)


def tick_size_to_native_ticks(
    tick_size: Decimal | str | float | int,
    *,
    price_scale: int = DEFAULT_PRICE_SCALE,
) -> int:
    return price_to_ticks(
        Decimal(str(tick_size)),
        price_scale=price_scale,
    )


# ============================================================================
# NATIVE MODULE LOADING
# ============================================================================


_REQUIRED_NATIVE_ATTRIBUTES = (
    "Side",
    "MarketQuoteUpdate",
    "MarketTradeUpdate",
    "OrderBookEngine",
    "OrderBookRegistry",
)


def load_native_module(
    module_name: str | None = None,
) -> ModuleType:
    """
    Load the pybind11 extension without hardcoding a single module name.

    Credentials are intentionally irrelevant here. They remain confined to
    alpaca_sip_stream.py / process environment.
    """

    name = (
        module_name
        or os.environ.get(
            "TRADING_NATIVE_MODULE",
            DEFAULT_NATIVE_MODULE,
        )
    )

    try:
        module = import_module(name)
    except Exception as exc:
        raise NativeBindingError(
            f"could not import native trading module {name!r}"
        ) from exc

    missing = [
        attr
        for attr in _REQUIRED_NATIVE_ATTRIBUTES
        if not hasattr(module, attr)
    ]

    if missing:
        raise NativeBindingError(
            "native module is missing required bindings: "
            + ", ".join(missing)
        )

    return module


# ============================================================================
# MARKET-DATA BRIDGE
# ============================================================================


class NativeMarketBridge:
    """
    Converts normalized Alpaca SIP events into C++ market-data update structs.

    The bridge never inserts SIP quotes into LimitOrderBook. The native
    OrderBookRegistry routes these updates only to ObservedMarketBook.
    """

    def __init__(
        self,
        registry: Any,
        *,
        native_module: ModuleType | Any | None = None,
    ) -> None:
        self.native = (
            native_module
            if native_module is not None
            else load_native_module()
        )
        self.registry = registry
        self._sequence = count(1)
        self._lock = RLock()

    @classmethod
    def create(
        cls,
        symbols: Iterable[str],
        *,
        native_module: ModuleType | Any | None = None,
    ) -> "NativeMarketBridge":
        native = (
            native_module
            if native_module is not None
            else load_native_module()
        )

        registry = native.OrderBookRegistry()

        normalized_symbols = _normalize_symbols(symbols)
        forbidden = sorted(
            set(normalized_symbols).intersection(VOLATILITY_FACTORS)
        )
        if forbidden:
            raise NonExecutableFactorError(
                "VIX/VXN are read-only statistical factors and cannot "
                f"be registered as executable order books: {forbidden!r}"
            )

        for symbol in normalized_symbols:
            registry.add_symbol(symbol)

        return cls(
            registry,
            native_module=native,
        )

    def _next_sequence(self) -> int:
        with self._lock:
            return next(self._sequence)

    def on_quote(
        self,
        event: AlpacaQuoteLike | LiveQuote,
    ) -> bool:
        quote = coerce_quote(
            event,
            sequence=self._next_sequence(),
        )

        update = self.native.MarketQuoteUpdate()

        update.symbol = quote.symbol
        update.timestamp_ns = quote.timestamp_ns
        update.received_ns = quote.received_ns

        update.bid_price_ticks = quote.bid_price_ticks
        update.bid_size = quote.bid_size
        update.bid_exchange = quote.bid_exchange

        update.ask_price_ticks = quote.ask_price_ticks
        update.ask_size = quote.ask_size
        update.ask_exchange = quote.ask_exchange

        update.sequence = quote.sequence

        accepted = bool(
            self.registry.on_market_quote(update)
        )

        if not accepted:
            raise MarketDataRejected(
                f"native market book rejected quote "
                f"{quote.symbol} sequence={quote.sequence}"
            )

        return True

    def on_trade(
        self,
        event: AlpacaTradeLike | LiveTrade,
    ) -> bool:
        trade = coerce_trade(
            event,
            sequence=self._next_sequence(),
        )

        update = self.native.MarketTradeUpdate()

        update.symbol = trade.symbol
        update.timestamp_ns = trade.timestamp_ns
        update.received_ns = trade.received_ns

        update.trade_id = trade.trade_id
        update.exchange = trade.exchange
        update.price_ticks = trade.price_ticks
        update.size = trade.size
        update.sequence = trade.sequence

        accepted = bool(
            self.registry.on_market_trade(update)
        )

        if not accepted:
            raise MarketDataRejected(
                f"native market book rejected trade "
                f"{trade.symbol} sequence={trade.sequence}"
            )

        return True

    def engine(self, symbol: str) -> Any:
        normalized = _normalize_symbol(symbol)

        engine = self.registry.get(normalized)

        if engine is None:
            raise KeyError(
                f"symbol is not registered: {normalized}"
            )

        return engine

    def market_snapshot(self, symbol: str) -> Any:
        return self.engine(symbol).market_snapshot()


# ============================================================================
# EVENT NORMALIZATION
# ============================================================================


def _normalize_symbol(symbol: str) -> str:
    normalized = symbol.strip().upper()

    if not normalized:
        raise ValueError("symbol cannot be empty")

    allowed = set(
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "0123456789.-_/"
    )

    if len(normalized) > 32:
        raise ValueError("symbol is too long")

    if any(ch not in allowed for ch in normalized):
        raise ValueError(
            f"unsupported symbol characters: {normalized!r}"
        )

    return normalized


def _normalize_symbols(
    symbols: Iterable[str],
) -> tuple[str, ...]:
    output: list[str] = []
    seen: set[str] = set()

    for raw in symbols:
        symbol = _normalize_symbol(raw)

        if symbol not in seen:
            output.append(symbol)
            seen.add(symbol)

    if not output:
        raise ValueError("at least one symbol is required")

    return tuple(output)


def coerce_quote(
    event: AlpacaQuoteLike | LiveQuote,
    *,
    sequence: int = 0,
) -> LiveQuote:
    result = LiveQuote(
        symbol=_normalize_symbol(event.symbol),
        timestamp_ns=int(event.timestamp_ns),
        received_ns=int(event.received_ns),
        bid_price_ticks=int(event.bid_price_ticks),
        bid_size=int(event.bid_size),
        bid_exchange=str(event.bid_exchange),
        ask_price_ticks=int(event.ask_price_ticks),
        ask_size=int(event.ask_size),
        ask_exchange=str(event.ask_exchange),
        sequence=(
            int(getattr(event, "sequence", 0))
            or sequence
        ),
    )

    result.validate()
    return result


def coerce_trade(
    event: AlpacaTradeLike | LiveTrade,
    *,
    sequence: int = 0,
) -> LiveTrade:
    result = LiveTrade(
        symbol=_normalize_symbol(event.symbol),
        timestamp_ns=int(event.timestamp_ns),
        received_ns=int(event.received_ns),
        trade_id=int(event.trade_id),
        exchange=str(event.exchange),
        price_ticks=int(event.price_ticks),
        size=int(event.size),
        sequence=(
            int(getattr(event, "sequence", 0))
            or sequence
        ),
    )

    result.validate()
    return result


def coerce_index(
    event: IndexEventLike | LiveIndex,
) -> LiveIndex:
    result = LiveIndex(
        symbol=_normalize_symbol(event.symbol),
        timestamp_ns=int(event.timestamp_ns),
        received_ns=int(event.received_ns),
        value_ticks=int(event.value_ticks),
        provider=str(getattr(event, "provider", "")),
        sequence=int(getattr(event, "sequence", 0)),
    )
    result.validate()
    return result


class VolatilityFactorBook:
    """
    Read-only latest-value store for VIX/VXN.

    This is deliberately separate from NativeMarketBridge and OrderBookRegistry:
    factors may influence features/signals/risk, but cannot be submitted,
    amended, cancelled, or matched as orders.
    """

    def __init__(self) -> None:
        self._states: dict[str, LiveIndex] = {}
        self._lock = RLock()
        self._updated_ns = 0

    def on_index(
        self,
        event: IndexEventLike | LiveIndex,
    ) -> LiveIndex:
        live = coerce_index(event)

        with self._lock:
            previous = self._states.get(live.symbol)

            if previous is not None:
                if live.timestamp_ns < previous.timestamp_ns:
                    raise MarketDataRejected(
                        f"stale {live.symbol} factor timestamp"
                    )
                if (
                    live.sequence > 0
                    and previous.sequence > 0
                    and live.sequence <= previous.sequence
                ):
                    raise MarketDataRejected(
                        f"out-of-sequence {live.symbol} factor update"
                    )

            self._states[live.symbol] = live
            self._updated_ns = time_ns()

        return live

    def latest(self, symbol: str) -> LiveIndex | None:
        normalized = _normalize_symbol(symbol)
        if normalized not in VOLATILITY_FACTORS:
            raise KeyError(
                f"not a configured volatility factor: {normalized}"
            )
        with self._lock:
            return self._states.get(normalized)

    def snapshot(self) -> VolatilityFactorSnapshot:
        with self._lock:
            return VolatilityFactorSnapshot(
                vix=self._states.get("VIX"),
                vxn=self._states.get("VXN"),
                updated_ns=self._updated_ns,
            )


# ============================================================================
# NATIVE STATUS CONVERSION
# ============================================================================


def _native_side(
    native: Any,
    side: Side,
) -> Any:
    """
    Support conventional pybind11 enum naming styles.

    Preferred binding:
        Side.Buy
        Side.Sell

    Also tolerates:
        Side.BUY
        Side.SELL
    """

    enum_type = native.Side

    if side is Side.BUY:
        for name in ("Buy", "BUY", "buy"):
            if hasattr(enum_type, name):
                return getattr(enum_type, name)
    else:
        for name in ("Sell", "SELL", "sell"):
            if hasattr(enum_type, name):
                return getattr(enum_type, name)

    raise NativeBindingError(
        "native Side enum does not expose Buy/Sell"
    )


def _native_status_to_python(
    status: Any,
) -> LiquidityOrderStatus:
    name = getattr(status, "name", None)

    if name is None:
        name = str(status)

    token = (
        str(name)
        .split(".")[-1]
        .replace("_", "")
        .lower()
    )

    mapping = {
        "new": LiquidityOrderStatus.NEW,
        "accepted": LiquidityOrderStatus.ACTIVE,
        "partiallyfilled":
            LiquidityOrderStatus.PARTIALLY_FILLED,
        "filled": LiquidityOrderStatus.FILLED,
        "cancelled": LiquidityOrderStatus.CANCELLED,
        "canceled": LiquidityOrderStatus.CANCELLED,
        "rejected": LiquidityOrderStatus.REJECTED,
    }

    return mapping.get(
        token,
        LiquidityOrderStatus.UNKNOWN,
    )


# ============================================================================
# LIQUIDITY PROVIDER
# ============================================================================


class LiquidityProvider:
    """
    Generates and manages synthetic liquidity inside C++ LimitOrderBook.

    This replaces the old model in which Python merely accumulated synthetic
    Order objects and a separate Python MarketSimulator decided fills.

    The native engine is now authoritative:

        provide_liquidity()
            -> OrderBookEngine.add_limit()
            -> C++ matching
            -> Execution[]

    Synthetic liquidity is useful for simulation/research. It is NOT an order
    router for Alpaca live brokerage.
    """

    def __init__(
        self,
        symbol: str,
        engine: Any,
        *,
        native_module: ModuleType | Any | None = None,
        config: LiquidityConfig | None = None,
        id_generator: Iterable[int] | None = None,
        id_start: int = 1_000_000_000,
    ) -> None:
        self.symbol = _normalize_symbol(symbol)

        if self.symbol in VOLATILITY_FACTORS:
            raise NonExecutableFactorError(
                f"{self.symbol} is a read-only volatility factor; "
                "LiquidityProvider cannot create orders for it"
            )

        self.native = (
            native_module
            if native_module is not None
            else load_native_module()
        )

        self.engine = engine

        self.config = (
            config
            if config is not None
            else LiquidityConfig()
        )
        self.config.validate()

        self._native_tick_size = (
            tick_size_to_native_ticks(
                self.config.tick_size
            )
        )

        # IDs should ultimately be issued by a process-wide authority shared
        # with strategy orders. An injected generator is supported so the
        # application can enforce that invariant.
        self._id_source = (
            iter(id_generator)
            if id_generator is not None
            else count(id_start)
        )

        self._rng = Random(
            self.config.random_seed
        )

        self._orders: dict[
            int,
            LiquidityOrder
        ] = {}

        self._active_ids: set[int] = set()

        self._lock = RLock()

        self._last_quote: LiveQuote | None = None

    # ------------------------------------------------------------------------
    # ORDER IDS
    # ------------------------------------------------------------------------

    def _next_order_id(self) -> int:
        with self._lock:
            order_id = int(next(self._id_source))

        if order_id <= 0:
            raise InvalidLiquidityOrder(
                "order IDs must be positive"
            )

        return order_id

    # ------------------------------------------------------------------------
    # VALIDATION
    # ------------------------------------------------------------------------

    def _validate_quantity(
        self,
        quantity: int,
    ) -> None:
        if isinstance(quantity, bool) or not isinstance(
            quantity,
            int,
        ):
            raise InvalidLiquidityOrder(
                "quantity must be an integer"
            )

        if quantity < self.config.min_quantity:
            raise InvalidLiquidityOrder(
                "quantity is below configured minimum"
            )

        if quantity > self.config.max_quantity:
            raise InvalidLiquidityOrder(
                "quantity exceeds configured maximum"
            )

    def _validate_price_ticks(
        self,
        price_ticks: int,
    ) -> None:
        if isinstance(price_ticks, bool) or not isinstance(
            price_ticks,
            int,
        ):
            raise InvalidLiquidityOrder(
                "price_ticks must be an integer"
            )

        if price_ticks <= 0:
            raise InvalidLiquidityOrder(
                "price_ticks must be positive"
            )

        if (
            price_ticks % self._native_tick_size
            != 0
        ):
            raise InvalidLiquidityOrder(
                "price does not align to configured tick size"
            )

    # ------------------------------------------------------------------------
    # CREATE / SUBMIT
    # ------------------------------------------------------------------------

    def create_order(
        self,
        side: Side,
        quantity: int,
        price_ticks: int,
        *,
        order_id: int | None = None,
    ) -> tuple[
        LiquidityOrder,
        tuple[Any, ...],
    ]:
        """
        Submit a synthetic limit order into the C++ matching engine.

        Returns:
            (python metadata, native executions)
        """

        if not isinstance(side, Side):
            raise InvalidLiquidityOrder(
                "side must be Side.BUY or Side.SELL"
            )

        self._validate_quantity(quantity)
        self._validate_price_ticks(price_ticks)

        oid = (
            self._next_order_id()
            if order_id is None
            else int(order_id)
        )

        if oid <= 0:
            raise InvalidLiquidityOrder(
                "order_id must be positive"
            )

        now = time_ns()

        metadata = LiquidityOrder(
            id=oid,
            symbol=self.symbol,
            side=side,
            quantity=quantity,
            price_ticks=price_ticks,
            status=LiquidityOrderStatus.NEW,
            filled_quantity=0,
            remaining_quantity=quantity,
            created_ns=now,
            updated_ns=now,
        )

        native_side = _native_side(
            self.native,
            side,
        )

        try:
            executions = tuple(
                self.engine.add_limit(
                    oid,
                    native_side,
                    quantity,
                    price_ticks,
                )
            )
        except Exception:
            rejected = replace(
                metadata,
                status=LiquidityOrderStatus.REJECTED,
                updated_ns=time_ns(),
            )

            with self._lock:
                self._orders[oid] = rejected

            raise

        with self._lock:
            self._orders[oid] = metadata

        refreshed = self.refresh_order(
            oid,
            allow_terminal_inference=True,
            executions=executions,
        )

        return refreshed, executions

    # ------------------------------------------------------------------------
    # QUOTE SYNTHETIC MARKET
    # ------------------------------------------------------------------------

    def provide_liquidity(
        self,
        quote: AlpacaQuoteLike | LiveQuote,
        *,
        levels: int | None = None,
        quantity: int | None = None,
    ) -> list[
        tuple[LiquidityOrder, tuple[Any, ...]]
    ]:
        """
        Reprice optional synthetic liquidity around current observed NBBO.

        SIP itself is not inserted as depth. These are OUR simulated orders.

        When replace_on_quote=True, currently active synthetic LP orders are
        cancelled before the new ladder is submitted.
        """

        live = coerce_quote(quote)

        if live.symbol != self.symbol:
            raise ValueError(
                f"quote symbol mismatch: "
                f"{live.symbol} != {self.symbol}"
            )

        self._last_quote = live

        level_count = (
            self.config.levels
            if levels is None
            else int(levels)
        )

        if level_count <= 0:
            raise ValueError("levels must be positive")

        if quantity is not None:
            self._validate_quantity(quantity)

        if self.config.replace_on_quote:
            self.cancel_all_active()

        results: list[
            tuple[LiquidityOrder, tuple[Any, ...]]
        ] = []

        for level in range(level_count):
            level_qty = (
                quantity
                if quantity is not None
                else self._rng.randint(
                    self.config.min_quantity,
                    self.config.max_quantity,
                )
            )

            distance = (
                self.config.inside_offset_ticks
                + level
                * self.config.level_spacing_ticks
            )

            distance_ticks = (
                distance
                * self._native_tick_size
            )

            buy_price = (
                live.bid_price_ticks
                - distance_ticks
            )

            sell_price = (
                live.ask_price_ticks
                + distance_ticks
            )

            if buy_price > 0:
                if (
                    not self.config.prevent_crossing_observed_market
                    or buy_price < live.ask_price_ticks
                ):
                    results.append(
                        self.create_order(
                            Side.BUY,
                            level_qty,
                            buy_price,
                        )
                    )

            if (
                not self.config.prevent_crossing_observed_market
                or sell_price > live.bid_price_ticks
            ):
                results.append(
                    self.create_order(
                        Side.SELL,
                        level_qty,
                        sell_price,
                    )
                )

        return results

    # ------------------------------------------------------------------------
    # RANDOM SYNTHETIC ORDER
    # ------------------------------------------------------------------------

    def generate_random_order(
        self,
        market_price_ticks: int,
        *,
        max_ticks_away: int = 5,
    ) -> tuple[
        LiquidityOrder,
        tuple[Any, ...],
    ]:
        if market_price_ticks <= 0:
            raise InvalidLiquidityOrder(
                "market_price_ticks must be positive"
            )

        if max_ticks_away <= 0:
            raise InvalidLiquidityOrder(
                "max_ticks_away must be positive"
            )

        side = self._rng.choice(
            [Side.BUY, Side.SELL]
        )

        quantity = self._rng.randint(
            self.config.min_quantity,
            self.config.max_quantity,
        )

        steps = self._rng.randint(
            1,
            max_ticks_away,
        )

        distance = (
            steps
            * self._native_tick_size
        )

        if side is Side.BUY:
            price_ticks = (
                market_price_ticks
                - distance
            )
        else:
            price_ticks = (
                market_price_ticks
                + distance
            )

        if price_ticks <= 0:
            raise InvalidLiquidityOrder(
                "generated price is non-positive"
            )

        return self.create_order(
            side,
            quantity,
            price_ticks,
        )

    # ------------------------------------------------------------------------
    # AMEND
    # ------------------------------------------------------------------------

    def amend(
        self,
        order_id: int,
        *,
        new_price_ticks: int,
        new_total_quantity: int,
    ) -> tuple[
        LiquidityOrder,
        tuple[Any, ...],
    ]:
        self._validate_price_ticks(
            new_price_ticks
        )
        self._validate_quantity(
            new_total_quantity
        )

        current = self.lookup_order(order_id)

        if current is None:
            raise LiquidityOrderNotFound(
                f"unknown liquidity order {order_id}"
            )

        if current.is_terminal:
            raise LiquidityOrderNotFound(
                f"liquidity order {order_id} is terminal"
            )

        executions = tuple(
            self.engine.amend(
                int(order_id),
                int(new_price_ticks),
                int(new_total_quantity),
            )
        )

        with self._lock:
            self._orders[order_id] = replace(
                current,
                quantity=new_total_quantity,
                price_ticks=new_price_ticks,
                updated_ns=time_ns(),
            )

        refreshed = self.refresh_order(
            order_id,
            allow_terminal_inference=True,
            executions=executions,
        )

        return refreshed, executions

    # ------------------------------------------------------------------------
    # CANCEL
    # ------------------------------------------------------------------------

    def cancel(
        self,
        order_id: int,
    ) -> LiquidityOrder:
        current = self.lookup_order(order_id)

        if current is None:
            raise LiquidityOrderNotFound(
                f"unknown liquidity order {order_id}"
            )

        if current.is_terminal:
            return current

        try:
            native_order = self.engine.cancel(
                int(order_id)
            )
        except Exception:
            # The order may already have filled due to a crossing internal
            # order. Refresh before deciding this is a hard cancellation fault.
            refreshed = self.refresh_order(
                order_id,
                allow_terminal_inference=True,
            )

            if refreshed.is_terminal:
                return refreshed

            raise

        result = self._metadata_from_native_order(
            current,
            native_order,
        )

        with self._lock:
            self._orders[order_id] = result
            self._active_ids.discard(order_id)

        return result

    def cancel_all_active(
        self,
    ) -> tuple[LiquidityOrder, ...]:
        with self._lock:
            active = tuple(self._active_ids)

        cancelled: list[LiquidityOrder] = []

        for order_id in active:
            try:
                cancelled.append(
                    self.cancel(order_id)
                )
            except LiquidityOrderNotFound:
                continue
            except Exception:
                # Do not silently convert a native failure into success.
                raise

        return tuple(cancelled)

    # ------------------------------------------------------------------------
    # RECONCILIATION
    # ------------------------------------------------------------------------

    def refresh_order(
        self,
        order_id: int,
        *,
        allow_terminal_inference: bool = True,
        executions: Sequence[Any] = (),
    ) -> LiquidityOrder:
        """
        Reconcile Python metadata against authoritative native book state.

        ``LimitOrderBook::get`` currently returns only active/resting orders.
        Therefore when an order disappears, fills supplied by the immediately
        preceding native operation are used to infer terminal fill state.

        For durable production reconciliation, expose completed-order history
        or an execution ledger from C++ rather than relying solely on this
        inference.
        """

        with self._lock:
            current = self._orders.get(order_id)

        if current is None:
            raise LiquidityOrderNotFound(
                f"unknown liquidity order {order_id}"
            )

        native_book = self.engine.simulated_book()
        native_order = native_book.get(
            int(order_id)
        )

        if native_order is not None:
            result = self._metadata_from_native_order(
                current,
                native_order,
            )

            with self._lock:
                self._orders[order_id] = result

                if result.is_terminal:
                    self._active_ids.discard(
                        order_id
                    )
                else:
                    self._active_ids.add(
                        order_id
                    )

            return result

        # No active native order exists.
        if not allow_terminal_inference:
            result = replace(
                current,
                status=LiquidityOrderStatus.UNKNOWN,
                updated_ns=time_ns(),
            )

            with self._lock:
                self._orders[order_id] = result
                self._active_ids.discard(order_id)

            return result

        filled_delta = 0

        for execution in executions:
            maker_id = int(
                getattr(execution, "maker_id", 0)
            )
            taker_id = int(
                getattr(execution, "taker_id", 0)
            )

            if order_id in (maker_id, taker_id):
                filled_delta += int(
                    getattr(
                        execution,
                        "quantity",
                        0,
                    )
                )

        filled = min(
            current.quantity,
            current.filled_quantity + filled_delta,
        )

        if filled >= current.quantity:
            status = LiquidityOrderStatus.FILLED
            remaining = 0
        else:
            # Disappearance with no complete execution evidence can occur after
            # explicit cancellation. Preserve a known cancellation, otherwise
            # mark UNKNOWN rather than falsely claiming a fill.
            if (
                current.status
                is LiquidityOrderStatus.CANCELLED
            ):
                status = (
                    LiquidityOrderStatus.CANCELLED
                )
                remaining = (
                    current.quantity
                    - current.filled_quantity
                )
            elif filled > current.filled_quantity:
                status = (
                    LiquidityOrderStatus.PARTIALLY_FILLED
                )
                remaining = (
                    current.quantity - filled
                )
            else:
                status = LiquidityOrderStatus.UNKNOWN
                remaining = (
                    current.quantity - filled
                )

        result = replace(
            current,
            status=status,
            filled_quantity=filled,
            remaining_quantity=remaining,
            updated_ns=time_ns(),
        )

        with self._lock:
            self._orders[order_id] = result
            self._active_ids.discard(order_id)

        return result

    def refresh_all(
        self,
    ) -> tuple[LiquidityOrder, ...]:
        with self._lock:
            order_ids = tuple(self._orders)

        output = []

        for order_id in order_ids:
            output.append(
                self.refresh_order(
                    order_id,
                    allow_terminal_inference=False,
                )
            )

        return tuple(output)

    def _metadata_from_native_order(
        self,
        current: LiquidityOrder,
        native_order: Any,
    ) -> LiquidityOrder:
        return replace(
            current,
            quantity=int(native_order.quantity),
            price_ticks=int(
                native_order.price_ticks
            ),
            status=_native_status_to_python(
                native_order.status
            ),
            filled_quantity=int(
                native_order.filled_quantity
            ),
            remaining_quantity=int(
                native_order.remaining_quantity
            ),
            updated_ns=time_ns(),
        )

    # ------------------------------------------------------------------------
    # READS
    # ------------------------------------------------------------------------

    def lookup_order(
        self,
        order_id: int,
    ) -> LiquidityOrder | None:
        with self._lock:
            return self._orders.get(
                int(order_id)
            )

    def active_orders(
        self,
    ) -> tuple[LiquidityOrder, ...]:
        with self._lock:
            return tuple(
                self._orders[order_id]
                for order_id
                in sorted(self._active_ids)
                if order_id in self._orders
            )

    def orders(
        self,
    ) -> tuple[LiquidityOrder, ...]:
        with self._lock:
            return tuple(
                self._orders[key]
                for key in sorted(self._orders)
            )

    def market_snapshot(self) -> Any:
        return self.engine.market_snapshot()

    def simulated_snapshot(
        self,
        max_levels: int = 0,
    ) -> Any:
        return self.engine.simulated_snapshot(
            int(max_levels)
        )


# ============================================================================
# MULTI-SYMBOL LIQUIDITY SERVICE
# ============================================================================


class LiquidityService:
    """
    Composition layer for one Alpaca SIP stream and many native books.

    Typical wiring:

        bridge = LiquidityService.create(UNIVERSE)

        stream = AlpacaSIPStream(
            UNIVERSE,
            on_quote=bridge.on_quote,
            on_trade=bridge.on_trade,
            ...
        )

    Quote callback sequence:

        Alpaca QuoteEvent
            -> NativeMarketBridge.on_quote()
            -> C++ ObservedMarketBook
            -> LiquidityProvider.provide_liquidity() [optional]

    ``auto_quote_liquidity`` should normally be False in production/live-data
    observation mode. Enable it only when synthetic depth is intentionally
    required for simulation.
    """

    def __init__(
        self,
        market_bridge: NativeMarketBridge,
        providers: Mapping[str, LiquidityProvider],
        *,
        factor_book: VolatilityFactorBook | None = None,
        auto_quote_liquidity: bool = False,
    ) -> None:
        self.market_bridge = market_bridge
        self.providers = dict(providers)
        self.factor_book = (
            factor_book
            if factor_book is not None
            else VolatilityFactorBook()
        )
        self.auto_quote_liquidity = bool(auto_quote_liquidity)

    @classmethod
    def create(
        cls,
        symbols: Iterable[str],
        *,
        native_module: ModuleType | Any | None = None,
        liquidity_config: LiquidityConfig | None = None,
        auto_quote_liquidity: bool = False,
        shared_id_start: int = 1_000_000_000,
    ) -> "LiquidityService":
        normalized = _normalize_symbols(symbols)

        executable = tuple(
            symbol
            for symbol in normalized
            if symbol not in VOLATILITY_FACTORS
        )

        if not executable:
            raise ValueError(
                "at least one executable equity symbol is required"
            )

        native = (
            native_module
            if native_module is not None
            else load_native_module()
        )

        bridge = NativeMarketBridge.create(
            executable,
            native_module=native,
        )

        # Process-wide monotonic source shared across all LPs.
        shared_ids = count(shared_id_start)

        providers: dict[
            str,
            LiquidityProvider
        ] = {}

        for symbol in executable:
            providers[symbol] = (
                LiquidityProvider(
                    symbol,
                    bridge.engine(symbol),
                    native_module=native,
                    config=liquidity_config,
                    id_generator=shared_ids,
                )
            )

        return cls(
            bridge,
            providers,
            auto_quote_liquidity=(
                auto_quote_liquidity
            ),
        )

    def provider(
        self,
        symbol: str,
    ) -> LiquidityProvider:
        normalized = _normalize_symbol(symbol)

        try:
            return self.providers[normalized]
        except KeyError as exc:
            raise KeyError(
                f"symbol is not configured: {normalized}"
            ) from exc

    def on_quote(
        self,
        event: AlpacaQuoteLike | LiveQuote,
    ) -> None:
        """
        Callback suitable for AlpacaSIPStream(on_quote=...).
        """

        live = coerce_quote(event)

        # Market observation reaches C++ first.
        self.market_bridge.on_quote(
            live
        )

        # Synthetic quoting is explicitly optional. Live SIP observation alone
        # should not mutate our simulated order book.
        if self.auto_quote_liquidity:
            self.provider(
                live.symbol
            ).provide_liquidity(live)

    def on_trade(
        self,
        event: AlpacaTradeLike | LiveTrade,
    ) -> None:
        """
        Callback suitable for AlpacaSIPStream(on_trade=...).
        """

        self.market_bridge.on_trade(
            event
        )

    def on_index(
        self,
        event: IndexEventLike | LiveIndex,
    ) -> LiveIndex:
        """
        Callback suitable for IndexFactorBridge(on_index=...).

        VIX/VXN update factor state only. They never reach OrderBookRegistry.
        """
        return self.factor_book.on_index(event)

    def factor(
        self,
        symbol: str,
    ) -> LiveIndex | None:
        return self.factor_book.latest(symbol)

    def factor_snapshot(
        self,
    ) -> VolatilityFactorSnapshot:
        return self.factor_book.snapshot()

    def cancel_all(
        self,
    ) -> dict[
        str,
        tuple[LiquidityOrder, ...]
    ]:
        return {
            symbol:
                provider.cancel_all_active()
            for symbol, provider
            in self.providers.items()
        }


# ============================================================================
# OPTIONAL ALPACA STREAM FACTORY
# ============================================================================


def build_alpaca_sip_stream(
    service: LiquidityService,
    *,
    stream_class: Any | None = None,
    bars: bool = True,
) -> Any:
    """
    Convenience factory connecting the previously created AlpacaSIPStream to
    this service.

    ``stream_class`` can be injected for tests. Otherwise the function imports
    AlpacaSIPStream lazily so liquidity.py does not own credential loading or
    WebSocket dependencies.
    """

    if stream_class is None:
        try:
            from alpaca_sip_stream import (
                AlpacaSIPStream,
            )
        except ImportError as exc:
            raise RuntimeError(
                "alpaca_sip_stream.py must be importable "
                "to build the live SIP stream"
            ) from exc

        stream_class = AlpacaSIPStream

    symbols = tuple(
        sorted(service.providers)
    )

    return stream_class(
        symbols,
        quotes=True,
        trades=True,
        bars=bars,
        on_quote=service.on_quote,
        on_trade=service.on_trade,
    )


def bind_index_factor_bridge(
    service: LiquidityService,
    *,
    bridge_class: Any | None = None,
) -> Any:
    """
    Construct the VIX/VXN factor bridge from the updated market-data module.

    The returned bridge accepts external index-provider observations and
    forwards normalized IndexEvent objects into service.on_index().
    """
    if bridge_class is None:
        try:
            from alpaca_sip_stream_with_vix_vxn_sqqq import (
                IndexFactorBridge,
            )
        except ImportError:
            try:
                from alpaca_sip_stream import IndexFactorBridge
            except ImportError as exc:
                raise RuntimeError(
                    "updated market-data module with IndexFactorBridge "
                    "must be importable"
                ) from exc

        bridge_class = IndexFactorBridge

    return bridge_class(on_index=service.on_index)


__all__ = [
    "DEFAULT_NATIVE_MODULE",
    "DEFAULT_EXECUTABLE_UNIVERSE",
    "INVERSE_EQUITY",
    "IndexEventLike",
    "LiveIndex",
    "NonExecutableFactorError",
    "PRIMARY_TARGET",
    "VOLATILITY_FACTORS",
    "VolatilityFactorBook",
    "VolatilityFactorSnapshot",
    "bind_index_factor_bridge",
    "coerce_index",
    "DEFAULT_PRICE_SCALE",
    "InvalidLiquidityOrder",
    "LiquidityConfig",
    "LiquidityError",
    "LiquidityOrder",
    "LiquidityOrderNotFound",
    "LiquidityOrderStatus",
    "LiquidityProvider",
    "LiquidityService",
    "LiveQuote",
    "LiveTrade",
    "MarketDataRejected",
    "NativeBindingError",
    "NativeMarketBridge",
    "Side",
    "build_alpaca_sip_stream",
    "coerce_quote",
    "coerce_trade",
    "load_native_module",
    "price_to_ticks",
    "ticks_to_price",
]
