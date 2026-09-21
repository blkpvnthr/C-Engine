#!/usr/bin/env python3
"""
engine.py

Top-level bootstrap for the complete market engine.

Data path
---------
Alpaca SIP
    -> normalized QuoteEvent / TradeEvent / BarEvent
    -> MarketDataFanout
        -> DailyHDF5Writer
        -> live MarketDataQueue
    -> Bar adapter
    -> StrategyCoordinator
        -> Turtle
        -> Dual MA
        -> APO mean reversion
    -> candidate arbitration
    -> OrderIntent
    -> OrderManager
    -> native C++ RiskEngine
    -> native C++ execution venue / LimitOrderBook

VIX/VXN
-------
A licensed external source publishes IndexEvent objects through
IndexFactorBridge. Those observations enter the same archive/live fan-out but
are never treated as executable equities.

Authority boundary
------------------
This file orchestrates components. It does not:
- approve risk;
- invent prices, quantities, timestamps, fills, or volatility-index values;
- fabricate Level-II depth from SIP NBBO;
- bypass the native C++ risk boundary;
- treat Metal output as execution authority.

Required local modules
----------------------
alpaca_sip_stream.py
market_data_store.py
strategies.py
order_manager.py

Native adapters
---------------
The application must provide a module exposing:

    build_risk_engine() -> object satisfying NativeRiskEngine
    build_execution_venue() -> object satisfying ExecutionVenue

Set its import name with:

    ENGINE_NATIVE_ADAPTER_MODULE=<module>

No permissive fallback is used when that module is absent.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib
import inspect
import logging
import os
import signal
from dataclasses import dataclass
from typing import Awaitable, Callable, Optional, Protocol

from alpaca_sip_stream import (
    AlpacaSIPStream,
    BarEvent,
    IndexEvent,
    IndexFactorBridge,
    MarketDataFanout,
    MarketDataQueue,
    QuoteEvent,
    TradeEvent,
)
from market_data_store import DailyHDF5Writer
from order_manager import (
    ExecutionUpdate,
    ExecutionVenue,
    NativeRiskEngine,
    OrderIntent,
    OrderManager,
    OrderStatus,
    OrderType,
    Side,
    TimeInForce,
)
from strategies import (
    APOConfig,
    Bar,
    DualMAConfig,
    SignalSide,
    StrategyCoordinator,
    StrategySetConfig,
    TradeCandidate,
    TurtleConfig,
    build_strategy_coordinator,
    group_candidates_by_direction,
)


LOGGER = logging.getLogger("market_engine")

MarketEvent = QuoteEvent | TradeEvent | BarEvent | IndexEvent


class EngineConfigurationError(RuntimeError):
    pass


class StrategyConflictError(RuntimeError):
    pass


class QuantityPolicy(Protocol):
    """Explicit position-sizing authority upstream of native risk.

    It proposes quantity only. Native C++ RiskEngine remains authoritative and
    may reject or reduce the proposed quantity.
    """

    def quantity_for(
        self,
        candidate: TradeCandidate,
    ) -> int | Awaitable[int]:
        ...


class ExitQuantityProvider(Protocol):
    """Returns authoritative quantity to close for a strategy exit candidate."""

    def quantity_to_close(
        self,
        candidate: TradeCandidate,
    ) -> int | Awaitable[int]:
        ...


class ExternalFactorSource(Protocol):
    """Licensed/provider-specific VIX/VXN source.

    Implementations publish actual observations into the supplied bridge and
    return only when stopped/cancelled.
    """

    async def run(
        self,
        bridge: IndexFactorBridge,
        stop_event: asyncio.Event,
    ) -> None:
        ...


async def _maybe_await(value):
    if inspect.isawaitable(value):
        return await value
    return value


@dataclass(frozen=True, slots=True)
class EngineConfig:
    symbols: tuple[str, ...]
    market_data_root: str
    archive_provider_label: str

    turtle_entry_lookback: int
    turtle_exit_lookback: int

    dual_ma_fast_period: int
    dual_ma_slow_period: int

    apo_fast_period: int
    apo_slow_period: int
    apo_entry_threshold_ticks: float
    apo_exit_threshold_ticks: float

    native_adapter_module: str

    alpaca_url: Optional[str] = None

    def validate(self) -> None:
        if not self.symbols:
            raise EngineConfigurationError("symbols are required")
        if not self.market_data_root.strip():
            raise EngineConfigurationError("market_data_root is required")
        if not self.archive_provider_label.strip():
            raise EngineConfigurationError(
                "archive_provider_label must explicitly identify provenance"
            )
        if not self.native_adapter_module.strip():
            raise EngineConfigurationError(
                "native_adapter_module is required"
            )

        # Strategy constructors perform the detailed mathematical validation.
        StrategySetConfig(
            turtle=TurtleConfig(
                entry_lookback=self.turtle_entry_lookback,
                exit_lookback=self.turtle_exit_lookback,
            ),
            dual_ma=DualMAConfig(
                fast_period=self.dual_ma_fast_period,
                slow_period=self.dual_ma_slow_period,
            ),
            apo=APOConfig(
                fast_period=self.apo_fast_period,
                slow_period=self.apo_slow_period,
                entry_threshold_ticks=self.apo_entry_threshold_ticks,
                exit_threshold_ticks=self.apo_exit_threshold_ticks,
            ),
        )


@dataclass(slots=True)
class EngineHealth:
    market_events_processed: int = 0
    bars_processed: int = 0
    strategy_candidates: int = 0
    conflicting_batches: int = 0
    order_intents_submitted: int = 0
    order_failures: int = 0


def load_native_adapters(
    module_name: str,
) -> tuple[NativeRiskEngine, ExecutionVenue]:
    """Load explicit pybind11/native adapters.

    The module must intentionally expose both factories. There is no Python
    allow-all risk engine and no implicit simulator fallback.
    """
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise EngineConfigurationError(
            f"cannot import native adapter module {module_name!r}"
        ) from exc

    risk_factory = getattr(module, "build_risk_engine", None)
    venue_factory = getattr(module, "build_execution_venue", None)

    if not callable(risk_factory):
        raise EngineConfigurationError(
            f"{module_name!r} must expose build_risk_engine()"
        )
    if not callable(venue_factory):
        raise EngineConfigurationError(
            f"{module_name!r} must expose build_execution_venue()"
        )

    risk_engine = risk_factory()
    venue = venue_factory()

    if risk_engine is None or venue is None:
        raise EngineConfigurationError(
            "native adapter factories must return concrete objects"
        )

    return risk_engine, venue


class CandidateRouter:
    """Converts non-conflicting strategy candidates into OrderIntent objects.

    All strategies remain active simultaneously.

    Conflict policy is intentionally conservative:
    - candidates in only one transaction direction may proceed independently;
    - simultaneous buy-direction and sell-direction candidates for the same
      symbol/bar are not netted, ranked, or guessed;
    - the conflicted batch is withheld from order creation and surfaced.

    This avoids silently choosing one strategy over another. A future
    portfolio-level policy may replace this class if it is explicitly tested
    and versioned.
    """

    def __init__(
        self,
        *,
        order_manager: OrderManager,
        entry_quantity_policy: QuantityPolicy,
        exit_quantity_provider: ExitQuantityProvider,
    ) -> None:
        self._orders = order_manager
        self._entry_quantity = entry_quantity_policy
        self._exit_quantity = exit_quantity_provider

    async def route(
        self,
        candidates: tuple[TradeCandidate, ...],
    ) -> tuple[int, bool]:
        if not candidates:
            return 0, False

        buy, sell = group_candidates_by_direction(candidates)

        if buy and sell:
            LOGGER.warning(
                "strategy conflict symbol=%s timestamp_ns=%d "
                "buy_strategies=%s sell_strategies=%s; withholding batch",
                candidates[0].symbol,
                candidates[0].timestamp_ns,
                [x.strategy.value for x in buy],
                [x.strategy.value for x in sell],
            )
            return 0, True

        submitted = 0

        # Multiple same-direction candidates are preserved as independent
        # strategy intents. The native risk engine sees every request and owns
        # aggregate exposure constraints.
        for candidate in candidates:
            intent = await self._to_order_intent(candidate)
            await self._orders.submit(intent)
            submitted += 1

        return submitted, False

    async def _to_order_intent(
        self,
        candidate: TradeCandidate,
    ) -> OrderIntent:
        if candidate.side in {SignalSide.LONG, SignalSide.SHORT}:
            quantity = int(
                await _maybe_await(
                    self._entry_quantity.quantity_for(candidate)
                )
            )
        else:
            quantity = int(
                await _maybe_await(
                    self._exit_quantity.quantity_to_close(candidate)
                )
            )

        if quantity <= 0:
            raise EngineConfigurationError(
                f"quantity provider returned {quantity} for "
                f"{candidate.strategy.value}/{candidate.symbol}"
            )

        if candidate.side in {
            SignalSide.LONG,
            SignalSide.EXIT_SHORT,
        }:
            side = Side.BUY
        else:
            side = Side.SELL

        # Market-vs-limit policy must not be invented here. This bootstrap uses
        # a MARKET intent only when explicitly selected by this router's
        # contract. If another execution policy is desired, replace the router
        # with a versioned implementation.
        return OrderIntent(
            symbol=candidate.symbol,
            side=side,
            quantity=quantity,
            order_type=OrderType.MARKET,
            time_in_force=TimeInForce.DAY,
            strategy_id=candidate.strategy.value,
            correlation_id=(
                f"{candidate.symbol}:"
                f"{candidate.timestamp_ns}:"
                f"{candidate.strategy.value}"
            ),
            created_ns=candidate.timestamp_ns,
        )


class MarketEngine:
    def __init__(
        self,
        *,
        config: EngineConfig,
        quantity_policy: QuantityPolicy,
        exit_quantity_provider: ExitQuantityProvider,
        factor_source: Optional[ExternalFactorSource] = None,
    ) -> None:
        config.validate()
        self.config = config
        self.health = EngineHealth()
        self._stop = asyncio.Event()

        risk_engine, venue = load_native_adapters(
            config.native_adapter_module
        )

        self.order_manager = OrderManager(
            risk_engine=risk_engine,
            venue=venue,
            on_audit=self._audit_order,
        )

        strategy_config = StrategySetConfig(
            turtle=TurtleConfig(
                entry_lookback=config.turtle_entry_lookback,
                exit_lookback=config.turtle_exit_lookback,
            ),
            dual_ma=DualMAConfig(
                fast_period=config.dual_ma_fast_period,
                slow_period=config.dual_ma_slow_period,
            ),
            apo=APOConfig(
                fast_period=config.apo_fast_period,
                slow_period=config.apo_slow_period,
                entry_threshold_ticks=config.apo_entry_threshold_ticks,
                exit_threshold_ticks=config.apo_exit_threshold_ticks,
            ),
        )
        self.strategies: StrategyCoordinator = (
            build_strategy_coordinator(strategy_config)
        )

        self.archive = DailyHDF5Writer(
            root=config.market_data_root,
            provider=config.archive_provider_label,
        )
        self.live_queue = MarketDataQueue()
        self.fanout = MarketDataFanout(
            live_queue=self.live_queue,
            archive=self.archive,
        )

        self.factor_bridge = IndexFactorBridge(
            on_index=self._publish_market_event
        )
        self.factor_source = factor_source

        stream_kwargs = dict(
            symbols=config.symbols,
            quotes=True,
            trades=True,
            bars=True,
            on_quote=self._publish_market_event,
            on_trade=self._publish_market_event,
            on_bar=self._publish_market_event,
            on_state=self._stream_state_changed,
        )
        if config.alpaca_url is not None:
            stream_kwargs["url"] = config.alpaca_url

        self.stream = AlpacaSIPStream(**stream_kwargs)

        self.router = CandidateRouter(
            order_manager=self.order_manager,
            entry_quantity_policy=quantity_policy,
            exit_quantity_provider=exit_quantity_provider,
        )

    async def run(self) -> None:
        """Start archive, consumer, market stream, and optional factor source."""
        self.archive.start()

        consumer_task = asyncio.create_task(
            self._consume_market_events(),
            name="market-engine-consumer",
        )
        stream_task = asyncio.create_task(
            self.stream.run_forever(),
            name="alpaca-sip-stream",
        )

        factor_task: Optional[asyncio.Task[None]] = None
        if self.factor_source is not None:
            factor_task = asyncio.create_task(
                self.factor_source.run(
                    self.factor_bridge,
                    self._stop,
                ),
                name="volatility-factor-source",
            )

        tasks = [consumer_task, stream_task]
        if factor_task is not None:
            tasks.append(factor_task)

        try:
            done, pending = await asyncio.wait(
                tasks,
                return_when=asyncio.FIRST_EXCEPTION,
            )

            for task in done:
                exc = task.exception()
                if exc is not None:
                    raise exc

            # If a task returned normally while the engine was not asked to
            # stop, treat that as an unexpected engine termination.
            if not self._stop.is_set():
                raise RuntimeError(
                    "engine component stopped unexpectedly"
                )
        finally:
            await self.stop()

            for task in tasks:
                if not task.done():
                    task.cancel()

            await asyncio.gather(
                *tasks,
                return_exceptions=True,
            )

            self.archive.close(drain=True)

    async def stop(self) -> None:
        if self._stop.is_set():
            return

        self._stop.set()
        await self.stream.stop()

    async def reconcile_execution(
        self,
        update: ExecutionUpdate,
    ) -> None:
        """Entry point for authoritative broker/order-book execution updates."""
        await self.order_manager.reconcile(update)

    async def _publish_market_event(
        self,
        event: MarketEvent,
    ) -> None:
        # MarketDataFanout archives first and then publishes to the latency-
        # oriented live queue. Archival queue saturation is surfaced as a fault.
        self.fanout.publish(event)

    async def _consume_market_events(self) -> None:
        while not self._stop.is_set():
            event = await self.live_queue.get()

            try:
                self.health.market_events_processed += 1

                if isinstance(event, BarEvent):
                    await self._process_bar(event)

                # Quotes/trades/factors remain available to other live
                # analytics through future fan-out subscribers. Strategy
                # definitions in strategies.py are bar-based.
            finally:
                self.live_queue.task_done()

    async def _process_bar(self, event: BarEvent) -> None:
        bar = Bar(
            symbol=event.symbol,
            timestamp_ns=event.timestamp_ns,
            open_ticks=event.open_ticks,
            high_ticks=event.high_ticks,
            low_ticks=event.low_ticks,
            close_ticks=event.close_ticks,
            volume=event.volume,
        )

        batch = self.strategies.on_bar(bar)
        self.health.bars_processed += 1
        self.health.strategy_candidates += len(batch.candidates)

        if not batch.candidates:
            return

        try:
            submitted, conflicted = await self.router.route(
                batch.candidates
            )
        except Exception:
            self.health.order_failures += 1
            LOGGER.exception(
                "candidate routing failed symbol=%s timestamp_ns=%d",
                batch.symbol,
                batch.timestamp_ns,
            )
            raise

        self.health.order_intents_submitted += submitted
        if conflicted:
            self.health.conflicting_batches += 1

    async def _audit_order(
        self,
        event: str,
        order,
    ) -> None:
        LOGGER.info(
            "order event=%s client_order_id=%s strategy=%s "
            "symbol=%s status=%s filled=%d approved=%d venue_order_id=%s",
            event,
            order.client_order_id,
            order.intent.strategy_id,
            order.intent.symbol,
            order.status.value,
            order.cumulative_filled_quantity,
            order.approved_quantity,
            order.venue_order_id,
        )

    async def _stream_state_changed(self, state) -> None:
        LOGGER.info("Alpaca SIP state=%s", state.value)


def _csv_symbols(value: str) -> tuple[str, ...]:
    symbols = tuple(
        part.strip().upper()
        for part in value.split(",")
        if part.strip()
    )
    if not symbols:
        raise argparse.ArgumentTypeError("at least one symbol is required")
    return symbols


def config_from_args() -> EngineConfig:
    parser = argparse.ArgumentParser(
        description="Activate the complete market engine."
    )

    parser.add_argument(
        "--symbols",
        required=True,
        type=_csv_symbols,
        help="Comma-separated executable equity universe.",
    )
    parser.add_argument(
        "--market-data-root",
        required=True,
        help="Root directory for daily HDF5 files.",
    )
    parser.add_argument(
        "--archive-provider-label",
        required=True,
        help="Explicit provenance label stored in HDF5 metadata.",
    )
    parser.add_argument(
        "--native-adapter-module",
        default=os.environ.get("ENGINE_NATIVE_ADAPTER_MODULE", ""),
        help=(
            "Python/pybind11 module exposing build_risk_engine() and "
            "build_execution_venue()."
        ),
    )

    parser.add_argument("--turtle-entry-lookback", required=True, type=int)
    parser.add_argument("--turtle-exit-lookback", required=True, type=int)

    parser.add_argument("--dual-ma-fast-period", required=True, type=int)
    parser.add_argument("--dual-ma-slow-period", required=True, type=int)

    parser.add_argument("--apo-fast-period", required=True, type=int)
    parser.add_argument("--apo-slow-period", required=True, type=int)
    parser.add_argument(
        "--apo-entry-threshold-ticks",
        required=True,
        type=float,
    )
    parser.add_argument(
        "--apo-exit-threshold-ticks",
        required=True,
        type=float,
    )

    parser.add_argument(
        "--alpaca-url",
        default=None,
        help=(
            "Optional explicit Alpaca WebSocket endpoint. "
            "If omitted, alpaca_sip_stream.py uses its configured SIP URL."
        ),
    )

    args = parser.parse_args()

    return EngineConfig(
        symbols=args.symbols,
        market_data_root=args.market_data_root,
        archive_provider_label=args.archive_provider_label,
        turtle_entry_lookback=args.turtle_entry_lookback,
        turtle_exit_lookback=args.turtle_exit_lookback,
        dual_ma_fast_period=args.dual_ma_fast_period,
        dual_ma_slow_period=args.dual_ma_slow_period,
        apo_fast_period=args.apo_fast_period,
        apo_slow_period=args.apo_slow_period,
        apo_entry_threshold_ticks=args.apo_entry_threshold_ticks,
        apo_exit_threshold_ticks=args.apo_exit_threshold_ticks,
        native_adapter_module=args.native_adapter_module,
        alpaca_url=args.alpaca_url,
    )


async def run_engine(
    *,
    config: EngineConfig,
    quantity_policy: QuantityPolicy,
    exit_quantity_provider: ExitQuantityProvider,
    factor_source: Optional[ExternalFactorSource] = None,
) -> None:
    """Programmatic activation entry point.

    Position sizing and exit inventory lookup are mandatory injected policies
    because this bootstrap must not invent quantities.
    """
    engine = MarketEngine(
        config=config,
        quantity_policy=quantity_policy,
        exit_quantity_provider=exit_quantity_provider,
        factor_source=factor_source,
    )

    loop = asyncio.get_running_loop()

    def request_stop() -> None:
        asyncio.create_task(engine.stop())

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, request_stop)
        except NotImplementedError:
            pass

    await engine.run()


def main() -> None:
    """CLI validates infrastructure but refuses to invent sizing policy.

    Use run_engine(...) from the application composition root after injecting
    an explicit QuantityPolicy and ExitQuantityProvider.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    config = config_from_args()
    config.validate()

    # Validate the native authority boundary before reporting readiness.
    load_native_adapters(config.native_adapter_module)

    raise EngineConfigurationError(
        "engine infrastructure is configured, but CLI activation requires "
        "explicit QuantityPolicy and ExitQuantityProvider implementations. "
        "Call run_engine(...) from your composition root; no quantity will "
        "be invented by engine.py."
    )


if __name__ == "__main__":
    main()
