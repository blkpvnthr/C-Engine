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
import time
from dataclasses import dataclass
from typing import Awaitable, Optional, Protocol

from cengine.engine_state import EngineState, EngineStateMachine
from cengine.event_bus import MarketEventBus
from cengine.execution_policy import ExecutionPolicy
from cengine.journal import AuditJournal
from cengine.metrics import PortfolioMetricsCollector
from cengine.nosql_journal import MongoDailyJournalStore
from cengine.safety import (
    MarketGateConfig,
    MarketSafetyGate,
    MarketSessionGate,
    SessionConfig,
)
from cengine.strategy_state import StrategyStateBook
from market_data.alpaca_sip_stream import (
    AlpacaSIPStream,
    BarEvent,
    IndexEvent,
    IndexFactorBridge,
    QuoteEvent,
    TradeEvent,
)
from market_data.market_data_store import DailyHDF5Writer
from order_manager import (
    TERMINAL_STATUSES,
    ExecutionUpdate,
    ExecutionVenue,
    NativeRiskEngine,
    OrderIntent,
    OrderManager,
    Side,
)
from strategies import (
    APOConfig,
    Bar,
    DualMAConfig,
    SignalSide,
    StrategyCoordinator,
    StrategyName,
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
    ) -> int | Awaitable[int]: ...


class ExitQuantityProvider(Protocol):
    """Returns authoritative quantity to close for a strategy exit candidate."""

    def quantity_to_close(
        self,
        candidate: TradeCandidate,
    ) -> int | Awaitable[int]: ...


class ExternalFactorSource(Protocol):
    """Licensed/provider-specific VIX/VXN source.

    Implementations publish actual observations into the supplied bridge and
    return only when stopped/cancelled.
    """

    async def run(
        self,
        bridge: IndexFactorBridge,
        stop_event: asyncio.Event,
    ) -> None: ...


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
    journal_path: str
    max_feed_age_ns: int
    max_bar_gap_ns: int
    reconciliation_interval_seconds: float
    session_timezone: str
    session_open_minute: int
    session_close_minute: int
    session_weekdays: tuple[int, ...]
    mongodb_uri: str
    mongodb_database: str
    mongodb_timeout_ms: int

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
            raise EngineConfigurationError("native_adapter_module is required")
        if not self.journal_path.strip():
            raise EngineConfigurationError("journal_path is required")
        MarketGateConfig(self.max_feed_age_ns, self.max_bar_gap_ns)
        if self.reconciliation_interval_seconds <= 0:
            raise EngineConfigurationError("reconciliation interval must be positive")
        SessionConfig(
            self.session_timezone,
            self.session_open_minute,
            self.session_close_minute,
            self.session_weekdays,
        )
        if not self.mongodb_uri.strip() or not self.mongodb_database.strip():
            raise EngineConfigurationError("MongoDB journal configuration is required")
        if self.mongodb_timeout_ms <= 0:
            raise EngineConfigurationError("MongoDB timeout must be positive")

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
        raise EngineConfigurationError(f"{module_name!r} must expose build_risk_engine()")
    if not callable(venue_factory):
        raise EngineConfigurationError(f"{module_name!r} must expose build_execution_venue()")

    risk_engine = risk_factory()
    venue = venue_factory()

    if risk_engine is None or venue is None:
        raise EngineConfigurationError("native adapter factories must return concrete objects")

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
        execution_policy: ExecutionPolicy,
        strategy_state: StrategyStateBook,
    ) -> None:
        self._orders = order_manager
        self._entry_quantity = entry_quantity_policy
        self._exit_quantity = exit_quantity_provider
        self._execution_policy = execution_policy
        self._strategy_state = strategy_state

    async def route(
        self,
        candidates: tuple[TradeCandidate, ...],
    ) -> tuple[int, bool]:
        if not candidates:
            return 0, False

        candidates = tuple(
            candidate
            for candidate in candidates
            if self._strategy_state.can_submit(candidate.strategy.value, candidate.symbol)
        )
        if not candidates:
            return 0, False

        exits = tuple(
            candidate
            for candidate in candidates
            if candidate.side in {SignalSide.EXIT_LONG, SignalSide.EXIT_SHORT}
        )
        # Risk reduction wins over new exposure. Entry signals in a batch that
        # contains an exit are withheld; exits are never blocked by entries.
        if exits:
            candidates = exits

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
            quantity = int(await _maybe_await(self._entry_quantity.quantity_for(candidate)))
        else:
            quantity = int(await _maybe_await(self._exit_quantity.quantity_to_close(candidate)))

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
        instruction = self._execution_policy.instruction_for(candidate)
        return OrderIntent(
            symbol=candidate.symbol,
            side=side,
            quantity=quantity,
            order_type=instruction.order_type,
            time_in_force=instruction.time_in_force,
            limit_price_ticks=instruction.limit_price_ticks,
            stop_price_ticks=instruction.stop_price_ticks,
            strategy_id=candidate.strategy.value,
            correlation_id=(
                f"{candidate.symbol}:{candidate.timestamp_ns}:{candidate.strategy.value}"
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
        execution_policy: ExecutionPolicy,
        factor_source: Optional[ExternalFactorSource] = None,
    ) -> None:
        config.validate()
        self.config = config
        self.health = EngineHealth()
        self._stop = asyncio.Event()
        self.state = EngineStateMachine()
        self.nosql_journal = MongoDailyJournalStore(
            config.mongodb_uri,
            config.mongodb_database,
            config.session_timezone,
            config.mongodb_timeout_ms,
        )
        self.journal = AuditJournal(config.journal_path, sinks=(self.nosql_journal,))
        self.market_gate = MarketSafetyGate(
            MarketGateConfig(config.max_feed_age_ns, config.max_bar_gap_ns)
        )
        self.session_gate = MarketSessionGate(
            SessionConfig(
                config.session_timezone,
                config.session_open_minute,
                config.session_close_minute,
                config.session_weekdays,
            )
        )
        self.strategy_state = StrategyStateBook()

        risk_engine, venue = load_native_adapters(config.native_adapter_module)
        self.venue = venue
        self._adapter_module = importlib.import_module(config.native_adapter_module)
        runtime_state = getattr(self._adapter_module, "STATE", None)
        self.portfolio = getattr(runtime_state, "positions", None)
        self.metrics = (
            PortfolioMetricsCollector(
                runtime_state.account,
                runtime_state.positions,
                runtime_state.reservations,
                runtime_state,
            )
            if runtime_state is not None
            else None
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
        self.strategies: StrategyCoordinator = build_strategy_coordinator(strategy_config)

        self.archive = DailyHDF5Writer(
            root=config.market_data_root,
            provider=config.archive_provider_label,
        )
        self.event_bus: MarketEventBus[MarketEvent] = MarketEventBus()
        self.strategy_events = self.event_bus.subscribe("strategies")
        for subscriber in (
            "order_book",
            "market_depth",
            "liquidity",
            "volatility",
            "stat_arb",
            "metal",
        ):
            self.event_bus.subscribe(subscriber)

        self.factor_bridge = IndexFactorBridge(on_index=self._publish_market_event)
        self.factor_source = factor_source

        if config.alpaca_url is None:
            self.stream = AlpacaSIPStream(
                symbols=config.symbols,
                quotes=True,
                trades=True,
                bars=True,
                on_quote=self._publish_market_event,
                on_trade=self._publish_market_event,
                on_bar=self._publish_market_event,
                on_state=self._stream_state_changed,
            )
        else:
            self.stream = AlpacaSIPStream(
                symbols=config.symbols,
                quotes=True,
                trades=True,
                bars=True,
                url=config.alpaca_url,
                on_quote=self._publish_market_event,
                on_trade=self._publish_market_event,
                on_bar=self._publish_market_event,
                on_state=self._stream_state_changed,
            )

        self.router = CandidateRouter(
            order_manager=self.order_manager,
            entry_quantity_policy=quantity_policy,
            exit_quantity_provider=exit_quantity_provider,
            execution_policy=execution_policy,
            strategy_state=self.strategy_state,
        )

    async def run(self) -> None:
        """Start archive, consumer, market stream, and optional factor source."""
        self.state.transition(EngineState.RECONCILING, "startup broker reconciliation")
        self.journal.append("engine_state", time.time_ns(), {"state": "reconciling"})
        reconcile = getattr(self.venue, "startup_reconcile", None)
        if not callable(reconcile):
            self.kill_switch("execution venue lacks startup reconciliation")
            raise EngineConfigurationError("execution venue lacks startup reconciliation")
        try:
            await reconcile()
        except Exception as exc:
            self.kill_switch(f"startup reconciliation failed: {exc}")
            raise
        self.state.transition(EngineState.READY, "broker state agrees")
        self.archive.start()
        self.state.transition(EngineState.RUNNING, "all startup gates passed")

        consumer_task = asyncio.create_task(
            self._consume_market_events(),
            name="market-engine-consumer",
        )
        stream_task = asyncio.create_task(
            self.stream.run_forever(),
            name="alpaca-sip-stream",
        )
        reconciliation_task = asyncio.create_task(
            self._reconciliation_loop(), name="broker-reconciliation"
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

        tasks = [consumer_task, stream_task, reconciliation_task]
        if factor_task is not None:
            tasks.append(factor_task)

        stop_task = asyncio.create_task(
            self._stop.wait(),
            name="market-engine-stop-waiter",
        )

        try:
            done, pending = await asyncio.wait(
                [*tasks, stop_task],
                return_when=asyncio.FIRST_COMPLETED,
            )

            # Propagate failures from engine components. The stop waiter is
            # only a lifecycle signal and cannot itself fail normally.
            for task in done:
                if task is stop_task:
                    continue
                exc = task.exception()
                if exc is not None:
                    raise exc

            # A component returning normally without a requested shutdown is
            # still an unexpected engine termination.
            component_finished = any(
                task is not stop_task for task in done
            )
            if component_finished and not self._stop.is_set():
                raise RuntimeError("engine component stopped unexpectedly")
        finally:
            await self.stop()

            if not stop_task.done():
                stop_task.cancel()

            for task in tasks:
                if not task.done():
                    task.cancel()

            await asyncio.gather(
                *tasks,
                stop_task,
                return_exceptions=True,
            )

            self.archive.close(drain=True)
            self.journal.close()

    async def stop(self) -> None:
        if self._stop.is_set():
            return

        if self.state.state not in {EngineState.KILLED, EngineState.STOPPING}:
            self.state.transition(EngineState.STOPPING, "stop requested")
        elif self.state.state is EngineState.KILLED:
            self.state.transition(EngineState.STOPPING, "killed engine stopping")
        self._stop.set()
        await self.stream.stop()
        self.state.transition(EngineState.STOPPED, "stopped")

    def kill_switch(self, reason: str) -> None:
        self.state.kill(reason)
        risk = getattr(self.order_manager, "_risk_engine", None)
        native_engine = getattr(risk, "engine", None)
        if native_engine is not None:
            native_engine.activate_kill_switch()
        self.journal.append("kill_switch", time.time_ns(), {"reason": reason})

    async def reconcile_execution(
        self,
        update: ExecutionUpdate,
    ) -> None:
        """Entry point for authoritative broker/order-book execution updates."""
        order = await self.order_manager.reconcile(update)
        if self.portfolio is not None:
            self.portfolio.apply_execution(order, update)
        self.strategy_state.reconcile(order, update)
        self.strategies.on_position(
            StrategyName(order.intent.strategy_id),
            order.intent.symbol,
            self.strategy_state.quantity(order.intent.strategy_id, order.intent.symbol),
        )
        self.journal.append("execution_update", update.event_ns, update)

    async def _reconciliation_loop(self) -> None:
        reconcile_status = getattr(self.venue, "reconcile_status", None)
        reconcile_snapshot = getattr(self.venue, "startup_reconcile", None)
        if not callable(reconcile_status):
            self.kill_switch("execution venue lacks runtime reconciliation")
            raise RuntimeError("execution venue lacks runtime reconciliation")
        while not self._stop.is_set():
            try:
                for order in self.order_manager.open_orders():
                    if order.venue_order_id is None:
                        continue
                    update = await reconcile_status(order.venue_order_id)
                    await self.reconcile_execution(update)
                if not callable(reconcile_snapshot):
                    raise RuntimeError("execution venue lacks broker snapshot reconciliation")
                await reconcile_snapshot()
                if self.metrics is not None:
                    metrics = self.metrics.snapshot()
                    self.journal.append("portfolio_metrics", metrics.timestamp_ns, metrics)
            except Exception as exc:
                self.kill_switch(f"runtime reconciliation failed: {exc}")
                raise
            await asyncio.sleep(self.config.reconciliation_interval_seconds)

    async def _publish_market_event(
        self,
        event: MarketEvent,
    ) -> None:
        # Persistence and every analytics consumer have independent streams.
        self.archive.submit(event)
        self.market_gate.observe(
            event.symbol, event.timestamp_ns, is_bar=isinstance(event, BarEvent)
        )
        self.journal.append("market_event", event.timestamp_ns, event)
        state = getattr(self._adapter_module, "STATE", None)
        if state is not None:
            state.on_market_event(event)
        self.event_bus.publish(event)

    async def _consume_market_events(self) -> None:
        while not self._stop.is_set():
            event = await self.strategy_events.queue.get()

            try:
                self.health.market_events_processed += 1

                if isinstance(event, BarEvent):
                    await self._process_bar(event)

                # Quotes/trades/factors remain available to other live
                # analytics through future fan-out subscribers. Strategy
                # definitions in strategies.py are bar-based.
            finally:
                self.strategy_events.queue.task_done()

    async def _process_bar(self, event: BarEvent) -> None:
        self.state.require_ordering_enabled()
        self.market_gate.require_fresh(event.symbol, event.received_ns)
        self.session_gate.require_open(event.timestamp_ns)
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
            submitted, conflicted = await self.router.route(batch.candidates)
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
        if self.portfolio is not None:
            self.portfolio.track_order(order)
        self.strategy_state.on_order(order)
        self.journal.append(
            "order_event",
            max(order.last_event_ns, time.time_ns()),
            {
                "event": event,
                "client_order_id": order.client_order_id,
                "status": order.status.value,
                "symbol": order.intent.symbol,
                "strategy_id": order.intent.strategy_id,
            },
        )
        if order.status in TERMINAL_STATUSES or order.status.value == "error":
            state = getattr(self._adapter_module, "STATE", None)
            if state is not None:
                state.reservations.release(order.client_order_id)

    async def _stream_state_changed(self, state) -> None:
        LOGGER.info("Alpaca SIP state=%s", state.value)


def _csv_symbols(value: str) -> tuple[str, ...]:
    symbols = tuple(part.strip().upper() for part in value.split(",") if part.strip())
    if not symbols:
        raise argparse.ArgumentTypeError("at least one symbol is required")
    return symbols


def _csv_ints(value: str) -> tuple[int, ...]:
    try:
        values = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from exc
    if not values:
        raise argparse.ArgumentTypeError("at least one integer is required")
    return values


def config_from_args() -> EngineConfig:
    parser = argparse.ArgumentParser(description="Activate the complete market engine.")

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
        help=("Python/pybind11 module exposing build_risk_engine() and build_execution_venue()."),
    )
    parser.add_argument("--journal-path", required=True)
    parser.add_argument("--max-feed-age-ns", required=True, type=int)
    parser.add_argument("--max-bar-gap-ns", required=True, type=int)
    parser.add_argument("--reconciliation-interval-seconds", required=True, type=float)
    parser.add_argument("--session-timezone", required=True)
    parser.add_argument("--session-open-minute", required=True, type=int)
    parser.add_argument("--session-close-minute", required=True, type=int)
    parser.add_argument("--session-weekdays", required=True, type=_csv_ints)
    parser.add_argument(
        "--mongodb-uri",
        default=os.environ.get("CENGINE_MONGODB_URI", ""),
        help="MongoDB URI; prefer CENGINE_MONGODB_URI to avoid shell history",
    )
    parser.add_argument(
        "--mongodb-database",
        default=os.environ.get("CENGINE_MONGODB_DATABASE", ""),
    )
    parser.add_argument("--mongodb-timeout-ms", required=True, type=int)

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
        journal_path=args.journal_path,
        max_feed_age_ns=args.max_feed_age_ns,
        max_bar_gap_ns=args.max_bar_gap_ns,
        reconciliation_interval_seconds=args.reconciliation_interval_seconds,
        session_timezone=args.session_timezone,
        session_open_minute=args.session_open_minute,
        session_close_minute=args.session_close_minute,
        session_weekdays=args.session_weekdays,
        mongodb_uri=args.mongodb_uri,
        mongodb_database=args.mongodb_database,
        mongodb_timeout_ms=args.mongodb_timeout_ms,
        alpaca_url=args.alpaca_url,
    )


async def run_engine(
    *,
    config: EngineConfig,
    quantity_policy: QuantityPolicy,
    exit_quantity_provider: ExitQuantityProvider,
    execution_policy: ExecutionPolicy,
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
        execution_policy=execution_policy,
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
