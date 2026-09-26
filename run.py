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
from collections import deque
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Optional, Protocol

from cengine.engine_state import EngineState, EngineStateMachine
from cengine.event_bus import MarketEventBus
from cengine.execution_policy import ConfiguredExecutionPolicy, ExecutionPolicy
from cengine.integrity import verify_source_integrity
from cengine.journal import AuditJournal
from cengine.metrics import PortfolioMetrics, PortfolioMetricsCollector
from cengine.nosql_journal import MongoDailyJournalStore
from cengine.portfolio import AccountState, PositionBook
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
    ManagedOrder,
    NativeRiskEngine,
    OrderIntent,
    OrderManager,
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


@dataclass(frozen=True, slots=True)
class DecisionReport:
    """Read-only snapshot of one bar's strategy decision and its routing outcome."""

    symbol: str
    timestamp_ns: int
    candidates: tuple[TradeCandidate, ...]
    submitted: int
    conflicted: bool


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
        bar_observer: Optional[Callable[["Bar"], None]] = None,
        on_decision: Optional[Callable[["DecisionReport"], None]] = None,
        on_metrics: Optional[Callable[[PortfolioMetrics], None]] = None,
        on_order_event: Optional[Callable[[str, ManagedOrder], None]] = None,
    ) -> None:
        config.validate()
        self.config = config
        self.health = EngineHealth()
        self._stop = asyncio.Event()
        # Observe-only telemetry hooks. They never influence routing, sizing,
        # risk, or execution; they only surface what the engine already decided.
        self._bar_observer = bar_observer
        self._on_decision = on_decision
        self._on_metrics = on_metrics
        self._on_order_event = on_order_event
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
                    if self._on_metrics is not None:
                        self._on_metrics(metrics)
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
        # SIP multiplexes independent channels; the gate enforces monotonicity
        # per (symbol, channel), never across channels (quotes and trades
        # interleave with non-monotonic cross-channel timestamps). Index factor
        # observations (VIX/VXN) get their own channel and are never executable.
        if isinstance(event, BarEvent):
            channel = "bar"
        elif isinstance(event, QuoteEvent):
            channel = "quote"
        elif isinstance(event, TradeEvent):
            channel = "trade"
        else:
            channel = "index"
        self.market_gate.observe(event.symbol, event.timestamp_ns, channel=channel)
        # High-frequency quote/trade ticks persist to HDF5 (the archive) and
        # drive the gate/strategies, but are NOT written to the audit journal:
        # journaling every tick performs a synchronous fsync + majority-write
        # Atlas round-trip on the event loop and saturates it at live SIP rates.
        # The audit journal records decision-relevant events only (bars, index
        # factors, orders, executions, portfolio metrics, engine state).
        if channel not in ("quote", "trade"):
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

        # Feed observers (e.g. an ATR estimator used by the sizing policy) with
        # the completed bar before strategies act on it. This is read-only
        # telemetry input; it never alters strategy or risk behaviour.
        if self._bar_observer is not None:
            self._bar_observer(bar)

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

        if self._on_decision is not None:
            self._on_decision(
                DecisionReport(
                    symbol=batch.symbol,
                    timestamp_ns=batch.timestamp_ns,
                    candidates=batch.candidates,
                    submitted=submitted,
                    conflicted=conflicted,
                )
            )

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
        if self._on_order_event is not None:
            self._on_order_event(event, order)
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


class ATRTracker:
    """Wilder Average True Range in price ticks, from completed bars only.

    True range and its running average are computed strictly from bars already
    delivered to :meth:`observe`. No future information is used, so the value
    read while sizing a signal bar reflects volatility up to and including that
    closed bar.
    """

    def __init__(self, period: int) -> None:
        if period < 1:
            raise ValueError("ATR period must be positive")
        self._period = period
        self._prev_close: dict[str, int] = {}
        self._window: dict[str, deque[int]] = {}
        self._atr: dict[str, float] = {}

    def observe(self, bar: "Bar") -> None:
        symbol = bar.symbol.strip().upper()
        prev_close = self._prev_close.get(symbol)
        if prev_close is None:
            true_range = bar.high_ticks - bar.low_ticks
        else:
            true_range = max(
                bar.high_ticks - bar.low_ticks,
                abs(bar.high_ticks - prev_close),
                abs(bar.low_ticks - prev_close),
            )
        self._prev_close[symbol] = bar.close_ticks

        if symbol in self._atr:
            # Wilder smoothing once the seed average exists.
            self._atr[symbol] = (
                self._atr[symbol] * (self._period - 1) + true_range
            ) / self._period
            return

        window = self._window.setdefault(symbol, deque(maxlen=self._period))
        window.append(true_range)
        if len(window) == self._period:
            self._atr[symbol] = sum(window) / self._period

    def atr_ticks(self, symbol: str) -> Optional[int]:
        value = self._atr.get(symbol.strip().upper())
        if value is None:
            return None
        return max(1, int(value))


class VolatilityScaledQuantityPolicy:
    """Proposes entry quantity from a volatility (ATR) risk budget.

    quantity = floor((equity * risk_fraction) / ATR_ticks)

    The proposal is bounded by current buying power so it is fundable, then
    handed to the native C++ RiskEngine, which independently approves, reduces,
    or rejects it. This class proposes; it never approves. When volatility or
    account equity is not yet observable it falls back to a small, explicit
    floor rather than fabricating a volatility estimate or returning zero
    (zero would abort the batch).
    """

    def __init__(
        self,
        *,
        account: AccountState,
        atr: ATRTracker,
        risk_fraction: float,
        fallback_shares: int,
    ) -> None:
        if not 0.0 < risk_fraction < 1.0:
            raise ValueError("risk_fraction must be in (0, 1)")
        if fallback_shares < 1:
            raise ValueError("fallback_shares must be >= 1")
        self._account = account
        self._atr = atr
        self._risk_fraction = risk_fraction
        self._fallback = fallback_shares

    def quantity_for(self, candidate: TradeCandidate) -> int:
        atr_ticks = self._atr.atr_ticks(candidate.symbol)
        try:
            account = self._account.snapshot()
        except RuntimeError:
            account = None

        if atr_ticks is None or account is None or account.equity_ticks <= 0:
            LOGGER.warning(
                "volatility sizing unavailable symbol=%s atr_ticks=%s "
                "equity_known=%s; proposing fallback=%d",
                candidate.symbol,
                atr_ticks,
                account is not None,
                self._fallback,
            )
            return self._fallback

        risk_budget_ticks = int(account.equity_ticks * self._risk_fraction)
        atr_quantity = max(1, risk_budget_ticks // atr_ticks)

        price_ticks = candidate.reference_price_ticks
        if price_ticks > 0 and account.buying_power_ticks > 0:
            affordable = account.buying_power_ticks // price_ticks
            if affordable >= 1:
                return min(atr_quantity, affordable)

        # Native risk remains authoritative and will bound an unaffordable
        # proposal; we still never propose a non-positive quantity.
        return atr_quantity


class StrategyOwnedExitProvider:
    """Closes exactly the quantity the exiting strategy currently owns.

    Ownership is fill-derived from the authoritative :class:`PositionBook`. No
    quantity is invented; if the strategy owns nothing the returned zero fails
    the batch closed, consistent with the engine's fail-closed reconciliation.
    """

    def __init__(self, positions: PositionBook) -> None:
        self._positions = positions

    def quantity_to_close(self, candidate: TradeCandidate) -> int:
        owned = self._positions.quantity(candidate.symbol, candidate.strategy.value)
        return abs(owned)


def _fmt_usd(ticks: int) -> str:
    """Ticks are integer cents (see AlpacaExecutionVenue._price_to_ticks)."""
    return f"${ticks / 100:,.2f}"


class ConsoleDecisionStream:
    """Human-readable live stream of decisions, orders, and portfolio metrics.

    Pure telemetry: it renders what the engine already decided. It holds no
    trading authority and mutates no engine state.
    """

    def __init__(self, stream=None) -> None:
        import sys

        self._out = stream if stream is not None else sys.stdout

    def _emit(self, tag: str, message: str) -> None:
        stamp = time.strftime("%H:%M:%S")
        print(f"{stamp} {tag:<8} {message}", file=self._out, flush=True)

    def banner(self, config: EngineConfig, *, live: bool) -> None:
        mode = "LIVE" if live else "PAPER"
        self._emit("ENGINE", f"starting [{mode}] symbols={','.join(config.symbols)}")
        self._emit(
            "ENGINE",
            f"session={config.session_timezone} "
            f"{config.session_open_minute}->{config.session_close_minute} "
            f"reconcile={config.reconciliation_interval_seconds:g}s "
            f"adapter={config.native_adapter_module}",
        )

    def on_decision(self, report: DecisionReport) -> None:
        for candidate in report.candidates:
            self._emit(
                "DECISION",
                f"{candidate.symbol:<6} {candidate.strategy.value:<18} "
                f"{candidate.side.value:<10} ref={_fmt_usd(candidate.reference_price_ticks)} "
                f":: {candidate.reason}",
            )
        outcome = (
            "CONFLICT (batch withheld)"
            if report.conflicted
            else f"submitted={report.submitted}"
        )
        self._emit(
            "ROUTE",
            f"{report.symbol:<6} candidates={len(report.candidates)} {outcome}",
        )

    def on_order_event(self, event: str, order: ManagedOrder) -> None:
        self._emit(
            "ORDER",
            f"{event:<10} {order.intent.symbol:<6} {order.intent.strategy_id:<18} "
            f"{order.intent.side.value:<4} qty={order.intent.quantity} "
            f"approved={order.approved_quantity} filled={order.cumulative_filled_quantity} "
            f"status={order.status.value} venue={order.venue_order_id}",
        )

    def on_metrics(self, metrics: PortfolioMetrics) -> None:
        self._emit(
            "METRICS",
            f"equity={_fmt_usd(metrics.equity_ticks)} "
            f"cash={_fmt_usd(metrics.cash_ticks)} "
            f"buying_power={_fmt_usd(metrics.buying_power_ticks)} "
            f"net_exp={_fmt_usd(metrics.net_exposure_ticks)} "
            f"gross_exp={_fmt_usd(metrics.gross_exposure_ticks)} "
            f"intraday_pnl={_fmt_usd(metrics.intraday_pnl_ticks)} "
            f"positions={metrics.position_count} open_orders={metrics.open_order_count} "
            f"drawdown={metrics.drawdown_bps}bps",
        )


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


# Executable equity universe: the NBBO order-book symbols only. VIX/VXN enter
# the engine as NormalizedIndex volatility factors through the IndexFactorBridge
# and are never treated as tradeable equities (see the module authority note).
DEFAULT_SYMBOLS: tuple[str, ...] = ("QQQ", "SQQQ")


def _load_env_file(path: str) -> None:
    """Populate os.environ from a KEY=VALUE file without overriding real env."""
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


def _mirror_env(primary: str, secondary: str) -> None:
    """Ensure two env names that carry the same secret share a value."""
    if os.environ.get(primary) and not os.environ.get(secondary):
        os.environ[secondary] = os.environ[primary]
    elif os.environ.get(secondary) and not os.environ.get(primary):
        os.environ[primary] = os.environ[secondary]


def load_runtime_env() -> None:
    """Load local credential files and reconcile equivalent variable names.

    Real process environment always wins over file contents. Alpaca's market
    data stream reads APCA_API_KEY_ID/APCA_API_SECRET_KEY while the trading
    venue reads ALPACA_API_KEY/ALPACA_SECRET_KEY; either pair is accepted and
    mirrored to the other. Atlas onboarding writes MONGODB_URI, which is
    mirrored to the engine's CENGINE_MONGODB_URI.
    """
    for path in (".env", "atlas-credentials.env"):
        _load_env_file(path)
    _mirror_env("ALPACA_API_KEY", "APCA_API_KEY_ID")
    _mirror_env("ALPACA_SECRET_KEY", "APCA_API_SECRET_KEY")
    _mirror_env("MONGODB_URI", "CENGINE_MONGODB_URI")


def _env_str(name: str, default: str) -> str:
    value = os.environ.get(name)
    return value if value is not None and value.strip() else default


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return int(value) if value is not None and value.strip() else default


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    return float(value) if value is not None and value.strip() else default


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


_ALPACA_FEED_PATHS: dict[str, str] = {
    "sip": "v2/sip",
    "iex": "v2/iex",
    "delayed_sip": "v2/delayed_sip",
    "boats": "v1beta1/boats",
    "overnight": "v1beta1/overnight",
}


def _alpaca_stream_url() -> Optional[str]:
    """Resolve the market-data WebSocket URL from the environment.

    ``CENGINE_ALPACA_URL`` (an explicit full URL) always wins. Otherwise
    ``CENGINE_ALPACA_FEED`` selects a feed (sip|iex|delayed_sip|boats|overnight)
    on the live or sandbox host (``CENGINE_ALPACA_SANDBOX=1``). Returns None to
    let the stream use its configured SIP default. A feed not covered by the
    account's data subscription is rejected by Alpaca at authentication.
    """
    explicit = os.environ.get("CENGINE_ALPACA_URL")
    if explicit and explicit.strip():
        return explicit.strip()
    feed = _env_str("CENGINE_ALPACA_FEED", "").strip().lower()
    if not feed:
        return None
    if feed not in _ALPACA_FEED_PATHS:
        raise EngineConfigurationError(
            f"unknown CENGINE_ALPACA_FEED {feed!r}; expected one of "
            f"{sorted(_ALPACA_FEED_PATHS)}"
        )
    sandbox = _env_str("CENGINE_ALPACA_SANDBOX", "").strip().lower() in {"1", "true", "yes"}
    host = "stream.data.sandbox.alpaca.markets" if sandbox else "stream.data.alpaca.markets"
    return f"wss://{host}/{_ALPACA_FEED_PATHS[feed]}"


def config_from_env() -> EngineConfig:
    """Build an EngineConfig from environment variables with operational defaults.

    Only *operational* defaults are supplied here (paths, session hours, feed
    tolerances, strategy periods). Trading sizing is never defaulted in the
    config; it is an injected policy. Every value is overridable via env.
    """
    symbols_raw = os.environ.get("CENGINE_SYMBOLS", "")
    symbols = _csv_symbols(symbols_raw) if symbols_raw.strip() else DEFAULT_SYMBOLS

    return EngineConfig(
        symbols=symbols,
        market_data_root=_env_str("CENGINE_MARKET_DATA_ROOT", "./state/market_data"),
        archive_provider_label=_env_str("CENGINE_ARCHIVE_PROVIDER_LABEL", "alpaca-sip"),
        turtle_entry_lookback=_env_int("CENGINE_TURTLE_ENTRY_LOOKBACK", 20),
        turtle_exit_lookback=_env_int("CENGINE_TURTLE_EXIT_LOOKBACK", 10),
        dual_ma_fast_period=_env_int("CENGINE_DUAL_MA_FAST_PERIOD", 20),
        dual_ma_slow_period=_env_int("CENGINE_DUAL_MA_SLOW_PERIOD", 50),
        apo_fast_period=_env_int("CENGINE_APO_FAST_PERIOD", 12),
        apo_slow_period=_env_int("CENGINE_APO_SLOW_PERIOD", 26),
        apo_entry_threshold_ticks=_env_float("CENGINE_APO_ENTRY_THRESHOLD_TICKS", 25.0),
        apo_exit_threshold_ticks=_env_float("CENGINE_APO_EXIT_THRESHOLD_TICKS", 5.0),
        native_adapter_module=_env_str("ENGINE_NATIVE_ADAPTER_MODULE", "native_adapters"),
        journal_path=_env_str("CENGINE_JOURNAL_PATH", "./state/audit.jsonl"),
        # Minute bars are timestamped at bar-open and delivered after bar-close,
        # so feed age must tolerate more than one bar interval. Bar-gap tolerance
        # absorbs low-liquidity minutes without silently accepting a dead feed.
        max_feed_age_ns=_env_int("CENGINE_MAX_FEED_AGE_NS", 180_000_000_000),
        max_bar_gap_ns=_env_int("CENGINE_MAX_BAR_GAP_NS", 900_000_000_000),
        reconciliation_interval_seconds=_env_float(
            "CENGINE_RECONCILIATION_INTERVAL_SECONDS", 15.0
        ),
        session_timezone=_env_str("CENGINE_SESSION_TIMEZONE", "America/New_York"),
        session_open_minute=_env_int("CENGINE_SESSION_OPEN_MINUTE", 570),
        session_close_minute=_env_int("CENGINE_SESSION_CLOSE_MINUTE", 960),
        session_weekdays=_csv_ints(_env_str("CENGINE_SESSION_WEEKDAYS", "0,1,2,3,4")),
        mongodb_uri=os.environ.get("CENGINE_MONGODB_URI", ""),
        mongodb_database=_env_str("CENGINE_MONGODB_DATABASE", "cengine"),
        mongodb_timeout_ms=_env_int("CENGINE_MONGODB_TIMEOUT_MS", 5000),
        alpaca_url=_alpaca_stream_url(),
    )


async def run_engine(
    *,
    config: EngineConfig,
    quantity_policy: QuantityPolicy,
    exit_quantity_provider: ExitQuantityProvider,
    execution_policy: ExecutionPolicy,
    factor_source: Optional[ExternalFactorSource] = None,
    bar_observer: Optional[Callable[["Bar"], None]] = None,
    on_decision: Optional[Callable[["DecisionReport"], None]] = None,
    on_metrics: Optional[Callable[[PortfolioMetrics], None]] = None,
    on_order_event: Optional[Callable[[str, ManagedOrder], None]] = None,
) -> None:
    """Programmatic activation entry point.

    Position sizing and exit inventory lookup are mandatory injected policies
    because this bootstrap must not invent quantities. The optional observer
    hooks are read-only telemetry and never influence engine decisions.
    """
    engine = MarketEngine(
        config=config,
        quantity_policy=quantity_policy,
        exit_quantity_provider=exit_quantity_provider,
        execution_policy=execution_policy,
        factor_source=factor_source,
        bar_observer=bar_observer,
        on_decision=on_decision,
        on_metrics=on_metrics,
        on_order_event=on_order_event,
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
    """No-argument composition root: run the paper engine and stream live.

    Reads configuration from the environment (:func:`config_from_env`) with safe
    operational defaults, injects an explicit volatility-scaled sizing policy
    and a strategy-owned exit provider, and streams decisions, order events, and
    portfolio metrics to stdout.

    This is a composition root, not the engine inventing anything: sizing is a
    proposal that the native C++ RiskEngine independently approves, reduces, or
    rejects, and exit quantity is the strategy's own fill-derived inventory.
    Paper trading is the enforced default; the live endpoint stays gated.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    load_runtime_env()

    # Source-integrity gate: refuse to run stale/divergent copies of the engine's own
    # code. Always warns if a critical module resolves from an installed shadow rather
    # than the repo; in strict mode (CENGINE_STRICT_SOURCE_INTEGRITY=1, for LIVE) it
    # additionally requires every source file to match its committed git blob and
    # fails closed otherwise.
    verify_source_integrity(strict=_env_bool("CENGINE_STRICT_SOURCE_INTEGRITY"))

    config = config_from_env()
    config.validate()

    missing = [
        name for name in ("ALPACA_API_KEY", "ALPACA_SECRET_KEY") if not os.environ.get(name)
    ]
    if missing:
        raise EngineConfigurationError(
            "Alpaca paper credentials are required: set "
            + ", ".join(missing)
            + ". Copy .env.example to .env and fill it, or export them. "
            "(APCA_API_KEY_ID / APCA_API_SECRET_KEY are accepted equivalently.)"
        )
    if not config.mongodb_uri.strip():
        raise EngineConfigurationError(
            "MongoDB journaling is required: set CENGINE_MONGODB_URI, or provide "
            "MONGODB_URI via atlas-credentials.env."
        )

    # Validate the native authority boundary and obtain authoritative state.
    load_native_adapters(config.native_adapter_module)
    adapter = importlib.import_module(config.native_adapter_module)
    state = getattr(adapter, "STATE", None)
    if state is None:
        raise EngineConfigurationError(
            f"native adapter {config.native_adapter_module!r} must expose STATE "
            "with account/positions for the no-argument composition root"
        )

    risk_fraction = _env_float("CENGINE_RISK_FRACTION", 0.02)
    atr_period = _env_int("CENGINE_ATR_PERIOD", 14)
    fallback_shares = _env_int("CENGINE_ATR_FALLBACK_SHARES", 1)

    atr = ATRTracker(atr_period)
    quantity_policy = VolatilityScaledQuantityPolicy(
        account=state.account,
        atr=atr,
        risk_fraction=risk_fraction,
        fallback_shares=fallback_shares,
    )
    exit_provider = StrategyOwnedExitProvider(state.positions)
    execution_policy = ConfiguredExecutionPolicy(OrderType.MARKET, TimeInForce.DAY)

    live = (
        os.environ.get("ALPACA_TRADING_BASE_URL", "").rstrip("/")
        == "https://api.alpaca.markets"
    )
    stream = ConsoleDecisionStream()
    stream.banner(config, live=live)
    LOGGER.info(
        "sizing=volatility-scaled risk_fraction=%.4f atr_period=%d fallback_shares=%d",
        risk_fraction,
        atr_period,
        fallback_shares,
    )

    try:
        asyncio.run(
            run_engine(
                config=config,
                quantity_policy=quantity_policy,
                exit_quantity_provider=exit_provider,
                execution_policy=execution_policy,
                bar_observer=atr.observe,
                on_decision=stream.on_decision,
                on_metrics=stream.on_metrics,
                on_order_event=stream.on_order_event,
            )
        )
    except KeyboardInterrupt:
        LOGGER.info("interrupted; shutting down")


if __name__ == "__main__":
    main()
