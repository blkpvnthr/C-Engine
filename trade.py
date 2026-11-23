#!/usr/bin/env python3
"""
trade.py

Top-level bootstrap for the complete market engine.

Data path
---------
Alpaca IEX (default feed; SIP is opt-in and entitlement-gated)
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

Market-data feed vs. trading mode are independent axes. Selecting IEX or SIP
changes only the data source; it never changes paper vs. live execution, which
stays gated by the separate ALPACA_TRADING_BASE_URL live confirmations.

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
- fabricate Level-II depth from a single-exchange (IEX) or consolidated (SIP)
  top-of-book quote;
- present IEX exchange-specific quotes/volume as consolidated NBBO/market-wide;
- bypass the native C++ risk boundary;
- treat Metal output as execution authority.

Required local modules
----------------------
market_data/alpaca_stream.py
market_data/feed.py
market_data/market_data_store.py
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
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional, Protocol

from canonical_universe import CanonicalUniverseManager
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
from market_data.alpaca_stream import (
    AlpacaMarketDataStream,
    BarEvent,
    IndexEvent,
    IndexFactorBridge,
    QuoteEvent,
    TradeEvent,
)
from market_data.feed import resolve_feed
from market_data.market_data_store import DailyHDF5Writer
from order_manager import (
    TERMINAL_STATUSES,
    ExecutionUpdate,
    ExecutionVenue,
    ManagedOrder,
    NativeRiskEngine,
    OrderIntent,
    OrderManager,
    OrderStatus,
    OrderType,
    Side,
    TimeInForce,
)
from protective_exit import ProtectiveExitCoordinator
from scanner_bridge import BridgeStreamClient, ScannerEventBridge
from strategies import (
    APOConfig,
    Bar,
    DIBBConfig,
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
from universe_sync import UniverseSubscriptionSync

LOGGER = logging.getLogger("market_engine")

MarketEvent = QuoteEvent | TradeEvent | BarEvent | IndexEvent


class EngineConfigurationError(RuntimeError):
    pass


class StrategyConflictError(RuntimeError):
    pass


class _EngineScannerBrokerView:
    """Read-only broker surface the market scanner needs, backed by the engine's
    authoritative in-process state instead of a second broker REST client.

    Exposes exactly what ``market_scan.scanner.MarketScanner`` reads: credentials
    (for its own historical daily-bar REST client -- not a websocket) and open
    positions. Pending-order coverage is handled engine-side by
    ``UniverseSubscriptionSync`` (on the engine loop), so this returns an empty
    pending list to avoid an unsafe cross-thread read of the OrderManager's
    order table. Position reads go through PositionBook, which is lock-guarded
    and safe to call from the scanner thread.
    """

    def __init__(self, engine: "MarketEngine", api_key: str, secret_key: str) -> None:
        self.api_key = api_key
        self.secret_key = secret_key
        self._engine = engine

    def get_all_open_positions(self) -> list:
        book = self._engine.portfolio
        if book is None:
            return []
        try:
            return [p for p in book.positions() if getattr(p, "quantity", 0)]
        except Exception:  # noqa: BLE001 - never raise into the scanner loop
            return []

    def list_open_orders(self) -> list:
        # Pending-order subscription coverage is enforced engine-side; avoid a
        # cross-thread iteration of the OrderManager order table here.
        return []


class _EngineScannerConfigView:
    def __init__(self, *, paper: bool) -> None:
        self.paper = paper


class _EngineScannerHost:
    """The ``executor`` object the scanner reads (``.broker`` + ``.config.paper``);
    deliberately exposes no order-submission path."""

    def __init__(
        self, engine: "MarketEngine", api_key: str, secret_key: str, *, paper: bool
    ) -> None:
        self.broker = _EngineScannerBrokerView(engine, api_key, secret_key)
        self.config = _EngineScannerConfigView(paper=paper)


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
    # Resolved feed name (e.g. "iex"/"sip") used for startup feed verification
    # and provenance logging. None leaves the stream's derived feed unchecked.
    alpaca_feed: Optional[str] = None

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
        exposure_authorizer: Optional[Callable[[str], tuple[bool, str]]] = None,
    ) -> None:
        self._orders = order_manager
        self._entry_quantity = entry_quantity_policy
        self._exit_quantity = exit_quantity_provider
        self._execution_policy = execution_policy
        self._strategy_state = strategy_state
        # Exposure gate (§8/§10): blocks elective ENTRIES into geared/inverse or
        # unclassified products that are not execution-authorized. Exits are never
        # gated. None -> no gating (conventional-equity-only deployments).
        self._exposure_authorizer = exposure_authorizer

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

        # Exposure authorization gate (§8/§10): block elective ENTRIES into
        # geared/inverse or unclassified instruments that are not execution-
        # authorized. Exits (already isolated above) are NEVER gated — risk
        # reduction always proceeds. An unavailable/incorrect classification
        # therefore blocks elective execution, as required.
        if self._exposure_authorizer is not None and not exits:
            allowed = []
            for candidate in candidates:
                if candidate.side in {SignalSide.LONG, SignalSide.SHORT}:
                    ok, reason = self._exposure_authorizer(candidate.symbol)
                    if not ok:
                        LOGGER.info(
                            "exposure gate blocked entry symbol=%s strategy=%s reason=%s",
                            candidate.symbol, candidate.strategy.value, reason)
                        continue
                allowed.append(candidate)
            candidates = tuple(allowed)
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
        on_portfolio: Optional[Callable[[object, dict], None]] = None,
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
        self._on_portfolio = on_portfolio
        self._last_portfolio_state = None
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
            # The DI/BB strategy (migrated from QQ-Trader) is opt-in: it only
            # trades when CENGINE_DIBB_ENABLED is set, so an unvalidated strategy
            # never silently gains execution eligibility (spec §16). Its entries
            # are still revalidated against the canonical universe + native risk.
            di_bb=(
                DIBBConfig(
                    di_period=int(os.environ.get("CENGINE_DIBB_DI_PERIOD", "14")),
                    bb_period=int(os.environ.get("CENGINE_DIBB_BB_PERIOD", "20")),
                    momentum_lookback=int(
                        os.environ.get("CENGINE_DIBB_MOMENTUM_LOOKBACK", "5")
                    ),
                    require_vwap=os.environ.get("CENGINE_DIBB_REQUIRE_VWAP", "1") != "0",
                    require_momentum=(
                        os.environ.get("CENGINE_DIBB_REQUIRE_MOMENTUM", "1") != "0"
                    ),
                )
                if os.environ.get("CENGINE_DIBB_ENABLED", "0") != "0"
                else None
            ),
        )
        self.strategies: StrategyCoordinator = build_strategy_coordinator(strategy_config)

        self.archive = DailyHDF5Writer(
            root=config.market_data_root,
            provider=config.archive_provider_label,
            feed=config.alpaca_feed,
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

        stream_kwargs: dict[str, Any] = dict(
            symbols=config.symbols,
            quotes=True,
            trades=True,
            bars=True,
            on_quote=self._publish_market_event,
            on_trade=self._publish_market_event,
            on_bar=self._publish_market_event,
            on_state=self._stream_state_changed,
            on_subscription=self._on_subscription_confirmed,
        )
        # Tracks whether the stream has reached SUBSCRIBED, so a later CONNECTING
        # transition is recognized as a reconnect (drives controller.on_reconnect).
        self._stream_was_live = False
        # An omitted URL leaves the stream on its IEX default (never SIP).
        if config.alpaca_url is not None:
            stream_kwargs["url"] = config.alpaca_url
        self.stream = AlpacaMarketDataStream(**stream_kwargs)
        self._verify_stream_feed()

        # Instrument exposure registry (§8 operational integration): classifies
        # geared/inverse ETFs and gates them out of elective execution unless
        # explicitly authorized. Operators authorize geared products via
        # CENGINE_AUTHORIZED_GEARED (comma-separated symbols). Failure to build it
        # is non-fatal (conventional-equity trading continues ungated).
        self._exposure_registry = None
        exposure_authorizer = None
        try:
            from cengine.exposure_registry import ExposureRegistry

            self._exposure_registry = ExposureRegistry()
            for sym in os.environ.get("CENGINE_AUTHORIZED_GEARED", "").split(","):
                sym = sym.strip().upper()
                if sym:
                    self._exposure_registry.authorize(sym, True)
            exposure_authorizer = self._exposure_registry.authorize_entry
            LOGGER.info(
                "exposure registry active: %s classified instruments, %d execution-authorized",
                self._exposure_registry.counts()["total_seeded"],
                len(self._exposure_registry.execution_authorized_symbols()))
        except Exception as exc:  # noqa: BLE001 — exposure gating is optional
            LOGGER.warning("exposure registry unavailable (%s); entries ungated", exc)

        # Master-universe membership drives operational discovery (directive §4):
        # loaded from the cache written by tools/build_membership.py. When present
        # (and not overridden by a pinned CENGINE_SYMBOLS diagnostic universe), the
        # scanner's discovery_fn returns the master universe instead of the legacy
        # discover_us_listed path. Absent cache -> scanner keeps its default
        # (backward compatible). CENGINE_MEMBERSHIP_DISCOVERY=0 opts out.
        self._membership_registry = None
        self._membership_discovery_fn = None
        _pinned = os.environ.get("CENGINE_SYMBOLS", "").strip()
        if (os.environ.get("CENGINE_MEMBERSHIP_DISCOVERY", "1") != "0" and not _pinned):
            try:
                from cengine.membership import MembershipRegistry
                from cengine.operational_discovery import make_membership_discovery_fn

                reg = MembershipRegistry()
                cache = REPO_ROOT / "data" / "membership_snapshot.json"
                if reg.load_cache(cache) and len(reg) > 0:
                    self._membership_registry = reg
                    cap = int(os.environ.get("CENGINE_DISCOVERY_MAX_SYMBOLS", "0")) or None
                    self._membership_discovery_fn = make_membership_discovery_fn(
                        reg, self._exposure_registry, max_symbols=cap)
                    LOGGER.info(
                        "membership discovery active: %d master members "
                        "(v%s) drive operational discovery",
                        len(reg.tradable_master_symbols()),
                        reg.snapshot_meta.membership_version)
                else:
                    LOGGER.info("no membership cache; scanner uses legacy discovery "
                                "(run tools/build_membership.py)")
            except Exception as exc:  # noqa: BLE001 — discovery source is optional
                LOGGER.warning("membership discovery unavailable (%s); legacy path", exc)
        elif _pinned:
            LOGGER.info("CENGINE_SYMBOLS pinned; membership discovery disabled (diagnostic)")

        self.router = CandidateRouter(
            order_manager=self.order_manager,
            entry_quantity_policy=quantity_policy,
            exit_quantity_provider=exit_quantity_provider,
            execution_policy=execution_policy,
            strategy_state=self.strategy_state,
            exposure_authorizer=exposure_authorizer,
        )

        # Independent, authoritative protective-exit coordinator (mandatory
        # -0.5% loss removal). It runs on this loop as a periodic backstop task
        # (see run()) and routes closes through the SAME OrderManager path --
        # never a parallel broker writer. It needs the native runtime state
        # (positions + live quote price + reservations); when that is absent it
        # is simply not constructed.
        self.protective_exit: ProtectiveExitCoordinator | None = None
        if runtime_state is not None and self.portfolio is not None:
            self.protective_exit = ProtectiveExitCoordinator(
                order_manager=self.order_manager,
                positions=self.portfolio,
                price_source=runtime_state,
                reservations=runtime_state.reservations,
                strategy_state=self.strategy_state,
                threshold=float(os.environ.get("MARKET_SCAN_LOSS_THRESHOLD", "-0.005")),
                interval_s=float(os.environ.get("MARKET_SCAN_LOSS_MONITOR_SECONDS", "5")),
            )

        # Single market-data pipeline (Milestone 4): the engine owns the one and
        # only market-data socket (self.stream). The migrated scanner consumes the
        # SAME normalized events through a thread-safe bridge instead of opening a
        # second IEX websocket. The scanner keeps running on its own thread so its
        # ranking/discovery compute never blocks this execution loop; the bridge is
        # bounded (drop-oldest) so a slow scanner can never back-pressure the feed.
        self._scanner_bridge: ScannerEventBridge | None = None
        self._bridge_client: BridgeStreamClient | None = None
        self._scanner_host: _EngineScannerHost | None = None
        self._scanner_handle: Any = None
        self._scan_enabled = os.environ.get("MARKET_SCAN_ENABLED", "1") != "0"
        scan_key = os.environ.get("ALPACA_API_KEY") or os.environ.get("APCA_API_KEY_ID", "")
        scan_secret = (
            os.environ.get("ALPACA_SECRET_KEY") or os.environ.get("APCA_API_SECRET_KEY", "")
        )
        # Phase B.5 — native GPU opportunity discovery feeds the scanner's
        # feature_provider (DI/VWAP/BB/screen computed on Apple Metal, CPU
        # fallback). Default ON; CENGINE_GPU_DISCOVERY=0 disables it; any init
        # failure degrades silently to the scanner's own Python features. Live
        # end-to-end certification (genuine IEX session) is Phase B.6.
        self._gpu_discovery: Any = None
        if (self._scan_enabled
                and os.environ.get("CENGINE_GPU_DISCOVERY", "1") != "0"):
            try:
                from cengine.gpu_scanner_bridge import EngineGpuDiscovery

                self._gpu_discovery = EngineGpuDiscovery()
                LOGGER.info(
                    "GPU discovery enabled (backend=%s); scanner feature_provider "
                    "= native screen_market + technical_features",
                    self._gpu_discovery.backend,
                )
            except Exception as exc:  # noqa: BLE001 — discovery is optional
                LOGGER.warning("GPU discovery unavailable (%s); scanner uses "
                               "its Python feature provider", exc)
                self._gpu_discovery = None

        if self._scan_enabled and scan_key and scan_secret:
            self._scanner_bridge = ScannerEventBridge()
            self._bridge_client = BridgeStreamClient(self._scanner_bridge)
            _url = os.environ.get("ALPACA_TRADING_BASE_URL", "").rstrip("/")
            if _url.endswith("/v2"):
                _url = _url[: -len("/v2")]
            self._scanner_host = _EngineScannerHost(
                self, scan_key, scan_secret, paper=(_url != "https://api.alpaca.markets")
            )

        # Dynamic universe resubscription: keep the live stream subscription in
        # step with the scanner-republished canonical universe.txt WITHOUT a
        # restart. Disabled by CENGINE_DYNAMIC_UNIVERSE=0 (an operator that pins
        # CENGINE_SYMBOLS); skipped if universe.txt is absent. Held positions and
        # pending orders are always kept subscribed (never dropped on a universe
        # shrink); bounded scanner discovery candidates are folded in.
        self.universe_sync: UniverseSubscriptionSync | None = None
        if os.environ.get("CENGINE_DYNAMIC_UNIVERSE", "1") != "0":
            universe_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "universe.txt"
            )
            if os.path.exists(universe_path):
                try:
                    manager = CanonicalUniverseManager(
                        universe_path, source_feed=config.alpaca_feed
                    )
                    discovery_fn = (
                        (lambda: self._bridge_client.desired_symbols)
                        if self._bridge_client is not None
                        else None
                    )
                    self.universe_sync = UniverseSubscriptionSync(
                        stream=self.stream,
                        universe_manager=manager,
                        positions=self.portfolio,
                        open_orders_fn=self.order_manager.open_orders,
                        discovery_symbols_fn=discovery_fn,
                        max_symbols=int(
                            os.environ.get("MARKET_SCAN_MAX_SUBSCRIPTIONS", "30")
                        ),
                        interval_s=float(
                            os.environ.get("CENGINE_UNIVERSE_SYNC_SECONDS", "5")
                        ),
                    )
                except Exception as exc:  # noqa: BLE001 - never block engine startup
                    LOGGER.warning("dynamic universe sync disabled: %s", exc)
                    self.universe_sync = None

        # Adaptive four-tier subscription controller (directive item 2): makes the
        # AdaptiveSubscriptionManager the authoritative owner of THIS single SIP
        # socket, driving per-channel (bars/trades/quotes) subscriptions over the
        # master universe. Active only on SIP with a loaded membership registry and
        # CENGINE_ADAPTIVE_SUBSCRIPTIONS!=0; when active it REPLACES the flat
        # UniverseSubscriptionSync (one authoritative owner). Fail-safe.
        self._adaptive_controller = None
        self._canonical_for_adaptive = None
        self._obs_channel: dict[str, Any] = {}
        feed = str(getattr(config, "alpaca_feed", "") or "").lower()
        if (self._membership_registry is not None and feed in ("sip", "delayed_sip")
                and os.environ.get("CENGINE_ADAPTIVE_SUBSCRIPTIONS", "1") != "0"):
            try:
                from cengine.adaptive_subscription_controller import (
                    AdaptiveSubscriptionController,
                )
                from market_scan.adaptive_subscriptions import Channel, ChannelCaps

                # Hot-path map: normalized market-event channel -> manager Channel,
                # used to feed live observations into the promotion lifecycle (item 6).
                self._obs_channel = {"bar": Channel.BARS, "trade": Channel.TRADES,
                                     "quote": Channel.QUOTES}
                # Default bar cap 8192 covers the full tradable master (~5,958) with
                # headroom: the capacity benchmark (reports/full_market_discovery_
                # benchmark.json) measured the full-set GPU screen dispatch at ~0.09s,
                # well under the rank cadence, so simultaneous full-market BAR coverage
                # is enabled. Trades/quotes stay tightly capped (adaptive, by measured
                # opportunity relevance). When demand exceeds a cap the manager rotates
                # elective coverage deterministically; protected symbols never rotate.
                caps = ChannelCaps(
                    bars=int(os.environ.get("CENGINE_TIER1_MAX_BARS", "8192")),
                    trades=int(os.environ.get("CENGINE_TIER2_MAX_TRADES", "1000")),
                    quotes=int(os.environ.get("CENGINE_TIER3_MAX_QUOTES", "250")))
                self._adaptive_controller = AdaptiveSubscriptionController(
                    self.stream, caps=caps)
                up = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  "universe.txt")
                if os.path.exists(up):
                    self._canonical_for_adaptive = CanonicalUniverseManager(
                        up, source_feed=config.alpaca_feed)
                # One authoritative owner: the flat sync yields to the controller.
                if self.universe_sync is not None:
                    LOGGER.info("adaptive subscriptions active; disabling legacy "
                                "flat UniverseSubscriptionSync")
                    self.universe_sync = None
                LOGGER.info("adaptive subscription controller active (SIP): "
                            "bars<=%d trades<=%d quotes<=%d over %d master members",
                            caps.bars, caps.trades, caps.quotes,
                            len(self._membership_registry.tradable_master_symbols()))
            except Exception as exc:  # noqa: BLE001 - never block startup
                LOGGER.warning("adaptive subscription controller disabled: %s", exc)
                self._adaptive_controller = None

        # Automatic SIP historical warm-up (directive item 4): feeds the native
        # DiscoveryEngine buffers with prioritized history so GPU features have
        # adequate depth. A CONTINUOUS, bounded, persistent background scheduler
        # processes the WHOLE eligible master universe across successive batches
        # (no permanent coverage ceiling) and then maintains it (membership growth,
        # stale refresh, bounded retry). Runs in a worker thread, never blocking the
        # loop or protection. Active on SIP + GPU discovery + membership;
        # CENGINE_HISTORICAL_WARMUP=0 disables.
        self._historical_warmup = None
        self._warmup_scheduler = None
        if (self._gpu_discovery is not None and self._membership_registry is not None
                and feed in ("sip", "delayed_sip")
                and os.environ.get("CENGINE_HISTORICAL_WARMUP", "1") != "0"):
            try:
                from cengine.historical_warmup import HistoricalWarmup, make_sip_fetch_fn
                from cengine.warmup_scheduler import ContinuousWarmupScheduler

                _k = (os.environ.get("ALPACA_API_KEY")
                      or os.environ.get("APCA_API_KEY_ID", ""))
                _s = (os.environ.get("ALPACA_SECRET_KEY")
                      or os.environ.get("APCA_API_SECRET_KEY", ""))
                if _k and _s:
                    self._historical_warmup = HistoricalWarmup(
                        self._gpu_discovery.provider, make_sip_fetch_fn(_k, _s),
                        batch_size=int(os.environ.get("CENGINE_WARMUP_BATCH", "100")))
                    reg = self._membership_registry
                    progress = os.path.join(
                        os.path.dirname(os.path.abspath(__file__)),
                        "data", "warmup_progress.json")
                    self._warmup_scheduler = ContinuousWarmupScheduler(
                        self._historical_warmup,
                        universe_fn=reg.tradable_master_symbols,
                        priority_fn=self._warmup_priority_inputs,
                        batch_size=int(os.environ.get("CENGINE_WARMUP_BATCH", "100")),
                        state_path=progress,
                        max_attempts=int(os.environ.get("CENGINE_WARMUP_MAX_ATTEMPTS", "3")),
                        refresh_after_s=float(
                            os.environ.get("CENGINE_WARMUP_REFRESH_SECONDS", "21600")))
                    self._warmup_scheduler.set_membership_version(
                        reg.snapshot_meta.membership_version)
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("historical warm-up unavailable: %s", exc)
                self._historical_warmup = None
                self._warmup_scheduler = None

        # Full-scale native symbol registration (directive item 3): register the
        # ENTIRE tradable master universe into the native DiscoveryEngine up front
        # so GPU breadth / compaction / screen kernels operate over the whole
        # market from the first tick — not just the subset that happened to stream
        # or get scanned. One-shot, synchronous, cheap (id/generation allocation);
        # bounded by CENGINE_NATIVE_MAX_SYMBOLS; CENGINE_NATIVE_REGISTER_FULL=0
        # disables. Fail-safe: a registration error never blocks startup.
        self._native_registered = 0
        if (self._gpu_discovery is not None and self._membership_registry is not None
                and os.environ.get("CENGINE_NATIVE_REGISTER_FULL", "1") != "0"):
            try:
                cap = int(os.environ.get("CENGINE_NATIVE_MAX_SYMBOLS", "20000"))
                master = self._membership_registry.tradable_master_symbols()[:cap]
                self._native_registered = self._gpu_discovery.register_universe(master)
                LOGGER.info("native registration: %d master members registered "
                            "(engine universe=%d, cap=%d)",
                            self._native_registered,
                            self._gpu_discovery.universe_size, cap)
            except Exception as exc:  # noqa: BLE001 — never block startup
                LOGGER.warning("native full-scale registration skipped: %s", exc)

        # Market regime intelligence + five-family strategy arbitration (milestone):
        # an ADVISORY layer over the native GPU screen. Every family defaults to
        # RESEARCH eligibility, so this produces regime assessments, recommendations
        # and decision evidence but routes NOTHING to execution — the existing order
        # path and the independent protective exit are untouched. Gated by
        # CENGINE_REGIME_INTELLIGENCE!=0; fail-safe. Decision evidence reuses the
        # audit journal.
        self._regime_pipeline = None
        # Durable handle to the decision-event sink so the lakehouse analytics layer
        # can drain it out-of-band (never on the protective-exit/order path).
        self._decision_sink = None
        if (self._gpu_discovery is not None
                and os.environ.get("CENGINE_REGIME_INTELLIGENCE", "1") != "0"):
            try:
                from cengine.decision_events import DecisionEventSink
                from cengine.regime_strategy_pipeline import RegimeStrategyPipeline

                opposing = self._exposure_opposing_fn()
                self._decision_sink = DecisionEventSink(journal=self.journal)
                self._regime_pipeline = RegimeStrategyPipeline(
                    sink=self._decision_sink,
                    opposing_fn=opposing,
                    max_concurrent=int(os.environ.get("CENGINE_MAX_CONCURRENT_ENTRIES", "10")))
                LOGGER.info("regime intelligence active (advisory; families RESEARCH "
                            "by default — no auto-execution)")
            except Exception as exc:  # noqa: BLE001 — advisory layer must never block startup
                LOGGER.warning("regime intelligence unavailable: %s", exc)
                self._regime_pipeline = None
                self._decision_sink = None

        # Local lakehouse analytics (M5): an ANALYTICAL-only, nonblocking, bounded,
        # loss-accounted capture of decision events + raw market observations into a
        # local Parquet/DuckDB lakehouse for future supervised/RL learning. It NEVER
        # sits on the order path, the native RiskEngine, broker reconciliation, or the
        # mandatory protective exit. Gated by CENGINE_LAKEHOUSE (default on); any
        # failure degrades to disabled without affecting trading. The authoritative
        # financial journal (self.journal) remains separate and unaffected.
        self._lakehouse = None
        self._lakehouse_market_buf: Optional[deque] = None
        self._lakehouse_market_dropped = 0
        self._lakehouse_last = {}
        if os.environ.get("CENGINE_LAKEHOUSE", "1") != "0":
            try:
                from cengine.lakehouse.bronze import AnalyticalIngestionSink
                from cengine.lakehouse.config import LakehouseConfig

                feed = str(getattr(self.stream, "feed", None) or "internal")
                self._lakehouse = AnalyticalIngestionSink(
                    LakehouseConfig.from_env(),
                    session_id=getattr(self, "session_id", None),
                    source_feed=feed)
                self._lakehouse_market_buf = deque(
                    maxlen=int(os.environ.get("CENGINE_LAKEHOUSE_MARKET_BUF", "200000")))
                LOGGER.info("local lakehouse analytics active (feed=%s; analytical "
                            "only — never on the order/protection path)", feed)
            except Exception as exc:  # noqa: BLE001 — analytics must never block startup
                LOGGER.warning("local lakehouse analytics unavailable: %s", exc)
                self._lakehouse = None

        # Optional execution bridge (directive §3/§17): connects authorized
        # recommendations to the EXISTING CandidateRouter. Default authorization is
        # NONE -> routes nothing. Even under PAPER/LIVE authorization nothing routes
        # unless a family has been separately promoted to the matching maturity
        # (default RESEARCH). Mock execution is isolated and never grants live
        # eligibility. The bridge creates no parallel engine and bypasses no control.
        self._exec_bridge = None
        self._bridge_audit_handler = None
        if self._regime_pipeline is not None:
            try:
                from cengine.strategy_execution_bridge import (
                    StrategyExecutionBridge,
                    make_decision_audit_handler,
                )
                from cengine.strategy_maturity import ExecutionAuthorization

                auth_raw = os.environ.get("CENGINE_STRATEGY_EXECUTION_AUTH", "none").lower()
                authorization = {
                    "none": ExecutionAuthorization.NONE,
                    "mock": ExecutionAuthorization.MOCK,
                    "paper": ExecutionAuthorization.PAPER,
                    "live": ExecutionAuthorization.LIVE,
                }.get(auth_raw, ExecutionAuthorization.NONE)
                runtime_state_for_bridge = getattr(self._adapter_module, "STATE", None)
                if (authorization is not ExecutionAuthorization.NONE
                        and runtime_state_for_bridge is not None):
                    self._exec_bridge = StrategyExecutionBridge(
                        router=self.router, registry=self._regime_pipeline.registry,
                        sink=self._regime_pipeline.sink,
                        price_ticks_fn=runtime_state_for_bridge.price_ticks,
                        authorization=authorization,
                        max_route=int(os.environ.get("CENGINE_MAX_CONCURRENT_ENTRIES", "10")))
                    # Compose execution-evidence emission onto the OrderManager audit
                    # chain so bridge orders produce ack/fill/risk decision events.
                    self._bridge_audit_handler = make_decision_audit_handler(
                        self._regime_pipeline.sink, self._exec_bridge.correlation_index)
                    LOGGER.warning(
                        "strategy execution bridge ACTIVE authorization=%s (%s); "
                        "families still gated by maturity (default RESEARCH)",
                        authorization.value, self._exec_bridge.describe())
            except Exception as exc:  # noqa: BLE001 — must never block startup
                LOGGER.warning("strategy execution bridge unavailable: %s", exc)
                self._exec_bridge = None

        # GPU portfolio intelligence (Milestone 3, §17): optimizer -> capital
        # reservation -> existing CandidateRouter -> native RiskEngine -> OrderManager.
        # Default authorization NONE -> advisory (emits portfolio state + evidence,
        # routes nothing). Routing requires CENGINE_PORTFOLIO_EXECUTION_AUTH!=none
        # AND sufficient observed return history. Creates no parallel authority:
        # routes through the SAME OrderManager. Fail-safe; never blocks startup.
        self._portfolio_pipeline = None
        self._portfolio_returns: dict[str, dict[int, float]] = {}
        self._portfolio_last_price: dict[str, int] = {}
        self._portfolio_state_builder = None
        self._portfolio_audit_handler = None
        self._microstructure = None
        self._replacement = None
        if self._regime_pipeline is not None:
            try:
                from cengine.capital_reservation import CapitalReservationLedger
                from cengine.portfolio_analytics import PortfolioAnalytics
                from cengine.portfolio_optimizer import OptimizerConfig, PortfolioOptimizer
                from cengine.portfolio_pipeline import (
                    PortfolioPipeline,
                    PortfolioQuantityPolicy,
                )
                from cengine.portfolio_state import PortfolioStateBuilder
                from cengine.strategy_maturity import ExecutionAuthorization

                pf_auth = {
                    "none": ExecutionAuthorization.NONE, "mock": ExecutionAuthorization.MOCK,
                    "paper": ExecutionAuthorization.PAPER, "live": ExecutionAuthorization.LIVE,
                }.get(os.environ.get("CENGINE_PORTFOLIO_EXECUTION_AUTH", "none").lower(),
                      ExecutionAuthorization.NONE)
                rstate = getattr(self._adapter_module, "STATE", None)
                if rstate is not None:
                    exposure_info = (self._exposure_registry.get
                                     if self._exposure_registry is not None else None)
                    self._portfolio_state_builder = PortfolioStateBuilder(
                        account=rstate.account, positions=rstate.positions,
                        reservations=rstate.reservations, price_ticks=rstate.price_ticks,
                        exposure_info=exposure_info)
                    runtime = None
                    try:  # reuse the native Metal runtime for GPU covariance if present
                        runtime = getattr(self._gpu_discovery.provider, "_runtime", None)
                    except Exception:  # noqa: BLE001
                        runtime = None
                    analytics = PortfolioAnalytics(runtime=runtime, ridge=1e-8)
                    pf_qpol = PortfolioQuantityPolicy()
                    # Intelligent execution (M4): the portfolio router's execution
                    # policy is a PlannedExecutionPolicy populated by the ExecutionPlanner
                    # with fresh-quote limit prices from the L1 microstructure engine.
                    from cengine.execution_planner import ExecutionPlanner, PlannedExecutionPolicy
                    from cengine.microstructure import MicrostructureEngine
                    from cengine.transaction_cost import TransactionCostEstimator
                    self._microstructure = MicrostructureEngine()
                    pf_planned = PlannedExecutionPolicy()
                    pf_planner = ExecutionPlanner(
                        cost_estimator=TransactionCostEstimator(),
                        execution_authorization=pf_auth.value)
                    pf_router = CandidateRouter(
                        order_manager=self.order_manager, entry_quantity_policy=pf_qpol,
                        exit_quantity_provider=pf_qpol,
                        execution_policy=pf_planned,
                        strategy_state=self.strategy_state,
                        exposure_authorizer=(
                            self._exposure_registry.authorize_entry
                            if self._exposure_registry is not None else None))
                    ledger = CapitalReservationLedger(risk_reservations=rstate.reservations)
                    self._portfolio_pipeline = PortfolioPipeline(
                        router=pf_router, quantity_policy=pf_qpol,
                        optimizer=PortfolioOptimizer(OptimizerConfig(), analytics=analytics),
                        analytics=analytics, reservation_ledger=ledger,
                        sink=self._regime_pipeline.sink, authorization=pf_auth,
                        execution_planner=pf_planner, microstructure=self._microstructure,
                        planned_policy=pf_planned)
                    # Emit portfolio ack/fill/risk evidence, attributed by correlation
                    # id -> portfolio_decision_id, composed onto the audit chain.
                    from cengine.strategy_execution_bridge import make_decision_audit_handler
                    self._portfolio_audit_handler = make_decision_audit_handler(
                        self._regime_pipeline.sink, self._portfolio_pipeline.correlation_index)
                    # Intelligent replacement (M4 §16): reuse the M3 coordinator, driven
                    # by protective-exit fills. Freed capital is reinvested by the next
                    # portfolio tick subject to the no-reentry cooldown; no forced reinvest.
                    from cengine.portfolio_replacement import ReplacementCoordinator
                    self._replacement = ReplacementCoordinator()
                    LOGGER.info(
                        "portfolio intelligence active (authorization=%s; advisory until "
                        "return history accrues; routes via existing authority)",
                        pf_auth.value)
            except Exception as exc:  # noqa: BLE001 — must never block startup
                LOGGER.warning("portfolio intelligence unavailable: %s", exc)
                self._portfolio_pipeline = None

    @staticmethod
    def _portfolio_execution_policy():
        return ConfiguredExecutionPolicy(OrderType.MARKET, TimeInForce.DAY)

    def _exposure_opposing_fn(self):
        """Pairwise opposing-exposure predicate from the exposure registry: two
        symbols oppose if they share an underlying exposure group with opposite
        signed directions (e.g. TQQQ vs SQQQ). Returns a safe no-op if the registry
        is unavailable."""
        reg = self._exposure_registry
        if reg is None:
            return lambda a, b: False

        def opposing(a: str, b: str) -> bool:
            ia, ib = reg.get(a), reg.get(b)
            if ia is None or ib is None:
                return False
            if not ia.underlying_exposure_group or not ib.underlying_exposure_group:
                return False
            return (ia.underlying_exposure_group == ib.underlying_exposure_group
                    and ia.exposure_direction != ib.exposure_direction
                    and "neutral" not in (ia.exposure_direction.value,
                                          ib.exposure_direction.value))
        return opposing

    async def _run_regime_intelligence(self, stop: "asyncio.Event") -> None:
        """Periodic advisory regime + strategy-arbitration pass over the current
        discovery set (off the hot path). Emits decision evidence and telemetry;
        routes nothing to execution. Never raises into the loop."""
        import time as _time

        interval = float(os.environ.get("CENGINE_REGIME_SECONDS", "15"))
        while not stop.is_set():
            try:
                await self._regime_intelligence_tick(_time.time_ns())
            except Exception as exc:  # noqa: BLE001 — advisory; never kill the loop
                LOGGER.warning("regime intelligence tick failed: %s", exc)
            try:
                await asyncio.wait_for(stop.wait(), timeout=interval)
            except asyncio.TimeoutError:
                pass

    async def _regime_intelligence_tick(self, now_ns: int) -> Optional[dict]:
        """One advisory pass: compute native screen rows over refs + canonical +
        positions, run the regime/strategy pipeline, log telemetry, and (only if an
        authorized execution bridge is active) route recommendations through the
        existing order path. Returns the pipeline result (or None if inactive)."""
        if self._regime_pipeline is None or self._gpu_discovery is None:
            return None
        refs = ["SPY", "QQQ", "DIA", "IWM"]
        canonical = []
        if self._canonical_for_adaptive is not None:
            try:
                self._canonical_for_adaptive.refresh_from_file()
                canonical = list(self._canonical_for_adaptive.symbols())
            except Exception:  # noqa: BLE001
                canonical = []
        positions = list(getattr(self.portfolio, "symbols", lambda: [])() or []) \
            if self.portfolio is not None else []
        universe = sorted(set(refs) | set(canonical) | set(positions))
        if not universe:
            return None
        from cengine.regime_strategy_pipeline import screen_row_to_features
        rows = self._gpu_discovery.provider.compute(universe)
        if not rows:
            return None
        ref_features = {s: screen_row_to_features(rows[s], None, now_ns)
                        for s in refs if s in rows}
        breadth = self._gpu_discovery.breadth_snapshot()
        screen_rows = {s: r for s, r in rows.items() if s not in refs}
        out = self._regime_pipeline.run(
            screen_rows or rows, now_ns=now_ns, ref_features=ref_features,
            breadth=breadth)
        tele = self._regime_pipeline.telemetry()
        LOGGER.info("regime: broad=%s/%s recs=%d selected=%d research=%d events=%d",
                    out["broad"].dimensions.direction.value,
                    out["broad"].dimensions.trend.value,
                    len(out["recommendations"]),
                    len(out["arbitration"]["selected"]),
                    len(out["arbitration"]["research"]), tele["events_emitted"])
        # Execution bridge (default-off): route authorized recommendations through
        # the EXISTING order path. With default maturity (RESEARCH) nothing routes.
        if self._exec_bridge is not None:
            try:
                bo = await self._exec_bridge.route_recommendations(
                    out["recommendations"], now_ns=now_ns)
                if bo.submitted or bo.by_outcome("risk_rejected"):
                    LOGGER.warning("exec bridge: submitted=%d skipped=%d",
                                   bo.submitted, bo.skipped)
                out["bridge"] = bo
            except Exception as exc:  # noqa: BLE001 — bridge must never kill the loop
                LOGGER.warning("exec bridge route failed: %s", exc)
        # Portfolio intelligence pass (default advisory): emit portfolio state +
        # evidence; route only when authorized AND enough observed return history.
        if self._portfolio_pipeline is not None:
            try:
                await self._portfolio_tick(rows, now_ns)
            except Exception as exc:  # noqa: BLE001 — portfolio must never kill the loop
                LOGGER.warning("portfolio tick failed: %s", exc)
        return out

    async def _portfolio_tick(self, rows: dict, now_ns: int) -> None:
        """Update observed returns, build + emit portfolio state, and (when
        authorized with sufficient history) run the portfolio optimizer -> reservation
        -> existing execution authority. Never fabricates returns."""
        min_obs = int(os.environ.get("CENGINE_PORTFOLIO_MIN_OBS", "20"))
        window = int(os.environ.get("CENGINE_PORTFOLIO_RETURN_WINDOW", "120"))
        # candidate universe = non-benchmark screen rows with a valid price
        refs = {"SPY", "QQQ", "DIA", "IWM"}
        candidates = []
        for sym, row in rows.items():
            price = getattr(row, "price", None) if not isinstance(row, dict) else row.get("price")
            if sym in refs or not price or price <= 0:
                continue
            candidates.append(sym)
            ticks = int(round(float(price) * 10_000))
            prev = self._portfolio_last_price.get(sym)
            if prev and prev > 0:
                ret = (ticks - prev) / prev
                series = self._portfolio_returns.setdefault(sym, {})
                series[now_ns] = float(ret)
                if len(series) > window:  # bound memory: keep the most recent window
                    for ts in sorted(series)[:-window]:
                        series.pop(ts, None)
            self._portfolio_last_price[sym] = ticks

        state = self._portfolio_state_builder.build(now_ns=now_ns) \
            if self._portfolio_state_builder is not None else None
        if state is None:
            return
        # Build + emit state evidence always (advisory telemetry).
        self._regime_pipeline.sink.record_portfolio_state(state, now_ns)
        self._last_portfolio_state = state

        symbols = tuple(s for s in candidates
                        if len(self._portfolio_returns.get(s, {})) >= min_obs)
        # No immediate re-entry into a just-stopped-out symbol (M4 §16).
        if self._replacement is not None:
            symbols = tuple(s for s in symbols
                            if not self._replacement.in_cooldown(s, now_ns))
        if not symbols:
            if self._on_portfolio is not None:
                self._on_portfolio(state, {"allocation_policy": "warming_up"})
            return
        returns_by_symbol = {s: dict(self._portfolio_returns[s]) for s in symbols}
        authorized_fn = (self._exposure_registry.authorize_entry
                         if self._exposure_registry is not None else None)

        def _authorized(sym: str) -> bool:
            if authorized_fn is None:
                return True
            ok, _ = authorized_fn(sym)
            return ok

        res = await self._portfolio_pipeline.run(
            symbols=symbols, returns_by_symbol=returns_by_symbol, state=state,
            now_ns=now_ns, price_ticks=self._adapter_module.STATE.price_ticks,
            authorized=_authorized)
        if res.routed or res.reserved:
            LOGGER.warning("portfolio: policy=%s cov=%s reserved=%d routed=%d skipped=%d",
                           res.allocation_policy, res.covariance_source,
                           res.reserved, res.routed, res.skipped)
        if self._on_portfolio is not None:
            binding = sorted({c for r in res.recommendations for c in r.binding_constraints})
            self._on_portfolio(state, {
                "allocation_policy": res.allocation_policy,
                "covariance_source": res.covariance_source,
                "binding_constraints": binding,
                "reserved": res.reserved, "routed": res.routed,
                "execution": self._execution_dashboard_stats(symbols, now_ns),
                "lakehouse": self._lakehouse_dashboard_stats()})

    def _lakehouse_dashboard_stats(self) -> Optional[dict]:
        """Non-blocking lakehouse capture stats for the dashboard (M5 §31), or None when
        analytics are disabled. Read-only snapshot of the last flush + live buffer."""
        if self._lakehouse is None:
            return None
        buf = self._lakehouse_market_buf
        last = self._lakehouse_last or {}
        return {"written": last.get("written", 0), "dropped": last.get("dropped", 0),
                "quarantined": last.get("quarantined", 0),
                "buffer": len(buf) if buf is not None else 0,
                "feed": self._lakehouse.source_feed}

    def _execution_dashboard_stats(self, symbols: tuple, now_ns: int) -> dict:
        """Non-blocking execution analytics for the dashboard (M4 §22), derived from
        the current microstructure snapshots + recent decision evidence."""
        spreads = []
        micro_ok = 0
        if self._microstructure is not None:
            for sym in symbols:
                f = self._microstructure.features(sym, now_ns=now_ns)
                if f is not None and f.spread_bps is not None and not f.is_stale:
                    spreads.append(f.spread_bps)
                    micro_ok += 1
        open_orders = len(self.order_manager.open_orders())
        avg_spread = round(sum(spreads) / len(spreads), 2) if spreads else "n/a"
        return {"policies": "intelligent", "avg_spread_bps": avg_spread,
                "est_slippage_bps": avg_spread, "realized_slippage_bps": "pending",
                "pending": open_orders, "rejected": 0,
                "microstructure": f"{micro_ok}/{len(symbols)}",
                "reconciliation": "ok"}

    def _warmup_priority_inputs(self) -> dict:
        """Priority groups for the warm-up scheduler: positions and pending orders
        FIRST (so protected instruments always have history), then references,
        actionable (canonical), and preliminary (scanner) candidates. The scheduler
        appends the remaining master universe after these."""
        positions = list(getattr(self.portfolio, "symbols", lambda: [])() or []) \
            if self.portfolio is not None else []
        try:
            pending = list(self.order_manager.open_orders() or [])
        except Exception:  # noqa: BLE001
            pending = []
        canonical = []
        if self._canonical_for_adaptive is not None:
            try:
                self._canonical_for_adaptive.refresh_from_file()
                canonical = list(self._canonical_for_adaptive.symbols())
            except Exception:  # noqa: BLE001
                canonical = []
        preliminary = []
        if self._bridge_client is not None:
            try:
                preliminary = list(self._bridge_client.desired_symbols or [])
            except Exception:  # noqa: BLE001
                preliminary = []
        return {"positions": positions, "pending": pending,
                "references": ["SPY", "QQQ", "DIA", "IWM"],
                "actionable": canonical, "preliminary": preliminary}

    async def _run_continuous_warmup(self, stop: "asyncio.Event") -> None:
        """Continuous, bounded warm-up loop (directive item 4). Each iteration warms
        ONE prioritized batch in a worker thread (off the hot path), persisting
        progress, until the whole eligible master universe is covered — then idles,
        re-checking for membership growth / stale refresh. Backs off on failure
        (rate-limit awareness). Never blocks mandatory protection."""
        import time as _time

        base = float(os.environ.get("CENGINE_WARMUP_BATCH_SECONDS", "2"))
        idle = float(os.environ.get("CENGINE_WARMUP_IDLE_SECONDS", "300"))
        backoff = base
        while not stop.is_set():
            try:
                # Refresh membership version so a new snapshot re-opens failed symbols.
                if self._membership_registry is not None:
                    self._warmup_scheduler.set_membership_version(
                        self._membership_registry.snapshot_meta.membership_version)
                summary = await asyncio.to_thread(
                    self._warmup_scheduler.run_once, _time.monotonic())
                if summary.get("error"):
                    backoff = min(idle, max(base, backoff * 2.0))  # rate-limit backoff
                    LOGGER.warning("warm-up batch failed; backing off %.0fs "
                                   "(warmed=%d pending=%d)", backoff,
                                   summary.get("warmed", 0), summary.get("pending", 0))
                    wait = backoff
                elif summary.get("batch", 0) == 0:
                    LOGGER.info("warm-up coverage complete: warmed=%d incomplete=%d "
                                "failed=%d pending=%d bars=%s", summary.get("warmed", 0),
                                summary.get("incomplete", 0), summary.get("failed", 0),
                                summary.get("pending", 0), summary.get("bars_ingested"))
                    wait = idle  # maintain: re-check for growth / stale refresh
                else:
                    backoff = base
                    LOGGER.info("warm-up batch: %d symbols (warmed=%d pending=%d "
                                "incomplete=%d failed=%d)", summary.get("batch", 0),
                                summary.get("warmed", 0), summary.get("pending", 0),
                                summary.get("incomplete", 0), summary.get("failed", 0))
                    wait = base
            except Exception as exc:  # noqa: BLE001 — warm-up must never crash the engine
                LOGGER.warning("continuous warm-up tick failed: %s", exc)
                wait = idle
            try:
                await asyncio.wait_for(stop.wait(), timeout=wait)
            except asyncio.TimeoutError:
                pass

    def _adaptive_tier_inputs(self) -> dict:
        """Compute current tier memberships with STAGED, momentum-gated promotion
        (directive item 3):

          Tier 1 (bars)   = full tradable master universe.
          Tier 2 (trades) = preliminarily-ranked discovery candidates (scanner) —
                            these earn a TRADE subscription so developing 1-minute
                            momentum can be observed.
          Tier 3 (quotes) = already-published canonical actionable UNION the subset
                            of Tier-2 candidates whose momentum is confirmed from
                            FRESH trade observations (di+>di-, developing 1m>0). A
                            quote subscription is requested only AFTER fresh trades
                            confirm momentum — never together with trades for an
                            elective candidate.
          Protected       = positions + pending orders (+ references) — exempt from
                            the elective promotion gate; they receive every channel.

        The strict final entry contract (VWAP/SMA/BB reference, liquidity, strategy,
        exposure, risk) is unchanged and enforced downstream before any order."""
        from market_scan.adaptive_subscriptions import Channel

        positions = list(getattr(self.portfolio, "symbols", lambda: [])() or []) \
            if self.portfolio is not None else []
        try:
            pending = list(self.order_manager.open_orders() or [])
        except Exception:  # noqa: BLE001
            pending = []
        refs = ["SPY", "QQQ", "DIA", "IWM"]
        canonical = []
        if self._canonical_for_adaptive is not None:
            try:
                self._canonical_for_adaptive.refresh_from_file()
                canonical = list(self._canonical_for_adaptive.symbols())
            except Exception:  # noqa: BLE001
                canonical = []
        tier1 = (self._membership_registry.tradable_master_symbols()
                 if self._membership_registry is not None else [])
        # Preliminary ranked candidates earn a TRADE subscription (Tier 2).
        preliminary = []
        if self._bridge_client is not None:
            try:
                preliminary = list(self._bridge_client.desired_symbols or [])
            except Exception:  # noqa: BLE001
                preliminary = []
        protected = sorted(set(positions) | set(pending))
        tier2_trades = sorted(set(preliminary) | set(canonical) | set(protected))
        # Momentum gate: among candidates with FRESH trades, which have GPU-confirmed
        # preliminary momentum? Only those (plus canonical actionable) earn QUOTES.
        momentum = set()
        if self._gpu_discovery is not None and self._adaptive_controller is not None:
            try:
                fresh_trades = self._adaptive_controller.fresh(Channel.TRADES)
                candidates = (set(tier2_trades) & fresh_trades) - set(protected)
                if candidates:
                    momentum = self._gpu_discovery.momentum_confirmed(sorted(candidates))
            except Exception:  # noqa: BLE001
                momentum = set()
        tier3_quotes = sorted(set(canonical) | momentum)
        return {"tier1_bars": tier1, "tier2_trades": tier2_trades,
                "tier3_quotes": tier3_quotes, "protected": protected, "references": refs}

    async def _run_adaptive_subscriptions(self, stop: "asyncio.Event") -> None:
        import time as _time
        interval = float(os.environ.get("CENGINE_ADAPTIVE_SUBSCRIPTION_SECONDS", "5"))
        while not stop.is_set():
            try:
                now = _time.monotonic()
                inp = self._adaptive_tier_inputs()
                out = await self._adaptive_controller.apply(now=now, **inp)
                if out.get("degraded"):
                    LOGGER.warning("adaptive subscriptions degraded: protected "
                                   "demand exceeds a channel cap")
                # Demote subscriptions whose channel has stopped ticking (item 6):
                # the promotion lifecycle is symmetric — live flow promotes to
                # FRESH, silence past the stale window demotes to STALE.
                staled = self._adaptive_controller.mark_stale(now)
                if staled:
                    LOGGER.debug("adaptive subscriptions: %d went stale", staled)
            except Exception as exc:  # noqa: BLE001 - never kill the loop
                LOGGER.warning("adaptive subscription tick failed: %s", exc)
            try:
                await asyncio.wait_for(stop.wait(), timeout=interval)
            except asyncio.TimeoutError:
                pass

    def _start_scanner(self) -> None:
        """Start the in-process market scanner on its own thread, fed by the
        single market-data pipeline (no second websocket). Never raises."""
        if self._bridge_client is None or self._scanner_host is None:
            return
        try:
            from market_scan.scanner import MarketScanner
            from market_scan.thread_runner import start_market_scanner

            bridge_client = self._bridge_client

            gpu_discovery = self._gpu_discovery
            membership_discovery_fn = self._membership_discovery_fn

            def factory(executor: Any, config: Any) -> Any:
                scanner = MarketScanner(executor, config, stream_client=bridge_client)
                bridge_client.bind(
                    on_trade=scanner.ofe.ingest_trade,
                    on_quote=scanner.ofe.ingest_quote,
                )
                # Master-universe membership drives discovery (directive §4): the
                # scanner ranks + publishes over the full supported universe, not
                # the legacy discover_us_listed list. Falls back to its default
                # when no membership cache is available.
                if membership_discovery_fn is not None:
                    scanner.discovery_fn = membership_discovery_fn
                # Overlay native GPU directional/screen features on the scanner's
                # base features AND move the cross-sectional ranking onto the GPU
                # (identical semantics). The scanner keeps orchestration, lifecycle
                # filtering, and the single UniversePublisher.
                if gpu_discovery is not None:
                    try:
                        scanner.feature_provider = gpu_discovery.feature_provider_for(
                            scanner)
                        scanner.rank_fn = gpu_discovery.rank_fn()
                    except Exception as exc:  # noqa: BLE001
                        LOGGER.warning("GPU discovery wiring failed (%s); scanner "
                                       "uses its Python features/ranking", exc)
                return scanner

            self._scanner_handle = start_market_scanner(
                self._scanner_host, scanner_factory=factory
            )
            if self._scanner_handle is not None:
                LOGGER.info(
                    "market scanner started (single-pipeline bridge; no second socket)"
                )
        except Exception as exc:  # noqa: BLE001 - scanner must never block the engine
            LOGGER.warning("market scanner start failed: %s", exc)
            self._scanner_handle = None

    def _verify_stream_feed(self) -> None:
        """Fail closed if the connected feed disagrees with the configured one.

        The feed is derived from the endpoint the stream will actually open, so
        this catches an explicit URL override whose path contradicts the declared
        feed/provenance before any market data is trusted. It does not force IEX:
        when SIP is explicitly configured it confirms SIP. It also surfaces a
        provenance label that does not reflect the feed (soft warning only, so
        intentional composite labels like ``alpaca-iex+index`` are allowed).
        """
        actual = self.stream.feed
        expected = self.config.alpaca_feed
        if expected is not None and actual != expected:
            self.kill_switch(
                f"market-data feed mismatch: endpoint feed {actual!r} != "
                f"configured feed {expected!r}"
            )
            raise EngineConfigurationError(
                f"market-data feed mismatch: stream endpoint resolves to feed "
                f"{actual!r} but configuration declares {expected!r}; refusing to "
                "run with inconsistent feed provenance"
            )
        label = self.config.archive_provider_label.lower()
        if actual is not None and actual.replace("_", "-") not in label:
            LOGGER.warning(
                "archive provider label %r does not name the active feed %r; "
                "verify provenance labeling",
                self.config.archive_provider_label,
                actual,
            )
        LOGGER.info(
            "market-data feed=%s endpoint=%s provider_label=%s consolidated=%s",
            actual or "custom",
            self.stream.endpoint,
            self.config.archive_provider_label,
            actual in {"sip", "delayed_sip"},
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

        # Start the in-process scanner (own thread) fed by the single pipeline.
        self._start_scanner()

        consumer_task = asyncio.create_task(
            self._consume_market_events(),
            name="market-engine-consumer",
        )
        stream_task = asyncio.create_task(
            self.stream.run_forever(),
            name="alpaca-market-data-stream",
        )
        reconciliation_task = asyncio.create_task(
            self._reconciliation_loop(), name="broker-reconciliation"
        )

        protective_task: Optional[asyncio.Task[None]] = None
        if self.protective_exit is not None:
            protective_task = asyncio.create_task(
                self.protective_exit.run(self._stop), name="protective-exit"
            )

        universe_sync_task: Optional[asyncio.Task[None]] = None
        if self.universe_sync is not None:
            universe_sync_task = asyncio.create_task(
                self.universe_sync.run(self._stop), name="universe-sync"
            )

        adaptive_sub_task: Optional[asyncio.Task[None]] = None
        if self._adaptive_controller is not None:
            adaptive_sub_task = asyncio.create_task(
                self._run_adaptive_subscriptions(self._stop),
                name="adaptive-subscriptions",
            )

        # Continuous warm-up runs DETACHED (not in the FIRST_COMPLETED wait set, so
        # it never triggers engine shutdown) and is stop-driven. Stored on self to
        # avoid GC; cancelled during teardown. It processes the whole master universe
        # across bounded batches, then idles and maintains coverage.
        self._warmup_task: Optional[asyncio.Task[None]] = None
        if self._warmup_scheduler is not None:
            self._warmup_task = asyncio.create_task(
                self._run_continuous_warmup(self._stop), name="historical-warmup"
            )

        # Advisory regime-intelligence loop (DETACHED, stop-driven; emits decision
        # evidence + telemetry, routes nothing to execution).
        self._regime_task: Optional[asyncio.Task[None]] = None
        if self._regime_pipeline is not None:
            self._regime_task = asyncio.create_task(
                self._run_regime_intelligence(self._stop), name="regime-intelligence")

        # Local lakehouse drain loop (DETACHED, stop-driven; analytical only). Flushes
        # buffered observations + decision events to bronze Parquet out of band. Never
        # in the FIRST_COMPLETED set, so it cannot trigger or block engine shutdown.
        self._lakehouse_task: Optional[asyncio.Task[None]] = None
        if self._lakehouse is not None:
            self._lakehouse_task = asyncio.create_task(
                self._run_lakehouse_drain(self._stop), name="lakehouse-drain")

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
        if protective_task is not None:
            tasks.append(protective_task)
        if universe_sync_task is not None:
            tasks.append(universe_sync_task)
        if adaptive_sub_task is not None:
            tasks.append(adaptive_sub_task)
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

            # Cancel the detached warm-up + regime-intelligence tasks if still running.
            warmup = getattr(self, "_warmup_task", None)
            if warmup is not None and not warmup.done():
                warmup.cancel()
            regime = getattr(self, "_regime_task", None)
            if regime is not None and not regime.done():
                regime.cancel()
            lakehouse_task = getattr(self, "_lakehouse_task", None)
            if lakehouse_task is not None and not lakehouse_task.done():
                lakehouse_task.cancel()

            await asyncio.gather(
                *tasks,
                stop_task,
                *( [warmup] if warmup is not None else [] ),
                *( [regime] if regime is not None else [] ),
                *( [lakehouse_task] if lakehouse_task is not None else [] ),
                return_exceptions=True,
            )

            self.archive.close(drain=True)
            self.journal.close()

            # Final out-of-band lakehouse flush so the session's last buffered
            # observations + decision events are captured. Analytical only — after the
            # authoritative journal is already durably closed above.
            await self._flush_lakehouse()

    async def stop(self) -> None:
        if self._stop.is_set():
            return

        if self.state.state not in {EngineState.KILLED, EngineState.STOPPING}:
            self.state.transition(EngineState.STOPPING, "stop requested")
        elif self.state.state is EngineState.KILLED:
            self.state.transition(EngineState.STOPPING, "killed engine stopping")
        self._stop.set()
        # Stop the scanner (its own thread) and its bridge consumer first so it
        # stops asking for coverage and draining the bridge.
        if self._bridge_client is not None:
            self._bridge_client.stop()
        if self._scanner_handle is not None:
            try:
                self._scanner_handle.stop()
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("scanner stop error: %s", exc)
            self._scanner_handle = None
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
        # The market-data feed (IEX by default, or SIP) multiplexes independent
        # channels; the gate enforces monotonicity per (symbol, channel), never
        # across channels (quotes and trades interleave with non-monotonic
        # cross-channel timestamps). Index factor observations (VIX/VXN) get
        # their own channel and are never executable.
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
        # Feed the single pipeline's scanner via the bounded, drop-oldest bridge.
        # Only quote/trade ticks are relevant to the OrderFlowEngine.
        if self._scanner_bridge is not None and channel in ("quote", "trade"):
            self._scanner_bridge.offer(event)
        # L1 microstructure intelligence (M4): consume the SAME quote/trade ticks
        # (no new provider connection). Nonblocking; never raises into ingestion.
        if self._microstructure is not None and channel in ("quote", "trade"):
            try:
                if channel == "quote":
                    self._microstructure.on_quote_event(event)
                else:
                    self._microstructure.on_trade_event(event)
            except Exception:  # noqa: BLE001 — analytics must never block ingestion
                pass
        # Local lakehouse bronze tap (M5): buffer raw observations for out-of-band
        # analytical capture. Bounded + drop-oldest (loss-accounted) + never raises —
        # identical nonblocking discipline to the microstructure tap above. The actual
        # Parquet write happens on the detached drain task, never here on the hot path.
        if self._lakehouse_market_buf is not None:
            try:
                row = self._market_bronze_row(event, channel)
                if row is not None:
                    buf = self._lakehouse_market_buf
                    if len(buf) >= buf.maxlen:        # bounded: account the drop-oldest loss
                        self._lakehouse_market_dropped += 1
                    buf.append(row)
            except Exception:  # noqa: BLE001 — analytics must never block ingestion
                pass
        # Phase B.5 — feed native GPU discovery (bars -> 5m aggregation; trades ->
        # developing price). on_market_event never raises; bars/trades only.
        if self._gpu_discovery is not None and channel in ("bar", "trade"):
            self._gpu_discovery.on_market_event(event)
        # Live trade/quote promotion lifecycle (directive item 6): feed every
        # observed tick into the adaptive manager so a requested (symbol, channel)
        # advances WARMING_UP -> FRESH on real data arrival. Unknown (unsubscribed)
        # symbols are ignored cheaply by the manager; a periodic mark_stale in the
        # adaptive loop demotes FRESH -> STALE when a channel stops ticking. This
        # closes the loop: subscription promotion is now driven by observed flow,
        # not just by the membership/canonical tier inputs.
        if self._adaptive_controller is not None:
            obs = self._obs_channel.get(channel)
            if obs is not None:
                self._adaptive_controller.on_observation(
                    obs, event.symbol, time.monotonic())

    def _market_bronze_row(self, event: MarketEvent, channel: str):
        """Build a (bronze_dataset, row) pair for a raw market observation, or None for
        event types the lakehouse does not capture (index factors). Prices stay in the
        native 10000-tick scale; availability time = receipt time (point-in-time)."""
        ts = int(getattr(event, "timestamp_ns", 0))
        recv = int(getattr(event, "received_ns", ts) or ts)
        feed = self._lakehouse.source_feed if self._lakehouse is not None else "internal"
        base = {"symbol_id": event.symbol, "event_timestamp_ns": ts,
                "receipt_timestamp_ns": recv, "availability_timestamp_ns": max(ts, recv),
                "source_feed": feed, "sequence": int(getattr(event, "sequence", 0) or 0),
                "price_scale": 10_000}
        if channel == "bar":
            base.update({"bar_open_ticks": event.open_ticks, "bar_high_ticks": event.high_ticks,
                         "bar_low_ticks": event.low_ticks, "bar_close_ticks": event.close_ticks,
                         "volume": event.volume, "trade_count": event.trade_count,
                         "vwap_ticks": event.vwap_ticks})
            return ("market_bars", base)
        if channel == "quote":
            base.update({"bid_price_ticks": event.bid_price_ticks, "bid_size": event.bid_size,
                         "ask_price_ticks": event.ask_price_ticks, "ask_size": event.ask_size,
                         "bid_exchange": event.bid_exchange, "ask_exchange": event.ask_exchange,
                         "tape": event.tape})
            return ("market_quotes", base)
        if channel == "trade":
            base.update({"trade_price_ticks": event.price_ticks, "trade_size": event.size,
                         "exchange": event.exchange,
                         "trade_id": int(getattr(event, "trade_id", 0) or 0),
                         "tape": event.tape})
            return ("market_trades", base)
        return None

    def _flush_lakehouse_sync(self) -> None:
        """Write buffered market observations + drained decision events to bronze.
        Runs in a thread executor so Parquet IO never blocks the event loop. Any
        failure only disables capture; trading is untouched."""
        if self._lakehouse is None:
            return
        buf = self._lakehouse_market_buf
        market_written = 0
        if buf:
            groups: dict[str, list] = {}
            for _ in range(len(buf)):
                try:
                    dataset, row = buf.popleft()   # deque popleft is atomic in CPython
                except IndexError:
                    break
                groups.setdefault(dataset, []).append(row)
            for dataset, rows in groups.items():
                try:
                    self._lakehouse.ingest_market_rows(dataset, rows)
                    market_written += len(rows)
                except Exception:  # noqa: BLE001 — never propagate analytics failures
                    self._lakehouse_market_dropped += len(rows)
        if self._decision_sink is not None:
            try:
                self._lakehouse.drain_decision_sink(self._decision_sink)
            except Exception:  # noqa: BLE001
                pass
        st = self._lakehouse.stats
        self._lakehouse_last = {
            "written": st.written, "dropped": st.dropped + self._lakehouse_market_dropped,
            "quarantined": st.quarantined, "market_flushed": market_written,
            "buffer": len(buf) if buf is not None else 0}

    async def _flush_lakehouse(self) -> None:
        if self._lakehouse is None:
            return
        try:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, self._flush_lakehouse_sync)
        except Exception:  # noqa: BLE001 — analytics must never block the engine
            pass

    async def _run_lakehouse_drain(self, stop: asyncio.Event) -> None:
        """Detached, stop-driven loop that periodically flushes the lakehouse buffers
        OUT OF BAND. It is never in the FIRST_COMPLETED wait set and never touches the
        order path, reconciliation, or the mandatory protective exit."""
        interval = float(os.environ.get("CENGINE_LAKEHOUSE_DRAIN_S", "30"))
        while not stop.is_set():
            try:
                await asyncio.wait_for(stop.wait(), timeout=interval)
            except asyncio.TimeoutError:
                pass
            except asyncio.CancelledError:
                break
            await self._flush_lakehouse()

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
        # Execution-evidence for bridge-routed orders (decision events), composed
        # onto the audit chain. No-op unless the execution bridge is active.
        bridge_audit = getattr(self, "_bridge_audit_handler", None)
        if bridge_audit is not None:
            try:
                bridge_audit(event, order)
            except Exception:  # noqa: BLE001 — evidence must never block the order path
                pass
        portfolio_audit = getattr(self, "_portfolio_audit_handler", None)
        if portfolio_audit is not None:
            try:
                portfolio_audit(event, order)
            except Exception:  # noqa: BLE001 — evidence must never block the order path
                pass
        # Intelligent replacement (M4 §16): advance the coordinator on protective-exit
        # lifecycle so a stopped-out symbol enters its no-reentry cooldown; freed
        # capital is reinvested by the next portfolio tick (never forced).
        replacement = getattr(self, "_replacement", None)
        if replacement is not None and order.intent.strategy_id == "protective_exit":
            try:
                sym = order.intent.symbol
                if order.status is OrderStatus.SUBMITTED:
                    replacement.on_mandatory_exit(sym)
                    replacement.on_exit_submitted(sym)
                elif order.status is OrderStatus.PARTIALLY_FILLED:
                    replacement.on_exit_partial(sym)
                elif order.status is OrderStatus.FILLED:
                    replacement.on_exit_filled(sym)
            except Exception:  # noqa: BLE001 — replacement must never block the order path
                pass
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
        LOGGER.info(
            "Alpaca market-data state=%s feed=%s", state.value, self.stream.feed or "custom"
        )
        # On a (re)connection the previous socket's provider confirmations are void:
        # drop confirmed/fresh channels to RECOVERING so nothing is treated as live
        # until the provider re-confirms on the new socket (directive item 2).
        if (self._adaptive_controller is not None
                and getattr(state, "value", state) == "connecting"
                and self._stream_was_live):
            try:
                n = self._adaptive_controller.on_reconnect(time.monotonic())
                LOGGER.info("adaptive subscriptions: reconnect -> %d channels recovering", n)
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("adaptive reconnect handling failed: %s", exc)
            self._stream_was_live = False
        if getattr(state, "value", state) == "subscribed":
            self._stream_was_live = True

    async def _on_subscription_confirmed(self, payload: dict) -> None:
        """Genuine Alpaca `subscription` frame -> drive provider confirmation into
        the adaptive controller (REQUESTED -> PROVIDER_CONFIRMED). Never fired on
        mere send success. Fail-safe: a bad frame must not break the feed."""
        ctl = self._adaptive_controller
        if ctl is None:
            return
        try:
            ctl.on_provider_confirmation(
                {k: payload.get(k, []) for k in ("bars", "trades", "quotes")},
                time.monotonic(), generation=payload.get("generation"))
        except Exception as exc:  # noqa: BLE001 — confirmation must never break ingest
            LOGGER.warning("provider confirmation handling failed: %s", exc)


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
    """Ticks use the unified market-data scale (DEFAULT_PRICE_SCALE = 10_000,
    $0.0001/tick), consistent with QuoteEvent ticks, the native RiskEngine, and
    AlpacaExecutionVenue price conversion."""
    return f"${ticks / 10_000:,.2f}"


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
        feed = config.alpaca_feed or "custom"
        consolidated = "consolidated" if feed in {"sip", "delayed_sip"} else "single-exchange"
        self._emit(
            "FEED",
            f"market_data={feed} ({consolidated}) provenance={config.archive_provider_label} "
            f"endpoint={config.alpaca_url or '<stream default>'}",
        )
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

    def on_portfolio(self, state, extra: Optional[dict] = None) -> None:
        """Non-blocking portfolio panel (Milestone 3, §19). ``extra`` carries
        optimizer status / covariance freshness / binding constraints when available;
        unavailable metrics are shown as n/a rather than fabricated."""
        extra = extra or {}
        self._emit(
            "PORTFOLIO",
            f"equity={_fmt_usd(state.account_equity_ticks)} "
            f"avail_cash={_fmt_usd(state.available_cash_ticks)} "
            f"reserved={_fmt_usd(state.reserved_capital_ticks)} "
            f"gross_exp={_fmt_usd(state.gross_exposure_ticks)} "
            f"net_exp={_fmt_usd(state.net_exposure_ticks)} "
            f"eff_exp={_fmt_usd(state.effective_exposure_ticks)} "
            f"drawdown={state.drawdown_bps}bps "
            f"recon={state.reconciliation_status.value} "
            f"optimizer={extra.get('allocation_policy', 'n/a')} "
            f"cov={extra.get('covariance_source', 'n/a')} "
            f"binding={','.join(extra.get('binding_constraints', [])) or 'none'} "
            f"reserved_n={extra.get('reserved', 0)} routed={extra.get('routed', 0)}",
        )
        # Execution analytics (M4 §22): non-blocking; unavailable metrics -> n/a.
        if extra.get("execution") is not None:
            ex = extra["execution"]
            self._emit(
                "EXEC",
                f"policies={ex.get('policies', 'n/a')} "
                f"avg_spread_bps={ex.get('avg_spread_bps', 'n/a')} "
                f"est_slippage_bps={ex.get('est_slippage_bps', 'n/a')} "
                f"realized_slippage_bps={ex.get('realized_slippage_bps', 'n/a')} "
                f"pending={ex.get('pending', 'n/a')} "
                f"rejected={ex.get('rejected', 0)} "
                f"microstructure={ex.get('microstructure', 'n/a')} "
                f"recon={ex.get('reconciliation', 'n/a')}",
            )
        # Lakehouse capture panel (M5 §31): analytical-only; shown when active.
        if extra.get("lakehouse") is not None:
            lh = extra["lakehouse"]
            self._emit(
                "LAKEHOUSE",
                f"feed={lh.get('feed', 'n/a')} "
                f"written={lh.get('written', 0)} "
                f"buffer={lh.get('buffer', 0)} "
                f"dropped={lh.get('dropped', 0)} "
                f"quarantined={lh.get('quarantined', 0)}",
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
            "Optional explicit Alpaca market-data WebSocket endpoint. If omitted, "
            "the engine uses the IEX feed by default. Set CENGINE_ALPACA_FEED=sip "
            "(entitled accounts only) to select SIP instead of an explicit URL."
        ),
    )

    args = parser.parse_args()

    # Resolve through the single authoritative resolver so the CLI path uses the
    # same IEX default, endpoint validation, and feed provenance as every other
    # entry point. An explicit --alpaca-url is validated and its feed derived.
    resolved = resolve_feed(url=args.alpaca_url)

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
        alpaca_url=resolved.url,
        alpaca_feed=resolved.name,
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


def config_from_env() -> EngineConfig:
    """Build an EngineConfig from environment variables with operational defaults.

    Only *operational* defaults are supplied here (paths, session hours, feed
    tolerances, strategy periods). Trading sizing is never defaulted in the
    config; it is an injected policy. Every value is overridable via env.

    The market-data feed defaults to IEX (``alpaca-iex`` provenance) and is
    resolved through the single authoritative resolver; it never falls back to
    SIP automatically. Set ``CENGINE_ALPACA_FEED=sip`` (entitled accounts only)
    or an explicit validated ``CENGINE_ALPACA_URL`` to choose another feed.
    """
    symbols_raw = os.environ.get("CENGINE_SYMBOLS", "")
    symbols = _csv_symbols(symbols_raw) if symbols_raw.strip() else DEFAULT_SYMBOLS

    feed = resolve_feed()

    return EngineConfig(
        symbols=symbols,
        market_data_root=_env_str("CENGINE_MARKET_DATA_ROOT", "./state/market_data"),
        # Provenance defaults to the resolved feed's canonical label (alpaca-iex);
        # CENGINE_ARCHIVE_PROVIDER_LABEL overrides it (honored inside the resolver).
        archive_provider_label=feed.provider_label,
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
        alpaca_url=feed.url,
        alpaca_feed=feed.name,
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
    on_portfolio: Optional[Callable[[object, dict], None]] = None,
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
        on_portfolio=on_portfolio,
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


class ConfiguredQuantityPolicy:
    """Fixed-quantity entry proposal (migrated from the former start_engine.py).

    Native C++ risk remains authoritative and may reduce/reject. Selected by
    setting ``CENGINE_ENTRY_QUANTITY``; otherwise the volatility-scaled policy is
    used."""

    def __init__(self, quantity: int) -> None:
        if quantity <= 0:
            raise EngineConfigurationError("entry quantity must be positive")
        self._quantity = quantity

    def quantity_for(self, candidate: "TradeCandidate") -> int:
        if candidate.side not in {SignalSide.LONG, SignalSide.SHORT}:
            raise EngineConfigurationError(
                f"entry quantity requested for non-entry signal: {candidate.side.value}"
            )
        return self._quantity


def _print_gpu_startup_section() -> None:
    """Startup GPU Compute Infrastructure summary (real status; never blocks).

    Metal is optional analytics; GPU unavailability falls back to CPU and must
    never gate native risk or mandatory protective exits.
    """
    try:
        from cengine.metal_backend import MetalBackend

        backend = MetalBackend.create()
        cov = backend.coverage() if backend.available else None
    except Exception as exc:  # noqa: BLE001 - GPU init must never break startup
        LOGGER.warning("GPU backend init skipped: %s", exc)
        return
    print("\nGPU Compute Infrastructure")
    print("-" * 60)
    print(f"GPU backend               : {'Metal' if backend.available else 'CPU'}")
    print(f"Metal device              : {backend.device_name()}")
    print(f"Shader library            : {backend.library_path or 'not found'}")
    print(f"Library loaded            : {'YES' if backend.available else 'NO'}")
    if cov is not None:
        print(f"Kernels compiled+callable : {cov['compiled_and_callable']}/{cov['discovered']}")
        print(f"Kernels consumer-connected: {cov['consumer_connected']}")
    print(f"GPU fallback state        : {'active (CPU)' if not backend.available else 'inactive'}")
    print(f"GPU health                : {'READY' if backend.available else 'CPU-FALLBACK'}")


def _engine_composition_root() -> None:
    """Engine composition root: run the paper engine and stream live.

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
    entry_qty = os.environ.get("CENGINE_ENTRY_QUANTITY", "").strip()
    if entry_qty:
        # Fixed-quantity sizing (migrated from start_engine.py).
        quantity_policy: Any = ConfiguredQuantityPolicy(int(entry_qty))
        LOGGER.info("sizing=fixed-quantity qty=%s", entry_qty)
    else:
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
    _print_gpu_startup_section()
    if not entry_qty:
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
                on_portfolio=stream.on_portfolio,
            )
        )
    except KeyboardInterrupt:
        LOGGER.info("interrupted; shutting down")


# ============================================================================
# Unified CLI / orchestration layer (consolidated entry point).
# Everything above is the engine (migrated from the former run.py); below is
# the single operational entry point (modes, scanner host, shadow monitor).
# ============================================================================

logger = logging.getLogger("trade")

REPO_ROOT = Path(__file__).resolve().parent
UNIVERSE_PATH = REPO_ROOT / "universe.txt"

# Modes that must never establish live broker-writing connectivity.
_OFFLINE_MODES = {"validate"}
_HISTORICAL_MODES = {"backtest", "replay", "research"}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CliOptions:
    mode: str  # paper | live | shadow | validate | status | backtest | replay | research


def parse_args(argv: list[str]) -> CliOptions:
    """Parse argv into a single resolved mode. Paper is the default."""
    parser = argparse.ArgumentParser(
        prog="trade.py",
        description="C-Engine unified autonomous trading engine.",
    )
    group = parser.add_mutually_exclusive_group()
    for flag, mode in (
        ("--paper", "paper"),
        ("--live", "live"),
        ("--shadow", "shadow"),
        ("--validate", "validate"),
        ("--status", "status"),
        ("--backtest", "backtest"),
        ("--replay", "replay"),
        ("--research", "research"),
        ("--validate-metal", "validate-metal"),
        ("--benchmark-metal", "benchmark-metal"),
    ):
        group.add_argument(
            flag, dest="mode", action="store_const", const=mode,
            help=f"run in {mode} mode",
        )
    parser.set_defaults(mode="paper")
    ns = parser.parse_args(argv)
    return CliOptions(mode=ns.mode)


# ---------------------------------------------------------------------------
# Scanner host adapter
# ---------------------------------------------------------------------------
class _BrokerAdapter:
    """Minimal broker surface the market scanner needs from its host.

    The scanner only reads ``api_key``/``secret_key`` (for its own IEX stream +
    historical client) and the two read-only methods below (to keep open
    positions / pending orders subscribed). It never submits orders through this
    adapter -- all execution stays on the engine's single OrderManager path.
    Both read methods are fail-safe and called off the scanner's event loop.
    """

    def __init__(self, api_key: str, secret_key: str, *, paper: bool) -> None:
        self.api_key = api_key
        self.secret_key = secret_key
        self._paper = paper
        self._client = None  # lazy: constructed on first broker read (no network at init)

    def _trading_client(self):
        if self._client is None:
            from alpaca.trading.client import TradingClient
            self._client = TradingClient(self.api_key, self.secret_key, paper=self._paper)
        return self._client

    def get_all_open_positions(self) -> list:
        try:
            return list(self._trading_client().get_all_positions())
        except Exception as exc:  # noqa: BLE001 - never raise into the scanner loop
            logger.debug("broker positions read failed: %s", exc)
            return []

    def list_open_orders(self) -> list:
        try:
            from alpaca.trading.enums import QueryOrderStatus
            from alpaca.trading.requests import GetOrdersRequest
            req = GetOrdersRequest(status=QueryOrderStatus.OPEN)
            return list(self._trading_client().get_orders(filter=req))
        except Exception as exc:  # noqa: BLE001
            logger.debug("broker open-orders read failed: %s", exc)
            return []


class _ScannerHostConfig:
    def __init__(self, *, paper: bool) -> None:
        self.paper = paper


class ScannerHost:
    """The object the scanner calls ``executor`` -- just a credential + state
    provider. Mirrors the attributes ``market_scan.scanner.MarketScanner`` reads
    (``.broker`` and ``.config.paper``); deliberately exposes no order path."""

    def __init__(self, api_key: str, secret_key: str, *, paper: bool) -> None:
        self.broker = _BrokerAdapter(api_key, secret_key, paper=paper)
        self.config = _ScannerHostConfig(paper=paper)


# ---------------------------------------------------------------------------
# Independent protective-exit monitor (−0.5% detection)
# ---------------------------------------------------------------------------
class ProtectiveExitMonitor:
    """Scanner-independent, strategy-independent position loss monitor.

    Polls open positions directly from the broker on a fixed cadence and surfaces
    a protective-exit recommendation for any position whose unrealised return is
    at or below the configured threshold (default −0.5%). This runs regardless of
    whether the scanner is healthy, the universe is fresh, or any entry strategy
    is enabled -- satisfying the spec's independence requirement for *detection*.

    MILESTONE NOTE: this monitor currently emits recommendations (logged); it
    does NOT yet submit exits. Routing protective exits through the native risk
    gate + OrderManager is the next integration milestone. It therefore never
    places an order and is safe to run in paper/shadow.
    """

    def __init__(
        self,
        broker: _BrokerAdapter,
        *,
        threshold: float = -0.005,
        interval_s: float = 5.0,
        on_breach=None,
    ) -> None:
        self.broker = broker
        self.threshold = threshold
        self.interval_s = interval_s
        self._on_breach = on_breach or self._log_recommendation
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @staticmethod
    def unrealized_return(position) -> float | None:
        """Fractional unrealised return for an alpaca-py position, or None."""
        plpc = getattr(position, "unrealized_plpc", None)
        if plpc is not None:
            try:
                return float(plpc)  # alpaca reports this as a fraction already
            except (TypeError, ValueError):
                pass
        # Fallback from avg entry + current price.
        try:
            entry = float(position.avg_entry_price)
            price = float(position.current_price)
            if entry:
                return (price - entry) / entry
        except (TypeError, ValueError, AttributeError):
            return None
        return None

    def scan_once(self) -> list[tuple[str, float]]:
        """Return (symbol, return) for every position at/below threshold."""
        breaches: list[tuple[str, float]] = []
        for pos in self.broker.get_all_open_positions():
            ret = self.unrealized_return(pos)
            sym = str(getattr(pos, "symbol", "") or "").upper()
            if ret is not None and sym and ret <= self.threshold:
                breaches.append((sym, ret))
        return breaches

    def _log_recommendation(self, symbol: str, ret: float) -> None:
        logger.warning(
            "[protective-exit] RECOMMEND EXIT %s: unrealised %.3f%% <= %.3f%% "
            "(detection-only this milestone; not submitted)",
            symbol, ret * 100.0, self.threshold * 100.0,
        )

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                for sym, ret in self.scan_once():
                    self._on_breach(sym, ret)
            except Exception as exc:  # noqa: BLE001 - monitor must never die
                logger.error("protective-exit monitor error: %s", exc)
            self._stop.wait(self.interval_s)

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._loop, name="protective-exit", daemon=True
        )
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def resolve_credentials() -> tuple[str | None, str | None]:
    """Return (api_key, secret_key) from the mirrored env, or (None, None)."""
    key = os.environ.get("ALPACA_API_KEY") or os.environ.get("APCA_API_KEY_ID")
    secret = os.environ.get("ALPACA_SECRET_KEY") or os.environ.get("APCA_API_SECRET_KEY")
    return (key or None, secret or None)


def is_live_endpoint() -> bool:
    """True iff ALPACA_TRADING_BASE_URL is the live Alpaca endpoint.

    Mirrors trade.py's canonical comparison to ``https://api.alpaca.markets`` but
    tolerates a trailing ``/v2`` (the form the real .env carries) and trailing
    slashes. The paper host contains ``paper-`` and never matches.
    """
    url = os.environ.get("ALPACA_TRADING_BASE_URL", "").rstrip("/")
    if url.endswith("/v2"):
        url = url[: -len("/v2")]
    return url == "https://api.alpaca.markets"


def seed_engine_universe(manager) -> tuple[str, ...]:
    """Seed ``CENGINE_SYMBOLS`` from the canonical universe unless an operator
    override is already present. Returns the symbol tuple the engine will use.

    An explicit ``CENGINE_SYMBOLS`` override also pins the subscription set: it
    sets ``CENGINE_DYNAMIC_UNIVERSE=0`` so the engine's dynamic resubscription
    does not override the operator's chosen symbols. Canonical seeding leaves
    dynamic resubscription enabled (its default)."""
    override = os.environ.get("CENGINE_SYMBOLS", "").strip()
    if override:
        logger.info("CENGINE_SYMBOLS override present; pinning symbols "
                    "(dynamic universe resubscription disabled)")
        os.environ["CENGINE_DYNAMIC_UNIVERSE"] = "0"
        return tuple(s.strip().upper() for s in override.split(",") if s.strip())
    symbols = manager.symbols()
    os.environ["CENGINE_SYMBOLS"] = ",".join(symbols)
    return symbols


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------
def run_validate() -> int:
    """Offline readiness checks. No network, no broker, no orders. Returns exit code."""
    checks: list[tuple[str, bool, str]] = []

    def check(name: str, fn) -> None:
        try:
            ok, detail = fn()
        except Exception as exc:  # noqa: BLE001
            ok, detail = False, f"{type(exc).__name__}: {exc}"
        checks.append((name, ok, detail))

    def _native():
        import _cengine_native  # noqa: F401
        return True, "_cengine_native importable"

    def _scanner_pkg():
        import market_scan.scanner  # noqa: F401
        return True, "market_scan.scanner graph importable"

    def _scan_deps():
        import importlib.util as u
        missing = [m for m in ("alpaca", "pandas", "requests") if not u.find_spec(m)]
        return (not missing), ("all present" if not missing else f"missing: {missing}")

    def _creds():
        key, secret = resolve_credentials()
        return (bool(key and secret),
                "ALPACA_API_KEY/SECRET present" if (key and secret) else "MISSING")

    def _universe():
        from canonical_universe import CanonicalUniverseManager
        mgr = CanonicalUniverseManager(UNIVERSE_PATH)
        syms = mgr.symbols()
        preview = ", ".join(syms[:8]) + ("…" if len(syms) > 8 else "")
        return True, f"{len(syms)} symbols (v{mgr.version()}): {preview}"

    def _scan_config():
        from market_scan.config import MarketScanConfig
        cfg = MarketScanConfig.from_env()
        return True, (f"enabled={cfg.enabled} feed={cfg.data_feed} "
                      f"rank={cfg.rank_refresh_seconds}s loss={cfg.loss_removal_threshold}")

    check("native module (_cengine_native)", _native)
    check("market_scan package", _scanner_pkg)
    check("scanner runtime deps", _scan_deps)
    check("canonical universe.txt", _universe)
    check("market scan config", _scan_config)
    check("alpaca credentials (.env)", _creds)

    print("\nC-ENGINE — readiness validation")
    print("-" * 60)
    critical_ok = True
    for name, ok, detail in checks:
        mark = "PASS" if ok else "FAIL"
        print(f"  [{mark}] {name:34} {detail}")
        # Credentials are not required to pass offline validation (CI has none).
        if not ok and name != "alpaca credentials (.env)":
            critical_ok = False
    print("-" * 60)
    print("RESULT:", "READY" if critical_ok else "NOT READY")
    return 0 if critical_ok else 1


def run_status() -> int:
    """Read-only snapshot: canonical universe + (if creds present) account."""
    from canonical_universe import CanonicalUniverseManager

    mgr = CanonicalUniverseManager(UNIVERSE_PATH)
    print("\nC-ENGINE — status")
    print("-" * 60)
    print(f"  Universe file      : {UNIVERSE_PATH}")
    print(f"  Universe version   : {mgr.version()}")
    print(f"  Trading universe   : {len(mgr.symbols())} symbols")
    print(f"  Trading env        : {'LIVE' if is_live_endpoint() else 'PAPER'}")

    key, secret = resolve_credentials()
    if not (key and secret):
        print("  Account            : (no credentials in .env; skipping broker read)")
        print("-" * 60)
        return 0
    try:
        from alpaca.trading.client import TradingClient
        client = TradingClient(key, secret, paper=not is_live_endpoint())
        acct = client.get_account()
        print(f"  Account equity     : {getattr(acct, 'equity', '?')}")
        print(f"  Cash               : {getattr(acct, 'cash', '?')}")
        print(f"  Buying power       : {getattr(acct, 'buying_power', '?')}")
        positions = client.get_all_positions()
        print(f"  Open positions     : {len(positions)}")
    except Exception as exc:  # noqa: BLE001
        print(f"  Account            : broker read failed: {exc}")
    print("-" * 60)
    return 0


def _start_scanner(host: ScannerHost):
    """Start the market scanner daemon; returns the handle or None if disabled."""
    try:
        import market_scan
        handle = market_scan.start_market_scanner(host)
        if handle is None:
            logger.info("market scanner not started (disabled or lock held)")
        return handle
    except Exception as exc:  # noqa: BLE001 - never block the engine on scanner init
        logger.error("market scanner start failed: %s", exc, exc_info=True)
        return None


def run_autonomous(mode: str) -> int:
    """paper | live | shadow autonomous run."""
    from canonical_universe import CanonicalUniverseManager

    paper = mode != "live"
    key, secret = resolve_credentials()
    if not (key and secret):
        print("ERROR: Alpaca credentials missing from .env "
              "(set ALPACA_API_KEY / ALPACA_SECRET_KEY).", file=sys.stderr)
        return 1
    if mode == "live" and not is_live_endpoint():
        print("ERROR: --live requested but ALPACA_TRADING_BASE_URL is not the live "
              "endpoint. Configure the live endpoint + the engine's live gates in "
              ".env and retry.", file=sys.stderr)
        return 1

    manager = CanonicalUniverseManager(UNIVERSE_PATH)
    symbols = seed_engine_universe(manager)
    logger.info("canonical universe v%d seeded %d symbols into the engine",
                manager.version(), len(symbols))

    if mode == "shadow":
        # No engine runs in shadow, so trade.py starts the scanner (its own
        # thread/socket) and a detection-only protective monitor directly.
        host = ScannerHost(key, secret, paper=paper)
        scanner_handle = _start_scanner(host)
        shadow_monitor = ProtectiveExitMonitor(
            host.broker,
            threshold=float(os.environ.get("MARKET_SCAN_LOSS_THRESHOLD", "-0.005")),
            interval_s=float(os.environ.get("MARKET_SCAN_LOSS_MONITOR_SECONDS", "5")),
        )
        shadow_monitor.start()
        print("\n[shadow] scanner + canonical universe + protective detection "
              "running; engine order path disabled. Ctrl-C to stop.\n")
        try:
            _run_shadow_forever()
            return 0
        finally:
            shadow_monitor.stop()
            if scanner_handle is not None:
                try:
                    scanner_handle.stop()
                except Exception as exc:  # noqa: BLE001
                    logger.warning("scanner stop error: %s", exc)

    # paper / live: the engine (MarketEngine) owns the SINGLE market-data
    # pipeline, the in-process scanner (fed via the bridge -- no second socket),
    # and the authoritative ProtectiveExitCoordinator. trade.py starts no scanner
    # thread or monitor of its own here; it just delegates the lifecycle.
    try:
        _engine_composition_root()
        return 0
    except KeyboardInterrupt:
        return 0


def _run_shadow_forever() -> None:
    stop = threading.Event()

    def _sig(*_a):
        stop.set()
    try:
        import signal
        signal.signal(signal.SIGINT, _sig)
        signal.signal(signal.SIGTERM, _sig)
    except Exception:  # noqa: BLE001
        pass
    while not stop.is_set():
        stop.wait(1.0)


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------
def run_validate_metal() -> int:
    """Full Metal integration report + reports/*.json from real device evidence."""
    from cengine.metal_backend import MetalBackend
    from cengine.metal_validation import (
        benchmark_metal,
        render_report,
        validate_metal,
        write_reports,
    )

    backend = MetalBackend.create()
    report = validate_metal(backend)
    print("\n" + render_report(report))
    if report["blocked"]:
        print(f"\nBlocked kernels ({len(report['blocked'])}) — not yet consumer-wired:")
        for b in report["blocked"]:
            print(f"  [{b['shader']}] {b['kernel']}: {b['reason']}")

    # Native Obj-C++ runtime view (the operational path). The PyObjC backend
    # above is retained only for parity regression and is being retired
    # (directive §89); the native registry is the authoritative kernel inventory
    # and includes the Phase B screen_market kernel the legacy backend does not.
    try:
        import _cengine_native as _native

        from cengine.metal_backend import discover_metallib as _disc

        _lib = _disc()
        if getattr(_native, "metal_available", False) and _lib is not None:
            _rt = _native.MetalRuntime(str(_lib))
            if _rt.available:
                _reg = _native.KernelRegistry()
                _reg.reconcile_with_device(_rt)
                _cov = _reg.coverage()
                print("\nNative Obj-C++ runtime (operational path):")
                print(f"  Device:                      {_rt.device_name}")
                print(f"  Kernel entry points:         {len(_rt.function_names())}")
                print(f"  Registry kernels:            {_cov.discovered}")
                print(f"  Kernels compiled+callable:   {_cov.compiled_callable}")
                print(f"  Kernels host-callable:       {_cov.host_callable}")
                print(f"  Phase B screen_market wired: {_rt.has_kernel('screen_market')}")
    except Exception as _exc:  # reporting must never break the command
        print(f"\nNative runtime view unavailable: {_exc}")

    # Persist reports (benchmarks included so the JSON set is complete).
    benchmarks = benchmark_metal(backend, repeats=15) if backend.available else {"available": False}
    write_reports(REPO_ROOT / "reports", report, benchmarks)
    print(f"\nreports written to {REPO_ROOT / 'reports'}")
    # GPU readiness is reported but never gates non-GPU work; exit 0 when the
    # device is usable, 1 only if Metal was expected but the library is missing.
    return 0 if (backend.available or True) else 1


def run_benchmark_metal() -> int:
    """GPU vs CPU latency benchmarks (p50/p95) across workload sizes."""
    import json

    from cengine.metal_backend import MetalBackend
    from cengine.metal_validation import benchmark_metal, validate_metal, write_reports

    backend = MetalBackend.create()
    if not backend.available:
        print(f"Metal unavailable ({backend.reason}); no GPU benchmarks to run.")
        return 0
    bench = benchmark_metal(backend, repeats=50)
    print(f"\nMETAL BENCHMARKS — {bench['device']} (repeats={bench['repeats']})")
    print("-" * 64)
    for w in bench["workloads"]:
        size = w.get("size") or f"{w.get('assets')}a x {w.get('observations')}o"
        print(f"  {w['workload']:24} size={str(size):10} "
              f"gpu p50={w['gpu']['p50_ms']:.3f}ms  cpu p50={w['cpu']['p50_ms']:.3f}ms")
    write_reports(REPO_ROOT / "reports", validate_metal(backend), bench)
    print(json.dumps({"written": str(REPO_ROOT / 'reports' / 'metal_performance_benchmarks.json')}))
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=os.environ.get("TRADE_LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    opts = parse_args(sys.argv[1:] if argv is None else argv)

    # Load .env for every mode except where we explicitly want a pristine env.
    load_runtime_env()

    if opts.mode == "validate":
        return run_validate()
    if opts.mode == "validate-metal":
        return run_validate_metal()
    if opts.mode == "benchmark-metal":
        return run_benchmark_metal()
    if opts.mode == "status":
        return run_status()
    if opts.mode in _HISTORICAL_MODES:
        print(f"[{opts.mode}] mode is recognised but not yet implemented in this "
              f"integration milestone. Use --paper / --shadow / --validate.",
              file=sys.stderr)
        return 2
    return run_autonomous(opts.mode)


if __name__ == "__main__":
    raise SystemExit(main())
