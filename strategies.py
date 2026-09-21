#!/usr/bin/env python3
"""
strategies.py

Concurrent strategy signal layer for the market engine.

Strategies:
    1. Turtle breakout
    2. Dual moving-average crossover
    3. APO (Absolute Price Oscillator) mean reversion

Architecture:
    normalized bars
          |
          +--> TurtleStrategy --------+
          +--> DualMAStrategy --------+--> StrategyCoordinator
          +--> APOMeanReversion ------+          |
                                                 v
                                          TradeCandidate
                                                 |
                                      caller maps candidate
                                          to OrderIntent
                                                 |
                                                 v
                                          OrderManager
                                                 |
                                      native C++ RiskEngine
                                                 |
                                      C++ order book / venue

Important:
- Strategies generate candidates only.
- They do not approve risk or submit/cancel orders.
- All parameters are explicit configuration. There are no hidden market
  assumptions or fabricated prices/timestamps.
- All three strategies can process every bar simultaneously.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum
from math import fsum
from typing import Iterable, Optional, Protocol, Sequence


class SignalSide(str, Enum):
    LONG = "long"
    SHORT = "short"
    EXIT_LONG = "exit_long"
    EXIT_SHORT = "exit_short"


class StrategyName(str, Enum):
    TURTLE = "turtle"
    DUAL_MA = "dual_ma"
    APO_MEAN_REVERSION = "apo_mean_reversion"


@dataclass(frozen=True, slots=True)
class Bar:
    symbol: str
    timestamp_ns: int
    open_ticks: int
    high_ticks: int
    low_ticks: int
    close_ticks: int
    volume: int = 0

    def validate(self) -> None:
        if not self.symbol.strip():
            raise ValueError("symbol is required")
        if self.timestamp_ns <= 0:
            raise ValueError("timestamp_ns must be positive")
        if min(
            self.open_ticks,
            self.high_ticks,
            self.low_ticks,
            self.close_ticks,
        ) <= 0:
            raise ValueError("OHLC prices must be positive")
        if self.low_ticks > self.high_ticks:
            raise ValueError("low_ticks cannot exceed high_ticks")
        if not (
            self.low_ticks <= self.open_ticks <= self.high_ticks
            and self.low_ticks <= self.close_ticks <= self.high_ticks
        ):
            raise ValueError("open/close must lie inside low/high")
        if self.volume < 0:
            raise ValueError("volume cannot be negative")


@dataclass(frozen=True, slots=True)
class TradeCandidate:
    strategy: StrategyName
    symbol: str
    side: SignalSide
    timestamp_ns: int
    reference_price_ticks: int
    reason: str

    # Strategy-derived measurements for audit/research. These are not risk
    # approvals and do not determine final order quantity.
    signal_value: Optional[float] = None
    fast_value: Optional[float] = None
    slow_value: Optional[float] = None
    upper_level_ticks: Optional[int] = None
    lower_level_ticks: Optional[int] = None


class Strategy(Protocol):
    @property
    def name(self) -> StrategyName:
        ...

    def on_bar(self, bar: Bar) -> tuple[TradeCandidate, ...]:
        ...


@dataclass(frozen=True, slots=True)
class TurtleConfig:
    entry_lookback: int
    exit_lookback: int

    def validate(self) -> None:
        if self.entry_lookback < 2:
            raise ValueError("entry_lookback must be >= 2")
        if self.exit_lookback < 1:
            raise ValueError("exit_lookback must be >= 1")
        if self.exit_lookback >= self.entry_lookback:
            raise ValueError(
                "exit_lookback must be shorter than entry_lookback"
            )


class TurtleStrategy:
    """Donchian-channel breakout/exit strategy.

    The current bar is compared against channels formed strictly from prior
    bars, avoiding look-ahead bias.

    Entry:
        close > prior N-bar high -> LONG
        close < prior N-bar low  -> SHORT

    Exit:
        long state and close < prior exit-window low  -> EXIT_LONG
        short state and close > prior exit-window high -> EXIT_SHORT

    Position state here is signal state only. Authoritative positions belong
    to the execution/account layer and should reconcile this state externally.
    """

    def __init__(self, config: TurtleConfig) -> None:
        config.validate()
        self.config = config
        self._bars: dict[str, deque[Bar]] = {}
        self._signal_state: dict[str, SignalSide] = {}

    @property
    def name(self) -> StrategyName:
        return StrategyName.TURTLE

    def on_bar(self, bar: Bar) -> tuple[TradeCandidate, ...]:
        bar.validate()
        symbol = bar.symbol.strip().upper()
        history = self._bars.setdefault(
            symbol,
            deque(maxlen=self.config.entry_lookback),
        )

        candidates: list[TradeCandidate] = []

        if len(history) >= self.config.entry_lookback:
            entry_high = max(x.high_ticks for x in history)
            entry_low = min(x.low_ticks for x in history)

            exit_history = list(history)[-self.config.exit_lookback:]
            exit_high = max(x.high_ticks for x in exit_history)
            exit_low = min(x.low_ticks for x in exit_history)

            state = self._signal_state.get(symbol)

            if state == SignalSide.LONG and bar.close_ticks < exit_low:
                candidates.append(
                    TradeCandidate(
                        strategy=self.name,
                        symbol=symbol,
                        side=SignalSide.EXIT_LONG,
                        timestamp_ns=bar.timestamp_ns,
                        reference_price_ticks=bar.close_ticks,
                        upper_level_ticks=exit_high,
                        lower_level_ticks=exit_low,
                        reason="close broke below prior Turtle exit channel",
                    )
                )
                self._signal_state.pop(symbol, None)

            elif state == SignalSide.SHORT and bar.close_ticks > exit_high:
                candidates.append(
                    TradeCandidate(
                        strategy=self.name,
                        symbol=symbol,
                        side=SignalSide.EXIT_SHORT,
                        timestamp_ns=bar.timestamp_ns,
                        reference_price_ticks=bar.close_ticks,
                        upper_level_ticks=exit_high,
                        lower_level_ticks=exit_low,
                        reason="close broke above prior Turtle exit channel",
                    )
                )
                self._signal_state.pop(symbol, None)

            elif state is None:
                if bar.close_ticks > entry_high:
                    candidates.append(
                        TradeCandidate(
                            strategy=self.name,
                            symbol=symbol,
                            side=SignalSide.LONG,
                            timestamp_ns=bar.timestamp_ns,
                            reference_price_ticks=bar.close_ticks,
                            upper_level_ticks=entry_high,
                            lower_level_ticks=entry_low,
                            reason="close broke above prior Turtle entry channel",
                        )
                    )
                    self._signal_state[symbol] = SignalSide.LONG

                elif bar.close_ticks < entry_low:
                    candidates.append(
                        TradeCandidate(
                            strategy=self.name,
                            symbol=symbol,
                            side=SignalSide.SHORT,
                            timestamp_ns=bar.timestamp_ns,
                            reference_price_ticks=bar.close_ticks,
                            upper_level_ticks=entry_high,
                            lower_level_ticks=entry_low,
                            reason="close broke below prior Turtle entry channel",
                        )
                    )
                    self._signal_state[symbol] = SignalSide.SHORT

        history.append(bar)
        return tuple(candidates)


@dataclass(frozen=True, slots=True)
class DualMAConfig:
    fast_period: int
    slow_period: int

    def validate(self) -> None:
        if self.fast_period < 1:
            raise ValueError("fast_period must be positive")
        if self.slow_period < 2:
            raise ValueError("slow_period must be >= 2")
        if self.fast_period >= self.slow_period:
            raise ValueError("fast_period must be less than slow_period")


class DualMAStrategy:
    """Dual simple-moving-average crossover.

    A candidate is emitted only on a confirmed sign change of fast - slow,
    rather than on every bar while one average remains above the other.
    """

    def __init__(self, config: DualMAConfig) -> None:
        config.validate()
        self.config = config
        self._closes: dict[str, deque[int]] = {}
        self._previous_difference: dict[str, float] = {}

    @property
    def name(self) -> StrategyName:
        return StrategyName.DUAL_MA

    def on_bar(self, bar: Bar) -> tuple[TradeCandidate, ...]:
        bar.validate()
        symbol = bar.symbol.strip().upper()
        closes = self._closes.setdefault(
            symbol,
            deque(maxlen=self.config.slow_period),
        )
        closes.append(bar.close_ticks)

        if len(closes) < self.config.slow_period:
            return ()

        values = list(closes)
        fast = fsum(values[-self.config.fast_period:]) / self.config.fast_period
        slow = fsum(values) / self.config.slow_period
        difference = fast - slow

        previous = self._previous_difference.get(symbol)
        self._previous_difference[symbol] = difference

        if previous is None:
            return ()

        if previous <= 0.0 and difference > 0.0:
            return (
                TradeCandidate(
                    strategy=self.name,
                    symbol=symbol,
                    side=SignalSide.LONG,
                    timestamp_ns=bar.timestamp_ns,
                    reference_price_ticks=bar.close_ticks,
                    signal_value=difference,
                    fast_value=fast,
                    slow_value=slow,
                    reason="fast MA crossed above slow MA",
                ),
            )

        if previous >= 0.0 and difference < 0.0:
            return (
                TradeCandidate(
                    strategy=self.name,
                    symbol=symbol,
                    side=SignalSide.SHORT,
                    timestamp_ns=bar.timestamp_ns,
                    reference_price_ticks=bar.close_ticks,
                    signal_value=difference,
                    fast_value=fast,
                    slow_value=slow,
                    reason="fast MA crossed below slow MA",
                ),
            )

        return ()


@dataclass(frozen=True, slots=True)
class APOConfig:
    fast_period: int
    slow_period: int
    entry_threshold_ticks: float
    exit_threshold_ticks: float

    def validate(self) -> None:
        if self.fast_period < 1:
            raise ValueError("fast_period must be positive")
        if self.slow_period < 2:
            raise ValueError("slow_period must be >= 2")
        if self.fast_period >= self.slow_period:
            raise ValueError("fast_period must be less than slow_period")
        if self.entry_threshold_ticks <= 0:
            raise ValueError("entry_threshold_ticks must be positive")
        if self.exit_threshold_ticks < 0:
            raise ValueError("exit_threshold_ticks cannot be negative")
        if self.exit_threshold_ticks >= self.entry_threshold_ticks:
            raise ValueError(
                "exit_threshold_ticks must be below entry_threshold_ticks"
            )


class APOMeanReversionStrategy:
    """Mean reversion using the Absolute Price Oscillator.

    APO = EMA_fast(close) - EMA_slow(close)

    Contrarian interpretation:
        APO <= -entry_threshold -> LONG
        APO >= +entry_threshold -> SHORT

    Exit:
        long signal state exits when APO >= -exit_threshold
        short signal state exits when APO <= +exit_threshold

    Thresholds are explicitly expressed in the engine's integer price ticks.
    No default thresholds are chosen here because their appropriate scale
    depends on instrument, bar interval, volatility, and price scale.
    """

    def __init__(self, config: APOConfig) -> None:
        config.validate()
        self.config = config
        self._fast_ema: dict[str, float] = {}
        self._slow_ema: dict[str, float] = {}
        self._samples: dict[str, int] = {}
        self._signal_state: dict[str, SignalSide] = {}

        self._fast_alpha = 2.0 / (config.fast_period + 1.0)
        self._slow_alpha = 2.0 / (config.slow_period + 1.0)

    @property
    def name(self) -> StrategyName:
        return StrategyName.APO_MEAN_REVERSION

    def on_bar(self, bar: Bar) -> tuple[TradeCandidate, ...]:
        bar.validate()
        symbol = bar.symbol.strip().upper()
        close = float(bar.close_ticks)

        if symbol not in self._fast_ema:
            self._fast_ema[symbol] = close
            self._slow_ema[symbol] = close
            self._samples[symbol] = 1
            return ()

        fast = (
            self._fast_alpha * close
            + (1.0 - self._fast_alpha) * self._fast_ema[symbol]
        )
        slow = (
            self._slow_alpha * close
            + (1.0 - self._slow_alpha) * self._slow_ema[symbol]
        )

        self._fast_ema[symbol] = fast
        self._slow_ema[symbol] = slow
        self._samples[symbol] += 1

        # Do not trade during EMA warm-up.
        if self._samples[symbol] < self.config.slow_period:
            return ()

        apo = fast - slow
        state = self._signal_state.get(symbol)

        if state == SignalSide.LONG:
            if apo >= -self.config.exit_threshold_ticks:
                self._signal_state.pop(symbol, None)
                return (
                    TradeCandidate(
                        strategy=self.name,
                        symbol=symbol,
                        side=SignalSide.EXIT_LONG,
                        timestamp_ns=bar.timestamp_ns,
                        reference_price_ticks=bar.close_ticks,
                        signal_value=apo,
                        fast_value=fast,
                        slow_value=slow,
                        reason="APO mean-reverted toward zero from below",
                    ),
                )
            return ()

        if state == SignalSide.SHORT:
            if apo <= self.config.exit_threshold_ticks:
                self._signal_state.pop(symbol, None)
                return (
                    TradeCandidate(
                        strategy=self.name,
                        symbol=symbol,
                        side=SignalSide.EXIT_SHORT,
                        timestamp_ns=bar.timestamp_ns,
                        reference_price_ticks=bar.close_ticks,
                        signal_value=apo,
                        fast_value=fast,
                        slow_value=slow,
                        reason="APO mean-reverted toward zero from above",
                    ),
                )
            return ()

        if apo <= -self.config.entry_threshold_ticks:
            self._signal_state[symbol] = SignalSide.LONG
            return (
                TradeCandidate(
                    strategy=self.name,
                    symbol=symbol,
                    side=SignalSide.LONG,
                    timestamp_ns=bar.timestamp_ns,
                    reference_price_ticks=bar.close_ticks,
                    signal_value=apo,
                    fast_value=fast,
                    slow_value=slow,
                    reason="negative APO exceeded mean-reversion entry threshold",
                ),
            )

        if apo >= self.config.entry_threshold_ticks:
            self._signal_state[symbol] = SignalSide.SHORT
            return (
                TradeCandidate(
                    strategy=self.name,
                    symbol=symbol,
                    side=SignalSide.SHORT,
                    timestamp_ns=bar.timestamp_ns,
                    reference_price_ticks=bar.close_ticks,
                    signal_value=apo,
                    fast_value=fast,
                    slow_value=slow,
                    reason="positive APO exceeded mean-reversion entry threshold",
                ),
            )

        return ()


@dataclass(frozen=True, slots=True)
class StrategyBatch:
    symbol: str
    timestamp_ns: int
    candidates: tuple[TradeCandidate, ...]

    @property
    def has_candidates(self) -> bool:
        return bool(self.candidates)


class StrategyCoordinator:
    """Runs all registered strategies against every incoming bar.

    The coordinator intentionally does not select a single "winning" strategy.
    All strategies receive the same bar and may emit candidates simultaneously.
    Conflict resolution and portfolio/risk constraints belong downstream.
    """

    def __init__(self, strategies: Sequence[Strategy]) -> None:
        if not strategies:
            raise ValueError("at least one strategy is required")

        names = [strategy.name for strategy in strategies]
        if len(set(names)) != len(names):
            raise ValueError("strategy names must be unique")

        self._strategies = tuple(strategies)
        self._last_timestamp: dict[str, int] = {}

    @property
    def strategies(self) -> tuple[Strategy, ...]:
        return self._strategies

    def on_bar(self, bar: Bar) -> StrategyBatch:
        bar.validate()
        symbol = bar.symbol.strip().upper()

        previous = self._last_timestamp.get(symbol)
        if previous is not None and bar.timestamp_ns <= previous:
            raise ValueError(
                f"non-monotonic bar timestamp for {symbol}: "
                f"{bar.timestamp_ns} <= {previous}"
            )

        self._last_timestamp[symbol] = bar.timestamp_ns

        candidates: list[TradeCandidate] = []
        for strategy in self._strategies:
            candidates.extend(strategy.on_bar(bar))

        return StrategyBatch(
            symbol=symbol,
            timestamp_ns=bar.timestamp_ns,
            candidates=tuple(candidates),
        )


@dataclass(frozen=True, slots=True)
class StrategySetConfig:
    turtle: TurtleConfig
    dual_ma: DualMAConfig
    apo: APOConfig


def build_strategy_coordinator(
    config: StrategySetConfig,
) -> StrategyCoordinator:
    """Construct all three strategies.

    Configuration is mandatory; this function intentionally supplies no
    trading-period or threshold defaults.
    """
    return StrategyCoordinator((
        TurtleStrategy(config.turtle),
        DualMAStrategy(config.dual_ma),
        APOMeanReversionStrategy(config.apo),
    ))


def candidate_direction(candidate: TradeCandidate) -> int:
    """Return directional intent for downstream conflict aggregation.

    +1 = buy/long pressure
    -1 = sell/short pressure

    Exit signals are represented as the transaction needed to close the
    strategy's signal state:
        EXIT_LONG  -> sell (-1)
        EXIT_SHORT -> buy (+1)

    This is not an order quantity and is not a risk decision.
    """
    if candidate.side in {SignalSide.LONG, SignalSide.EXIT_SHORT}:
        return 1
    return -1


def group_candidates_by_direction(
    candidates: Iterable[TradeCandidate],
) -> tuple[tuple[TradeCandidate, ...], tuple[TradeCandidate, ...]]:
    """Return (buy_direction, sell_direction) without discarding conflicts."""
    buy: list[TradeCandidate] = []
    sell: list[TradeCandidate] = []

    for candidate in candidates:
        if candidate_direction(candidate) > 0:
            buy.append(candidate)
        else:
            sell.append(candidate)

    return tuple(buy), tuple(sell)


__all__ = [
    "APOConfig",
    "APOMeanReversionStrategy",
    "Bar",
    "DualMAConfig",
    "DualMAStrategy",
    "SignalSide",
    "Strategy",
    "StrategyBatch",
    "StrategyCoordinator",
    "StrategyName",
    "StrategySetConfig",
    "TradeCandidate",
    "TurtleConfig",
    "TurtleStrategy",
    "build_strategy_coordinator",
    "candidate_direction",
    "group_candidates_by_direction",
]
