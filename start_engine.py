#!/usr/bin/env python3

from __future__ import annotations

import asyncio
import importlib
import os

from cengine.portfolio import PositionBook
from run import (
    EngineConfig,
    EngineConfigurationError,
    ExitQuantityProvider,
    QuantityPolicy,
    run_engine,
)
from strategies import SignalSide, TradeCandidate


def required_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise EngineConfigurationError(
            f"required environment variable {name} is not configured"
        )
    return value


def required_int(name: str) -> int:
    raw = required_env(name)
    try:
        value = int(raw)
    except ValueError as exc:
        raise EngineConfigurationError(
            f"{name} must be an integer"
        ) from exc

    if value <= 0:
        raise EngineConfigurationError(
            f"{name} must be greater than zero"
        )

    return value


def required_float(name: str) -> float:
    raw = required_env(name)
    try:
        return float(raw)
    except ValueError as exc:
        raise EngineConfigurationError(
            f"{name} must be numeric"
        ) from exc


class ConfiguredQuantityPolicy(QuantityPolicy):
    """Explicit entry proposal; native C++ risk remains authoritative."""

    def __init__(self, quantity: int) -> None:
        if quantity <= 0:
            raise EngineConfigurationError(
                "entry quantity must be positive"
            )
        self._quantity = quantity

    def quantity_for(self, candidate: TradeCandidate) -> int:
        if candidate.side not in {
            SignalSide.LONG,
            SignalSide.SHORT,
        }:
            raise EngineConfigurationError(
                f"entry quantity requested for non-entry signal: "
                f"{candidate.side.value}"
            )

        return self._quantity


class StrategyPositionExitProvider(ExitQuantityProvider):
    """Close only inventory belonging to the strategy issuing the exit."""

    def __init__(self, positions: PositionBook) -> None:
        self._positions = positions

    def quantity_to_close(self, candidate: TradeCandidate) -> int:
        strategy_id = candidate.strategy.value

        quantity = int(
            self._positions.quantity(
                candidate.symbol,
                strategy_id,
            )
        )

        if candidate.side == SignalSide.EXIT_LONG:
            if quantity <= 0:
                raise EngineConfigurationError(
                    f"{candidate.symbol}/{strategy_id}: "
                    f"EXIT_LONG has no strategy-owned long position"
                )
            return quantity

        if candidate.side == SignalSide.EXIT_SHORT:
            if quantity >= 0:
                raise EngineConfigurationError(
                    f"{candidate.symbol}/{strategy_id}: "
                    f"EXIT_SHORT has no strategy-owned short position"
                )
            return abs(quantity)

        raise EngineConfigurationError(
            f"exit quantity requested for non-exit signal: "
            f"{candidate.side.value}"
        )


def build_config() -> EngineConfig:
    symbols = tuple(
        symbol.strip().upper()
        for symbol in required_env("CENGINE_SYMBOLS").split(",")
        if symbol.strip()
    )

    if not symbols:
        raise EngineConfigurationError(
            "CENGINE_SYMBOLS contains no symbols"
        )

    alpaca_url = os.environ.get("ALPACA_STREAM_URL", "").strip() or None

    config = EngineConfig(
        symbols=symbols,
        market_data_root=required_env("CENGINE_MARKET_DATA_ROOT"),
        archive_provider_label=required_env(
            "CENGINE_ARCHIVE_PROVIDER_LABEL"
        ),
        native_adapter_module=required_env(
            "ENGINE_NATIVE_ADAPTER_MODULE"
        ),
        turtle_entry_lookback=required_int(
            "CENGINE_TURTLE_ENTRY_LOOKBACK"
        ),
        turtle_exit_lookback=required_int(
            "CENGINE_TURTLE_EXIT_LOOKBACK"
        ),
        dual_ma_fast_period=required_int(
            "CENGINE_DUAL_MA_FAST_PERIOD"
        ),
        dual_ma_slow_period=required_int(
            "CENGINE_DUAL_MA_SLOW_PERIOD"
        ),
        apo_fast_period=required_int(
            "CENGINE_APO_FAST_PERIOD"
        ),
        apo_slow_period=required_int(
            "CENGINE_APO_SLOW_PERIOD"
        ),
        apo_entry_threshold_ticks=required_float(
            "CENGINE_APO_ENTRY_THRESHOLD_TICKS"
        ),
        apo_exit_threshold_ticks=required_float(
            "CENGINE_APO_EXIT_THRESHOLD_TICKS"
        ),
        alpaca_url=alpaca_url,
    )

    config.validate()
    return config


async def main() -> None:
    config = build_config()

    adapter_module = importlib.import_module(
        config.native_adapter_module
    )

    state = getattr(adapter_module, "STATE", None)
    positions = getattr(state, "positions", None)

    if positions is None:
        raise EngineConfigurationError(
            f"{config.native_adapter_module!r} does not expose "
            "STATE.positions"
        )

    entry_policy = ConfiguredQuantityPolicy(
        required_int("CENGINE_ENTRY_QUANTITY")
    )

    exit_provider = StrategyPositionExitProvider(
        positions
    )

    await run_engine(
        config=config,
        quantity_policy=entry_policy,
        exit_quantity_provider=exit_provider,
    )


if __name__ == "__main__":
    asyncio.run(main())
