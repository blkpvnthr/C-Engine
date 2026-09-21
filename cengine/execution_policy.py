"""Explicit conversion of candidates into executable order parameters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from order_manager import OrderType, TimeInForce
from strategies import TradeCandidate


@dataclass(frozen=True, slots=True)
class ExecutionInstruction:
    order_type: OrderType
    time_in_force: TimeInForce
    limit_price_ticks: int | None = None
    stop_price_ticks: int | None = None


class ExecutionPolicy(Protocol):
    def instruction_for(self, candidate: TradeCandidate) -> ExecutionInstruction: ...


@dataclass(frozen=True, slots=True)
class ConfiguredExecutionPolicy:
    """Policy values are mandatory constructor inputs; there are no trading defaults."""

    order_type: OrderType
    time_in_force: TimeInForce

    def instruction_for(self, candidate: TradeCandidate) -> ExecutionInstruction:
        if self.order_type is not OrderType.MARKET:
            raise ValueError("non-market policies must provide authoritative per-candidate prices")
        return ExecutionInstruction(self.order_type, self.time_in_force)
