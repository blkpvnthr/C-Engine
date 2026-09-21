"""Composition adapters joining Python contracts to native risk and Alpaca paper."""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Any

import _cengine_native as native

from cengine.execution import AlpacaExecutionVenue
from cengine.portfolio import AccountState, PositionBook
from market_data.alpaca_sip_stream import QuoteEvent, TradeEvent
from order_manager import OrderIntent, RiskDecision, Side


@dataclass(slots=True)
class RuntimeState:
    account: AccountState = field(default_factory=AccountState)
    positions: PositionBook = field(default_factory=PositionBook)
    quotes: dict[str, QuoteEvent] = field(default_factory=dict)
    trades: dict[str, TradeEvent] = field(default_factory=dict)

    def on_market_event(self, event: Any) -> None:
        if isinstance(event, QuoteEvent):
            self.quotes[event.symbol] = event
        elif isinstance(event, TradeEvent):
            self.trades[event.symbol] = event


STATE = RuntimeState()


class NativeRiskAdapter:
    policy_version = "cengine-native-v1"

    def __init__(self, state: RuntimeState = STATE) -> None:
        self.state = state
        self.engine = native.RiskEngine()

    def evaluate(self, intent: OrderIntent) -> RiskDecision:
        quote = self.state.quotes.get(intent.symbol)
        decided_ns = time.time_ns()
        if quote is None:
            return RiskDecision(
                False,
                self._decision_id(intent),
                self.policy_version,
                decided_ns,
                "authoritative quote unavailable",
            )
        try:
            account = self.state.account.snapshot()
        except RuntimeError as exc:
            return RiskDecision(
                False, self._decision_id(intent), self.policy_version, decided_ns, str(exc)
            )
        request = native.RiskRequest()
        digest = hashlib.sha256(intent.client_order_id.encode()).digest()[:8]
        request.order_id = int.from_bytes(digest, "big") or 1
        request.symbol = intent.symbol
        request.side = native.Side.BUY if intent.side is Side.BUY else native.Side.SELL
        request.quantity = intent.quantity
        request.price_ticks = intent.limit_price_ticks or (
            quote.ask_price_ticks if intent.side is Side.BUY else quote.bid_price_ticks
        )
        market = native.RiskMarketState()
        market.symbol = intent.symbol
        market.bid_ticks = quote.bid_price_ticks
        market.ask_ticks = quote.ask_price_ticks
        market.timestamp_ns = quote.timestamp_ns
        trade = self.state.trades.get(intent.symbol)
        if trade is not None:
            market.last_ticks = trade.price_ticks
        request.market = market
        native_account = native.RiskAccountState()
        native_account.position = self.state.positions.quantity(intent.symbol)
        native_account.cash_ticks = account.cash_ticks
        native_account.buying_power_ticks = account.buying_power_ticks
        native_account.gross_exposure_ticks = sum(
            abs(p.intent.quantity * (p.intent.limit_price_ticks or request.price_ticks))
            for p in self.state.positions.open_orders()
        )
        request.account = native_account
        open_state = native.OpenOrderRiskState()
        open_orders = self.state.positions.open_orders()
        open_state.count = len(open_orders)
        open_state.total_remaining_quantity = sum(x.remaining_quantity() for x in open_orders)
        request.open_orders = open_state
        request.now_ns = max(intent.created_ns, quote.received_ns, decided_ns)
        result = self.engine.evaluate(request)
        return RiskDecision(
            bool(result.accepted),
            self._decision_id(intent),
            self.policy_version,
            decided_ns,
            result.message,
            intent.quantity if result.accepted else None,
        )

    @staticmethod
    def _decision_id(intent: OrderIntent) -> str:
        return hashlib.sha256((intent.client_order_id + ":risk").encode()).hexdigest()[:24]


def build_risk_engine() -> NativeRiskAdapter:
    return NativeRiskAdapter()


def build_execution_venue() -> AlpacaExecutionVenue:
    return AlpacaExecutionVenue(account_state=STATE.account)
