"""Alpaca Trading API adapter with paper trading as the enforced default."""

from __future__ import annotations

import asyncio
import json
import os
import time
import urllib.error
import urllib.request
from typing import Any, Awaitable, Callable, Optional

from cengine.portfolio import (
    AccountSnapshot,
    AccountState,
    BrokerOrder,
    BrokerPosition,
    PositionBook,
)
from order_manager import ExecutionAck, ExecutionUpdate, OrderIntent, OrderStatus, Side

PAPER_URL = "https://paper-api.alpaca.markets"
LIVE_URL = "https://api.alpaca.markets"


class ExecutionConfigurationError(RuntimeError):
    pass


def trading_base_url() -> str:
    url = os.environ.get("ALPACA_TRADING_BASE_URL", PAPER_URL).rstrip("/")
    _validate_base_url(url)
    return url


def _validate_base_url(url: str) -> None:
    if url == LIVE_URL:
        gates = (
            os.environ.get("CENGINE_ENABLE_LIVE") == "YES_I_ACCEPT_LIVE_TRADING_RISK",
            os.environ.get("CENGINE_LIVE_CONFIRMATION") == "LIVE_ORDERS_MAY_LOSE_MONEY",
            os.environ.get("CENGINE_LIVE_RECONCILIATION_CERTIFIED")
            == "PAPER_REPLAY_AND_BROKER_STATE_AGREE",
        )
        if not all(gates):
            raise ExecutionConfigurationError(
                "live endpoint blocked; all three explicit live-trading gates are required"
            )
    elif url != PAPER_URL:
        raise ExecutionConfigurationError(
            "trading endpoint must be the Alpaca paper URL or explicitly gated live URL"
        )


class AlpacaExecutionVenue:
    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        base_url: Optional[str] = None,
        on_update: Optional[Callable[[ExecutionUpdate], Awaitable[None]]] = None,
        account_state: Optional[AccountState] = None,
        position_book: Optional[PositionBook] = None,
    ) -> None:
        self.api_key = api_key or os.environ.get("ALPACA_API_KEY", "")
        self.secret_key = secret_key or os.environ.get("ALPACA_SECRET_KEY", "")
        self.base_url = (base_url or trading_base_url()).rstrip("/")
        _validate_base_url(self.base_url)
        if not self.api_key or not self.secret_key:
            raise ExecutionConfigurationError("Alpaca credentials are required")
        self.on_update = on_update
        self.account_state = account_state
        self.position_book = position_book
        self._sequence = 0
        self._reported_fills: dict[str, int] = {}
        self._reported_fill_notional_ticks: dict[str, int] = {}

    async def sync_account(self) -> AccountSnapshot:
        data = await self._request("GET", "/v2/account")
        snapshot = AccountSnapshot(
            cash_ticks=self._price_to_ticks(data["cash"]),
            buying_power_ticks=self._price_to_ticks(data["buying_power"]),
            equity_ticks=self._price_to_ticks(data["equity"]),
            realized_pnl_ticks=0,
            updated_ns=time.time_ns(),
        )
        if self.account_state is not None:
            self.account_state.replace_from_broker(snapshot)
        return snapshot

    async def startup_reconcile(
        self,
    ) -> tuple[AccountSnapshot, tuple[BrokerPosition, ...], tuple[BrokerOrder, ...]]:
        account, raw_positions, raw_orders = await asyncio.gather(
            self.sync_account(),
            self._request("GET", "/v2/positions"),
            self._request("GET", "/v2/orders?status=open&nested=false&limit=500"),
        )
        positions = tuple(
            BrokerPosition(
                symbol=str(item["symbol"]).upper(),
                quantity=int(float(item["qty"])),
                average_entry_price_ticks=self._price_to_ticks(item["avg_entry_price"]),
            )
            for item in raw_positions
        )
        orders = tuple(self._broker_order(item) for item in raw_orders)
        if self.position_book is not None:
            self.position_book.replace_from_broker(positions, orders, time.time_ns())
            self.position_book.assert_consistent()
        return account, positions, orders

    async def submit(self, intent: OrderIntent) -> ExecutionAck:
        payload: dict[str, Any] = {
            "symbol": intent.symbol,
            "qty": str(intent.quantity),
            "side": intent.side.value,
            "type": intent.order_type.value,
            "time_in_force": intent.time_in_force.value,
            "client_order_id": intent.client_order_id,
        }
        if intent.limit_price_ticks is not None:
            payload["limit_price"] = self._ticks_to_price(intent.limit_price_ticks)
        if intent.stop_price_ticks is not None:
            payload["stop_price"] = self._ticks_to_price(intent.stop_price_ticks)
        data = await self._request("POST", "/v2/orders", payload)
        return ExecutionAck(intent.client_order_id, str(data["id"]), time.time_ns())

    async def cancel(self, *, client_order_id: str, venue_order_id: str) -> None:
        await self._request("DELETE", f"/v2/orders/{venue_order_id}")

    async def replace(
        self,
        venue_order_id: str,
        *,
        quantity: Optional[int] = None,
        limit_price_ticks: Optional[int] = None,
        stop_price_ticks: Optional[int] = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        if quantity is not None:
            if quantity <= 0:
                raise ValueError("quantity must be positive")
            payload["qty"] = str(quantity)
        if limit_price_ticks is not None:
            payload["limit_price"] = self._ticks_to_price(limit_price_ticks)
        if stop_price_ticks is not None:
            payload["stop_price"] = self._ticks_to_price(stop_price_ticks)
        if not payload:
            raise ValueError("replacement fields are required")
        return await self._request("PATCH", f"/v2/orders/{venue_order_id}", payload)

    async def status(self, venue_order_id: str) -> dict[str, Any]:
        return await self._request("GET", f"/v2/orders/{venue_order_id}")

    async def reconcile_status(self, venue_order_id: str) -> ExecutionUpdate:
        data = await self.status(venue_order_id)
        self._sequence += 1
        status_map = {
            "new": OrderStatus.SUBMITTED,
            "accepted": OrderStatus.SUBMITTED,
            "partially_filled": OrderStatus.PARTIALLY_FILLED,
            "filled": OrderStatus.FILLED,
            "canceled": OrderStatus.CANCELED,
            "expired": OrderStatus.EXPIRED,
            "rejected": OrderStatus.REJECTED,
        }
        raw = str(data["status"])
        if raw not in status_map:
            raise RuntimeError(f"unsupported Alpaca order status {raw!r}")
        cumulative = int(float(data.get("filled_qty") or 0))
        previous = self._reported_fills.get(venue_order_id, 0)
        if cumulative < previous:
            raise RuntimeError("broker cumulative fill moved backwards")
        fill_delta = cumulative - previous
        price = data.get("filled_avg_price")
        cumulative_notional = (
            cumulative * self._price_to_ticks(price) if price and cumulative else 0
        )
        previous_notional = self._reported_fill_notional_ticks.get(venue_order_id, 0)
        delta_price = (
            (cumulative_notional - previous_notional) // fill_delta if fill_delta else None
        )
        update = ExecutionUpdate(
            client_order_id=str(data["client_order_id"]),
            venue_order_id=str(data["id"]),
            status=status_map[raw],
            event_ns=time.time_ns(),
            cumulative_filled_quantity=cumulative,
            last_fill_quantity=fill_delta,
            last_fill_price_ticks=delta_price,
            reason=str(data.get("rejected_at") or ""),
            venue_sequence=self._sequence,
        )
        self._reported_fills[venue_order_id] = cumulative
        self._reported_fill_notional_ticks[venue_order_id] = cumulative_notional
        if self.on_update is not None:
            await self.on_update(update)
        return update

    @staticmethod
    def _broker_order(data: dict[str, Any]) -> BrokerOrder:
        return BrokerOrder(
            client_order_id=str(data["client_order_id"]),
            venue_order_id=str(data["id"]),
            symbol=str(data["symbol"]).upper(),
            side=Side(str(data["side"])),
            quantity=int(float(data["qty"])),
            filled_quantity=int(float(data.get("filled_qty") or 0)),
            status=str(data["status"]),
        )

    @staticmethod
    def _ticks_to_price(ticks: int) -> str:
        if ticks <= 0:
            raise ValueError("price ticks must be positive")
        return f"{ticks / 100:.2f}"

    @staticmethod
    def _price_to_ticks(price: Any) -> int:
        from decimal import ROUND_HALF_UP, Decimal

        return int((Decimal(str(price)) * 100).to_integral_value(rounding=ROUND_HALF_UP))

    async def _request(
        self, method: str, path: str, payload: Optional[dict[str, Any]] = None
    ) -> dict[str, Any]:
        def call() -> dict[str, Any]:
            body = None if payload is None else json.dumps(payload).encode()
            req = urllib.request.Request(
                self.base_url + path,
                data=body,
                method=method,
                headers={
                    "APCA-API-KEY-ID": self.api_key,
                    "APCA-API-SECRET-KEY": self.secret_key,
                    "Content-Type": "application/json",
                },
            )
            try:
                with urllib.request.urlopen(req, timeout=15) as response:
                    raw = response.read()
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", "replace")
                raise RuntimeError(f"Alpaca {method} {path} failed: {exc.code} {detail}") from exc
            return {} if not raw else json.loads(raw)

        return await asyncio.to_thread(call)
