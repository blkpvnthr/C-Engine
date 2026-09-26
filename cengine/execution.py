"""Alpaca Trading API adapter with paper trading as the enforced default."""

from __future__ import annotations

import asyncio
import json
import os
import time
import urllib.error
import urllib.request
from decimal import ROUND_HALF_UP, Decimal
from typing import Any, Awaitable, Callable, Optional

from cengine.portfolio import AccountSnapshot, AccountState
from order_manager import (
    ExecutionAck,
    ExecutionUpdate,
    OrderIntent,
    OrderStatus,
)

PAPER_URL = "https://paper-api.alpaca.markets"
LIVE_URL = "https://api.alpaca.markets"


class ExecutionConfigurationError(RuntimeError):
    """Raised when the execution venue is configured unsafely or incompletely."""


def trading_base_url() -> str:
    """Return the configured Alpaca trading endpoint.

    Paper trading is the default.

    Live trading is rejected unless both explicit live-trading gates are set.
    Arbitrary trading endpoints are rejected.
    """

    url = os.environ.get(
        "ALPACA_TRADING_BASE_URL",
        PAPER_URL,
    ).rstrip("/")

    if url == LIVE_URL:
        gates = (
            os.environ.get("CENGINE_ENABLE_LIVE")
            == "YES_I_ACCEPT_LIVE_TRADING_RISK",
            os.environ.get("CENGINE_LIVE_CONFIRMATION")
            == "LIVE_ORDERS_MAY_LOSE_MONEY",
        )

        if not all(gates):
            raise ExecutionConfigurationError(
                "live endpoint blocked; both explicit "
                "live-trading gates are required"
            )

    elif url != PAPER_URL:
        raise ExecutionConfigurationError(
            "trading endpoint must be the Alpaca paper URL "
            "or explicitly gated live URL"
        )

    return url


class AlpacaExecutionVenue:
    """Alpaca execution adapter.

    Responsibilities:
    - submit orders
    - cancel orders
    - replace orders
    - query authoritative order state
    - synchronize authoritative account state
    - convert cumulative Alpaca fill state into incremental ExecutionUpdate
      objects suitable for OrderManager reconciliation

    This adapter is not a risk authority. Native C++ RiskEngine remains the
    authoritative risk boundary upstream of this venue.
    """

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        base_url: Optional[str] = None,
        on_update: Optional[
            Callable[[ExecutionUpdate], Awaitable[None]]
        ] = None,
        account_state: Optional[AccountState] = None,
    ) -> None:
        self.api_key = api_key or os.environ.get(
            "ALPACA_API_KEY",
            "",
        )
        self.secret_key = secret_key or os.environ.get(
            "ALPACA_SECRET_KEY",
            "",
        )

        self.base_url = (
            base_url or trading_base_url()
        ).rstrip("/")

        if self.base_url == LIVE_URL:
            # Enforce the live gates even when the caller supplied the URL
            # directly rather than through ALPACA_TRADING_BASE_URL.
            trading_base_url()

        elif self.base_url != PAPER_URL:
            raise ExecutionConfigurationError(
                "only Alpaca paper is allowed by default"
            )

        if not self.api_key or not self.secret_key:
            raise ExecutionConfigurationError(
                "Alpaca credentials are required"
            )

        self.on_update = on_update
        self.account_state = account_state
        self._sequence = 0

    async def sync_account(self) -> AccountSnapshot:
        """Refresh authoritative account state from Alpaca."""

        data = await self._request(
            "GET",
            "/v2/account",
        )

        snapshot = AccountSnapshot(
            cash_ticks=self._price_to_ticks(data["cash"]),
            buying_power_ticks=self._price_to_ticks(
                data["buying_power"]
            ),
            equity_ticks=self._price_to_ticks(data["equity"]),
            realized_pnl_ticks=0,
            updated_ns=time.time_ns(),
        )

        if self.account_state is not None:
            self.account_state.replace_from_broker(snapshot)

        return snapshot

    async def submit(
        self,
        intent: OrderIntent,
    ) -> ExecutionAck:
        """Submit an already risk-approved order to Alpaca."""

        payload: dict[str, Any] = {
            "symbol": intent.symbol,
            "qty": str(intent.quantity),
            "side": intent.side.value,
            "type": intent.order_type.value,
            "time_in_force": intent.time_in_force.value,
            "client_order_id": intent.client_order_id,
        }

        if intent.limit_price_ticks is not None:
            payload["limit_price"] = self._ticks_to_price(
                intent.limit_price_ticks
            )

        if intent.stop_price_ticks is not None:
            payload["stop_price"] = self._ticks_to_price(
                intent.stop_price_ticks
            )

        data = await self._request(
            "POST",
            "/v2/orders",
            payload,
        )

        return ExecutionAck(
            intent.client_order_id,
            str(data["id"]),
            time.time_ns(),
        )

    async def cancel(
        self,
        *,
        client_order_id: str,
        venue_order_id: str,
    ) -> None:
        """Request cancellation of an Alpaca order.

        Cancellation is not considered complete until reconciliation observes
        the authoritative terminal state from Alpaca.
        """

        del client_order_id

        await self._request(
            "DELETE",
            f"/v2/orders/{venue_order_id}",
        )

    async def replace(
        self,
        venue_order_id: str,
        *,
        quantity: Optional[int] = None,
        limit_price_ticks: Optional[int] = None,
        stop_price_ticks: Optional[int] = None,
    ) -> dict[str, Any]:
        """Replace supported fields on an existing Alpaca order."""

        payload: dict[str, Any] = {}

        if quantity is not None:
            if quantity <= 0:
                raise ValueError("quantity must be positive")

            payload["qty"] = str(quantity)

        if limit_price_ticks is not None:
            payload["limit_price"] = self._ticks_to_price(
                limit_price_ticks
            )

        if stop_price_ticks is not None:
            payload["stop_price"] = self._ticks_to_price(
                stop_price_ticks
            )

        if not payload:
            raise ValueError(
                "replacement fields are required"
            )

        return await self._request(
            "PATCH",
            f"/v2/orders/{venue_order_id}",
            payload,
        )

    async def status(
        self,
        venue_order_id: str,
    ) -> dict[str, Any]:
        """Return Alpaca's authoritative state for an order."""

        if not venue_order_id:
            raise ValueError(
                "venue_order_id is required"
            )

        return await self._request(
            "GET",
            f"/v2/orders/{venue_order_id}",
        )

    async def reconcile_status(
        self,
        venue_order_id: str,
        *,
        previous_cumulative_filled_quantity: int = 0,
        previous_average_fill_price_ticks: Optional[float] = None,
    ) -> ExecutionUpdate:
        """Convert Alpaca cumulative order state into an incremental update.

        Alpaca's REST order representation exposes cumulative filled quantity
        and cumulative average fill price.

        OrderManager, however, requires the newly filled quantity and the price
        attributable to that quantity whenever cumulative quantity increases.

        Given:

            previous quantity = Q0
            previous average  = P0
            new quantity      = Q1
            new average       = P1

        the newly observed notional is:

            P1 * Q1 - P0 * Q0

        and therefore the average price attributable to the newly observed
        quantity is:

            (P1 * Q1 - P0 * Q0) / (Q1 - Q0)

        This preserves cumulative quantity and notional exactly enough for the
        engine's integer-tick reconciliation model.

        It does not claim that multiple exchange fills observed between polls
        were a single physical execution event.
        """

        if not venue_order_id:
            raise ValueError(
                "venue_order_id is required"
            )

        if previous_cumulative_filled_quantity < 0:
            raise ValueError(
                "previous cumulative fill cannot be negative"
            )

        if (
            previous_cumulative_filled_quantity > 0
            and previous_average_fill_price_ticks is not None
            and previous_average_fill_price_ticks <= 0
        ):
            raise ValueError(
                "previous average fill price must be positive"
            )

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

        raw_status = str(data["status"])

        if raw_status not in status_map:
            raise RuntimeError(
                f"unsupported Alpaca order status "
                f"{raw_status!r}"
            )

        filled_decimal = Decimal(
            str(data.get("filled_qty") or "0")
        )

        if filled_decimal < 0:
            raise RuntimeError(
                "Alpaca returned a negative filled quantity"
            )

        # C-Engine currently represents order quantities as integers.
        if (
            filled_decimal
            != filled_decimal.to_integral_value()
        ):
            raise RuntimeError(
                "fractional Alpaca fill unsupported by "
                "integer quantity model: "
                f"{filled_decimal}"
            )

        cumulative = int(filled_decimal)

        if (
            cumulative
            < previous_cumulative_filled_quantity
        ):
            raise RuntimeError(
                "Alpaca cumulative filled quantity "
                "moved backwards"
            )

        fill_delta = (
            cumulative
            - previous_cumulative_filled_quantity
        )

        last_fill_quantity = 0
        last_fill_price_ticks: Optional[int] = None

        if fill_delta > 0:
            broker_average = data.get(
                "filled_avg_price"
            )

            if broker_average is None:
                raise RuntimeError(
                    "Alpaca reported a new fill without "
                    "filled_avg_price"
                )

            new_average_ticks = (
                Decimal(str(broker_average))
                * Decimal("100")
            )

            if new_average_ticks <= 0:
                raise RuntimeError(
                    "Alpaca average fill price must "
                    "be positive"
                )

            if (
                previous_cumulative_filled_quantity
                == 0
            ):
                delta_average_ticks = (
                    new_average_ticks
                )

            else:
                if (
                    previous_average_fill_price_ticks
                    is None
                ):
                    raise RuntimeError(
                        "previous average fill price "
                        "required for incremental "
                        "reconciliation"
                    )

                previous_average_ticks = Decimal(
                    str(
                        previous_average_fill_price_ticks
                    )
                )

                new_notional_ticks = (
                    new_average_ticks
                    * Decimal(cumulative)
                )

                previous_notional_ticks = (
                    previous_average_ticks
                    * Decimal(
                        previous_cumulative_filled_quantity
                    )
                )

                delta_notional_ticks = (
                    new_notional_ticks
                    - previous_notional_ticks
                )

                delta_average_ticks = (
                    delta_notional_ticks
                    / Decimal(fill_delta)
                )

            last_fill_quantity = fill_delta

            last_fill_price_ticks = int(
                delta_average_ticks.to_integral_value(
                    rounding=ROUND_HALF_UP
                )
            )

            if last_fill_price_ticks <= 0:
                raise RuntimeError(
                    "derived fill price must be positive"
                )

        update = ExecutionUpdate(
            client_order_id=str(
                data["client_order_id"]
            ),
            venue_order_id=str(data["id"]),
            status=status_map[raw_status],
            event_ns=time.time_ns(),
            cumulative_filled_quantity=cumulative,
            last_fill_quantity=last_fill_quantity,
            last_fill_price_ticks=last_fill_price_ticks,
            reason=str(
                data.get("rejected_at") or ""
            ),
            venue_sequence=self._sequence,
        )

        # Validate at the venue boundary before exposing the update to the
        # rest of the engine.
        update.validate()

        if self.on_update is not None:
            await self.on_update(update)

        return update

    @staticmethod
    def _ticks_to_price(
        ticks: int,
    ) -> str:
        """Convert integer cent ticks to Alpaca's decimal price format."""

        if ticks <= 0:
            raise ValueError(
                "price ticks must be positive"
            )

        return f"{ticks / 100:.2f}"

    @staticmethod
    def _price_to_ticks(
        price: Any,
    ) -> int:
        """Convert an Alpaca decimal price to integer cent ticks."""

        ticks = int(
            (
                Decimal(str(price))
                * Decimal("100")
            ).to_integral_value(
                rounding=ROUND_HALF_UP
            )
        )

        return ticks

    async def _request(
        self,
        method: str,
        path: str,
        payload: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Perform an authenticated Alpaca Trading API request."""

        def call() -> dict[str, Any]:
            body = (
                None
                if payload is None
                else json.dumps(payload).encode(
                    "utf-8"
                )
            )

            req = urllib.request.Request(
                self.base_url + path,
                data=body,
                method=method,
                headers={
                    "APCA-API-KEY-ID": self.api_key,
                    "APCA-API-SECRET-KEY": self.secret_key,
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                },
            )

            try:
                with urllib.request.urlopen(
                    req,
                    timeout=15,
                ) as response:
                    raw = response.read()

            except urllib.error.HTTPError as exc:
                detail = (
                    exc.read()
                    .decode(
                        "utf-8",
                        "replace",
                    )
                )

                raise RuntimeError(
                    f"Alpaca {method} {path} failed: "
                    f"{exc.code} {detail}"
                ) from exc

            except urllib.error.URLError as exc:
                raise RuntimeError(
                    f"Alpaca {method} {path} "
                    f"connection failed: {exc.reason}"
                ) from exc

            if not raw:
                return {}

            try:
                decoded = json.loads(raw)

            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"Alpaca {method} {path} "
                    "returned invalid JSON"
                ) from exc

            if not isinstance(decoded, dict):
                raise RuntimeError(
                    f"Alpaca {method} {path} "
                    "returned a non-object response"
                )

            return decoded

        return await asyncio.to_thread(call)


__all__ = [
    "AlpacaExecutionVenue",
    "ExecutionConfigurationError",
    "LIVE_URL",
    "PAPER_URL",
    "trading_base_url",
]
