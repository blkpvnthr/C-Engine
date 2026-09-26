from __future__ import annotations

from typing import Any

import pytest

from cengine.execution import AlpacaExecutionVenue
from order_manager import OrderStatus


class StubAlpacaVenue(AlpacaExecutionVenue):
    """Alpaca venue whose REST responses are supplied by the test."""

    def __init__(self, responses: list[dict[str, Any]]) -> None:
        super().__init__(
            api_key="test-key",
            secret_key="test-secret",
        )
        self._responses = iter(responses)

    async def _request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        assert method == "GET"
        assert path.startswith("/v2/orders/")
        assert payload is None
        return next(self._responses)


@pytest.mark.asyncio
async def test_full_fill_from_zero() -> None:
    venue = StubAlpacaVenue(
        [
            {
                "id": "venue-1",
                "client_order_id": "client-1",
                "status": "filled",
                "filled_qty": "10",
                "filled_avg_price": "123.45",
            }
        ]
    )

    update = await venue.reconcile_status(
        "venue-1",
        previous_cumulative_filled_quantity=0,
        previous_average_fill_price_ticks=None,
    )

    assert update.client_order_id == "client-1"
    assert update.venue_order_id == "venue-1"
    assert update.status is OrderStatus.FILLED
    assert update.cumulative_filled_quantity == 10
    assert update.last_fill_quantity == 10
    assert update.last_fill_price_ticks == 12345


@pytest.mark.asyncio
async def test_partial_then_full_fill_derives_incremental_price() -> None:
    venue = StubAlpacaVenue(
        [
            {
                "id": "venue-2",
                "client_order_id": "client-2",
                "status": "partially_filled",
                "filled_qty": "4",
                "filled_avg_price": "100.00",
            },
            {
                "id": "venue-2",
                "client_order_id": "client-2",
                "status": "filled",
                "filled_qty": "10",
                "filled_avg_price": "103.00",
            },
        ]
    )

    first = await venue.reconcile_status(
        "venue-2",
        previous_cumulative_filled_quantity=0,
        previous_average_fill_price_ticks=None,
    )

    assert first.status is OrderStatus.PARTIALLY_FILLED
    assert first.cumulative_filled_quantity == 4
    assert first.last_fill_quantity == 4
    assert first.last_fill_price_ticks == 10000

    second = await venue.reconcile_status(
        "venue-2",
        previous_cumulative_filled_quantity=4,
        previous_average_fill_price_ticks=10000.0,
    )

    assert second.status is OrderStatus.FILLED
    assert second.cumulative_filled_quantity == 10
    assert second.last_fill_quantity == 6

    # Cumulative broker notional:
    #     10 * $103 = $1,030
    #
    # Previous notional:
    #      4 * $100 = $400
    #
    # Newly observed six shares therefore represent:
    #     ($1,030 - $400) / 6 = $105
    assert second.last_fill_price_ticks == 10500


@pytest.mark.asyncio
async def test_duplicate_poll_has_zero_fill_delta() -> None:
    venue = StubAlpacaVenue(
        [
            {
                "id": "venue-3",
                "client_order_id": "client-3",
                "status": "partially_filled",
                "filled_qty": "4",
                "filled_avg_price": "100.00",
            }
        ]
    )

    update = await venue.reconcile_status(
        "venue-3",
        previous_cumulative_filled_quantity=4,
        previous_average_fill_price_ticks=10000.0,
    )

    assert update.status is OrderStatus.PARTIALLY_FILLED
    assert update.cumulative_filled_quantity == 4
    assert update.last_fill_quantity == 0
    assert update.last_fill_price_ticks is None


@pytest.mark.asyncio
async def test_canceled_order_without_new_fill() -> None:
    venue = StubAlpacaVenue(
        [
            {
                "id": "venue-4",
                "client_order_id": "client-4",
                "status": "canceled",
                "filled_qty": "0",
                "filled_avg_price": None,
            }
        ]
    )

    update = await venue.reconcile_status(
        "venue-4",
        previous_cumulative_filled_quantity=0,
        previous_average_fill_price_ticks=None,
    )

    assert update.status is OrderStatus.CANCELED
    assert update.cumulative_filled_quantity == 0
    assert update.last_fill_quantity == 0
    assert update.last_fill_price_ticks is None


@pytest.mark.asyncio
async def test_rejected_order_without_fill() -> None:
    venue = StubAlpacaVenue(
        [
            {
                "id": "venue-5",
                "client_order_id": "client-5",
                "status": "rejected",
                "filled_qty": "0",
                "filled_avg_price": None,
            }
        ]
    )

    update = await venue.reconcile_status(
        "venue-5",
        previous_cumulative_filled_quantity=0,
        previous_average_fill_price_ticks=None,
    )

    assert update.status is OrderStatus.REJECTED
    assert update.cumulative_filled_quantity == 0
    assert update.last_fill_quantity == 0


@pytest.mark.asyncio
async def test_backward_cumulative_quantity_is_rejected() -> None:
    venue = StubAlpacaVenue(
        [
            {
                "id": "venue-6",
                "client_order_id": "client-6",
                "status": "partially_filled",
                "filled_qty": "3",
                "filled_avg_price": "100.00",
            }
        ]
    )

    with pytest.raises(
        RuntimeError,
        match="Alpaca cumulative filled quantity moved backwards",
    ):
        await venue.reconcile_status(
            "venue-6",
            previous_cumulative_filled_quantity=4,
            previous_average_fill_price_ticks=10000.0,
        )


@pytest.mark.asyncio
async def test_fractional_quantity_is_rejected() -> None:
    venue = StubAlpacaVenue(
        [
            {
                "id": "venue-7",
                "client_order_id": "client-7",
                "status": "partially_filled",
                "filled_qty": "1.5",
                "filled_avg_price": "100.00",
            }
        ]
    )

    with pytest.raises(
        RuntimeError,
        match="fractional Alpaca fill unsupported",
    ):
        await venue.reconcile_status(
            "venue-7",
            previous_cumulative_filled_quantity=0,
            previous_average_fill_price_ticks=None,
        )
