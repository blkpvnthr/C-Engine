"""Atomic capital reservation ledger (Milestone 3, §16)."""

from __future__ import annotations

from cengine.capital_reservation import CapitalReservationLedger
from cengine.reservations import RiskReservationBook
from order_manager import Side


def test_buy_reserve_within_available():
    led = CapitalReservationLedger()
    ok, reason = led.try_reserve(decision_id="d1", symbol="A", side=Side.BUY,
                                 quantity=10, price_ticks=1_000_000,
                                 base_available_ticks=50_000_000)
    assert ok and reason == "reserved"
    assert led.reserved_buy_ticks() == 10_000_000
    assert led.available_ticks(50_000_000) == 40_000_000
    assert led.outstanding_count() == 1


def test_double_spend_prevention():
    led = CapitalReservationLedger()
    base = 15_000_000
    ok1, _ = led.try_reserve(decision_id="d1", symbol="A", side=Side.BUY, quantity=10,
                             price_ticks=1_000_000, base_available_ticks=base)
    ok2, reason2 = led.try_reserve(decision_id="d2", symbol="B", side=Side.BUY, quantity=10,
                                   price_ticks=1_000_000, base_available_ticks=base)
    assert ok1 is True
    assert ok2 is False
    assert "insufficient_capital" in reason2
    assert led.reserved_buy_ticks() == 10_000_000


def test_sell_never_consumes_buying_power():
    led = CapitalReservationLedger()
    ok, _ = led.try_reserve(decision_id="s1", symbol="A", side=Side.SELL, quantity=100,
                            price_ticks=1_000_000, base_available_ticks=1)
    assert ok is True
    assert led.reserved_buy_ticks() == 0


def test_idempotent_by_decision_id():
    led = CapitalReservationLedger()
    led.try_reserve(decision_id="d1", symbol="A", side=Side.BUY, quantity=10,
                    price_ticks=1_000_000, base_available_ticks=50_000_000)
    ok, reason = led.try_reserve(decision_id="d1", symbol="A", side=Side.BUY, quantity=10,
                                 price_ticks=1_000_000, base_available_ticks=50_000_000)
    assert ok is True and reason == "already_reserved"
    assert led.reserved_buy_ticks() == 10_000_000  # not double-counted
    assert led.outstanding_count() == 1


def test_terminal_states_release_capital():
    for mark in ("mark_filled", "mark_rejected", "mark_canceled"):
        led = CapitalReservationLedger()
        led.try_reserve(decision_id="d1", symbol="A", side=Side.BUY, quantity=10,
                        price_ticks=1_000_000, base_available_ticks=50_000_000)
        getattr(led, mark)("d1")
        assert led.reserved_buy_ticks() == 0, mark
        assert led.outstanding_count() == 0, mark


def test_submitted_and_partial_keep_capital_held():
    for mark in ("mark_submitted", "mark_partial"):
        led = CapitalReservationLedger()
        led.try_reserve(decision_id="d1", symbol="A", side=Side.BUY, quantity=10,
                        price_ticks=1_000_000, base_available_ticks=50_000_000)
        getattr(led, mark)("d1")
        assert led.reserved_buy_ticks() == 10_000_000, mark
        assert led.outstanding_count() == 1, mark


def test_unknown_outcome_does_not_release():
    led = CapitalReservationLedger()
    led.try_reserve(decision_id="d1", symbol="A", side=Side.BUY, quantity=10,
                    price_ticks=1_000_000, base_available_ticks=50_000_000)
    led.mark_unknown("d1")
    assert led.reserved_buy_ticks() == 10_000_000
    assert led.outstanding_count() == 1


def test_invalid_quantity_or_price():
    led = CapitalReservationLedger()
    ok, reason = led.try_reserve(decision_id="d1", symbol="A", side=Side.BUY, quantity=0,
                                 price_ticks=1_000_000, base_available_ticks=50_000_000)
    assert ok is False and reason == "invalid_quantity_or_price"
    ok2, reason2 = led.try_reserve(decision_id="d2", symbol="A", side=Side.BUY, quantity=10,
                                   price_ticks=0, base_available_ticks=50_000_000)
    assert ok2 is False and reason2 == "invalid_quantity_or_price"


def test_propagates_to_risk_reservation_book():
    risk = RiskReservationBook()
    led = CapitalReservationLedger(risk_reservations=risk)
    led.try_reserve(decision_id="d1", symbol="A", side=Side.BUY, quantity=10,
                    price_ticks=1_000_000, base_available_ticks=50_000_000)
    assert risk.gross_notional_ticks() == 10_000_000
    led.mark_filled("d1")
    assert risk.gross_notional_ticks() == 0
