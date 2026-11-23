"""Versioned PortfolioState + builder (Milestone 3, §4)."""

from __future__ import annotations

from cengine.portfolio import AccountSnapshot, AccountState
from cengine.portfolio_state import (
    PORTFOLIO_STATE_VERSION,
    PortfolioState,
    PortfolioStateBuilder,
    ReconciliationStatus,
)
from cengine.reservations import RiskReservation, RiskReservationBook
from order_manager import Side


def _state(**over) -> PortfolioState:
    base = dict(
        portfolio_version=PORTFOLIO_STATE_VERSION, observation_timestamp=1,
        account_equity_ticks=1_000_000, settled_cash_ticks=1_000_000,
        available_cash_ticks=1_000_000, buying_power_ticks=1_000_000,
        reserved_capital_ticks=0, pending_buy_exposure_ticks=0,
        pending_sell_exposure_ticks=0, realized_pl_ticks=0, unrealized_pl_ticks=0,
        gross_exposure_ticks=0, net_exposure_ticks=0, effective_exposure_ticks=0,
        reconciliation_status=ReconciliationStatus.RECONCILED,
    )
    base.update(over)
    return PortfolioState(**base)


class _Pos:
    def __init__(self, symbol, quantity, cost_ticks):
        self.symbol = symbol
        self.quantity = quantity
        self.cost_ticks = cost_ticks


class _Positions:
    def __init__(self, positions):
        self._positions = list(positions)

    def positions(self):
        return list(self._positions)

    def open_orders(self):
        return ()


class _Exposure:
    def __init__(self, target_daily_leverage, underlying_exposure_group):
        self.target_daily_leverage = target_daily_leverage
        self.underlying_exposure_group = underlying_exposure_group


def _account(equity, cash=None, bp=None, updated_ns=1):
    acct = AccountState()
    acct.replace_from_broker(AccountSnapshot(
        cash_ticks=cash if cash is not None else equity,
        buying_power_ticks=bp if bp is not None else equity,
        equity_ticks=equity, realized_pnl_ticks=0, updated_ns=updated_ns))
    return acct


# ---------------------------------------------------------------- allocatable
def test_allocatable_blocks_stale_and_unreconciled():
    for status in (ReconciliationStatus.STALE, ReconciliationStatus.UNRECONCILED):
        ok, reason = _state(reconciliation_status=status).allocatable()
        assert ok is False
        assert status.value in reason


def test_allocatable_reconciled_with_cash():
    ok, reason = _state().allocatable()
    assert ok is True and reason == "allocatable"


def test_allocatable_blocks_no_capital():
    ok, reason = _state(available_cash_ticks=0, buying_power_ticks=0).allocatable()
    assert ok is False and reason == "no_available_capital"


# ------------------------------------------------------------------- builder
def test_builder_exposure_and_pricing():
    positions = _Positions([_Pos("AAA", 10, 9_000_000), _Pos("BBB", -4, 4_400_000)])
    builder = PortfolioStateBuilder(
        account=_account(1_000_000_000), positions=positions,
        reservations=RiskReservationBook(), price_ticks=lambda s: 1_000_000)
    state = builder.build(now_ns=100, account_observed_ns=100)
    # AAA long 10 @ 1M = +10M ; BBB short 4 @ 1M = -4M
    assert state.gross_exposure_ticks == 14_000_000
    assert state.net_exposure_ticks == 6_000_000
    # no leverage info -> effective == net (leverage +1)
    assert state.effective_exposure_ticks == 6_000_000
    assert state.reconciliation_status is ReconciliationStatus.RECONCILED


def test_builder_drawdown_and_high_watermark():
    acct = _account(1_000_000_000)
    builder = PortfolioStateBuilder(
        account=acct, positions=_Positions([]), reservations=RiskReservationBook(),
        price_ticks=lambda s: 1_000_000)
    first = builder.build(now_ns=100, account_observed_ns=100)
    assert first.drawdown_bps == 0
    assert first.portfolio_high_watermark_ticks == 1_000_000_000
    # equity drops 10% -> drawdown ~1000 bps, HWM retained
    acct.replace_from_broker(AccountSnapshot(
        cash_ticks=900_000_000, buying_power_ticks=900_000_000,
        equity_ticks=900_000_000, realized_pnl_ticks=0, updated_ns=2))
    second = builder.build(now_ns=200, account_observed_ns=200)
    assert second.portfolio_high_watermark_ticks == 1_000_000_000
    assert second.drawdown_bps == 1000


def test_builder_staleness():
    builder = PortfolioStateBuilder(
        account=_account(1_000_000_000), positions=_Positions([]),
        reservations=RiskReservationBook(), price_ticks=lambda s: 1_000_000,
        max_staleness_ns=10)
    fresh = builder.build(now_ns=100, account_observed_ns=95)
    assert fresh.reconciliation_status is ReconciliationStatus.RECONCILED
    stale = builder.build(now_ns=1_000, account_observed_ns=100)
    assert stale.reconciliation_status is ReconciliationStatus.STALE


def test_builder_effective_exposure_signed_for_inverse():
    positions = _Positions([_Pos("SQQQ", 10, 9_000_000)])
    info = {"SQQQ": _Exposure(-3.0, "QQQ")}
    builder = PortfolioStateBuilder(
        account=_account(1_000_000_000), positions=positions,
        reservations=RiskReservationBook(), price_ticks=lambda s: 1_000_000,
        exposure_info=lambda s: info.get(s))
    state = builder.build(now_ns=100, account_observed_ns=100)
    # market value +10M (long the inverse ETF); effective exposure signed -30M
    assert state.net_exposure_ticks == 10_000_000
    assert state.effective_exposure_ticks == -30_000_000
    assert state.effective_exposure_ticks != state.net_exposure_ticks
    assert state.positions[0].exposure_group == "QQQ"


def test_pending_buy_reduces_available_sell_does_not_add():
    reservations = RiskReservationBook()
    reservations.reserve(RiskReservation("b1", "AAA", Side.BUY, 5, 1_000_000))
    reservations.reserve(RiskReservation("s1", "BBB", Side.SELL, 3, 1_000_000))
    builder = PortfolioStateBuilder(
        account=_account(100_000_000), positions=_Positions([]),
        reservations=reservations, price_ticks=lambda s: 1_000_000)
    state = builder.build(now_ns=100, account_observed_ns=100)
    assert state.pending_buy_exposure_ticks == 5_000_000
    assert state.pending_sell_exposure_ticks == 3_000_000
    # available cash = settled cash - pending BUY; SELL proceeds NOT added
    assert state.available_cash_ticks == 100_000_000 - 5_000_000


def test_no_account_is_unreconciled():
    builder = PortfolioStateBuilder(
        account=AccountState(), positions=_Positions([]),
        reservations=RiskReservationBook(), price_ticks=lambda s: 1_000_000)
    state = builder.build(now_ns=100)
    assert state.reconciliation_status is ReconciliationStatus.UNRECONCILED
