# Capital Reservation & Portfolio State (Milestone 3, §4/§16)

## CapitalReservationLedger (§16)

`cengine/capital_reservation.py` prevents multiple portfolio recommendations from
spending the same buying power. It wraps the existing
`cengine/reservations.py` `RiskReservationBook` (the native risk path's ledger)
with pre-order capital accounting and an explicit per-reservation lifecycle.

- **Atomic / no double-spend** — `try_reserve` checks a BUY's notional against the
  remaining availability (`base_available − outstanding BUY reservations`) under a
  lock; the over-budget reservation fails with `insufficient_capital`.
- **SELL never consumes buying power** — SELL reservations are tracked but reduce
  exposure, so they do not draw down availability.
- **Lifecycle states** — `RESERVED → SUBMITTED → PARTIAL → FILLED` (terminal),
  plus `REJECTED` / `CANCELED` (terminal) and `UNKNOWN`.
- **Unknown-outcome no-release** — `mark_unknown` keeps capital reserved; capital
  is released only on confirmed terminal, non-unknown states (`FILLED` / `REJECTED`
  / `CANCELED`). An unknown broker outcome never frees capital prematurely.
- **Idempotent** — reservations are keyed by `decision_id`; re-reserving the same
  id is a no-op (safe across restarts / duplicate recommendations).
- **Bridge** — optionally propagates reserve/release to the native-path
  `RiskReservationBook` so downstream native risk sees the same reserved capital.

## PortfolioState (§4)

`cengine/portfolio_state.py` (`PortfolioState`, `PortfolioStateBuilder`) assembles a
versioned, point-in-time snapshot from the authoritative broker-reconciled inputs
(`AccountState`, `PositionBook`, `RiskReservationBook`, live prices) + the exposure
registry.

Fields: `portfolio_version`, `observation_timestamp`, `account_equity`,
`settled_cash`, `available_cash`, `buying_power`, `reserved_capital`,
`pending_buy_exposure`, `pending_sell_exposure`, `realized_pl`, `unrealized_pl`,
`gross_exposure`, `net_exposure`, `effective_exposure` (signed, leverage-adjusted),
`positions`, `open_order_count`, `portfolio_high_watermark`, `drawdown`,
`reconciliation_status`.

Key rules:

- **Broker-reconciled is authoritative.** Submitted sell orders are NOT treated as
  completed liquidations: pending-SELL proceeds are **not** added to available cash;
  pending-BUY reservations **reduce** available cash (conservative).
- **Effective exposure** uses the exposure registry's signed daily leverage
  (`market_value × target_daily_leverage`), so an inverse/geared ETF contributes a
  signed figure distinct from net market value.
- **Fail closed.** `allocatable()` returns `(False, …)` when the state is `STALE`
  or `UNRECONCILED` (or there is no available capital). Mandatory protective exits
  are NOT governed by this gate — they run independently.

Evidence: `reports/capital_reservation_validation.json`.
