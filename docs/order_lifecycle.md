# Order Lifecycle, Idempotency & Recovery (Milestone 4, §11)

The authoritative order lifecycle lives in `order_manager.py` (`OrderManager`) and is
**reused** by every path (strategy, portfolio/execution planner, protective exit).

## States

`CREATED → RISK_PENDING → APPROVED → SUBMITTING → SUBMITTED → PARTIALLY_FILLED →
FILLED`, with `RISK_REJECTED / REJECTED / CANCEL_PENDING → CANCELED / EXPIRED / ERROR`.
Transitions are validated against `_ALLOWED_TRANSITIONS`; `TERMINAL_STATUSES` =
{`RISK_REJECTED, FILLED, CANCELED, REJECTED, EXPIRED`}.

## Idempotency & recovery (already enforced)

- **Duplicate client_order_id** → `DuplicateClientOrderId` (registry-locked).
- **Per-order serialization** via one `asyncio.Lock` per client order id.
- **Duplicate / stale venue events** — `venue_sequence ≤ last_venue_sequence` returns
  the current snapshot without re-applying.
- **Out-of-order fills** without a sequence (`event_ns < last_event_ns`) → rejected.
- **Monotonic fills** — cumulative fill may not exceed approved or move backwards.
- **Average fill price** derived only from an explicitly priced `last_fill_quantity`
  equal to the cumulative delta; missing prices are never invented.
- **Ack ≠ fill** — acknowledgement only advances to `SUBMITTED`.
- **Partial vs complete** — `FILLED ⇔ cumulative == approved`; `PARTIALLY_FILLED ⇔
  0 < cumulative < approved`.

Client order ids are stable and broker-compatible. **An order whose broker outcome is
unknown is never blindly resubmitted** — capital reservations for unknown outcomes are
held, not released (`CapitalReservationLedger.mark_unknown`).

## Recovery across restart / reconnect

`AlpacaExecutionVenue.startup_reconcile()` GETs account + open positions + open orders
and rebuilds the `PositionBook`; the reconciliation loop polls `reconcile_status(...)`
per open order. (In-memory `OrderManager` tables are rebuilt from broker state on
restart; the authoritative financial journal retains order events for recovery.)

## Traceability

`ExecutionPlan (execution_plan_id / portfolio_decision_id)` → `OrderIntent
(correlation_id)` → `BROKER_ACK` → `PARTIAL_FILL`/`COMPLETE_FILL` →
`TRANSACTION_COST_ANALYSIS`, all linked by a composed decision-event audit handler.
See `reports/order_lifecycle_validation.json`.
