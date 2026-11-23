# Canonical Event Envelope

Every lakehouse record is wrapped in a single versioned envelope
(`ENVELOPE_SCHEMA_VERSION = "event-envelope-1.0.0"`, `envelope.py`) that **reuses**
C-Engine's existing identifiers rather than minting incompatible new ones, and that
makes point-in-time correctness structural rather than best-effort.

## The time distinction (why it exists)

The envelope carries four timestamps, and the difference between two of them is the
whole point:

| Field | Meaning |
|-------|---------|
| `event_timestamp_ns` | when the market **event occurred** |
| `receipt_timestamp_ns` | when C-Engine **received** it |
| `processing_timestamp_ns` | when C-Engine **processed** it |
| `availability_timestamp_ns` | when it became **usable for a decision** |

Training and feature serving read **as-of `availability_timestamp_ns`**. This prevents
look-ahead: a feature cannot be used before the engine could actually have known it.
For a decision event, availability equals event time (decisions are computed from
already-available observations). For a market bar, event time is the bar's **open** and
availability is its **close** (a bar is only usable once the interval ends).

## Identifier reuse

The envelope threads the engine's own ids so lineage is preserved end-to-end:

- `correlation_id`, `causation_id` (the originating `decision_id`)
- `strategy_decision_id`, `portfolio_decision_id`, `execution_plan_id`,
  `order_intent_id`, `broker_order_id`
- `strategy_family`, `strategy_version`, `regime_version`, `feature_version`,
  `configuration_version`

`envelope_from_decision_row(row, …)` maps a `DecisionEventSink.to_row()` dict into the
envelope: it reads `timestamp_ns`, the nested `payload`, and the typed decision columns,
sets `causation_id = decision_id`, derives the strategy/portfolio/execution ids from the
event type, and copies `correlation_id` from the payload.

## Deterministic identity

`event_id = sha256(event_type | ts | decision_id | canonical(payload))[:32]`. Identical
input always yields the same `event_id`, which makes:

- **dedup** trivial (silver `QUALIFY row_number() OVER (PARTITION BY event_id) = 1`), and
- **re-ingestion idempotent** across restarts (same `event_id` → same content-hashed
  part file → no duplication).

## Serialization (`to_row()`)

The stable identity + time + lineage fields are emitted as typed top-level columns; the
flexible, schema-varying body is stored as a single `payload_json` string. Silver
projects typed columns out of `payload_json` on demand, so the bronze schema stays
stable even as payloads evolve.
