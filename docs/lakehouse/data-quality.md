# Data Quality & Attribution

## Quality framework (`quality.py`, `QUALITY_VERSION = "data-quality-1.0.0"`)

`DataQualityValidator.profile()` runs automated DuckDB checks over the materialized
Parquet and returns a machine-readable profile (`total_checks / passed / failed / checks`).
The M5 evidence run passed **14/14** (`reports/lakehouse_data_quality.json`).

### Bronze checks (`validate_bronze`), per decision/execution dataset
- **`pk_uniqueness`** — no duplicate `event_id`.
- **`required_timestamps`** — `event_timestamp_ns` and `availability_timestamp_ns` present.
- **`point_in_time_availability`** — no row with `availability < event` (the leakage
  guard; see [point-in-time.md](point-in-time.md)).
- **`row_count`** — descriptive volume.

### Silver fills checks (`validate_silver_fills`)
- **`price_validity`** — no negative fill prices.
- **`financial_scale_consistency`** — no suspiciously small tick prices (1–99), a guard
  against a cents-scale regression; everything stays on `DEFAULT_PRICE_SCALE = 10000`.

### Referential integrity (`validate_referential_integrity`)
- **`referential_integrity`** — every `execution_plans.portfolio_decision_id` resolves to
  a `portfolio_decisions` row.

## Quarantine (never silently discard)

Records that fail envelope construction or validation are written to
`quarantine/<dataset>/reason=<reason>/` as JSON with the reason (`parquet_writer.quarantine`),
and `IngestionStats.quarantined` is incremented. Bad data is isolated and auditable, never
dropped on the floor. A non-dict payload, for example, is quarantined rather than crashing
ingestion.

## Daily attribution (`attribution.py`, `ATTRIBUTION_VERSION = "daily-attribution-1.0.0"`)

`DailyAttribution.attribute()` produces an end-of-session attribution over the gold/silver
datasets and persists it to `gold/daily_performance`:

- Gross/net return, winners, win rate, per-strategy-family net return.
- Execution quality from TCA (implementation shortfall, realized slippage).
- A **loss-source taxonomy** (`LOSS_SOURCES`): `POOR_OPPORTUNITY_SELECTION`,
  `REGIME_MISMATCH`, `ENTRY_TIMING`, `POSITION_SIZING`, `EXPOSURE_CONCENTRATION`,
  `EXECUTION_COST`, `ADVERSE_SELECTION`, `EXIT_TIMING`, `DATA_QUALITY`.

### Honest about causality
Attribution cleanly separates **prediction quality** (label vs realized), **strategy
quality** (net return by family/regime), and **execution quality** (shortfall/slippage).
It carries an explicit `causal_note`: causal attribution is treated as **uncertain** unless
supported by sufficient evidence — the counts are descriptive, not causal claims. The stage
never crashes the session (it is failure-isolated and returns an `error` field if a query
fails).
