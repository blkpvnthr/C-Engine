# Point-in-Time Correctness

The platform's single most important property: a learning dataset must contain only what
the engine could actually have known at decision time. Four failure modes are guarded.

## 1. Look-ahead / leakage

Every record carries `availability_timestamp_ns` (see
[event-envelope.md](event-envelope.md)). Features are served **as-of** availability, not
event time.

`PointInTimeFeatureStore` (`feature_store.py`, `FEATURE_STORE_VERSION =
"feature-store-1.0.0"`):

- `as_of(symbol, as_of_ns, feature_set)` returns the most recent feature row whose
  `availability_timestamp_ns <= as_of_ns` — nothing from the future.
- `training_snapshot(decisions, feature_sets)` uses a DuckDB **`ASOF LEFT JOIN`** to
  attach, to each decision, the latest feature available at that decision's timestamp.
  The join predicate is `feature.availability_timestamp_ns <= decision_ts`, so a decision
  can never pull a later feature value.

Verified in `tests/test_lakehouse.py`
(`test_feature_store_asof_excludes_future`, `test_training_snapshot_is_leakage_free`): a
decision at `t+5` attached to regimes at `t` and `t+100` resolves to the `t` value.

## 2. Availability < event (invalid records)

The data-quality check `point_in_time_availability` (`quality.py`) counts any bronze row
where `availability_timestamp_ns < event_timestamp_ns` and fails if any exist. The 15,480
real SIP bars ingested for the M5 evidence run had **zero** violations
(`reports/lakehouse_point_in_time_integrity.json`).

## 3. Survivorship bias

Historical ingestion (`historical.py`) records the **as-of universe snapshot**
(`record_universe_snapshot`) to `bronze/universe_membership` for every run, so a future
training job reconstructs exactly which symbols were in scope at that time rather than
using today's survivors.

## 4. Label contamination / censoring

Realized outcomes are assigned only to **closed** positions. A recommendation with no
completed lifecycle keeps a `label_status = "censored_no_outcome"` and `NULL` label — it
is never backfilled with a fabricated outcome. See
[supervised-and-rl.md](supervised-and-rl.md).

## Net effect

Because availability time is structural (set at capture, checked by quality, and the key
every read joins on), point-in-time correctness does not depend on the discipline of the
query author — it is enforced by the data model.
