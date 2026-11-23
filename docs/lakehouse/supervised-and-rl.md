# Supervised & Offline-RL Datasets

The gold layer (`gold.py`, `GOLD_SCHEMA_VERSION = "gold-1.0.0"`) turns captured
experience into learning-ready datasets — **honestly**. M5 **builds** these datasets;
it does **not** train on them (that is M6).

## Position outcomes (the label source)

`build_position_outcomes()` reconstructs position lifecycles from silver `fills` using
**FIFO lot attribution** per symbol:

- The first fill(s) per symbol open a long lot; later fills close against the oldest open
  lot (FIFO), deterministic and documented (`lot_attribution = "FIFO"`).
- Closed lots get a realized `gross_return = (exit - entry) / entry` (net cost is applied
  via the TCA join). Example: entry `1_000_000` → exit `1_010_000` ticks ⇒ `+1%`.
- Still-open quantity is written to `open_positions` as `OPEN_NO_REALIZED_OUTCOME` — it is
  **never** labeled as a closed outcome.

## Supervised datasets

`build_supervised()` LEFT JOINs silver `strategy_recommendations` (the point-in-time
features) to `position_outcomes` (the labels):

- Recommendations with a realized outcome get `label_status = "realized"`,
  `label_net_return`, and `label_positive_return ∈ {0,1}`.
- Recommendations **without** an outcome (rejected, or not yet closed) keep
  `label_status = "censored_no_outcome"` and `NULL` labels. **No fabricated labels.**
- Rejected opportunities are retained (for selection modeling) but are never relabeled as
  executions.

## Offline-RL transitions

`build_offline_rl()` builds logged transitions from executed portfolio decisions +
realized outcomes, with strict integrity rules:

- `action_probability_if_known` is **always `NULL`** — the behavior policy is
  deterministic/unknown and a propensity is never invented (so downstream importance
  weighting stays honest).
- `reward` is the realized net return **only for closed positions**; unexecuted or
  rejected actions produce **no** fabricated reward. `terminal = true` iff the position
  closed.
- **Mandatory exits and hard constraints are environment dynamics, not agent actions.**
  The -0.5% protective exit is part of the world the policy acts in; it is never modeled
  as a discretionary agent decision.
- Costs are not double-counted: net return already incorporates cost via the TCA path, so
  reward does not re-subtract it.

## Reproducibility

`ModelGovernanceRegistry.dataset_version(path)` hashes the dataset's Parquet part
checksums + sizes into a stable 24-char version, letting M6 pin the exact training data.
See [governance.md](governance.md).

## What M6 will do (not M5)

Train supervised/contextual-bandit/offline-RL models on these datasets, validate them
out-of-sample, and — only then — promote them through the governance registry. M5 stops
at reproducible, point-in-time-correct, honestly-labeled data.
