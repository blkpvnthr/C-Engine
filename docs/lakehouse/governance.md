# Model Governance Metadata

M5 prepares the reproducible, auditable metadata that the **next** milestone (M6) will
need to train and promote models — and it does **nothing more**. No model is trained,
validated, or activated in M5.

## Registry (`governance.py`, `GOVERNANCE_VERSION = "model-governance-1.0.0"`)

`ModelGovernanceRegistry` is an append-only local JSONL registry under
`<root>/_model_registry/models.jsonl` (small, versioned; never committed to git).

- `register(record)` appends a `ModelRecord` and stamps `created_ns`.
- `list_models()` returns all recorded models.
- `dataset_version(path)` returns a stable 24-char hash over the dataset's Parquet part
  paths + sizes — the reproducible dataset pin.

## `ModelRecord` fields

Identity & provenance: `model_id`, `model_version`, `model_type`,
`training_dataset_version`, `feature_schema_version`, `training_period`,
`validation_period`, `out_of_sample_period`.

Configuration & results: `hyperparameters`, `validation_metrics`.

Applicability scoping: `applicable_strategy_families`, `applicable_regimes`,
`applicable_instrument_classes`.

Lifecycle: `artifact_checksum`, `rollback_reference`, and the two gates below.

## The hard stop: NONE / UNAPPROVED

Every record defaults to:

```
promotion_status = "NONE"
approval_status  = "UNAPPROVED"
```

M5 **never** changes these. Nothing the lakehouse does can put a model into production —
there is no code path in M5 that trains a model or flips either flag. This is asserted in
`tests/test_lakehouse.py::test_governance_defaults_to_unpromoted_unapproved` and recorded
in `reports/lakehouse_model_governance.json`.

## Reproducibility contract for M6

Because each dataset has a stable `dataset_version` and each layer/dataset has an
append-only manifest of content-checksummed parts, M6 can:

1. Pin the exact training/validation/out-of-sample data by `dataset_version`.
2. Record the pin, hyperparameters, and out-of-sample metrics in a `ModelRecord`.
3. Only then — under its own validation + approval process — move `promotion_status` and
   `approval_status` forward.

M5 hands M6 trustworthy, reproducible inputs and the ledger to govern them. It does not
cross the line into learning.
