# Model Governance (Milestone 6)

M6 turns the passive M5 metadata ledger into a **full operational registry**: GPU-aware,
reproducibility-complete model cards; atomic, checksum-verified artifact publication; an
explicit maturity state machine with authorized transitions; and always-legal rollback.

The cardinal invariant, enforced structurally rather than by convention: **no model
auto-promotes itself to live financial authority.** Live states require explicit operator
approval, and nothing in this layer submits orders, opens a broker, bypasses the native
`RiskEngine`, or touches the mandatory -0.5% protective exit (that exit is independent
environment dynamics, not a model action).

Source modules:

- `cengine/learning/registry/model_registry.py` — `LearningModelRegistry`, `ModelCard`
- `cengine/learning/registry/maturity.py` — `ModelMaturity`, `check_transition`
- `cengine/learning/registry/artifacts.py` — `ArtifactStore`, `ArtifactRef`
- `cengine/lakehouse/governance.py` — the M5 `ModelGovernanceRegistry` / `ModelRecord` it extends

---

## Extends M5 governance

M5 (`cengine/lakehouse/governance.py`, `GOVERNANCE_VERSION = "model-governance-1.0.0"`)
recorded model metadata only — `promotion_status` stayed `NONE`, `approval_status` stayed
`UNAPPROVED`, and no code path trained or promoted anything. M6 consumes that foundation
and adds the operational machinery:

`LearningModelRegistry` (`REGISTRY_VERSION = "learning-model-registry-1.0.0"`) is an
**append-only JSONL ledger** at `<root>/_learning_models/registry.jsonl`. Cards are never
mutated in place — every state revision appends a new line (`_append()` writes, flushes,
and `os.fsync`s), so the full lifecycle of a model is an auditable history. `<root>` is the
lakehouse root (`CENGINE_LAKEHOUSE_ROOT`, default `data/lakehouse/`); the ledger and
artifact blobs are never committed to Git.

Every append is **mirrored** into the M5 registry for continuity:
`ModelCard.to_model_record()` projects the rich card onto the M5 `ModelRecord`, and
`_append()` best-effort calls `ModelGovernanceRegistry.register()` (writing
`<root>/_model_registry/models.jsonl`). A failure in the mirror never blocks the M6 ledger
of record.

Query surface:

| Method | Returns |
| --- | --- |
| `latest(model_id)` | last revision of a model |
| `list_latest()` | newest revision of every model |
| `by_status(ModelMaturity)` | models currently in a maturity state |
| `champion(model_type=None)` | newest `ACTIVE` model (falls back to newest `SHADOW`) |
| `challenger(model_type)` | newest `VALIDATED`/`SHADOW` of a type that is not the champion |

---

## ModelCard fields

`ModelCard` (`card_version = REGISTRY_VERSION`) carries the full reproducibility and
governance record. Fields, grouped:

**Identity & reproducibility**
`model_id`, `model_version`, `model_type`, `model_architecture`
(e.g. `mlx_quantile[13, 64, 32, 3]` | `mlx_mlp[13, 64, 32, 1]` | `sklearn_hgbr`, built from
`[n_features, *hidden, outputs]`), `framework` (`mlx` | `sklearn` |
`numpy`), `framework_version`, `numerical_precision` (default `float32`), `random_seed`,
`hyperparameters`, `software_versions`.

**GPU / device provenance** — the honesty contract made explicit on every card:
`training_device` and `inference_device` (`gpu` for Metal, else `cpu`),
`hardware_benchmark_reference`. These record where the work **actually ran**, populated from
the trainer's observed `fit.device` — never a silent GPU claim. A small or
CPU-forced workload that fell back to CPU (see `DevicePolicy`,
`CENGINE_LEARNING_FORCE_CPU`) is recorded as `cpu`.

**Data pinning** — `dataset_version`, `dataset_checksum`, `feature_schema_version`
(`feature-schema-1.0.0`), `label_definition_version` (`label-def-1.0.0`),
`training_period`, `validation_period`, `out_of_sample_period`. These come from the
point-in-time `DatasetCertificate` (`cengine/learning/datasets/certification.py`,
`dataset-cert-1.0.0`) so the exact chronological splits are reproducible.

**Evaluation** — `validation_metrics`, `calibration_metrics`, `economic_metrics` (which
carries the `baseline_comparison` and the `beats_baseline` flag).

**Applicability** — `applicable_regimes`, `applicable_instrument_classes`,
`applicable_strategy_families`, `execution_environment` (default `research`).

**Artifacts** — `artifacts` (name → `ArtifactRef` dict) and a combined `artifact_checksum`
computed at `register()` by hashing each attached artifact's checksum.

**Governance / lifecycle** — `maturity` (default `ModelMaturity.UNTRAINED`),
`promotion_status`, `approval_status`, `rollback_reference`, `revision`, `created_ns`,
`transition_reason`.

---

## ArtifactStore: atomic, checksummed publish + corrupt rejection

`ArtifactStore` publishes model blobs under `<root>/_learning_models/<model_id>/` using the
same recipe as the lakehouse `ParquetWriter`: **write to a temp file → flush → `os.fsync`
→ `os.replace`** into the final path. The final filename embeds the content checksum
(`<model_version>-<sha256[:16]><suffix>`), so a half-written blob can never masquerade as a
published one. Publication is idempotent — identical content already on disk returns the
existing `ArtifactRef` without rewriting.

- `save_json(...)` / `save_npz(...)` serialize params/weights (npz is portable across
  CPU/MLX). `save_npz` records `framework` + `framework_version` on the ref.
- `ArtifactRef` captures `path`, `checksum` (full sha256), `bytes`, `kind`
  (`npz` | `json` | `bin`), and framework provenance.
- `load_bytes(ref)` **re-hashes on every load** and raises `ArtifactVerificationError` on
  mismatch — a corrupted or incompatible blob is **rejected**, never allowed to influence a
  decision. `verify(ref)` is the non-raising probe.
- `recover_incomplete()` sweeps orphaned `*.tmp` files from an interrupted publish.

Blobs live under the lakehouse root and are never committed to Git.

---

## ModelMaturity lifecycle

`ModelMaturity` (`MATURITY_VERSION = "model-maturity-1.0.0"`) is an **independent
dimension** from strategy maturity and from financial-execution authorization. The states,
in forward order, are:

```
UNTRAINED → RESEARCH → TRAINED → VALIDATED → SHADOW → PAPER_ELIGIBLE →
PAPER_CERTIFIED → LIVE_ELIGIBLE → LIVE_APPROVED → ACTIVE
```

plus the safety sinks `DEGRADED`, `BLOCKED`, and the terminal `RETIRED`.

Legal transitions are declared in `_LEGAL`. Key properties:

- **Safety sinks are reachable from any non-terminal state** — a model can always be sent
  to `DEGRADED` / `BLOCKED` / `RETIRED`. `RETIRED` is terminal; `DEGRADED` can recover to
  `ACTIVE` or `SHADOW`; `BLOCKED` can return to `RESEARCH`.
- **Independent validation precedes promotion past `SHADOW`** — the only path to `SHADOW`
  is `TRAINED → VALIDATED → SHADOW`, and `VALIDATED` is only granted on a genuine
  out-of-sample result (see baseline gate below).
- **`SHADOW` is not live.** The live states are `LIVE_APPROVED` and `ACTIVE`
  (`_LIVE_STATES` / `is_live()`).

### Promotion authorization — no auto-live

`check_transition(src, dst, *, approved)` raises `PromotionError` if a transition is either
illegal (`can_transition` is false) or unauthorized. The transitions in `_APPROVAL_REQUIRED`
— and **any** transition whose destination is a live state — demand `approved=True`:

```
LIVE_ELIGIBLE → LIVE_APPROVED      (requires explicit approval)
LIVE_APPROVED → ACTIVE             (requires explicit approval)
```

`requires_explicit_approval()` returns true for these, so live financial authority is
**never** granted automatically. In `LearningModelRegistry.transition()`, a live-bound
transition only sets `approval_status = "APPROVED"` / `promotion_status = "PROMOTED"` when
`approved=True` is passed. `approve()` records an explicit operator approval as its own
ledger revision.

On the CLI (`cengine/learning/cli.py`), `promote --to <state>` refuses without `--approve`
for live states and prints `promotion refused: ...`. **M6 does not promote to live** — M7
certification is the milestone that carries a model through `PAPER_CERTIFIED →
LIVE_ELIGIBLE → LIVE_APPROVED → ACTIVE` under operator sign-off.

### Rollback — always legal

`LearningModelRegistry.rollback(model_id, to_reference=..., reason=...)` demotes a model to
`DEGRADED` and records the `rollback_reference` — the last approved model or the
deterministic baseline (`to_reference="baseline"`). It appends a new revision and never
requires approval: reverting to safety is always permitted. The corresponding decision
event is `record_model_rollback` (`cengine/decision_events.py`).

---

## What the registry and card is honest about today

The one genuinely-trainable task on real data, `forward_return`, trained on the GPU
(`training_device = "gpu"`) over a certified chronological split, but its out-of-sample
edge was **marginal and did not beat the sklearn baseline** (champion rank-IC ≈ 0.003 vs
baseline ≈ 0.018; see `reports/supervised_model_validation.json`). Per the baseline gate in
`cengine/learning/supervised/trainer.py`, it therefore stayed at `TRAINED` — **not**
transitioned to `VALIDATED`/`SHADOW`, and nowhere near live. The card records the real
metrics, the real device, and the honest maturity. No economic edge is claimed.
