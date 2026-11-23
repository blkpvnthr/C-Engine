# Inference Integration (Milestone 6)

How a trained learning model is served into the live engine as a **recommendation-only**
signal. Every prediction is point-in-time reproducible, schema/maturity/staleness/OOD
gated, and wrapped in a hard "always-return, never-raise" contract. The inference path is
strictly **off** the trading-critical path: it submits no orders, never bypasses the
native financial authority, and never touches the mandatory `-0.5%` protective exit.

Source modules:

- `cengine/learning/inference/contract.py` — the versioned request/response contract.
- `cengine/learning/inference/service.py` — `LearningInferenceService`.
- `cengine/learning/inference/loader.py` — verified artifact reconstruction (`load_model`).
- `cengine/decision_events.py` — decision-event evidence emitters (M6 §32).
- `trade.py` — shadow wiring (`LearningInferenceService` + detached learning loop).

---

## 1. The versioned contract

`cengine/learning/inference/contract.py` pins
`INFERENCE_CONTRACT_VERSION = "inference-contract-1.0.0"`.

### `InferenceRequest` (frozen dataclass)

Required fields: `task` (e.g. `"forward_return_prediction"`), `symbol_id`, `features`
(an **ordered tuple** of floats, schema-aligned), `feature_schema_version`,
`observation_timestamp_ns`, and `availability_timestamp_ns`.

The two timestamps are carried separately on purpose: `observation_timestamp_ns` is when
the market state existed; `availability_timestamp_ns` is when the feature row became
usable. Staleness is judged against **availability**, so a prediction can be reproduced
and point-in-time audited after the fact.

Optional fields let the caller pin or attribute a model and horizon: `model_id`,
`model_version` (both `None` ⇒ the service picks the champion), `strategy_family`,
`regime_version`, `portfolio_version`, `prediction_horizon`.

### `InferenceResponse` (frozen dataclass)

A response is **always** returned. Fields: `task`, `symbol_id`, `prediction`
(`Optional[float]`), `prediction_uncertainty` (`Optional[float]`), `applicability`,
`data_quality`, `model_maturity`, `model_id`, `model_version`, `device`
(`"gpu" | "cpu" | "none"`), `inference_timestamp_ns`, `expiration_timestamp_ns`,
`abstention_reason` (`Optional[str]`), and a free-form `extra` dict.

Helpers: the `abstained` property is `True` when there is no prediction or an abstention
reason is set; `to_dict()` flattens the response for logging/decision events.

### `Applicability` enum

`APPLICABLE`, `OUT_OF_DISTRIBUTION`, `UNSUPPORTED_INSTRUMENT`, `UNSUPPORTED_REGIME`,
`SCHEMA_INCOMPATIBLE`, `STALE`, `NO_MODEL`, `LOW_CONFIDENCE`.

### `abstain(...)` — the universal fallback

`abstain(request, *, reason, applicability=Applicability.NO_MODEL.value,
model_maturity="untrained", now_ns=0, model_id=None, model_version=None)` builds a
no-prediction response with `data_quality="abstained"` and `device="none"`. Every failure
path in the service funnels through this; nothing raises into the caller.

---

## 2. `LearningInferenceService`

`cengine/learning/inference/service.py`. Constructed as
`LearningInferenceService(config=None, *, registry=None, device_policy=None,
ttl_ns=DEFAULT_TTL_NS, max_staleness_ns=DEFAULT_MAX_STALENESS_NS, min_confidence=0.0)`.

- `DEFAULT_TTL_NS` = 5 minutes — predictions carry an `expiration_timestamp_ns` and go
  stale downstream after this window.
- `DEFAULT_MAX_STALENESS_NS` = 10 minutes — the inbound-feature freshness bound.
- `config` defaults to `LakehouseConfig.from_env()`; `registry` to a
  `LearningModelRegistry` over that config; `device_policy` to a default `DevicePolicy`
  (GPU-first with a measurable CPU fallback — see `gpu_inference.md`).
- `self.stats` tracks `{"served", "abstained", "fallbacks", "load_errors"}`.

### Champion loading + persistent model cache

`_resolve_card(request)` selects the model card:

- If `request.model_id` is set, it loads that id's latest card; if a specific
  `model_version` is pinned, it scans the registry ledger for the exact row (returns
  `None` if that version is not found).
- Otherwise it asks the registry for the **champion** of the task's model family, mapped
  by `_task_model_type(task)`:
  - `forward_return_prediction` → `mlx_quantile_regressor`
  - `probability_positive_return` → `mlx_classifier`
  - `strategy_net_return` → `mlx_regressor`
  - anything else → `mlx_regressor`

`LearningModelRegistry.champion()` returns the latest `ACTIVE` model, falling back to the
latest `SHADOW` model, or `None`.

`_get_model(card)` keys a persistent cache on
`(model_id, model_version, revision)` so a champion is loaded **once** and reused across
calls (see `gpu_inference.md` for why this matters for latency). Loading goes through
`load_model(card, config=...)`.

### Verified artifact reconstruction

`cengine/learning/inference/loader.py::load_model` reconstructs a fitted model from its
registry `ModelCard`:

- Weights load **only** through the checksum-verifying `ArtifactStore.load_npz`
  (`cengine/learning/registry/artifacts.py`, sha256-verified). A corrupt or incompatible
  blob raises `ArtifactVerificationError`, which `load_model` re-raises as the typed
  `ModelLoadError` — a bad model is **never** fed into a decision.
- The class is rebuilt from the card's `framework` + `model_type`: MLX models
  (`mlx_regressor` / `mlx_quantile_regressor` / `mlx_classifier`) are reconstructed from
  the saved `meta::config` (hidden sizes, `n_features`, quantiles); `sklearn` and `numpy`
  baselines are rehydrated from their blobs. An unknown framework/type raises
  `ModelLoadError`.

### Gating sequence (`predict`)

`predict(request, *, now_ns=None, train_feature_ranges=None)` runs every check inside one
`try/except`, and any miss returns an `abstain(...)` response — it **never** raises:

1. **No model** → `card is None` ⇒ `reason="no_model_registered"`,
   `applicability=NO_MODEL`.
2. **Maturity gate** → the card's `maturity` must be in `_SERVING`
   = `{SHADOW, PAPER_ELIGIBLE, PAPER_CERTIFIED, LIVE_ELIGIBLE, LIVE_APPROVED, ACTIVE}`.
   Anything earlier (`UNTRAINED`/`RESEARCH`/`TRAINED`/`VALIDATED`) ⇒
   `reason="model_not_serving(maturity=...)"`. Only shadow-or-later models ever serve.
3. **Schema gate** → if the request and card both carry a `feature_schema_version` and
   they differ ⇒ `reason="feature_schema_incompatible"`, `applicability=SCHEMA_INCOMPATIBLE`.
4. **Staleness gate** → if `now_ns - availability_timestamp_ns > max_staleness_ns` ⇒
   `reason="stale_features"`, `applicability=STALE`.
5. **OOD tagging** → when `train_feature_ranges` is supplied, `_is_ood` flags a row that
   falls outside the training min/max by more than one feature span; the response is still
   produced but tagged `applicability=OUT_OF_DISTRIBUTION`.
6. **Inference** → `model.predict_with_uncertainty(X)` yields point prediction + an
   uncertainty estimate (for the quantile regressor, the half-spread between the outer
   quantiles). `device` is read off the model.

Any `ModelLoadError` increments `load_errors` + `fallbacks` and abstains with
`model_load_error:...`; any other exception increments `fallbacks` and abstains with
`inference_error:<ExceptionType>`. The caller always gets a well-formed response.

### Abstention is the honest default today

Per the Milestone 6 data reality: only `forward_return_prediction` has enough **real**
data (≈262k SIP bars) to train and evaluate, and its out-of-sample edge was **marginal**
(rank-IC ≈0.01–0.02) and did **not** beat the sklearn baseline, so it stayed at `TRAINED`
and was **not** shadow-promoted. All outcome/execution/bandit/RL tasks are
`INSUFFICIENT_DATA` and remain research-only. Because `_SERVING` requires `SHADOW` or
later, the live service currently resolves no serving champion and returns
`no_model_registered` / `model_not_serving` abstentions by design. No economic edge is
claimed; the integration is validated for correctness, not promoted for alpha.

---

## 3. Decision-event evidence

Every inference is recorded as point-in-time evidence through
`cengine/decision_events.py` (`DECISION_EVENT_SCHEMA_VERSION = "decision-events-1.0.0"`),
using the Milestone 6 event types:

- `INFERENCE_REQUEST` via `record_inference_request(req, now_ns)` — task, model id/version,
  observation/availability timestamps, horizon, schema version.
- `PREDICTION` via `record_prediction(resp, now_ns)` — prediction, uncertainty,
  applicability, data quality, model maturity, device, abstention reason, expiration.
- `SHADOW_RECOMMENDATION` via `record_shadow_recommendation(...)` — champion vs challenger
  scores and which was selected (the ranking signal; see §4).
- `MODEL_DISAGREEMENT`, `DRIFT_DETECTION`, `TRAINING_RESULT`, `MODEL_PROMOTION`,
  `MODEL_ROLLBACK` for the surrounding lifecycle.

`DecisionEventSink.emit` is O(1), bounded (`deque(maxlen=...)`), and **never raises**, so
evidence capture cannot perturb ingestion, strategy evaluation, or the protective-exit
path. Rows drain to the local Parquet/DuckDB lakehouse for offline analysis.

---

## 4. Shadow wiring in `trade.py`

The adaptive learning layer is gated by `CENGINE_LEARNING` (default `"1"`/on). During
engine construction `trade.py` instantiates the service:

```python
from cengine.learning.inference.service import LearningInferenceService
self._learning = LearningInferenceService()
```

It is logged as "shadow inference; recommendation-only — never on the order/protection
path", and any construction failure degrades to `self._learning = None` without affecting
trading.

**Where learning runs.** A detached, stop-driven loop `_run_learning_cycle(stop)` keeps
the shadow service warm and (only when `CENGINE_LEARNING_TRAINING=1`, default off) runs
the daily cycle in a **thread executor** so GPU training never runs on the event loop.
Its interval is `CENGINE_LEARNING_CYCLE_S` (default 3600s). This task is **never** part of
the `FIRST_COMPLETED` wait set and never touches the order path, native `RiskEngine`,
reconciliation, or the mandatory protective exit.

**Role in entry/exit.** Learning only improves **ranking / eligible-policy selection**. It
does not relax or replace the entry qualification contract
(`DI+ > DI-` **AND** price `>` VWAP | SMA20 | BB-mid **AND** developing-1m `>` prev-close),
which is preserved exactly as before. A shadow prediction is advisory evidence only.

**Never on the protective-exit path.** The mandatory `-0.5%` protective exit is owned by
the independent `ProtectiveExitCoordinator` (`protective_exit.py`), which runs on its own
periodic backstop task with threshold `MARKET_SCAN_LOSS_THRESHOLD` (default `-0.005`). It
is environment dynamics — independent of MLX, GPU, training, and inference. The inference
service is never called from, and can never block or delay, that coordinator.

**Dashboard.** `_learning_dashboard_stats()` reports only genuinely-available metrics:
backend (`mlx`/`cpu`), inference device (`gpu`/`cpu`), active model version + maturity,
challenger version, served/abstained counts, GPU fallbacks, and the last cycle summary.
It returns `None` when learning is disabled and never raises.

---

## 5. Maturity guardrails (why serving is gated)

`cengine/learning/registry/maturity.py` (`MATURITY_VERSION = "model-maturity-1.0.0"`)
defines the lifecycle `UNTRAINED → RESEARCH → TRAINED → VALIDATED → SHADOW →
PAPER_ELIGIBLE → PAPER_CERTIFIED → LIVE_ELIGIBLE → LIVE_APPROVED → ACTIVE`, plus the
safety sinks `DEGRADED`, `BLOCKED`, `RETIRED`.

- No model auto-promotes to live. Transitions into `LIVE_APPROVED` / `ACTIVE` require
  explicit operator approval (`requires_explicit_approval`), enforced by
  `check_transition(..., approved=...)`.
- Rollback to a safety sink (or the deterministic baseline) is always legal.
- **Milestone 6 does not promote any model to live.** M7 is responsible for certification.
  M6 ships the serving path, the gating, and the evidence trail — nothing more.
