# Daily Adaptation Cycle (Milestone 6)

`DailyLearningCycle` (`cengine/learning/orchestration/daily_cycle.py`,
`DAILY_CYCLE_VERSION = "daily-learning-cycle-1.0.0"`) is the post-session learning workflow.
It is **failure-isolated, idempotent, recommendation-only**, and runs **off the
trading-critical path**: it submits no orders, promotes nothing to live, and never touches
the mandatory -0.5% protective exit, the native `RiskEngine`, or the order path.

Every stage is wrapped by `_stage()`, which captures any exception into `report.errors` and
returns `None` — **the cycle never aborts** on a single stage failure.

---

## The pipeline

```
materialize (M5 pipeline) → eligibility assessment → train ELIGIBLE tasks only →
baseline-gated promote → shadow deploy → (optional) drift → decision evidence
```

### 1. Materialize (M5 lakehouse)

`_materialize()` runs the M5 `LakehousePipeline` (silver → gold → quality → attribution,
`compact=False`) so training sees fresh gold/quality/attribution data. Controlled by the
`materialize=True` argument to `run()`. Output is condensed to `ok`/`error` per stage.

### 2. Eligibility assessment (honest; no fabricated data)

`assess_all(config)` (`cengine/learning/datasets/eligibility.py`,
`ELIGIBILITY_VERSION = "dataset-eligibility-1.0.0"`) profiles each learning dataset straight
from the lakehouse Parquet via DuckDB and assigns a per-task `EligibilityState`:

```
INSUFFICIENT_DATA · DATA_QUALITY_FAILED · RESEARCH_ELIGIBLE ·
TRAINING_ELIGIBLE · VALIDATION_ELIGIBLE
```

Crucially, raw bar counts are **not** treated as independent trading outcomes — a task that
needs labeled lifecycles, rewards, or action support is assessed against those (not against
serially-dependent bar counts). Point-in-time violations short-circuit to
`DATA_QUALITY_FAILED`. A task with insufficient evidence receives **no fabricated examples
and no training run** — it stays research-only while evidence accumulates.

### 3. Train eligible tasks only

The cycle's `_TRAINABLE` map lists the tasks it can train today and their trainer kwargs:

```python
_TRAINABLE = {
    "forward_return_prediction":   dict(classification=False),
    "probability_positive_return": dict(classification=True),
}
```

A task is trained only if its eligibility is in
`{TRAINING_ELIGIBLE, VALIDATION_ELIGIBLE}`; anything else is appended to `report.skipped`
with its state and `reason="not_training_eligible"`. Training calls
`train_forward_return_model(...)` (`cengine/learning/supervised/trainer.py`) inside
`_stage()`, so a training failure is isolated like any other stage.

### 4. Baseline-gated promotion (inside the trainer)

Promotion is **not** automatic — it is gated on beating the baseline out-of-sample, and the
gate lives inside the trainer, not the cycle:

- The MLX GPU champion (or a CPU baseline, if MLX is unavailable) is fit on TRAIN with an
  early-stopping validation slice, then evaluated OOS.
- An independent sklearn baseline (`SklearnRegressor`/`SklearnClassifier`) and a
  `ConstantBaseline` floor are evaluated on the **same** OOS rows.
- `_beats(...)` requires a real improvement (regression: higher rank-IC **and** non-worse
  directional accuracy; classification: lower Brier by a margin).
- Only if `beats_baseline` does the trainer transition `TRAINED → VALIDATED → SHADOW`.
  Otherwise the model **stays `TRAINED`** (reason:
  `did_not_beat_baseline_oos (kept TRAINED, not shadow-deployed)`), keeping the incumbent
  champion / deterministic baseline.

### 5. Shadow deployment

A challenger promoted to `SHADOW` is registered and available to the inference service, but
`SHADOW` is **not live** — it runs alongside the champion for comparison only. The
champion/challenger harness (`cengine/learning/orchestration/champion_challenger.py`,
`champion-challenger-1.0.0`) scores champion and challenger over **identical** observations
and only ever emits a *recommendation* (`retain_champion` /
`challenger_outperforms_shadow`); it never auto-promotes and never generates duplicate real
orders.

### 6. Optional drift

When `run_drift=True`, `_drift()` confirms the drift-monitoring module
(`cengine/learning/monitoring/drift.py`, `drift-monitor-1.0.0`) is available. The
`DriftMonitor` is advisory: `DriftReport.grants_trading_authority` is hard-wired `False`,
and no reference distribution is fabricated here.

### 7. Decision evidence

`_emit_training()` writes evidence through the nonblocking decision sink
(`cengine/decision_events.py`): `record_training_result(...)` for every trained model, and
`record_model_promotion(..., approved=False, reason="auto_shadow_beats_baseline")` only when
a model genuinely beat its baseline. The sink is best-effort — a sink failure never perturbs
the cycle. A compact run row (`version`, timestamps, counts of trained/skipped/errors) is
appended to `<root>/_checkpoints/learning_cycle_runs.jsonl` by `_record_run()`.

The run returns a `DailyCycleReport.to_dict()` with `materialization`, `eligibility`,
`trained`, `skipped`, `errors`, and a computed `duration_s`.

---

## Detached, off the critical path — training OFF in-session

The engine wires the cycle as a **detached, stop-driven loop** in
`trade.py::_run_learning_cycle`:

- It is **never** in the engine's `FIRST_COMPLETED` wait set and never touches the order
  path, `RiskEngine`, reconciliation, or the protective exit.
- The learning layer as a whole is gated by `CENGINE_LEARNING` (default on).
- **In-session training defaults OFF.** Heavy training only runs when
  `CENGINE_LEARNING_TRAINING=1`; otherwise the loop keeps the shadow inference service warm
  and skips the cycle body. Training belongs to the **out-of-session** daily run.
- When enabled, the cycle body runs in a worker thread (`run_in_executor`) so GPU training
  never executes on the asyncio event loop.
- Any exception in the loop is swallowed — *learning must never perturb the engine.*

Cadence is `CENGINE_LEARNING_CYCLE_S` (default 3600s).

### Standalone invocation

Out of session, the same cycle runs directly:

```
python -m cengine.learning daily-cycle            # run_daily_cycle(config)
python -m cengine.learning profile-datasets       # eligibility report only
python -m cengine.learning train-supervised       # train the forward-return model
```

---

## Honest data reality (what the cycle actually does today)

Over the real lakehouse, only `forward_return_prediction` reaches `VALIDATION_ELIGIBLE`
(≈262k real SIP bars; chronological split). Every other task —
`strategy_net_return`, `probability_positive_return`, `regime_transition`,
`execution_quality`, `policy_bandit`, `offline_rl_execution` — is `INSUFFICIENT_DATA` and is
**skipped**, remaining research-only (their infrastructure is validated on labeled synthetic
fixtures, never fabricated real labels/rewards; IPS/DR is gated off when behavior
probabilities are unknown).

And the one trainable task, when trained, **did not beat the baseline out-of-sample** and so
stayed `TRAINED` — not shadow-promoted (see
`reports/daily_adaptation_validation.json` and `reports/supervised_model_validation.json`).
No economic edge is claimed. The cycle's value today is the **validated, honest machinery**:
materialize → gate on real eligibility → train only what the data supports → promote only on
a genuine out-of-sample win → leave everything else research-only.
