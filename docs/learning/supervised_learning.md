# Supervised Learning (Milestone 6)

Look-ahead-free supervised models that **rank** eligible opportunities. Everything here is
**recommendation-only**: a trained model emits a score or probability that the detached
learning loop can use to re-order already-qualified candidates. It submits no orders, never
bypasses the native financial authority, and has no relationship to the mandatory **−0.5%**
protective exit (which is environment dynamics, independent of any model, GPU, or training).

The one task that currently has enough **real** data to train *and* evaluate is
`forward_return_prediction`. Its out-of-sample edge is **marginal and is not claimed
profitable** (see [The honest result](#the-honest-result)). Every other supervised/outcome
task is `INSUFFICIENT_DATA` and stays research-only.

## Pipeline at a glance

```
bronze market_bars (Parquet)
  → build_bar_features        (DuckDB window functions, look-ahead-free)   features/builder.py
  → certify_training_dataset  (chronological point-in-time split)          datasets/certification.py
  → fit champion + baselines  (MLX GPU vs sklearn/constant CPU)            supervised/{mlx_models,baselines}.py
  → evaluate OOS              (shared metrics)                             supervised/base.py
  → baseline-gated promotion  (beat the baseline, or stay TRAINED)         supervised/trainer.py
  → checksummed artifact + ModelCard                                       registry/{artifacts,model_registry}.py
```

Entry qualification is **untouched**. The existing rule — `DI+ > DI-` AND
`price > VWAP | SMA20 | BB-mid` AND `developing-1m > prev-close` — still decides *whether*
an opportunity is eligible. Learning only improves the *ranking* / eligible-policy selection
among candidates that already passed that gate.

## Features and the builder (`cengine/learning/features/builder.py`)

`build_bar_features()` constructs a point-in-time feature/label matrix directly from the
lakehouse `bronze/market_bars` Parquet via DuckDB window functions, returning a
`BarFeatureMatrix`:

| Field | Meaning |
|---|---|
| `X` | `(n, n_features)` float32 features, look-ahead-free |
| `y_return` | `(n,)` forward log return over `horizon_bars` (regression target) |
| `y_positive` | `(n,)` int {0,1}, `y_return > 0` (classification target) |
| `timestamps_ns` | `(n,)` int64 decision time = **bar close** |
| `symbols` | `(n,)` symbol ids |
| `horizon_bars` | forward horizon, default `3` (three 5-minute bars ≈ 15 min) |

**Look-ahead discipline (the load-bearing property).** Every feature at bar *t* uses only
bars `<= t`; the forward-return label uses bars `> t` via `lead(close, {horizon})` and is a
*target*, never an input. All windows are `PARTITION BY symbol_id, session` so (a) overnight
gaps never contaminate intraday returns and (b) a window can never span two instruments.
Rows missing the forward close (end-of-session tail) are dropped, and a final
`np.isfinite(...)` guard removes any residual non-finite values from degenerate warm-up
windows.

**What the label is — and is not.** `y_return` is the **theoretical** forward bar return.
It is explicitly *not* a realized strategy return under the −0.5% protective exit. The net-
of-exit realized label requires genuine execution lifecycles and is sourced separately by
`gold_training_frame()` from the `gold/supervised` dataset with honest, censored labels
(`label_status = 'realized'`); that dataset does not yet have enough realized outcomes to
train on, so it remains research-only.

## Feature schema (`cengine/learning/features/schema.py`)

A `FeatureSchema` pins the exact ordered columns a model was trained on, so inference builds
tensors identically and an incompatible schema is **rejected** rather than silently mis-fed.

- `FEATURE_SCHEMA_VERSION = "feature-schema-1.0.0"`
- `BAR_FEATURE_COLUMNS` (13, ordered):
  `ret_1, ret_5, ret_15, mom_10, mom_20, vol_10, vol_20, sma_ratio_10, sma_ratio_20,
  range_10, rsi_14, volume_ratio_10, vwap_dist`
- `fingerprint()` → sha256 over `(version | ordered columns)[:16]`; `is_compatible(other)`
  compares fingerprints. Names mirror the engine's existing DI/ADX/ATR/VWAP/SMA/BB technical
  vocabulary so lakehouse features and learning features stay aligned.

## Models

All models implement the uniform `LearningModel` interface in
`cengine/learning/supervised/base.py` (`fit` / `predict` / `predict_with_uncertainty` /
`params` / `load_params`), so the trainer, registry, and inference service treat MLX neural
models and CPU baselines identically. Standardization (`Standardizer`, z-score) is **fit on
training data only** — a point-in-time requirement.

### MLX GPU neural models (`cengine/learning/supervised/mlx_models.py`)

GPU-first neural models trained with MLX autodiff on the Metal GPU, with a documented CPU
fallback via `DevicePolicy` and an explicit `device` record on every fit. Import is guarded:
on a machine without MLX the module still imports and construction raises a clear error so
the trainer degrades to a CPU baseline instead of crashing. All three share a configurable
ReLU `_MLP` (default hidden `(64, 32)`).

| Class | `model_type` | Head / loss | `predict` returns |
|---|---|---|---|
| `MLXRegressor` | `mlx_regressor` | 1 output, MSE | point forward-return estimate |
| `MLXBinaryClassifier` | `mlx_classifier` | 1 logit, numerically-stable BCE-with-logits | `P(y=1)` |
| `MLXQuantileRegressor` | `mlx_quantile_regressor` | `len(quantiles)` outputs, pinball loss | median quantile (point) |

`MLXQuantileRegressor` is the champion architecture for `forward_return_prediction`: default
quantiles `(0.1, 0.5, 0.9)` give a point estimate (0.5), an **uncertainty** band
(`predict_with_uncertainty` → half the 0.9−0.1 spread), and an explicit **downside**
(`predict_downside` → lowest quantile) — useful context for a risk-aware ranker, never an
order.

For training mechanics (the `value_and_grad` loop, device streams, early stopping, npz
serialization, and the honest CPU-comparable benchmark), see
[metal_gpu_training.md](metal_gpu_training.md).

### CPU baselines (`cengine/learning/supervised/baselines.py`)

Independent, deliberately simple non-neural models. **Every learned GPU model must beat an
appropriate baseline before promotion** — a neural model is never preferred merely for
running on the GPU.

| Class | `model_type` | What it is |
|---|---|---|
| `ConstantBaseline` | `constant_baseline` | predicts the training-mean target — the floor every model must clear |
| `HistoricalMeanBaseline` | `historical_mean_baseline` | per-instrument historical mean |
| `SklearnRegressor` | `sklearn_regressor` | `HistGradientBoostingRegressor` (default) or `Ridge` |
| `SklearnClassifier` | `sklearn_classifier` | `HistGradientBoostingClassifier` (default) or `LogisticRegression` |

The sklearn gradient-boosted tree (`kind="hgbr"` / `"hgbc"`, 200 iterations, depth 4,
`l2_regularization=1.0`) is the primary comparison opponent. sklearn estimators serialize
via joblib (pickle) **only** inside the checksum-verified `ArtifactStore`, which are locally
produced and sha256-verified before any deserialization — no artifact is ever loaded from an
untrusted/remote source. MLX weights use npz with `allow_pickle=False`.

## Shared metrics (`cengine/learning/supervised/base.py`)

Every model and baseline is scored the same way so comparisons are fair:

- **Regression** (`regression_metrics`): `mse`, `rmse`, `mae`, `r2`, `directional_accuracy`
  (sign hit-rate), `rank_ic` (rank/Spearman-like correlation — the primary ranking metric).
- **Classification** (`classification_metrics`): `brier`, `log_loss`, `accuracy`,
  `base_rate`.

## Trainer and baseline-gated promotion (`cengine/learning/supervised/trainer.py`)

`train_forward_return_model(...)` runs the full loop and returns a `TrainingReport`.
`LABEL_DEFINITION_VERSION = "label-def-1.0.0"`.

1. **Build** the look-ahead-free matrix; require `n >= 500`, else `insufficient_bar_data`.
2. **Certify** the dataset (`certify_training_dataset`) for a valid chronological
   point-in-time split; an uncertified dataset aborts training.
3. **Split chronologically** from certified boundaries (`_chrono_masks`): train ≤
   `training_end`, OOS ≥ `out_of_sample_start`; carve the last 15% of train as a validation
   slice for early stopping. Require `n_train >= 200` and `n_oos >= 100`.
4. **Fit** the MLX GPU champion (`_build_champion` — `MLXQuantileRegressor` for regression,
   `MLXBinaryClassifier` for classification; falls back to the sklearn model as champion if
   MLX is unavailable) plus the independent sklearn baseline, both on TRAIN only.
5. **Evaluate OOS** with the shared metrics; also record the `ConstantBaseline` floor.
6. **Gate promotion** on `_beats(...)`:
   - regression: champion `rank_ic` > baseline `rank_ic + 1e-3` **AND**
     `directional_accuracy` not worse by more than 0.01;
   - classification: champion `brier` < baseline `brier − 1e-4`.
7. **Register** a `ModelCard` (real metrics, device, dataset version + checksum, schema and
   label versions, hyperparameters, seed, software versions, `execution_environment =
   "research"`) with a checksummed npz weights artifact, then transition maturity:
   - always → `TRAINED`;
   - only **if it beats the baseline OOS** → `VALIDATED` → `SHADOW` (auto-shadow, which is
     still off the trading-critical path and serves no orders);
   - otherwise it **stays `TRAINED`**, with `reason = "did_not_beat_baseline_oos (kept
     TRAINED, not shadow-deployed)"`.

Nothing here promotes to live. M6 stops at `SHADOW` at most; live certification is M7.

## The honest result

On real SIP market bars (≈262k raw bars → **138,549 train / 46,183 OOS** after the look-
ahead-free windows and finite guards, chronological split), the MLX quantile-regression
champion trained successfully on the GPU but its OOS predictive edge is **marginal**, and it
**did not beat the sklearn baseline** (source: `reports/supervised_model_validation.json`):

| Model (OOS) | rank_ic | directional_accuracy | r² |
|---|---|---|---|
| MLX champion (GPU) | 0.0026 | 0.5012 | −0.043 |
| sklearn HGBR baseline | 0.0179 | 0.5190 | −0.0019 |
| constant floor | −0.0002 | 0.5258 | −0.0012 |

Because the champion's rank-IC (0.0026) did not exceed the baseline's (0.018) by the gate
margin, it **stayed `TRAINED`** and was **not shadow-promoted**. The best rank-IC any model
reached (~0.01–0.02) is economically marginal. **No economic edge is claimed.** The verdict
recorded is: *"GPU training works; predictive edge is MARGINAL and NOT claimed profitable."*

This is the intended, falsification-first outcome: the infrastructure is validated, the
honest signal is reported, and a model that does not clear its baseline is not deployed.

## Scope honesty

- **One trainable task on real data.** Only `forward_return_prediction` has enough real
  labeled data to train and evaluate. Its edge is marginal and unpromoted.
- **Everything else is `INSUFFICIENT_DATA`.** Outcome/execution/bandit/RL tasks remain
  research-only; their infrastructure is exercised on labeled **synthetic fixtures**, never
  fabricated labels or rewards presented as real.
- **Theoretical vs realized.** Bar-feature labels are theoretical forward returns, distinct
  from net-of-exit realized returns (which need genuine execution lifecycles, not yet
  available in sufficient quantity).
- **Recommendation-only.** Models re-rank already-qualified candidates. They submit no
  orders, never bypass the native financial authority, and never touch the −0.5% protective
  exit. No model auto-promotes to live; live certification is a separate, gated milestone.

## Key source references

- `cengine/learning/features/builder.py` — `build_bar_features`, `BarFeatureMatrix`, `gold_training_frame`
- `cengine/learning/features/schema.py` — `FeatureSchema`, `BAR_FEATURE_COLUMNS`, `bar_feature_schema`
- `cengine/learning/supervised/base.py` — `LearningModel`, `Standardizer`, `TrainConfig`, metrics
- `cengine/learning/supervised/mlx_models.py` — `MLXRegressor`, `MLXBinaryClassifier`, `MLXQuantileRegressor`
- `cengine/learning/supervised/baselines.py` — `ConstantBaseline`, `SklearnRegressor`, `SklearnClassifier`
- `cengine/learning/supervised/trainer.py` — `train_forward_return_model`, `TrainingReport`
- `reports/supervised_model_validation.json`, `reports/supervised_dataset.json` — the honest results
