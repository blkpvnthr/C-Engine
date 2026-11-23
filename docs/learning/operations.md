# Learning Platform Operations (Milestone 6)

Operator guide for the local adaptive ML / constrained offline-RL learning layer
(`cengine/learning/`). Everything here is **recommendation-only**: no command submits
orders, opens a broker connection, bypasses the native financial authority
(`RiskEngine` / constraints / reservations / exposure authorization), or touches the
mandatory **−0.5% protective exit**. The protective exit is environment dynamics,
fully independent of MLX / Metal / GPU, training, inference, the registry, and the
lakehouse — no forecast, reward, or policy can defer or suppress it.

The strict elective-entry contract is preserved and cannot be overridden by any model:

```
DI+ > DI-  AND  price > (VWAP | SMA20 | BB-mid)  AND  developing-1m > prev-1m-close
```

Learning only improves **ranking** and **eligible-policy selection** within that
contract.

## Honest data reality

Only **`forward_return_prediction`** currently has enough genuine data to train and
evaluate (~262k real SIP bars). Its out-of-sample edge is **marginal** (rank-IC
≈ 0.01–0.02) and it **did not beat the sklearn baseline**, so it stayed at maturity
**TRAINED** and was **not** shadow-promoted. No economic edge is claimed.

Every outcome / execution / bandit / RL task is **INSUFFICIENT_DATA** and remains
research-only. Their infrastructure is validated on labeled **synthetic fixtures**,
never on fabricated production labels. No fabricated labels, rewards, or behavior-policy
probabilities are ever created; IPS / DR estimators stay gated off when behavior
probabilities are unknown. This is local-only: Parquet / PyArrow / DuckDB + MLX. No
cloud, no CUDA, no external model APIs.

## Invocation

```
python -m cengine.learning <command> [options]
```

Entry point: `cengine/learning/__main__.py` → `cengine/learning/cli.py:main`. All
subcommand imports are lazy, so a missing optional dependency only affects the command
that needs it. Output is JSON on stdout.

Global option:

| Flag | Meaning |
|---|---|
| `--root PATH` | Lakehouse root; sets `CENGINE_LAKEHOUSE_ROOT` for the process. Defaults to `LakehouseConfig.from_env()` (`./data/lakehouse`). |

The CLI calls `config.ensure_dirs()` before dispatch, so the lakehouse layer roots
(`bronze/ silver/ gold/ quarantine/ _manifests/ _checkpoints`) exist.

## Commands

All commands live in `cengine/learning/cli.py`. Exit code `0` = success unless noted.

### `hardware`
`inspect_hardware()` (`cengine/learning/hardware.py`). Reports the **actual** runtime
Apple-silicon / Metal / MLX capabilities — never hardcoded specs: platform, unified
memory, `mlx_installed`, `mlx_metal_available`, device name, GPU architecture,
`max_recommended_working_set_gib`, GPU-probed `supported_precisions` (float32 / float16 /
bfloat16 round-tripped through a real Metal matmul), and `native_metallib_present`
(coexistence signal for the compiled native `build-metal/cengine.metallib`). Report
schema version `learning-hardware-1.0.0`.

### `profile-datasets`
`assess_all()` (`cengine/learning/datasets/eligibility.py`). Profiles every learning
dataset directly from the lakehouse Parquet via DuckDB and assigns a per-task
`EligibilityState` (`insufficient_data` / `data_quality_failed` / `research_eligible` /
`training_eligible` / `validation_eligible`) with honest reasons. Raw bar counts are
**not** treated as independent trading outcomes — each task declares what it actually
needs (labeled outcomes, instruments, sessions, action support). Report version
`dataset-eligibility-1.0.0`. This is the authority on what may be trained.

### `validate-datasets`
`certify_training_dataset()` (`cengine/learning/datasets/certification.py`). Point-in-time
certification of one dataset with chronological train / validation / out-of-sample
boundaries pinned against `FEATURE_SCHEMA_VERSION` (`feature-schema-1.0.0`) and
`LABEL_DEFINITION_VERSION` (`label-def-1.0.0`). Exit code `1` if the dataset is not
certifiable.

| Flag | Default |
|---|---|
| `--layer` | `bronze` |
| `--dataset` | `market_bars` |

### `train-supervised`
`train_forward_return_model()` (`cengine/learning/supervised/trainer.py`). The one
genuinely-trainable task on real data. Builds look-ahead-free features
(`cengine/learning/features/builder.py`, DuckDB window functions partitioned by
`(symbol, session)`), certifies chronological splits, fits the MLX GPU champion
(`MLXQuantileRegressor` / `MLXBinaryClassifier`) **and** an independent CPU sklearn
baseline, evaluates OOS, and **gates promotion on beating the baseline**. A challenger
auto-advances only as far as **SHADOW**, and only if it genuinely beats the baseline
out-of-sample; otherwise it stays **TRAINED**. Nothing here promotes to live.

| Flag | Default | Meaning |
|---|---|---|
| `--horizon` | `3` | forward-return horizon in bars (3 × 5-min = 15 min) |
| `--epochs` | `40` | training epochs (early-stopped on a tail validation slice) |
| `--batch-size` | `4096` | minibatch size |
| `--classification` | off | train `probability_positive_return` (P(return>0)) instead of regression |
| `--no-register` | off | evaluate without writing a `ModelCard` / artifact |

### `train-bandit`
Guard stub (`cmd_train_bandit`). Contextual-bandit training over authorized policies
requires eligible gold `offline_rl` **reward** data with action support. That data does
not exist (genuine execution outcomes only), so the command reports insufficiency and
directs you to `profile-datasets`. Returns `2` if the module itself is unavailable.

### `train-offline-rl`
Guard stub (`cmd_train_offline_rl`). Constrained offline RL requires eligible gold
`offline_rl` **transitions** with action support; currently insufficient. Same
behavior as `train-bandit`.

### `replay`
`ReplayEnvironment.run()` (`cengine/learning/environment/replay.py`). Deterministic,
event-driven historical replay of genuine SIP bars producing **labeled simulated**
offline-RL transitions (`execution_environment='replay_simulated'`). No future leakage
(fills at/after the next bar), conservative fills, and the **mandatory −0.5% protective
exit is enforced as environment dynamics** every bar on every open position — never a
policy action. This is an explicitly approximate simulator, comparable to but never a
silent replacement for the native `RiskEngine` + `OrderManager`.

| Flag | Required | Meaning |
|---|---|---|
| `--symbol` | yes | symbol to replay under a built-in momentum/vol policy |

### `registry`
`LearningModelRegistry.list_latest()` (`cengine/learning/registry/model_registry.py`).
Lists the latest `ModelCard` revision per `model_id` from the append-only JSONL ledger
at `<root>/_learning_models/registry.jsonl`.

### `compare-models`
`champion()` / `challenger()`. Champion = latest `ACTIVE` (falling back to `SHADOW`)
for a model type; challenger = latest `VALIDATED`/`SHADOW` of that type that is not the
champion.

| Flag | Default |
|---|---|
| `--model-type` | `mlx_quantile_regressor` |

### `shadow`
Transition a model to `SHADOW` (`reg.transition(..., ModelMaturity.SHADOW)`). Shadow
models serve predictions off the critical path but carry **no** financial authority.

| Flag | Required |
|---|---|
| `--model-id` | yes |

### `promote`
Authorized maturity transition (`cengine/learning/registry/maturity.py`). Legal
forward transitions only; `DEGRADED`/`BLOCKED`/`RETIRED` are safety sinks. Transitions
into `LIVE_APPROVED` / `ACTIVE` (any live financial authority) **require `--approve`** —
live authority is never granted automatically. Illegal/unauthorized transitions print
`promotion refused:` and return `2`. **M6 does not promote to live; M7 certifies.**

| Flag | Required | Meaning |
|---|---|---|
| `--model-id` | yes | target model |
| `--to` | yes | target maturity state (enum value) |
| `--approve` | for live | explicit operator approval |
| `--reason` | no | audit reason |

### `rollback`
`reg.rollback()`. Demotes to `DEGRADED` and records the rollback target (last approved
model or the deterministic baseline). **Always legal** — never refused.

| Flag | Default |
|---|---|
| `--model-id` | (required) |
| `--to-reference` | `baseline` |
| `--reason` | `cli_rollback` |

### `daily-cycle`
`run_daily_cycle()` (`cengine/learning/orchestration/daily_cycle.py`, version
`daily-learning-cycle-1.0.0`). The failure-isolated, idempotent post-session workflow:

```
lakehouse materialization (M5 silver→gold→quality→attribution)
  → dataset eligibility assessment
  → challenger training on ELIGIBLE tasks only
  → independent OOS evaluation + baseline-gated promotion (inside the trainer)
  → registry publication + (shadow) deployment
  → optional drift check
  → learning decision evidence
```

INSUFFICIENT_DATA tasks are skipped (never trained on fabricated data); a challenger
that fails to beat the baseline keeps the incumbent / deterministic baseline. Each stage
is isolated so the cycle never aborts; a run summary is appended to
`<root>/_checkpoints/learning_cycle_runs.jsonl`. Nothing here promotes to live or
touches the protective exit.

### `benchmark`
`run_all_benchmarks()` (`cengine/learning/benchmarks/gpu_benchmarks.py`, version
`gpu-benchmarks-1.0.0`). CPU-vs-MLX-GPU certification with correct synchronization:
warm-up/compile excluded, `mx.eval()` forces completion before timing stops, and
workloads where the GPU is **slower** (small, sync-dominated batches) are reported
honestly, not hidden. Covers training throughput, inference latency (p50/p95/p99 per
batch size with a crossover note), and peak unified-memory use. Falls back to clearly
marked CPU-only numbers when MLX/Metal is unavailable.

## trade.py environment gates

The live engine wires a **detached** learning loop and shadow inference service strictly
off the trading-critical path (`trade.py`). Any failure degrades the learning layer to
disabled without affecting trading.

| Env var | Default | Effect |
|---|---|---|
| `CENGINE_LEARNING` | `1` | Master switch. `0` disables the entire learning layer (no `LearningInferenceService`, no detached loop). Read at `trade.py:1033`. |
| `CENGINE_LEARNING_TRAINING` | `0` | In-session training switch. Off by default — heavy training belongs to the out-of-session `daily-cycle`. When `1`, the detached loop runs `DailyLearningCycle` in a worker thread (`run_in_executor`), never on the event loop. Read at `trade.py:2074`. |
| `CENGINE_LEARNING_CYCLE_S` | `3600` | Interval (seconds) of the detached stop-driven learning loop (`_run_learning_cycle`, `trade.py:2073`). |
| `CENGINE_LEARNING_FORCE_CPU` | unset | `DevicePolicy` override (`cengine/learning/hardware.py`). Any non-empty value other than `0` forces every placement onto the CPU, recorded with reason `forced_cpu`. Used for deterministic CPU reproduction and in tests. |

The learning loop (`_run_learning_cycle`, task name `learning-cycle`) is created with
`asyncio.create_task` but is deliberately **not** in the engine's `FIRST_COMPLETED` wait
set, so it can neither trigger nor block shutdown, and it never touches the order path,
native `RiskEngine`, reconciliation, or the protective exit. See
[gpu_workload_scheduling.md](gpu_workload_scheduling.md) for the scheduling model.

## Maturity lifecycle (reference)

```
UNTRAINED → RESEARCH → TRAINED → VALIDATED → SHADOW → PAPER_ELIGIBLE
  → PAPER_CERTIFIED → LIVE_ELIGIBLE → LIVE_APPROVED → ACTIVE
```

Safety sinks `DEGRADED` / `BLOCKED` / `RETIRED` are reachable from non-terminal states;
`RETIRED` is terminal. `LIVE_APPROVED` and `ACTIVE` require explicit operator approval.
Rollback to the last approved model or the deterministic baseline is always legal.
Defined in `cengine/learning/registry/maturity.py` (version `model-maturity-1.0.0`).

## Scope honesty

M6 builds and validates the learning **infrastructure**. It does not certify an economic
edge and does not promote any model to live. The only real-data model (forward-return)
is marginal and sub-baseline; all other tasks are research-only on synthetic fixtures.
Live certification is a separate, gated milestone (M7).
