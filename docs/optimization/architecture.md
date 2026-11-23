# Milestone 6B — Autonomous Optimization Architecture

The Milestone 6B optimization platform (`cengine/optimization/`) is a **governed,
closed-loop, performance-driven ML/RL optimization layer** that sits on top of the Milestone
5 lakehouse and the Milestone 6 learning layer and turns *genuine recorded behavior* into
*bounded, out-of-sample, cost-and-risk-gated experiments* whose qualified winners become
**shadow-eligible only**.

This document describes the local-data substrate it builds on, the subpackage map, the
module-level data flow, and the structural safety boundaries. For the overview, the closed
loop in prose, and the invariants in one place, see [README.md](README.md).

Package version constants:

| Constant | Module | Value |
|---|---|---|
| `OPTIMIZATION_PLATFORM_VERSION` | `__init__.py` | `optimization-platform-1.0.0` |
| `OPTIMIZATION_CONFIG_VERSION` | `config.py` | `optimization-config-1.0.0` |
| `OPTIMIZATION_CONTRACTS_VERSION` | `contracts.py` | `optimization-contracts-1.0.0` |
| `META_OPTIMIZER_VERSION` | `controller.py` | `meta-optimizer-1.0.0` |
| `ATTRIBUTION_ENGINE_VERSION` | `attribution.py` | `performance-attribution-1.0.0` |
| `DATA_SUFFICIENCY_VERSION` | `data_sufficiency.py` | `data-sufficiency-controller-1.0.0` |
| `HYPOTHESIS_GENERATOR_VERSION` | `hypotheses.py` | `hypothesis-generator-1.0.0` |
| `SEARCH_VERSION` | `search.py` | `search-spaces-1.0.0` |
| `EXPERIMENT_PLANNER_VERSION` | `experiments.py` | `experiment-planner-1.0.0` |
| `ECONOMIC_EVAL_VERSION` | `economic.py` | `optimization-economic-eval-1.0.0` |
| `OPTIMIZATION_MEMORY_VERSION` | `memory.py` | `optimization-memory-1.0.0` |
| `VALIDATION_VERSION` | `validation.py` | `oos-validation-1.0.0` |
| `ORCHESTRATION_VERSION` | `orchestration.py` | `autonomous-orchestration-1.0.0` |

---

## 1. The local-data substrate (reused, not rebuilt)

The platform reads from two lower layers and **owns no new storage** except an append-only
research ledger.

### Milestone 5 lakehouse (`cengine/lakehouse/`)

- **Local only** — Parquet (zstd) + PyArrow + DuckDB; no cloud, no CUDA, no external APIs.
- Three layers under `LakehouseConfig.root` (default `data/lakehouse/`, overridable by
  `CENGINE_LAKEHOUSE_ROOT`): **bronze** (immutable raw/engine/execution events), **silver**
  (validated/normalized), **gold** (point-in-time features + FIFO position outcomes).
- Prices are integer ticks end-to-end: `DEFAULT_PRICE_SCALE = 10_000`.
- Datasets are addressed via `config.dataset_path(layer, dataset)` and read through DuckDB
  `read_parquet(..., union_by_name=true)` globs.

The optimization layer touches two datasets directly:

| Dataset | Layer | Used by | Purpose |
|---|---|---|---|
| `market_bars` | bronze | `validation.py`, `exits/counterfactual.py`, `exits/experiment_matrix.py`, supervised features | Genuine 1-minute (`timeframe='1Min'`) bars — the replay environment |
| `position_outcomes` | gold | `attribution.py` | Genuine closed-position outcomes (currently `INSUFFICIENT_DATA`) |

### Milestone 6 learning layer (`cengine/learning/`)

- **Economic metrics + bootstrap CIs** (`learning.evaluation`): `economic_metrics`,
  `bootstrap_metric_ci`, and the data-label constants `DATA_LABEL_REAL` (`"real"`),
  `DATA_LABEL_SYNTHETIC` (`"synthetic"`), `DATA_LABEL_HISTORICAL_REPLAY`
  (`"historical_replay"`).
- **Dataset eligibility** (`learning.datasets.eligibility`): `EligibilityState`,
  `assess_all`, `default_task_specs` — reused verbatim by the data-sufficiency controller.
- **Models** (`learning.supervised`): MLX GPU models (`MLXQuantileRegressor`, `MLXRegressor`,
  `mlx_is_available`) and the sklearn baseline (`SklearnRegressor`), plus feature building
  (`learning.features.builder.build_bar_features`).

### Genuine data footprint

The research environment is **multi-timeframe historical replay** over **~6.76M real
1-minute bars** across **AAPL / SPY / QQQ / MSFT / NVDA (2020–2026)**. 1-minute is canonical;
5-minute (aggregated from 1m, look-ahead-free) is the strategic timeframe used for strategic
entry generation in the exit matrix. All replay results carry the `historical_replay` data
label and are **simulated, not actual trading performance**.

### The only new storage: the optimization ledger

`memory.OptimizationMemory` writes an append-only JSONL file under
`<root>/_optimization/optimization_memory.jsonl` (local, gitignored), and
`orchestration.AutonomousOptimizationCycle` appends cycle summaries to
`<root>/_optimization/cycle_runs.jsonl`. These are research records; they are never the
order-management database.

---

## 2. Subpackage map

```
cengine/optimization/
├── __init__.py          # public API + OPTIMIZATION_PLATFORM_VERSION
├── __main__.py          # `python -m cengine.optimization` -> cli.main
├── cli.py               # standalone research CLI (7 subcommands, all non-ordering)
├── config.py            # OptimizationConfig + load_config (CENGINE_OPT_* ; validate())
├── contracts.py         # Diagnostic/Hypothesis/ExperimentManifest/CollectionTask, enums, safe_ticker
├── attribution.py       # PerformanceAttributionEngine  (1 · measure -> diagnose)
├── data_sufficiency.py  # DataSufficiencyController       (2 · eligibility gate + collection tasks)
├── hypotheses.py        # HypothesisGenerator             (3 · diagnostic-driven + standing)
├── search.py            # SearchSpace + random/grid/successive-halving/TPE (4a · bounded spaces)
├── experiments.py       # ExperimentPlanner + ExperimentManifest builder   (4b · plan)
├── supervised_runner.py # make_supervised_runner          (5 · OOS runner on genuine bars)
├── economic.py          # evaluate_candidate + compare_to_baselines (6 · gated eval)
├── memory.py            # OptimizationMemory              (persistent ledger; selection intelligence)
├── validation.py        # oos_windows + purged_walk_forward (OOS discipline; holdout reserved)
├── controller.py        # MetaOptimizer                   (the closed loop)
├── orchestration.py     # AutonomousOptimizationCycle + run_autonomous_cycle (operational face)
└── exits/               # protective-exit research (the empirical core)
    ├── adaptive.py          # AdaptiveExitPolicy, AdaptiveExitConfig, ExitDecision (candidates)
    ├── state_machine.py     # ProtectiveExitStateMachine (deterministic candidate SM)
    ├── counterfactual.py    # analyze() — post -0.5%-exit recovery on genuine 1m bars
    ├── experiment_matrix.py # run_exit_matrix() — EXIT-00..15 on genuine 1m bars
    └── runner.py            # make_exit_runner — OOS exit runner for the MetaOptimizer

risk/protective_exit_candidate.hpp   # native candidate emergency stop (INACTIVE)
bindings/python_bindings.cpp         # exposes evaluate_emergency_exit / clamp_emergency_threshold
```

---

## 3. Data flow: lakehouse → attribution → hypotheses → experiments → evaluation → memory

```mermaid
flowchart LR
    subgraph SUB["Substrate (M5 + M6)"]
        direction TB
        BARS["bronze/market_bars<br/>genuine 1m bars"]
        OUT["gold/position_outcomes<br/>closed-position outcomes"]
        LRN["learning layer<br/>metrics · eligibility · MLX models"]
    end

    subgraph OPT["cengine/optimization"]
        direction TB
        ATTR["attribution.py<br/>PerformanceAttributionEngine<br/>-> Diagnostic[]"]
        SUFF["data_sufficiency.py<br/>DataSufficiencyController<br/>-> eligible tasks + CollectionTask[]"]
        HYP["hypotheses.py<br/>HypothesisGenerator<br/>-> Hypothesis[] (diagnostic + standing)"]
        PLAN["experiments.py + search.py<br/>ExperimentPlanner.plan()<br/>-> bounded, deduped configs"]
        RUN["runners<br/>exits/runner.py · supervised_runner.py<br/>OOS replay on genuine bars"]
        EVAL["economic.py<br/>evaluate_candidate -> gates<br/>compare_to_baselines -> verdict"]
        MEM[("memory.py<br/>optimization_memory.jsonl")]
        CTRL{{"controller.py<br/>MetaOptimizer.run()"}}
    end

    HOLD["Milestone 7<br/>certification holdout (untouched)"]

    OUT --> ATTR
    BARS --> ATTR
    LRN --> SUFF
    OUT --> SUFF
    ATTR -->|diagnostics| CTRL
    SUFF -->|eligibility| CTRL
    CTRL --> HYP --> PLAN
    MEM -.history / dedupe / failures.-> PLAN
    PLAN --> RUN
    BARS --> RUN
    LRN --> RUN
    RUN --> EVAL
    EVAL -->|accept(shadow) / reject / defer| CTRL
    CTRL ==>|record every outcome| MEM
    MEM -.effectiveness.-> CTRL
    EVAL -. never accessed .-> HOLD

    classDef sub fill:#eef,stroke:#88a;
    classDef store fill:#efe,stroke:#8a8;
    class BARS,OUT,LRN sub;
    class MEM store;
```

### Stage 1 — Attribution (`attribution.py`)

`PerformanceAttributionEngine.attribute()` queries `gold/position_outcomes` via DuckDB and
slices by dimension (`symbol_id`, `entry_decision_id`) and by exit behavior. It emits
descriptive `Diagnostic` records (never causal claims — each carries a `causal_note` and a
`recommended_experiment_family`). It flags:

- dimension buckets with materially negative mean `net_return` (`_by_dimension`), and
- **near-threshold exit clustering** around -0.5% (`_exit_quality`) → a candidate
  premature-liquidation signal with family `protective_exit_matrix`.

If there are no / too few (`< 30`) genuine closed positions — **the current reality** — it
emits a single `INSUFFICIENT_DATA` diagnostic, explicitly labeled a *data-coverage gap, not a
weakness in the engine*. It never crashes the cycle (all query errors are caught).

### Stage 2 — Data-sufficiency gate (`data_sufficiency.py`)

`DataSufficiencyController.assess()` runs the M6 `assess_all` over `default_task_specs()` and
classifies every learning task by `EligibilityState`:

| State | Meaning | Effect |
|---|---|---|
| `TRAINING_ELIGIBLE` / `VALIDATION_ELIGIBLE` | enough genuine data | task is **experiment-eligible** |
| `RESEARCH_ELIGIBLE` | below training threshold | collection task emitted |
| `INSUFFICIENT_DATA` | too few observations | collection task emitted |
| `DATA_QUALITY_FAILED` | handled by quarantine pipeline | no collection task |

For every deficiency it emits a prioritized `CollectionTask` (priority = economic-importance
prior × recoverability bonus) — **it never fabricates observations**. The controller consults
`is_experiment_eligible(task)` before launching any task-bound experiment. Today
`forward_return_prediction` and `regime_transition` are historically recoverable from SIP
bars; outcome/execution/bandit/RL tasks are not (they need genuine live lifecycles).

### Stage 3 — Hypotheses (`hypotheses.py`)

`HypothesisGenerator` turns diagnostics into testable `Hypothesis` records via the `_FAMILY`
table (experiment-family → intervention / expected improvement / methodology / named search
space). Each hypothesis carries fixed `acceptance_criteria` and non-negotiable
`risk_constraints` ("hard risk limits are not tunable", "native C++ financial authority
retained", "mandatory emergency protection preserved").

`standing()` adds research hypotheses that need no prior diagnostic: the
**protective-exit matrix always runs on genuine bars** (it does not require closed-position
outcomes), and the **supervised-model search fires whenever `forward_return_prediction` is
eligible**. This keeps the optimizer researching even while attribution is data-blocked.

### Stage 4 — Bounded planning (`search.py` + `experiments.py`)

`search.py` defines the **only tunable surface**: versioned `SearchSpace`s
(`default_search_spaces()`): `supervised_model`, `rl_reward`, `protective_exit`,
`entry_policy`, `strategy_policy`. Hard risk limits and mandatory emergency protection are
**deliberately not parameters in any space.** Search methods: `random_search`, `grid_search`,
`successive_halving` (multi-fidelity), and `tpe_suggest` (a dependency-free TPE-style
surrogate — honestly a lightweight sampler, not a full GP).

`ExperimentPlanner.plan(hypothesis, budget)` samples the hypothesis's space (`method` ∈
`random|grid|tpe`), **validates + clips to bounds**, drops configs that duplicate memory
(`memory.dedupe`) or match a repeatedly-failing fingerprint (`memory.repeated_failures`), and
returns at most `max_configs_per_hypothesis` (default 8). `ExperimentPlanner.manifest(...)`
builds the reproducible `ExperimentManifest` record for each run.

### Stage 5 — OOS runners on genuine data

Runners are **pluggable callables keyed by search-space name**; the controller wires in two
genuine-data runners (`orchestration._runners`):

- **`exits/runner.py` (`make_exit_runner`)** — multi-timeframe replay on genuine 1m bars.
  Candidate exit configs are sanity-checked on the **train** window and scored
  **out-of-sample on the validation** window against the EXIT-00 (-0.5%) incumbent **and** a
  simple -1.0% baseline on the *same* window. The untouched holdout is never accessed.
  Research-only policies (`recovery_prob`, `rl`, `hybrid`) cause the runner to decline →
  deferred (honest).
- **`supervised_runner.py` (`make_supervised_runner`)** — tunes the forward-return model's
  family / architecture / horizon over `supervised_model`, trained + scored OOS on genuine
  bars (chronological 60/20/20), using MLX on GPU with a sklearn CPU fallback, and scored
  *economically* (a paper long-if-predicted-positive net-return series) against the sklearn
  baseline.

A runner returns `{"evaluation", "baselines", "data_label", "runtime_s", "device", ...}`, or
`None` to decline. A missing runner or ineligible data → the hypothesis is **deferred**, not
faked.

### Stage 6 — Independent economic evaluation (`economic.py`)

`evaluate_candidate(net_returns, ...)` runs the M6 economic metrics + dependence-aware
bootstrap CI and produces a `CandidateEvaluation` with a fixed-formula `economic_score`
(net return minus drawdown / tail-risk / turnover / instability penalties) **and mandatory
risk gates**:

- `insufficient_sample` (n < `min_sample`, default 30),
- `drawdown_exceeds_limit` (|max drawdown| > 25%),
- `unauthorized_risk_expansion`.

**A high return can never buy its way past a violated hard gate** — the scoring method is
fixed before candidates are observed. `compare_to_baselines(candidate, baselines)` then
applies the **baseline hierarchy**: a candidate must pass its gates **and** beat the
`incumbent` **and** a simple/deterministic baseline on score to be `eligible_for_shadow` —
otherwise `reject`. It never promotes to live; the verdict note restates "shadow-eligible
only (M7 certifies)", and synthetic data is never genuine economic evidence.

### Memory (`memory.py`) — the outer loop

`OptimizationMemory` is what distinguishes autonomous optimization from scheduled retraining.
The controller records **every** experiment (config fingerprint, score, economic metrics,
decision, rejection reason, data label, compute cost) to the JSONL ledger. That history
powers selection intelligence consumed back at Stage 4: `seen_fingerprints`/`dedupe` (never
repeat work), `history_for` (feeds TPE), `repeated_failures` (deprioritize losing regions),
and `effectiveness()` (experiments attempted/completed, improvements identified/retained,
compute-per-improvement) — the optimizer scoring **its own** experiment-selection process.
It improves its selection *policy*; it never modifies its own executable code.

---

## 4. The control loop (`controller.py` + `orchestration.py`)

`MetaOptimizer.run()` sequences the stages: attribution → diagnostics → sufficiency →
hypotheses (diagnostic-driven + standing, capped at `max_hypotheses`) →
`_process_hypothesis` per hypothesis. For each hypothesis it checks the eligibility gate
(`_SPACE_TASK` maps space → gating task; `protective_exit` is `None` because it runs on
genuine bars via its own runner), plans bounded configs, invokes the runner per config,
evaluates each, records to memory, and keeps the best shadow-eligible candidate. A single
runner exception is caught and recorded as an error — **one failure never aborts the loop**.

`AutonomousOptimizationCycle.run()` is the operational face: it loads the governed
`OptimizationConfig`, assembles the genuine-data runners, runs one `MetaOptimizer` cycle, and
persists a cycle summary. `run_autonomous_cycle(config, symbol="AAPL")` is the one-call entry
point used by the `optimize` CLI command.

---

## 5. The protective-exit research subpackage (`exits/`)

The empirical core. All candidates are shadow/research-only; the production -0.5% exit is
preserved untouched until Milestone 7 certification.

| Module | Role |
|---|---|
| `adaptive.py` | `AdaptiveExitPolicy` — candidate policy where -0.5% becomes a *reassessment* threshold with an adaptive stop (−1.0% default, −2.5% max), HOLD/REDUCE/EXIT/REPLACE decisions, an independent −3% emergency trigger, risk-normalized sizing (`risk_normalized_notional`: a wider stop → smaller position), and **fallback to the incumbent -0.5% on invalid intelligence**. `PRODUCTION_EXIT_PCT = -0.005` is the preserved control. |
| `state_machine.py` | `ProtectiveExitStateMachine` — deterministic candidate SM with strict precedence (a triggered emergency/adaptive boundary always wins from any state), idempotent/duplicate-safe transitions, stale-data fallback, and reconcile-before-replace. Emits intended actions only — never orders. |
| `counterfactual.py` | `analyze()` — on genuine 1m bars, measures what happened *after* a hypothetical -0.5% exit at the §16 horizons (1/5/15/30/60 min). Strict evidence class: observed subsequent prices + simulated outcomes, **not genuine executions**; point-in-time censoring; a recovery is not proof holding was the superior portfolio decision. |
| `experiment_matrix.py` | `run_exit_matrix()` — the **EXIT-00..EXIT-15** matrix on genuine 1m bars. Strategic 5-min entries (no look-ahead) + tactical 1m exit simulation, conservative fills (stop-before-target within a bar, gap fills), independent −3% emergency, two evaluations: **A)** exit-isolation and **B)** risk-normalized portfolio. ML/RL/portfolio-aware policies (EXIT-08/09/10/11/13) are marked `RESEARCH_ONLY` until their models/context exist. |
| `runner.py` | `make_exit_runner()` — the OOS runner that closes the loop (Stage 5 above). |

### Native candidate emergency stop (INACTIVE)

`risk/protective_exit_candidate.hpp` implements a **pure, stateless, crash-independent**
emergency liquidation trigger in C++ so it can remain operational if Python/MLX/GPU fail:

- `evaluate_emergency_exit(unrealized_return, threshold = -0.03)` → `EmergencyExitDecision`
  (`triggered`, `unrealized_return`, `threshold`, `overrides_tighter_stop`), exposed to Python
  as `_cengine_native.evaluate_emergency_exit` (the `emergency-check` CLI command).
- It is **a CANDIDATE only: NOT wired into `RiskEngine::evaluate` or any live path.**
- `overrides_tighter_stop` is always `false` — **a tighter active stop always takes
  precedence.**
- `clamp_emergency_threshold(proposed, current)` lets the optimizer only **tighten**, never
  loosen (a more-negative threshold is looser and is rejected).
- `kCandidateEmergencyThreshold = -0.03` is a liquidation *level*, not a guaranteed realized
  maximum loss. Milestone 7 owns any activation decision.

---

## 6. Structural safety boundaries

| Boundary | Where enforced |
|---|---|
| Production -0.5% exit preserved | `config.validate()` rejects `production_exit_pct != -0.005`; `adaptive.PRODUCTION_EXIT_PCT = -0.005` |
| No live activation / risk expansion / hard-stop tuning / live exploration | `config.validate()` hard-rejects each toggle |
| Hard limits not tunable | absent from every `SearchSpace` in `search.py` |
| Native C++ is sole financial authority | no module submits orders; all outputs are recommendations/records |
| Emergency candidate inactive & never loosened | native header not wired to `evaluate`; `clamp_emergency_threshold` |
| No fabricated evidence | `attribution` emits `INSUFFICIENT_DATA`; `data_sufficiency` emits collection tasks; runners decline → deferred |
| SQL-injection safety | `contracts.safe_ticker` validates every symbol before DuckDB interpolation |
| OOS discipline + untouched holdout | `validation.oos_windows` / `purged_walk_forward` reserve the holdout; runners use train+validation only |
| Simulated ≠ live | all replay carries the `historical_replay` data label; Milestone 7 certifies |

The authoritative analytical interface remains the M5 lakehouse; the authoritative financial
journal is separate and untouched. Milestone 7 owns independent economic certification and
controlled live deployment — and the certification holdout this platform never reads.
</content>
