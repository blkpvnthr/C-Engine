# Autonomous Optimization Operations (Milestone 6B)

Operator guide for the governed autonomous optimization platform (`cengine/optimization/`).
Everything here is **non-ordering, research / shadow-only**: no command submits orders, opens
a broker connection, bypasses the native financial authority (`RiskEngine`), expands risk, or
relaxes the production **−0.5% protective exit**. See `native_risk_authority.md` for the
authority boundary and `milestone7_boundary.md` for what Milestone 7 owns.

This layer builds on — and does not rebuild — the Milestone 5 lakehouse (Parquet / DuckDB)
and the Milestone 6 learning layer. It is **local-only** (no cloud, no CUDA, no external
APIs), **GPU-first via MLX / Metal with a CPU fallback**.

## Honest data reality

- **Genuine data.** ~6.76M real 1-minute bars across **AAPL, SPY, QQQ, MSFT, NVDA**
  (2020–2026); 1m canonical, 5m strategic. Multi-timeframe historical replay is the primary
  research environment.
- **Protective-exit research** runs on these genuine bars today (counterfactual + the
  EXIT-00..15 matrix + the OOS runner).
- **Outcome / execution / bandit / RL tasks remain `INSUFFICIENT_DATA`** — no genuine
  execution lifecycles exist yet (no live session has produced closed positions). Their
  infrastructure is validated but untrained / research-only.
- **All historical-replay results are SIMULATED, not actual trading performance.** No
  economic edge is claimed. No fabricated data, orders, rewards, or behavior-policy
  probabilities are ever created; SQL inputs are validated (`safe_ticker`).

## Invocation

```
python -m cengine.optimization <command> [options]
```

Entry point: `cengine/optimization/__main__.py` → `cengine/optimization/cli.py:main`. Each
command lazy-imports its module so commands stay independent. All output is JSON on stdout.

A global `--root <path>` sets the lakehouse root (equivalent to `CENGINE_LAKEHOUSE_ROOT`)
before the config is loaded.

### Commands

| Command | Purpose | Options |
| --- | --- | --- |
| `attribute` | Performance-attribution diagnostics from genuine gold `position_outcomes` (emits `INSUFFICIENT_DATA` honestly when no closed positions exist). | — |
| `data-sufficiency` | Per-task eligibility + prioritized data-collection tasks (the gate the optimizer consults). | — |
| `memory` | Optimization-memory effectiveness (attempts, improvements, compute per improvement, duplicate rate). | — |
| `counterfactual` | Post −0.5%-exit recovery on genuine 1m bars for one symbol. | `--symbol` (req), `--max-events` (default 2000) |
| `exit-matrix` | EXIT-00..15 protective-exit matrix on genuine 1m bars. | `--symbol` (req), `--risk-normalized` |
| `optimize` | Run one full autonomous optimization cycle (OOS) for a symbol. | `--symbol` (default AAPL) |
| `validation-windows` | Show the chronological train / validation / reserved-holdout split for a symbol. | `--symbol` (req) |
| `emergency-check` | Read-only evaluation of the **inactive** native candidate emergency exit. Computes; takes no action. | `--unrealized-return` (req) |

Examples:

```
python -m cengine.optimization optimize --symbol AAPL
python -m cengine.optimization counterfactual --symbol SPY --max-events 2000
python -m cengine.optimization exit-matrix --symbol QQQ --risk-normalized
python -m cengine.optimization validation-windows --symbol MSFT
python -m cengine.optimization data-sufficiency
python -m cengine.optimization memory
python -m cengine.optimization emergency-check --unrealized-return -0.031
```

## Configuration (`CENGINE_OPT_*`)

The governed config is defined in `cengine/optimization/config.py`
(`OptimizationConfig` / `load_config()`), validated on load. Only a few knobs are exposed as
environment overrides; the safety invariants are **not** environment-tunable.

| Env var | Field | Default | Meaning |
| --- | --- | --- | --- |
| `CENGINE_OPT_ENABLED` | `enabled` | `True` | Master switch. When off, a cycle returns `skipped: "optimization disabled"`. |
| `CENGINE_OPT_TRAIN_IN_SESSION` | `training_during_market_hours` | `False` | Keep training/search off the trading-critical path during market hours. |
| `CENGINE_OPT_DEVICE` | `preferred_device` | `metal_gpu` | Preferred compute device (CPU fallback documented). |
| `CENGINE_OPT_MAX_CONFIGS` | `max_configs_per_hypothesis` | `8` | Per-hypothesis experiment budget. |
| `CENGINE_OPT_SEARCH` | `search_method` | `tpe` | Planner method: `random` / `grid` / `tpe` (successive-halving is available in `search.py`). |
| `CENGINE_LAKEHOUSE_ROOT` | (lakehouse) | lakehouse default | Root for data + the `_optimization/` ledgers; also settable via `--root`. |

Boolean env vars read as false only for `""`, `0`, `false`, `False`; any other value is true.

### Hard invariants (not configurable via env)

`validate()` rejects a config that would violate any of these (raises `ConfigError`):
production/live activation off; no autonomous risk expansion; no hard-stop / mandatory-
protection tuning; no live exploration; `production_exit_pct == -0.005`;
`candidate_emergency_pct` never looser than `-0.030`; `initial_stop_pct` not wider than
`max_stop_pct`; `milestone_7_certification_required = True`. Details in
`native_risk_authority.md`.

## OOS windows

`cengine/optimization/validation.py` derives chronological splits from a symbol's actual data
span. `oos_windows(config, symbol)` returns:

- **train** — default first 60% of the span.
- **validation** — the next 20%, after an **embargo** (default 3 days).
- **holdout** — the remaining tail, after a second embargo. This is the **UNTOUCHED
  certification set**. The optimizer is only ever handed `train` + `validation`; the holdout
  must never be accessed for feature selection, tuning, reward optimization, or experiment
  selection. **Milestone 7 owns it.**

`purged_walk_forward(config, symbol, ...)` provides expanding-window, embargoed folds and
reserves the final 20% as the untouched holdout (never returned as a fold).

The protective-exit runner (`exits/runner.py`) enforces this discipline operationally:
candidate configs are fit/sanity-checked on TRAIN and scored OOS on VALIDATION, always
against the EXIT-00 (−0.5%) incumbent and a simple baseline on the same window; the holdout is
never read.

Inspect a split:

```
python -m cengine.optimization validation-windows --symbol AAPL
```

## Reports & ledgers

Written under `<lakehouse_root>/_optimization/` (local, gitignored):

| File | Writer | Contents |
| --- | --- | --- |
| `optimization_memory.jsonl` | `memory.py` (`OptimizationMemory.record`) | Append-only, `fsync`-flushed ledger: every experiment's fingerprint, config, score, economic metrics, decision, rejection reason, data label, regime, compute cost. Drives dedupe, TPE priors, failure deprioritization, and effectiveness. |
| `cycle_runs.jsonl` | `orchestration.py` (`AutonomousOptimizationCycle._record`) | Best-effort compact summary per cycle: version, symbol, start time, duration, outcome counts. |

`ExperimentManifest` (`contracts.py`) is the reproducible per-experiment record
(hyperparameters, train/eval windows, reward definition, seed, device, economic + risk
metrics, statistical uncertainty, decision, data label). CLI summaries print JSON to stdout.

## Troubleshooting

| Symptom | Likely cause | Resolution |
| --- | --- | --- |
| `{"error": "no data for symbol"}` from `validation-windows`, or `"no 1Min bars for symbol"` / `"insufficient 1Min bars for window"` | No genuine 1m bars for that symbol in the lakehouse, or fewer than the minimum (`<500` rows / `<30` entries). | Point `--root` / `CENGINE_LAKEHOUSE_ROOT` at the populated lakehouse; use a symbol in the genuine universe (AAPL/SPY/QQQ/MSFT/NVDA). |
| `attribute` returns only an `INSUFFICIENT_DATA` diagnostic | No genuine closed-position outcomes yet (no live session). Expected. | Not a defect — a data-coverage gap. The cycle proceeds via standing hypotheses (protective-exit on genuine bars). |
| `optimize` outcomes show many `deferred` | Task-bound spaces lack eligible data, or no runner registered / configs deduped by memory. | Check `data-sufficiency`; execution/bandit/RL tasks are `INSUFFICIENT_DATA` by design. |
| EXIT ids `EXIT-08/09/10/11/13` show `RESEARCH_ONLY` | Those policies need a trained model / portfolio context that does not yet exist. | Expected; deterministic policies still run genuinely. |
| Candidates always `rejected` in `optimize` | Correct, honest result: no candidate beat the incumbent OOS under the risk gates. The −0.5% exit is retained. | No action; this is the intended falsification outcome. |
| `ConfigError` on load | An override attempted to cross a hard invariant (e.g. relax the −0.5% exit, loosen emergency, enable activation). | Remove the offending override; hard limits are not tunable. |
| Cycle returns `skipped: "optimization disabled"` | `CENGINE_OPT_ENABLED` is off. | Set `CENGINE_OPT_ENABLED=1`. |
| `emergency-check` runs but "nothing happens" | By design — it is read-only and the native candidate stop is **INACTIVE**. | It reports `triggered`/`threshold` only; activation is a Milestone 7 concern. |
| A runner error during `optimize` | One runner raised; it is isolated. | The error is recorded under `errors` and the loop continues; inspect the JSON `details.errors`. |

## Related

- `daily_workflow.md` — daily/weekly cadence, scheduler reliability, off-critical-path.
- `native_risk_authority.md` — the native financial authority and inactive emergency stop.
- `milestone7_boundary.md` — M6B scope vs Milestone 7 certification and deployment.
