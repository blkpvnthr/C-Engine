# Reward Engineering

How C-Engine turns genuine historical experience into learnable rewards for the
Milestone 6 offline-RL research framework — and, just as importantly, what it
*refuses* to turn into a reward.

Everything described here is **recommendation-only research infrastructure**. No
reward signal here submits an order, touches a broker, bypasses the native
financial authority, or interacts with the mandatory `-0.5%` protective exit. The
reward is a label for value learning, not an instruction to trade.

---

## 1. Where rewards come from

Rewards reach a learner by one of two provenance paths, and the path is always
recorded on the data:

1. **Logged transitions** — the gold `offline_rl_transitions` view, loaded by
   `TransitionLoader.load()` into a versioned, immutable
   `TransitionDataset` (`cengine/learning/offline_rl/transitions.py`,
   `TRANSITIONS_VERSION = "offline-rl-transitions-1.0.0"`). These are genuine
   decisions the engine actually made; `synthetic=False`.

2. **Deterministic historical replay** — `ReplayEnvironment.run()`
   (`cengine/learning/environment/replay.py`,
   `REPLAY_ENV_VERSION = "replay-env-1.0.0"`) replays real SIP bars in strict
   time order under a candidate policy to produce *labeled simulated* experience.
   Every transition it emits is tagged
   `execution_environment = "replay_simulated"`, and the result is explicitly an
   approximate simulator — comparable to, never a silent replacement for, the
   authoritative native `RiskEngine` + `OrderManager` path.

Both paths produce the same transition shape (state, action, reward, next-state,
terminal/truncated, action duration, exit reason). Neither fabricates a reward.

> **Data reality (M6).** Only `forward_return_prediction` has enough real data
> (262k SIP bars) to train and evaluate, and its out-of-sample edge is marginal
> (rank-IC ~0.01–0.02; it did **not** beat the sklearn baseline, so it stayed
> `TRAINED`, not shadow-promoted). All outcome / execution / bandit / RL tasks —
> the ones that consume the engineered rewards below — are
> `INSUFFICIENT_DATA` and remain research-only. Their reward plumbing is
> validated on explicitly **labeled synthetic fixtures**
> (`make_synthetic_dataset`). No economic edge is claimed from any of it.

---

## 2. The reward definition: one realized net return per closed position

The reward is the **realized net return of a closed position**, expressed as a
fraction (`0.01 == +1%`). It is computed once, at close, and it already contains
every cost.

In the replay environment (`_close`, `replay.py`), the per-trade reward is:

```
gross = exit_price / entry_price - 1.0
net   = gross - 2 * cost_frac
```

where `cost_frac = (fees_bps + half_spread_bps + slippage_bps) / 1e4`
(`ReplayConfig`, defaults: fees `1.0`, half-spread `2.0`, slippage `1.0` bps per
side). The `2 *` applies the crossing cost on **both** entry and exit. The entry
price itself is already loaded with the crossing cost
(`entry_price = next_open * (1 + cost_frac)`), and the exit fill is conservative:
a passive limit fills only when a bar trades *through* it, market orders fill at
the next bar's open adjusted by half-spread + slippage, and fills occur at or
after the next bar (configurable `latency_bars`, default `1`). This is the
"net performance only" principle made concrete — gross metrics are never the
reward.

For logged transitions, the reward is whatever was realized when the position
closed. It is loaded verbatim; it is never recomputed, smoothed, or imputed.

---

## 3. No double-counting

The single most important reward-engineering rule here is that a given economic
effect is counted **exactly once**. The design makes double-counting structurally
hard rather than relying on reviewer vigilance:

- **Costs are subtracted once.** `cost_frac` is applied once to the entry price
  and once (via `2 * cost_frac`) to the round-trip return. Fees, half-spread, and
  slippage are bundled into that single `cost_frac`; there is no separate
  per-bar cost drag *and* a round-trip cost — only the round-trip.

- **Realized profit is not also marked-to-market.** The reward is emitted **only
  at close** (`_close`), carrying the whole realized net return of the position.
  While a position is open, the environment does **not** accrue a per-bar
  mark-to-market reward. If it did, the sum of per-bar MTM rewards plus a
  terminal realized reward would count the same price move twice. Instead the
  episode's value is the single realized number, attributed to the entry
  decision.

- **One transition per completed trade.** `ReplayResult.per_trade_returns` is
  exactly `[t["reward"] for t in res.transitions]`, and
  `net_return = sum(per_trade_returns)`. `gross_return` is reconstructed *from*
  the net by adding back `trades * 2 * cost_frac` — it is a derived view of the
  same trades, not an independently accumulated quantity, so the two can never
  disagree about how many trades or costs exist.

For logged data, `reward` corresponds to a closed-position outcome and unclosed
positions carry no reward at all (see §5) — so there is never a half-counted open
position feeding a learner.

---

## 4. Variable action duration

Positions do not close on a fixed horizon, so the reward carries its own
duration. Every emitted transition records:

```python
"action_duration_bars": int(exit_i - entry_i),
"exit_reason": exit_reason,   # "protective_exit" | "max_hold" | "policy_exit" | "end_of_data"
```

A single logged reward therefore spans a *variable* number of bars, and the
learner is told how many. This matters for value learning because:

- **Discounting is per-transition, not per-bar-uniform.** The offline-RL learners
  (`FittedQEvaluator`, `ConservativeQLearner`, `ImplicitQLearner`) bootstrap with
  `targets = r + gamma * not_term * Q(s', a')` (`fqe.py`, `cql_iql.py`), where
  `not_term = (~terminal)`. A terminal closed position contributes only its
  realized `r`; an intermediate step bootstraps through the next state. The
  episode structure — not a fixed window — defines what is terminal.

- **Episodes are reconstructed in time order.** `build_dataset_from_records`
  orders rows within each `episode_id` by `decision_timestamp`, links
  `next_state`/`next_action`, marks the first step `is_initial`, and forces the
  last logged step of an episode `terminal` so bootstrapping stops cleanly. The
  policy value is then estimated at **initial** states only
  (`is_initial & reward_known`), so each episode contributes its value once
  regardless of how many bars it lasted.

- **Truncation is distinguished from termination.** A `max_hold` force-flat
  (default `max_hold_bars = 12`) or `end_of_data` close is marked
  `truncated=True`; a natural policy/protective exit is a true terminal. Keeping
  the two separate avoids treating an artificially cut episode as if the policy
  chose to end it.

---

## 5. No fabricated rewards

Unknown is represented as unknown — never as zero, never as an average.

- **`reward` is `NaN` where unrealized**, masked by a boolean `reward_known`
  array on the `TransitionDataset`. The reward-dependent learners train on the
  `reward_known` subset only (`sub = dataset.subset(dataset.reward_known)` in
  `cql_iql.py`; `rk = dataset.reward_known` in `fqe.py`). An unclosed position
  contributes no label and cannot silently bias a value estimate toward zero.

- **No fabricated behavior probabilities.**
  `action_probability_if_known` is **always `None`** on every transition — the
  logging/behavior policy is unknown and is never invented. If any row arrives
  with a non-null behavior probability it is flagged as a data-integrity
  violation (`point_in_time_violations`, note
  `"fabricated_behavior_probabilities_detected"`) and the `TrainingGuard` returns
  `DATA_QUALITY_FAILED`. Because the probability is unknown, every estimator that
  needs it (IPS / weighted-IS / doubly-robust) is **gated off** upstream; the
  shipped reward-based methods (FQE / CQL / IQL) and the reward-free method (BC)
  deliberately do not require it. See
  [model_validation.md](model_validation.md) §4 for the evaluation side of this
  gate.

- **Synthetic fixtures are labeled synthetic.** `make_synthetic_dataset` builds
  *learnable* reward signal (reward depends on state + action so value learning
  has something to find) but always sets `synthetic=True`. The `TrainingGuard`
  then marks any result on it research-only and **non-promotable** —
  `profitability_claim_allowed` can never be earned from synthetic evidence.

---

## 6. Mandatory exits are environment dynamics, not rewards or actions

The mandatory `-0.5%` protective exit is **part of the world**, not part of the
policy and not a reward the agent optimizes.

- **In the replay environment** (`replay.py`,
  `PROTECTIVE_EXIT_THRESHOLD = -0.005`), the protective exit is enforced *every
  bar on every open position*, **before** the policy is consulted:

  ```python
  # --- MANDATORY PROTECTIVE EXIT (environment dynamics; not a policy action) ---
  if unreal_low < self.rc.protective_exit_threshold:
      exit_price = entry_price * (1.0 + self.rc.protective_exit_threshold)
      ...
  ```

  It can never be suppressed or deferred by the policy, and it is identical to the
  production invariant. It is independent of MLX/GPU, of training, and of
  inference — a rule of the environment the policy lives in.

- **In the learnable action space**, reserved mandatory-exit labels
  (`MANDATORY_EXIT_ACTIONS`: `mandatory_exit`, `mandatory_protective_exit`,
  `protective_exit`, `protective_stop`, `stop_loss_exit`, `forced_exit`,
  `risk_mandatory_exit`) are **excluded** from the action space and the rows that
  carry them are dropped from the learnable set (counted in
  `DataQualityReport.excluded_mandatory_exits`). No learner can model, imitate, or
  recommend the protective exit as a policy choice.

This separation is what keeps entry qualification and risk control authoritative:
learning only improves *ranking* and *eligible-policy selection*. The engine's
entry rule (DI+ > DI− **and** price > VWAP | SMA20 | BB-mid **and** developing-1m
> prev-close) and the protective exit are preserved exactly; reward engineering
never touches them.

---

## 7. What each learner does with the reward

| Method | Needs reward? | Needs behavior probs? | Role |
|---|---|---|---|
| Behavior Cloning (`BehaviorCloner`, `bc.py`) | No | No | Imitates logged actions; reference / FQE target |
| Fitted-Q Evaluation (`FittedQEvaluator`, `fqe.py`) | Yes | No | Value of a fixed target policy |
| Conservative Q-Learning (`ConservativeQLearner`, `cql_iql.py`) | Yes | No | Pessimistic value learning |
| Implicit Q-Learning (`ImplicitQLearner`, `cql_iql.py`) | Yes | No | Expectile value learning |

BC is in `_REWARD_FREE_METHODS`, so the `TrainingGuard` lets it proceed without any
realized reward; all others require `n_reward_known > 0` or the guard returns
`INSUFFICIENT_DATA` (`"no_realized_reward (closed-position reward required; none
present)"`). No shipped method is in `_BEHAVIOR_PROB_METHODS` — that set is empty
by design, because behavior probabilities are unknown.

All four run **GPU-first via MLX** (Apple M5 Max / Metal) with a documented,
measurable numpy CPU fallback selected by `DevicePolicy` (`hardware.py`): small
workloads (`n_elements < min_gpu_elements`, default 50k) run on CPU because they
are host/device-sync dominated, and the chosen device is recorded on every result
(`device`, `backend`). `CENGINE_LEARNING_FORCE_CPU` forces CPU. This is a
placement decision only — it never changes the reward, the mandatory exit, or
whether a model may be promoted.

---

## 8. Invariants summary

- Reward = realized **net** return of a **closed** position; costs already in.
- Each cost / realized gain counted **once**; no per-bar MTM accrual stacked on a
  terminal realized reward.
- Variable action duration carried explicitly (`action_duration_bars`,
  `exit_reason`, `terminal` vs `truncated`); value estimated at initial states.
- Unknown reward = `NaN` + `reward_known` mask; never fabricated, never zero-filled.
- Behavior probabilities always `None`; IPS/DR gated off; fabrication is a
  data-quality failure.
- Synthetic data is labeled and non-promotable.
- Mandatory `-0.5%` exit = environment dynamics, enforced pre-policy, excluded
  from every learnable action space; entry qualification preserved.

### References

- `cengine/learning/environment/replay.py` — replay env, `_close`, `_default_context`
- `cengine/learning/offline_rl/transitions.py` — `TransitionDataset`,
  `DataQualityReport`, `TrainingGuard`, `MANDATORY_EXIT_ACTIONS`,
  `make_synthetic_dataset`
- `cengine/learning/offline_rl/fqe.py`, `cengine/learning/offline_rl/cql_iql.py`,
  `cengine/learning/offline_rl/bc.py` — reward consumers
- `cengine/learning/hardware.py` — `DevicePolicy` (GPU-first + CPU fallback)
- `cengine/learning/datasets/eligibility.py` — task specs & `EligibilityState`
- [model_validation.md](model_validation.md) — calibration, OPE, champion/challenger
