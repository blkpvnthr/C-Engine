# Model Validation

How C-Engine decides whether a learned model is *trustworthy enough to act on* —
calibration of its probabilities, economic evaluation of its decisions with
honest uncertainty, offline policy evaluation of challengers, and the
champion/challenger comparison that gates promotion. The governing stance
throughout: **never claim profitability from synthetic evidence**, and never let a
model promote itself.

All of this is **recommendation-only research infrastructure**. Validation
scores models; it submits no orders, never bypasses the native financial
authority, and never touches the mandatory `-0.5%` protective exit (which is
environment dynamics — see [reward_engineering.md](reward_engineering.md) §6).

> **Data reality (M6).** Only `forward_return_prediction` has enough real data
> (262k SIP bars) to be validated on real outcomes, and it is a **marginal**
> result (OOS rank-IC ~0.01–0.02; it did not beat the sklearn baseline and stayed
> `TRAINED`). Every other task is `INSUFFICIENT_DATA`; its validation machinery is
> exercised on **labeled synthetic fixtures**, which by rule can never produce a
> profitability verdict. M6 does **not** promote anything to live — M7 certifies.

---

## 1. Calibration (`cengine/learning/calibration/calibration.py`)

Classifier probabilities are only useful if a reported `0.7` means a ~70%
empirical frequency. Calibration scores and, where warranted, recalibrates the
probabilistic outputs of the supervised models.

### Scores

- **Brier score** (`brier_score`) and **log loss** (`log_loss`) — thin wrappers
  over the shared `supervised.base.classification_metrics` so the metric math
  lives in exactly one place.
- **Reliability curve** (`reliability_curve` → `ReliabilityCurve`) bins predicted
  confidence against observed frequency, with uniform or quantile bins; empty bins
  are `nan` and excluded.
- **ECE / MCE** — `ReliabilityCurve.ece()` is the count-weighted mean
  `|accuracy − confidence|`; `.mce()` is the worst populated bin. Also exposed as
  `expected_calibration_error` / `maximum_calibration_error`, and bundled with the
  classifier metrics by `calibration_report`.

### Recalibrators — fit on VALIDATION ONLY

Two post-hoc calibrators, both of which **must be fit on a held-out validation
split** — never on the training data the base model saw (that leaks), and never on
the test data used to judge calibration (that invalidates the evaluation):

- **`IsotonicCalibrator`** — non-parametric monotonic isotonic regression
  (scikit-learn, CPU). Degenerate validation data (<2 samples, a single observed
  class, or constant inputs) falls back to an identity pass-through flagged
  `research_only = True`; a valid fit on fewer than `min_samples` (default 50)
  points is likewise flagged `research_only` — usable but **non-promotable**.
- **`TemperatureScaler`** — single-parameter temperature `T > 0` on the logits,
  fit by a bounded golden-section line search minimizing validation log loss (pure
  numpy; `T > 1` softens overconfidence, `T < 1` sharpens). Same `research_only`
  guards.

### Uncertainty-head checks

- **`quantile_coverage`** — per-quantile empirical coverage of a
  quantile-regression head plus a quantile-crossing (monotonicity) check.
- **`prediction_interval_coverage`** — PICP (fraction inside the interval) and
  MPIW (mean width) vs a nominal level.

### Abstention

`apply_abstention` turns low-confidence predictions into abstentions and reports
whether doing so is **justified** — `justified` is `True` only when selective
accuracy on the retained set actually exceeds full-set accuracy *and* retained
coverage ≥ `min_coverage`. Abstention is earned by measured performance, not
assumed.

### Two confidences — do not conflate

`confidence_from_proba` yields **model confidence** (`max(p, 1−p) ∈ [0.5, 1]`) — a
calibratable probability, the only thing these tools calibrate or abstain on. It
is **distinct** from the engine's heuristic **ranking confidence** (the scanner's
ordinal priority score), which is a preference ordering, not a frequency, and must
never be fed into the calibrators.

### Device

Calibration is **CPU** by design (`CALIBRATION_DEVICE = "cpu"`), documented in the
module. These are small, serial, latency-insensitive post-processing steps over a
validation set; isotonic regression is CPU-only scikit-learn and temperature
scaling is a scalar 1-D search. There is no MLX/Metal path to fall back *from*, so
the `DevicePolicy` GPU-first rule does not apply here — this is a documented fixed
placement, not a silent GPU claim.

---

## 2. Economic evaluation (`cengine/learning/evaluation/economic.py`)

Models are judged on the economics of their *decisions*, over a series of
**per-decision NET returns** (`economic_metrics` → `EconomicMetrics`,
`ECONOMIC_METRICS_VERSION = "economic-metrics-1.0.0"`). Gross metrics are never
reported here; inputs must already include commissions, slippage, spread, and
borrow — a gross input is a caller error.

Reported: net return (simple and compounded), net expectancy, profit factor,
Sharpe, Sortino, max drawdown (compound or additive), avg win / avg loss, win
rate, and optional turnover from a weight series.

### Dependence-aware uncertainty — enforced structurally

Trading decisions are serially dependent: 24k bars are **not** 24k independent
experiments, and naive i.i.d. confidence intervals understate the true
uncertainty. So:

- **`sample_size_report`** computes an honest **effective sample size** (ESS) from
  the integrated autocorrelation time (`ESS = n / IACT`, IACT truncated at the
  first non-positive autocorrelation). Positive serial correlation shrinks the
  effective sample, and a low ESS raises a `research-only` warning.
- **`bootstrap_metric_ci`** uses a **stationary** bootstrap (Politis & Romano,
  geometric random block length) or a **moving-block** bootstrap (fixed circular
  blocks) — both preserve short-range dependence, so the interval does not
  understate uncertainty. The interval is returned *with* the ESS so a reader can
  see how much independent information actually backs it.

`evaluate_returns` bundles metrics + a dependence-aware expectancy CI (optionally
a Sharpe CI) + the sample-size report + an honest verdict in one call
(`EconomicEvaluation`).

### Provenance gate

Every economic result carries a `data_label` of `'real'` or `'synthetic'`
(`validate_data_label` raises on anything else — an unlabeled result could be
mistaken for a real claim). `profitability_claim_allowed` is `True` **only** when
the label is `'real'` **and** `n ≥ 30` decisions **and** `ESS ≥ 20`. The verdict
text is explicit:

- `'synthetic'` → "SYNTHETIC — illustrative only. No profitability may be inferred
  from synthetic evidence."
- real but too small / too dependent → "RESEARCH-ONLY — … Not a deployable
  profitability claim."
- real, adequate, expectancy CI excludes zero on the low side →
  "POSITIVE-EXPECTANCY (real) … Further falsification still required before
  deployment."
- real, CI includes zero → "INCONCLUSIVE (real) — no significant edge
  demonstrated."

---

## 3. Offline policy evaluation (`cengine/learning/evaluation/ope.py`)

OPE estimates the value a *candidate* policy would have earned on the **same
historical decisions** the engine actually logged — so a challenger can be ranked
against the champion/baseline without touching live capital, submitting an order,
or disturbing the protective exit (`OPE_VERSION =
"offline-policy-evaluation-1.0.0"`).

### Direct Method + Fitted-Q (what we can actually run)

- **`DirectMethodEvaluator`** — fit a reward model on **realized rewards only**,
  then average its predictions under the target policy:
  `V_DM(π) = (1/N) Σ_i Σ_a π(a|s_i) · r̂(s_i, a)`. Reward models:
  `PerActionMeanRewardModel` (context-free, robust default; actions with no
  realized reward fall back to the global mean and that fallback is surfaced as
  extrapolation risk) and `LinearRewardModel` (context-aware least squares). This
  estimator needs **no** behavior probabilities, so it is the workhorse when the
  logging policy is unknown — but it is model-based, hence always paired with the
  support audit below.
- **`FittedQEvaluator`** — the FQE hook generalizes the Direct Method to
  multi-step transitions (`Q(s,a) ← r + γ·Q(s', π(s'))`). With single-step
  (bandit) logs and no next-state it **provably reduces** to the Direct Method
  (`Q = r̂`, γ unused), and that reduction is reported in the result diagnostics.
  Genuine multi-step backups raise `NotImplementedError` rather than fabricate
  next-states — the current gold `offline_rl` feed does not provide them.

(Separately, `offline_rl/fqe.py` provides the MLX/CPU neural FQE used for value
estimation during training; the evaluation-side `FittedQEvaluator` here is the
OPE hook over logged decisions.)

### IPS / doubly-robust — GATED OFF

`ips_value` and `doubly_robust_value` mathematically require the logging policy's
propensities. The gold feed records `action_probability_if_known` as **always
NULL**, so both return `status = "unavailable"` with reason "behavior
probabilities unknown", diagnostics `{"gated_off": True}` — **never** a value
computed from a fabricated propensity. A fabricated propensity would produce a
fabricated value, so the honest answer is "unavailable."

### Action-support / extrapolation audit

`action_support_audit` → `ActionSupportAudit` reports, per action: logged support,
the fraction of target-policy action mass that falls on actions with **no realized
logged reward** (out-of-distribution `extrapolation_fraction`), realized-reward
coverage, and a warning when the target policy wanders into unsupported actions.
Off-policy estimates are only trustworthy where the target chooses actions the
logging policy actually explored.

### Provenance

Like economic metrics, every `OPEResult` carries a `data_label` and
`profitability_claim_allowed` (real **and** ≥ 30 realized-reward rows). Synthetic
OPE is "ordering shown for plumbing only."

---

## 4. Champion / challenger (`cengine/learning/orchestration/champion_challenger.py`)

The `ChampionChallengerHarness` runs the current champion, a newly-trained
challenger, and (implicitly) the deterministic baseline over **identical
observations** and records where they differ
(`CHAMPION_CHALLENGER_VERSION = "champion-challenger-1.0.0"`). It is
**SHADOW-ONLY**: it never generates duplicate real orders, and a challenger that
fails validation stays inactive.

Per evaluation it records:

- `mean_abs_prediction_diff`, `rank_correlation` (via a Spearman-style rank
  correlation), and `sign_disagreement_rate` between champion and challenger
  predictions;
- when realized outcomes are supplied, `champion_economic` and
  `challenger_economic` from a prediction-driven *long-if-positive* paper
  strategy, scored through `economic_metrics` with the correct `data_label`;
- a `recommendation` — and it **never auto-promotes**: the most a challenger can
  earn is `"challenger_outperforms_shadow"`, a recommendation for *independent
  review*, when its net expectancy beats the champion's on real data.

Champion / challenger selection comes from the registry
(`LearningModelRegistry.champion()` = latest `ACTIVE`, falling back to `SHADOW`;
`.challenger()` = latest `VALIDATED`/`SHADOW` that is not the champion), so the
comparison always reflects the lifecycle state, not an ad-hoc pick.

The `compare_policies` helper (in `ope.py`) is the OPE-side analogue: it scores a
candidate vs a baseline with the **identical estimator on the identical logged
decisions** (`identical_decisions=True`), returning a dependence-aware bootstrap CI
on the per-decision value *difference* for the Direct Method, and inheriting the
IPS/DR gate otherwise. Its verdict only calls a candidate superior when the
value-difference CI excludes zero on real data — and even then appends "confirm
with falsification before any promotion."

---

## 5. Maturity lifecycle — no self-promotion

Model validation feeds the maturity state machine
(`cengine/learning/registry/maturity.py`, `ModelMaturity`):

```
UNTRAINED → RESEARCH → TRAINED → VALIDATED → SHADOW → PAPER_ELIGIBLE →
PAPER_CERTIFIED → LIVE_ELIGIBLE → LIVE_APPROVED → ACTIVE
          (safety sinks: DEGRADED, BLOCKED; terminal: RETIRED)
```

The cardinal rules, enforced in the transition table, not by convention:

- **No model auto-promotes to live.** Independent validation must precede any
  promotion past `SHADOW`.
- **Explicit operator approval is required** for `LIVE_APPROVED` / `ACTIVE`
  (`_APPROVAL_REQUIRED`; `requires_explicit_approval` / `is_live`). An illegal or
  unauthorized transition raises `PromotionError`.
- **Rollback is always legal** — to the last approved model or the deterministic
  baseline (`DEGRADED`/`BLOCKED` are reachable from any non-terminal state).

M6 stops at building and validating; it does **not** promote to live. The
`forward_return_prediction` model, being marginal and not beating baseline, stayed
`TRAINED`. M7 is where certification happens.

---

## 6. The stance: never claim profitability from synthetic evidence

This is not a slogan — it is implemented as a gate in multiple independent places,
so no single oversight can leak a false claim:

| Mechanism | Where | Effect |
|---|---|---|
| `data_label` required `'real'`/`'synthetic'` | `economic.py`, `ope.py` | Unlabeled result raises; synthetic can't be mistaken for real |
| `profitability_claim_allowed` | `EconomicMetrics`, `OPEResult`, `PolicyComparison` | `True` only on real + adequate n + adequate ESS |
| Dependence-aware CI + ESS | `economic.py` | A real but too-autocorrelated series is "research-only," not a claim |
| `synthetic` ⇒ non-promotable | `offline_rl/transitions.py` `TrainingGuard` | Synthetic data can never produce a promoted model |
| `research_only` calibrators | `calibration.py` | Thin/degenerate fits are usable but non-promotable |
| IPS/DR gated off | `ope.py` | No value from fabricated propensities |
| No self-promotion | `registry/maturity.py` | Explicit approval + legal rollback |

The honest conclusion for M6: the infrastructure is validated and the one real,
trainable task (`forward_return_prediction`) produced a marginal, baseline-losing
edge that was correctly **not** promoted. Everything else is research-only on
labeled synthetic fixtures. **No economic edge is claimed.**

### References

- `cengine/learning/calibration/calibration.py` — Brier/log-loss/reliability/
  ECE/MCE, isotonic + temperature (validation-only), coverage, abstention
- `cengine/learning/evaluation/economic.py` — `economic_metrics`,
  `evaluate_returns`, stationary/moving-block bootstrap, ESS
- `cengine/learning/evaluation/ope.py` — Direct Method, FQE hook, IPS/DR gate,
  support audit, `compare_policies`
- `cengine/learning/orchestration/champion_challenger.py` — shadow comparison
- `cengine/learning/registry/model_registry.py`,
  `cengine/learning/registry/maturity.py` — champion/challenger selection &
  lifecycle
- [reward_engineering.md](reward_engineering.md) — how the rewards being evaluated
  are built
