# C-Engine — Milestone 8 Final Alpha-Discovery Report

_Autonomous alpha discovery and economic optimization, built on the M1–M7 stack. All economic
results are **historical-replay simulations on genuine Alpaca SIP bars** — not live-trading
performance and not a profitability claim. No live orders were placed. Negative results are
preserved, not reinterpreted._

## Headline

- **No positive, statistically credible out-of-sample net expectancy was found** on the tested
  universe under the mandatory −0.5% protective exit.
- The failure is **not specific to momentum**: every strategy family, every entry filter, every
  higher-timeframe/regime conditioning, and a supervised ML opportunity filter remained negative
  out-of-sample.
- The **P0 accounting audit** corrected the M7 drawdown interpretation (a compounded per-trade
  artifact) but the corrected, capital-constrained portfolio economics are still negative.
- **Cash (NO_TRADE) dominates** every active policy.
- The autonomous discovery loop genuinely **rejected 6 challengers and deferred 2**; the incumbent
  is retained and the loop continues gathering evidence. Independent certification is **deferred**
  (there is no positive edge to certify).

Per CLAUDE.md the decision is **REJECT** the current entries/strategies as a deployable edge while
**ACCEPTING** the alpha-discovery infrastructure as a sound, honest research substrate.

## P0 — Replay accounting audit (§3)

The M7 "~90–98% drawdown" was computed by compounding a flat series of **independent per-trade net
returns** (`economic_metrics(..., drawdown_mode="compound")`), i.e. an **all-in, sequential,
fully-reinvested single-position** curve. The strided entries overlap in time and, in reality, form
a **capital-constrained portfolio of concurrent positions**.

| Accounting | Drawdown | Interpretation |
|---|---|---|
| M7 compound per-trade | ~100% | all-in sequential single-position artifact |
| Capital-constrained portfolio (10%/pos, max 10 concurrent) | **80.3%** | executable, time-indexed equity |

Cash conservation verified (cash never negative, min $10,669). Per-trade economics are **unchanged
and still negative** (expectancy −0.00108, PF 0.64). M7's "−0.2 Sharpe" was a **per-trade** figure,
not annualized; the portfolio Sharpe (annualized) is −5.2. The correction clarifies magnitude and
basis; it does **not** rescue the strategy. Evidence: `m7_accounting_audit.json`.

### Why it fails (§3/§16)

Mean max-adverse-excursion **−0.69%** exceeds the −0.5% stop while max-favorable is only **+0.58%**
— a near-symmetric intrabar swing against a tight stop; 45.7% of trades are stopped out. Adverse >
favorable ⇒ the entry has **no directional edge** over the hold window. This is an **entry-quality**
problem, not an exit-tuning problem. Evidence: `incumbent_failure_attribution.json`.

## Research results (all out-of-sample, development)

| Study (§) | Result | Verdict |
|---|---|---|
| Entry-quality ablation (§5) | best filter −0.00081 vs −0.00116 baseline; combo overfits | no positive OOS filter |
| Multi-timeframe confirmation (§6) | cuts drawdown 80%→66%, expectancy still negative | reduces damage, no edge |
| Regime conditioning (§7) | only `down_midvol` positive on TRAIN, overfits on VAL | abstention still negative |
| Strategy families (§8) | ALL 5 negative (MEAN_REVERSION least-bad −0.00087) | exit dominates every archetype |
| Microstructure (§9) | genuine L1 INSUFFICIENT_DATA (bars only); bar-proxy −0.00104 | deferred to live/paper L1 |
| Supervised ML filter (§10) | top-decile −0.00071, rank-IC ≤0.09; MLX on GPU | no economic edge |
| Contextual bandit (§11) | direct-method −0.00095 vs uniform −0.00109; no IPS | INSUFFICIENT_DATA |
| Offline RL (§12) | 0 genuine execution transitions | INSUFFICIENT_DATA |
| Portfolio-aware (§13) | cash +0% dominates all_eligible −80% / selective −70% | NO_TRADE is correct |
| Protective-exit (§14) | wider stops better in isolation, still negative; prod unchanged | counterfactual only |

Evidence: the correspondingly-named JSON files in this directory.

## Autonomous discovery loop (§4/§16/§17/§22)

One governed cycle: attribution → 3 hypotheses → data-sufficiency → optimization memory →
challengers → frozen-gate evaluation → **0 accepted / 6 rejected / 2 deferred** → persisted to
`OptimizationMemory` (14 rows). The loop runs off-session in the continuous runtime
(`AlphaDiscoveryService`, P9 research priority, GPU-admission-gated, shadow-only). Evidence:
`experiment_history.json`, `challenger_registry.json`, `continuous_research_validation.json`.

## Independent validation (§20)

- The M7 sealed holdout is **consumed** and is never reused.
- All M8 development used only the 5 canonical 1-minute symbols; **26 five-minute-only instruments**
  (broad/sector/leveraged/inverse/defensive) form a cohort **untouched** by M7 and M8 development.
- One **pre-registered** incumbent test on that fresh cohort (68,853 trades) independently
  corroborates the negative finding (expectancy −0.001, PF 0.67, not positive).
- Multiple-testing/data-snooping caveats recorded; no best-of-search config is described as
  certified. **Independent certification: deferred** (no positive challenger to certify).

Evidence: `holdout_integrity.json`, `multiple_testing_audit.json`.

## Compute (§18)

Honest, completion-measured (mx.eval) GPU vs CPU: batched linear inference favors the **CPU** at
small size (0.57×); large feature standardization favors the **GPU** (9×). CPU wins are reported,
not hidden. Research workloads never touch financial protection. Evidence:
`metal_research_benchmarks.json`.

## Invariants preserved

- Production mandatory −0.5% protective exit + event-driven path + 5-second backstop: **unchanged**.
- Native C++ RiskEngine remains the sole financial authority.
- −3% emergency candidate: **inactive**.
- No new engine/optimizer/lakehouse; no live activation; no fabricated performance; sealed holdout
  protected from optimization.

## Conclusion

The alpha-discovery infrastructure is complete, honest and operational, and it did its job: it
falsified the incumbent and every nearby alternative on genuine data, corrected a real accounting
defect, and refused to manufacture an edge. **No deployable edge exists yet.** The autonomous loop
is retained to keep gathering evidence and evaluating new hypotheses; a positive result is a
research objective, not a delivered claim.
