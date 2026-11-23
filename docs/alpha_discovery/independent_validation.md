# Independent Validation Protocol (Milestone 8, §20)

This document specifies how C-Engine Milestone 8 separates *development* from
*certification* so that no result is contaminated by the search that produced it.
The protocol enforces a strict data firewall between four data roles, treats the
Milestone 7 sealed holdout as permanently consumed, reserves a fresh instrument
cohort that development never touched, and runs exactly one pre-registered
confirmatory test on that cohort.

The headline outcome is a negative one, honestly reported: **no development
challenger reached the frozen challenger gates, so there is nothing to certify.
Independent certification is DEFERRED.** The single confirmatory test does not
rescue the strategy — it independently corroborates the negative finding on data
that played no part in development.

Evidence: `reports/alpha_discovery/holdout_integrity.json`,
`reports/alpha_discovery/multiple_testing_audit.json`,
`reports/alpha_discovery/challenger_registry.json`,
`reports/alpha_discovery/final_alpha_discovery_report.md`.

---

## 1. Four separated data roles

Every analysis in M8 is assigned exactly one data role. Data never flows backward
across the firewall — a role used for selection can never later be used for
certification.

| Role | Purpose | Universe | Timeframe |
|------|---------|----------|-----------|
| Training (TRAIN) | Fit thresholds, filter parameters, penalty weights | Chronological first 70% per symbol | 1Min |
| Development-validation (VAL) | Measure selected configurations; reject overfit | Chronological last 30% per symbol | 1Min |
| Purged walk-forward | Time-ordered out-of-sample re-evaluation | Canonical dev universe | 1Min |
| Final independent certification | Confirm a passing challenger on untouched data | Fresh cohort (reserved) | 5Min |

- **TRAIN / VAL** splits are chronological per symbol — the validation window is
  strictly later in time than the training window, so no future information leaks
  backward. Thresholds are fit on TRAIN only and measured on VAL.
- **Purged walk-forward** re-evaluates survivors in time order on out-of-sample
  segments.
- **Final certification** is the only role permitted to touch the reserved fresh
  cohort, and only for a pre-registered test — never for a parameter search.

---

## 2. The M7 sealed holdout is consumed

The Milestone 7 certification already spent its sealed holdout on the five
canonical 1-minute symbols (AAPL, MSFT, NVDA, SPY, QQQ). Under this protocol that
holdout is treated as **consumed**:

- `status: "consumed"`, `reused_as_new_holdout: false`.
- It is **never** re-opened as a fresh holdout for M8. Re-using a holdout that has
  already informed a decision would turn it into an in-sample set and silently
  inflate any result measured on it.
- All M8 *development* (§5–§14) was conducted on those same five 1-minute symbols
  as the working universe — TRAIN/VAL splits, filter scans, family comparisons,
  regime buckets, ML and bandit experiments. Because development reused the
  development universe, no M8 development number may be presented as an
  independent out-of-sample result.

---

## 3. The fresh cohort (reserved, untouched)

A separate cohort of **26 five-minute-only instruments** shares **no symbol** with
the 1-minute development/holdout universe and was touched by neither M7
certification nor M8 development:

| Group | Instruments |
|-------|-------------|
| Broad / mega-cap | AMZN, GOOGL, META, TSLA, AMD, PLTR |
| Index / broad ETFs | DIA, IWM |
| Sector ETFs | XLE, XLF, XLK, XLP, XLV, SMH, SOXX, GLD |
| Leveraged (long) | SOXL, TQQQ, UPRO |
| Inverse / leveraged-inverse | DOG, PSQ, SH, SDS, SQQQ, SOXS |
| Defensive / rates | TLT |

Key properties (`holdout_integrity.json`):

- `fresh_cohort_size: 26`, `fresh_cohort_timeframe: "5Min"`.
- `fresh_cohort_untouched_by_development: true`.
- The cohort is deliberately heterogeneous — broad, sector, leveraged, inverse and
  defensive exposures — so a confirmatory test is not a near-replica of the
  development universe.

Because the cohort is 5-minute-only while development is 1-minute-only, it is
genuinely out-of-universe *and* out-of-timeframe relative to everything the search
has seen.

---

## 4. One pre-registered confirmatory test

A fresh cohort is a scarce resource: the moment it is used to pick among many
candidates, it is consumed by selection bias and stops being independent.

Under this protocol the cohort is spent on **exactly one pre-registered test**, not
a search:

- **Why only one.** No development challenger reached the frozen
  `ChallengerGates` (see §5 below), so there was no winning configuration to carry
  forward and tune. With nothing to optimize against the cohort, the correct use is
  a single confirmatory probe of the incumbent, decided in advance.
- **What was run.** The pre-registered incumbent strategy was replayed once on the
  full 26-instrument cohort — **68,853 trades**.
- **Result.** Net expectancy ≈ **−0.001**, profit factor ≈ **0.67**, **not
  positive**. The finding is directionally identical to development: the entry has
  no out-of-sample net edge under the mandatory −0.5% protective exit.

This test **corroborates a negative finding**. It is not a certification and must
never be cited as one. The cohort was not used to select a best-of-many
configuration.

---

## 5. Why certification is DEFERRED

Certification requires a challenger that first passes the frozen `ChallengerGates`
on development data. The genuine M8 run produced none:

| Development outcome | Count |
|---------------------|-------|
| Accepted | 0 |
| Rejected | 6 |
| Deferred | 2 |

- The challenger registry is anti-retrofitting fingerprinted
  (`gate_fingerprint: 19b52641f3e837e7`); the gates were **not** relaxed to
  manufacture a pass.
- The incumbent itself fails every economic gate under corrected portfolio
  accounting (net expectancy −0.001075, PF 0.641, annualized Sharpe −5.201,
  max drawdown 0.803; fails net-expectancy, PF, Sharpe, Sortino, drawdown, and the
  beat-cash / beat-incumbent / beat-baselines requirements).

Because the certification entry condition (a gate-passing challenger) is never met,
**independent certification is DEFERRED — there is no positive edge to certify.**
No best-of-search development configuration is described as certified anywhere in
M8.

---

## 6. Multiple-testing, data-snooping, and selection-bias controls

M8 ran roughly **400 development comparisons** across families
(`multiple_testing_audit.json`):

| Development family | Comparison surface |
|--------------------|--------------------|
| Entry-quality filter scans | 11 features × 8 thresholds × 2 directions |
| Multi-timeframe gates | 5 |
| Regime buckets | ~9 |
| Strategy-family archetypes | 5 |
| Supervised models | 4 |
| Bandit contexts | regime × family |

With that many comparisons, **the best figure among them is optimistic by
construction.** The protocol records this explicitly and refuses to launder a
selection-biased maximum into a claim:

- **Selection-bias warning (recorded).** The best development number is optimistic;
  a Bonferroni-style correction would widen every development confidence interval
  substantially. No best-of-search configuration is described as certified.
- **Data-snooping controls (enforced):**
  - Chronological TRAIN/VAL splits — no future leakage.
  - Thresholds fit on TRAIN, measured on VAL only.
  - Sealed M7 holdout never accessed.
  - Fresh cohort reserved; single pre-registered incumbent test only.

---

## 7. Dependence-aware caveats

The replayed trade series is **not** a set of independent draws, so naive
per-trade statistics overstate confidence:

- **Overlapping, concurrent positions.** Strided entries overlap in time and form a
  capital-constrained portfolio of concurrent positions (§3 accounting audit), so
  trade outcomes are serially and cross-sectionally dependent. Per-trade standard
  errors treating each trade as independent are therefore too narrow.
- **Regime clustering.** Outcomes cluster by market regime and time-of-day; a block
  of trades in one regime is effectively far fewer independent observations than its
  raw count.
- **Effective sample < nominal sample.** Both the ~400-comparison correction and the
  dependence structure push the *effective* breadth below the nominal trade and
  comparison counts. Where the gates require "adequate effective sample," this is
  assessed on effective, not nominal, breadth.

These caveats only *widen* uncertainty around results that are already negative —
they give no reason to upgrade any finding.

---

## 8. Dashboard separation

The alpha-discovery dashboard panel (§24) keeps development and certification
results in strictly separate fields so a development maximum can never be misread as
a certified result:

- `BEST_DEVELOPMENT_NET_EXPECTANCY` — development-only, selection-biased.
- `INCUMBENT_NET_EXPECTANCY` — incumbent baseline.
- `INDEPENDENT_CERTIFICATION_STATUS` — reports **DEFERRED**; no fabricated metrics.

Development results are never promoted into the certification field.

---

## 9. Summary

- Four data roles — training, development-validation, purged walk-forward, final
  certification — are firewalled; selection data never becomes certification data.
- The M7 sealed holdout is **consumed** and never reused.
- A **26-instrument, 5-minute fresh cohort** (no symbol overlap, out-of-timeframe)
  is reserved and untouched by M7 and M8 development.
- Because **0 development challengers reached the frozen gates**, the cohort was not
  consumed by a selection search; **one pre-registered incumbent test (68,853
  trades)** independently corroborated the negative finding (expectancy ≈ −0.001,
  PF ≈ 0.67, not positive).
- Multiple-testing, data-snooping, and dependence-aware caveats are recorded; the
  effective sample is treated as below nominal.
- **Independent certification is DEFERRED — no positive edge exists to certify.**
  No best-of-search development configuration is described as certified.

---

*Provenance: all economic figures above are historical-replay simulations on
genuine Alpaca SIP bars. They are not live performance and not a profitability
claim; no live orders were placed. Negative results are preserved as reported and
are never reinterpreted.*
