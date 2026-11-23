# Dynamic Universe Management & Opportunity Rotation — Design

**Branch:** `feature/dynamic-universe-rotation` (based on `feature/m7-trading-certification` @ `24c008f`, which carries M5 lakehouse + M6 learning + M6B autonomous optimization + M7 runtime/calendar).
**Date:** 2026-11-23
**Status:** Approved in principle (see Decisions). Autonomous execution authorized; clarification only for safety / data-integrity / architecture-compatibility forks.

---

## 1. Problem & Framing

The engine must maintain a **permanent baseline** of 30 securities while continuously discovering,
ranking, promoting, demoting, and reassessing additional securities — keeping at most **30 unique
tickers in the active scanning window** without permanently restricting the engine to 30.

**Key finding from codebase exploration:** ~80% of this already exists. This is a *gap-driven
extension*, not a greenfield build. The following already work and are **reused as-is**:

| Capability | Existing component |
|---|---|
| Canonical universe file + versioned snapshots | `universe.py`, `canonical_universe.py` (`CanonicalUniverseManager`) |
| Opportunity ranking (rel-strength, momentum, vol-expansion, liquidity, order-flow incl. seller exhaustion, regime) | `market_scan/ranking.py` (`score_candidate`, `rank`); GPU mirror `cengine/gpu_scanner_bridge.py` |
| Market scanner (discovery + rank cadences, single socket via bridge) | `market_scan/scanner.py`, `scanner_bridge.py` |
| Incremental resubscription without restart | `market_data/alpaca_stream.py` (`update_symbols`, `set_channel_subscriptions`); `universe_sync.py` (`UniverseSubscriptionSync`); `cengine/adaptive_subscription_controller.py` |
| Position/order protection (never unsubscribe held/pending) | `universe_sync.py`, adaptive tiers; tests `test_held_position_never_unsubscribed`, `test_pending_order_symbol_never_unsubscribed` |
| Hard-budget allocator (priority groups, no silent truncation) | `market_scan/subscription_manager.py` |
| Broker asset validation | `cengine/membership.py` (`fetch_alpaca_assets`, `MembershipRegistry`, `tradable_master_symbols`) |
| Broker reconciliation (positions/open orders) | `cengine/execution.py` (`startup_reconcile`, `reconcile_status`); `trade.py` loops |
| Lakehouse bronze→silver→gold + feature store + envelope | `cengine/lakehouse/*` |
| Decision-event contract (nonblocking sink → bronze) | `cengine/decision_events.py` |
| Learning layer (contextual bandits, registry, maturity, champion/challenger, daily cycle) | `cengine/learning/*` |
| Autonomous optimization (memory, hypotheses, experiments, validation) | `cengine/optimization/*` |
| Market-calendar / scheduling / supervisor | `cengine/runtime/*` (M7) |
| Deterministic & learning replay harnesses | `cengine/replay.py`, `cengine/learning/environment/replay.py` |

### Decisions (locked)
- **Scope:** Gap-driven extension. Reuse all working components. Do **not** consolidate the two
  subscription owners (`UniverseSubscriptionSync` flat path / `AdaptiveSubscriptionController` SIP
  path) unless a concrete correctness defect requires it. One subscription owner per runtime mode.
- **Baseline:** Config-driven in a dedicated **`baseline_universe.txt`**, seeded with the exact
  spec-30. `UPRO` is **not** baseline (dynamic-only). `universe.txt` stays the mutable working
  universe. Permanence = **membership**, not continuous subscription. Only operator config edits may
  add/remove baseline members; the Opportunity Finder never modifies the baseline. Invalid updates
  are rejected without overwriting last-good; changes are audited.
- **ML/RL:** Persistence **plus** a **shadow-only learned challenger** (contextual bandit over
  existing features) vs. the deterministic champion. Wire into existing M6B daily cycle, registry,
  maturity, governance. No production override. If data-sufficiency gates fail, emit collection
  requirements and defer fitting — never fabricate.
- **Replay:** Full static-vs-dynamic comparison (policies A–E) over the existing replay harness,
  chronological walk-forward OOS, net-of-cost, regime breakdown. Report as-measured; no unsupported
  improvement claims (per CLAUDE.md).

---

## 2. Architecture

A new **selection/rotation layer** computes the Active Universe and feeds the **existing**
subscription owner. It never owns a socket and never opens a second data connection.

```
                 baseline_universe.txt (operator config, permanent)
                          │
                          ▼
   BaselineRegistry ──────┐
                          │          scanner ranking (live_context.json / scanner._scored)
   MembershipRegistry ────┤                    │  (Opportunity Universe)
   (master/tradable)      │                    ▼
   positions/pending ─────┼──────►  DynamicUniverseManager ──► rotation cycle (13 steps)
   (protected)            │                    │
   M7 calendar/session ───┘                    ├──► Active Universe (≤30, protection-inclusive)
                                               │        │
                                               │        ├──► sole writer of universe.txt
                                               │        │     (CanonicalUniverseManager picks it up)
                                               │        │        │
                                               │        │        ▼
                                               │        │   EXISTING owner: UniverseSubscriptionSync
                                               │        │   OR AdaptiveSubscriptionController → stream
                                               │        │   (adds protection, enforces cap — no change)
                                               │        │
                                               │        ├──► DecisionEventSink (universe events)
                                               │        │        └► bronze → silver → gold → feature store → learning
                                               │        │
                                               │        └──► dashboard [DYNAMIC UNIVERSE MANAGER] + structured logs
                                               │
                                               └──► shadow challenger (learned policy) — recommendation only
```

### 2.1 New modules (grouped in a cohesive subpackage)

New package **`cengine/dynamic_universe/`** (distinct from top-level `universe.py`; avoids shadowing):

| Module | Responsibility |
|---|---|
| `config.py` | `UniverseRotationConfig` dataclass mirroring `UNIVERSE_CONFIG` (active_universe_size, baseline/opportunity target slots, refresh/opportunity-refresh/revisit/residence intervals, protect flags); env overrides (`CENGINE_UNIVERSE_*`). |
| `baseline.py` | `BaselineRegistry` — loads/validates/audits `baseline_universe.txt`, seeds spec-30 on first run, permanence guarantees, reject-invalid-keep-last-good, change audit log. Reuses `universe.build_universe` for normalize/validate/classify. |
| `layers.py` | Immutable, versioned `UniverseLayers` snapshot: `baseline`, `opportunity`, `master` (= baseline ∪ opportunity ∪ held ∪ pending, deduped), `active` (≤30). |
| `lifecycle.py` | Candidate lifecycle state machine `DISCOVERED→ELIGIBLE→RANKED→ACTIVE→ROTATED_OUT→REASSESSMENT→ACTIVE|EXPIRED`; per-symbol `CandidateState` with promotion/demotion reasons, min-residence, hysteresis, baseline-revisit deadlines, last-evaluation timestamps. Emergency bypass hook. |
| `rotation.py` | `UniverseRotationScheduler` — pure, deterministic implementation of the 13-step priority-aware, starvation-resistant round-robin. Inputs: baseline (ranked + staleness), opportunity ranking, protected set, mandatory references, config, lifecycle state. Output: `RotationResult` (active set ≤30, adds, removes, per-symbol reasons, coverage report). No I/O. |
| `manager.py` | `DynamicUniverseManager` — orchestrates one cycle: gather positions/pending (protected), refresh baseline, pull opportunity ranking, run scheduler, publish active universe atomically (sole writer of `universe.txt`), emit decision events, update dashboard stats, drive shadow challenger. Nonblocking async `run(stop)` loop; M7-calendar aware; persistent state recovery. |
| `state.py` | Persistent rotation state (last-evaluation times, lifecycle, residence) under `data/dynamic_universe/state.json`; atomic write; restored on startup. |

### 2.2 Integration rules (correctness-critical)

1. **Single writer of `universe.txt`.** Today the scanner's `UniversePublisher` writes both
   `universe.txt` and `live_context.json`. To avoid competing writers, the scanner continues to
   publish its **ranking** to `live_context.json` (the Opportunity source), and the
   `DynamicUniverseManager` becomes the **sole writer of `universe.txt` active membership**. The
   publisher's `universe.txt` write is gated/handed to the manager. Reuses `universe.write_canonical_universe` (atomic temp+rename).
2. **Protection-inclusive active set.** The scheduler reserves protected symbols (open positions,
   pending entry/exit, incomplete executions, unresolved reconciliation) and mandatory references
   **first**, so `active ⊇ protected` and `|active| ≤ 30`. The existing owner then re-adds
   protection on top (a no-op, since already included) and never overflows the 30 cap.
3. **Baseline permanence ≠ subscription.** The scheduler always considers the **full baseline** as
   candidates from `BaselineRegistry`, independent of what is currently in `universe.txt`. Dropping a
   baseline symbol from the active window never removes baseline membership and never liquidates a
   position (liquidation stays with existing execution/risk systems).
4. **Degraded state.** If protected + mandatory references exceed the subscription budget, reduce
   discretionary slots; if required monitoring still exceeds budget, enter a degraded state, **block
   new entries** (via the existing `market_scan/health.py` fail-closed gate), keep the independent
   `ProtectiveExitMonitor` risk path, and never silently exceed broker limits or abandon positions.
5. **One owner per mode.** The manager detects whether `AdaptiveSubscriptionController` is active
   (SIP) or `UniverseSubscriptionSync` (flat); it feeds the active universe to whichever is live and
   never instantiates a third owner.

### 2.3 Decision-event & lakehouse extension (§9)

- Add `DecisionEventType` members: `UNIVERSE_SELECTION`, `UNIVERSE_PROMOTION`, `UNIVERSE_DEMOTION`,
  `UNIVERSE_REJECTION`, `BASELINE_REVISIT`, `ROTATION_CYCLE`. Add matching `DecisionEventSink`
  record methods (O(1) nonblocking).
- Capture per spec §9: timestamp, symbol, baseline membership, opportunity score, ranking, selection
  reason, removal reason, market regime, feature snapshot reference, model version, active membership,
  subscription residence duration, subsequent price performance, trade decisions, actual execution
  outcomes, observable missed-opportunity outcomes. Reuse `EventEnvelope` (event-time vs
  availability-time) for point-in-time integrity.
- Map events to the existing bronze datasets (`universe_membership`, `subscription_events`; add a
  `universe_selection` dataset in `bronze._EVENT_DATASET`). Add silver `_TRANSFORMS` entries and a
  gold dataset linking selection → downstream `position_outcomes` via stable ids. Expose via
  `DuckDBWarehouse._VIEWS` and the point-in-time `feature_store`.
- **No fabrication:** censored/unobserved outcomes stay NULL; no rewards for unmonitored securities.

### 2.4 Learned shadow policy (§9, Option 2)

New subpackage **`cengine/learning/universe_selection/`**:
- `policy.py` — deterministic champion wrapper + contextual-bandit challenger built on existing
  `cengine/learning/contextual_bandits/linucb.py` (and/or `neural_bandit.py`). Context = existing
  candidate features + regime; arms = select/rotate decisions and allocation ratio.
- `reward.py` — bounded, attribution-windowed reward from downstream net-of-cost outcomes
  (net return, risk-adjusted, drawdown, conversion, churn, coverage, latency, capital efficiency).
  Distinguishes realized execution from counterfactual; never fabricates.
- `dataset.py` — builds training samples from the gold universe-selection + outcomes datasets via
  the point-in-time feature store (leakage-free ASOF joins).
- Governance: register in `cengine/learning/registry/model_registry.py`; maturity gated via
  `registry/maturity.py` (**SHADOW only** this milestone); evaluated via existing
  `orchestration/champion_challenger.py`; trained in `orchestration/daily_cycle.py` when
  `datasets/eligibility.py` + `cengine/optimization/data_sufficiency.py` gates pass. Challenger may
  emit a full alternative active universe but **never** touches production subscriptions or orders.
- Feed outcomes/experiment results into `cengine/optimization/memory.py` so the autonomous cycle can
  propose future allocation/rotation/residence/threshold experiments.

### 2.5 Observability (§10)

- `trade.py`: add `_dynamic_universe_dashboard_stats()` (must never raise) and render the
  `[DYNAMIC UNIVERSE MANAGER]` block (baseline/dynamic/master/active counts, active subscriptions
  N/30, protected, active baseline/opportunities, pending rotation, last/next rotation, coverage
  status HEALTHY/DEGRADED).
- Structured logging of every promotion, demotion, subscription update, validation failure, rotation
  decision; report baseline symbols not evaluated within `baseline_max_revisit_seconds`.

### 2.6 Historical replay comparison (§12)

New **`tools/universe_policy_comparison.py`** (+ `cengine/dynamic_universe/replay_comparison.py`
core) driving the existing replay harness over genuine local 1m/5m bars:
- Policies: **A** static-30, **B** equal 15/15, **C** flexible 20/10, **D** adaptive deterministic
  (ranked + baseline revisit + hysteresis), **E** learned shadow challenger (data permitting).
- Chronological walk-forward (train/validation/OOS/holdout), regime breakdown, identical controls
  (capital, costs/slippage, position sizing, risk, caps, latency, warm-up, broker eligibility).
- Point-in-time discovery only (no look-ahead / survivorship); dynamic policies get access to
  eligible non-baseline securities (replay restricted to baseline cannot validate discovery).
- Metrics: full trading set (net/gross, Sharpe/Sortino/Calmar, maxDD, PF, win rate, expectancy,
  turnover, capital utilization), universe-selection set (conversion, missed opportunities,
  false-positive rate, baseline coverage, revisit duration, churn, residence, window utilization,
  dynamic contribution to net P&L), operational (selection latency p50/p95, reconciliation latency,
  data freshness). Bootstrap CIs (time-series-appropriate), cost/allocation/frequency sensitivity.
- Artifacts (reproducible commands, config manifests, machine-readable results, equity/drawdown
  curves, attribution) persisted via existing research/lakehouse infra. Results feed M6B memory.

---

## 3. Phase decomposition

Delivered in one deliverable, implemented in these internal phases (each TDD, tests green + commit
before the next):

- **P1 — Baseline permanence + 4-layer model.** `config.py`, `baseline.py`, `layers.py`,
  `baseline_universe.txt` seed, permanence tests.
- **P2 — Rotation scheduler + candidate lifecycle.** `lifecycle.py`, `rotation.py` (pure, fully
  unit-tested: cap, slot allocation, starvation resistance, protection, revisit, hysteresis,
  residence, dedup, degraded state).
- **P3 — Manager + engine integration.** `manager.py`, `state.py`, single-writer handoff of
  `universe.txt`, feed existing subscription owner, nonblocking loop, M7-calendar awareness,
  restart recovery, integration tests (no second socket, positions never dropped, ≤30 enforced).
- **P4 — Decision persistence + lakehouse.** `decision_events` extension, bronze/silver/gold,
  feature-store linkage, persistence tests, no-fabrication tests.
- **P5 — Observability.** Dashboard block, structured logging, coverage reporting, tests.
- **P6 — Learned shadow policy.** `cengine/learning/universe_selection/*`, champion/challenger
  shadow, registry/maturity/governance, daily-cycle wiring, eligibility-gated training, tests.
- **P7 — Replay comparison + evidence.** Policies A–E harness, walk-forward, metrics, artifacts,
  engineering report.
- **Cross-cutting (finalize):** full regression suite, README + Mermaid diagrams + config docs,
  paper-trading smoke test, final engineering report.

---

## 4. Testing strategy (§12)

Unit + integration + regression. Mandatory coverage:
- Exact preservation of all baseline symbols; baseline never permanently removed by the finder.
- Active universe never exceeds 30 unique tickers (property test over random candidate pools).
- Dynamic promotion/removal; baseline rotation + reinsertion; baseline revisit within interval.
- Protection of positions / pending entry / pending exit / incomplete execution / unresolved recon.
- Mandatory reference handling; duplicate prevention; invalid/unavailable symbol rejection.
- Insufficient opportunity candidates (fill with baseline); stale data; streaming subscription
  failure; concurrent universe updates; broker reconnection; engine restart + state recovery;
  continuous operation across sessions; **starvation prevention**.
- ML/RL persistence + no-fabrication; champion/challenger shadow isolation (no real orders).
- Historical replay reproducibility; static-vs-dynamic comparison under identical controls.

All existing tests must continue to pass (regression gate).

---

## 5. Non-goals / constraints

- No parallel trading engine; no second socket/subscription owner/competing publisher.
- No modification to execution modes or risk controls; protective exit stays independent.
- No unvalidated learning policy overriding production risk controls (shadow-only this milestone).
- Backward compatible: `CENGINE_SYMBOLS` override still pins + disables dynamic sync; dynamic
  behavior gated by env flags so the static path remains the default-safe fallback.
- No blocking operations in the primary execution loop; all new work is out-of-band/nonblocking.
- No performance claims without supporting evidence.

---

## 6. Outstanding risks / open items

- **R1 — universe.txt write handoff.** Changing the scanner publisher from universe.txt writer to
  ranking-only publisher must not break `market_scan/health.py`'s fail-closed entry gate (which
  reads both files for freshness/consistency). Mitigation: manager writes `universe.txt` on the same
  cadence/atomicity contract; health gate semantics preserved; covered by `test_market_scan_health`.
- **R2 — Two subscription owners.** Must robustly detect the active owner per mode and feed only it.
  Mitigation: reuse the existing mutual-exclusion (`trade.py` sets `universe_sync=None` when adaptive
  is active); manager keys off the same flag.
- **R3 — Data sufficiency for learned policy.** Likely insufficient real universe-selection history
  at first. Mitigation: ship full pipeline + emit collection requirements; defer fitting behind
  eligibility gates; champion (deterministic) remains production.
- **R4 — Replay realism.** Replay must use point-in-time discovery, not unrestricted history.
  Mitigation: reuse `record_universe_snapshot` / point-in-time feature store; document coverage,
  missing data, and limitations explicitly.
