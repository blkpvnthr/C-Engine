# Dynamic Universe Management & Opportunity Rotation — Engineering Report

**Branch:** `feature/dynamic-universe-rotation` (base `24c008f`, M5+M6+M6B+M7).
**Date:** 2026-11-23.
**Status:** Implementation + unit/integration validation complete; gated OFF by
default. Real-data replay comparison is harness-complete but **not run** (no
historical bars in this worktree — see Limitations). No performance improvement
is claimed on real data.

---

## 1. Files created / modified

**New package `cengine/dynamic_universe/`** (selection/rotation decision layer):
`config.py`, `baseline.py`, `layers.py`, `lifecycle.py`, `rotation.py`,
`manager.py`, `state.py`, `events.py`, `observability.py`, `replay_comparison.py`.

**New package `cengine/learning/universe_selection/`** (shadow policy):
`reward.py`, `policy.py`, `orchestration.py`, `__init__.py`.

**New config:** `baseline_universe.txt` (operator-managed permanent baseline, spec-30).

**New infra:** `conftest.py` (worktree import bootstrap — removes the scikit-build
editable finder so `cengine.*`/`market_data.*` resolve to this worktree).

**Modified (reuse, not rebuild):**
- `market_scan/publisher.py` — `publish()` gains `members` (active window →
  `universe.txt`) distinct from `ranked_candidates` (entry-eligible →
  `entry_eligible_count`); preserves the fail-closed gate's consistency contract.
- `market_scan/scanner.py` — builds the manager when
  `CENGINE_DYNAMIC_UNIVERSE_ROTATION=1`; at publish the manager's active window
  becomes membership; fail-closed fallback; `decision_sink` wiring.
- `trade.py` — passes the engine `DecisionEventSink` to the scanner;
  `_dynamic_universe_dashboard_stats()` added to the dashboard observer.
- `cengine/decision_events.py` — universe event types + `record_universe_rotation()`.
- `cengine/lakehouse/{bronze,silver,gold,warehouse}.py` — `universe_selection`
  dataset routing + silver transform + gold builder + warehouse views.
- `cengine/learning/datasets/eligibility.py`, `cengine/optimization/data_sufficiency.py`
  — `universe_selection` task spec + sufficiency gating.

**Docs:** `docs/dynamic_universe/architecture.md` (+ mermaid), design spec and
plans under `docs/superpowers/`.

Total: 42 files, +4978 / −14 lines.

## 2. Architecture changes

A new **selection/rotation layer** computes the active universe (≤30) and feeds
the **existing** publisher/subscription owner. Authority is separated so the
fail-closed entry gate is untouched:

- `DynamicUniverseManager` = sole **selection** authority (immutable
  `ActiveUniverseSnapshot`, generation id; writes no files).
- `UniversePublisher` = sole **writer** + ~3s heartbeat, consuming the snapshot.
- `UniverseSubscriptionSync` / `AdaptiveSubscriptionController` = subscription
  owner (unchanged; one per runtime mode).

No second socket, no competing publisher, no parallel engine. See
`architecture.md` for the data-flow diagram.

## 3. Configuration parameters

`UniverseRotationConfig` (env `CENGINE_UNIVERSE_*`): `active_universe_size=30`,
`baseline_target_slots=20`, `opportunity_target_slots=10`,
`refresh_interval_seconds=60`, `opportunity_refresh_seconds=300`,
`baseline_max_revisit_seconds=300`, `candidate_min_residence_seconds=180`,
`protect_open_positions=True`, `protect_pending_orders=True`.

Gate/paths: `CENGINE_DYNAMIC_UNIVERSE_ROTATION` (default **0/off**),
`CENGINE_BASELINE_UNIVERSE_PATH`, `CENGINE_DYNAMIC_UNIVERSE_STATE`.

## 4. Test results

Full suite: **1031 passed, 10 skipped, 0 regressions** (baseline before this work
was 894; the editable-install conftest fix brought all `cengine.*` tests onto the
worktree copies). 70 tests are dedicated to this feature, covering every §12
requirement that does not depend on live market data:

- baseline preservation (all 30, UPRO excluded, reject-invalid-keep-last-good);
- active window never exceeds 30 (200-pool property test);
- promotion/removal, baseline rotation + reinsertion, baseline revisit;
- protection of positions/pending; degraded state (no silent truncation);
- duplicate prevention; starvation resistance; anti-churn residence;
- restart + state recovery; stalled-manager fail-closed;
- decision-event persistence (no fabrication); routing to bronze;
- observability block + coverage; shadow policy learning on synthetic data;
- data-sufficiency deferral (defers + emits collection requirement, never fakes);
- replay comparison math (dynamic beats static when candidates carry alpha; cost
  penalizes churn; walk-forward chronological; bootstrap deterministic).

## 5. Historical replay results

The comparison harness (`replay_comparison.py`, policies A–E, walk-forward OOS,
net-of-cost metrics, block-bootstrap CIs) is implemented and **validated on
synthetic panels** where the correct policy is knowable:

| Policy | synthetic net | baseline coverage | churn |
|---|---|---|---|
| A static | one-time entry cost only (~−0.0005), 0 thereafter | 1.00 | ~0 |
| C flexible / D adaptive | positive (captures candidate alpha) | lower | higher |

These are **synthetic** results that validate the harness's isolation of the
selection variable — they are **not** a claim about real markets.

**Real-data run: NOT performed.** This worktree contains no historical 1m/5m bars
(they live in the primary checkout's local, gitignored store). A genuine run also
requires reconstructing point-in-time opportunity-score panels by replaying the
ranking over history. The harness is ready; the reproducible entry point is
`cengine.dynamic_universe.replay_comparison.compare_policies(panel, cap=30)` with
a `ReplayPanel` built from genuine bars. Per project policy, no real performance
figures are reported until that run is executed on real data.

## 6. Performance comparison

None claimed on real data (see §5). Synthetic validation only.

## 7. Outstanding limitations

1. **Real-data replay not executed** — needs historical bars + a ranking-over-
   history panel builder (follow-up with the data pipeline). Harness is ready.
2. **Learned policy is untrained** — data-sufficiency gates DEFER training until
   genuine logged selections + realized rewards accumulate; the cycle emits a
   collection requirement rather than fabricating. Shadow-only regardless.
3. **Replay attribution is equal-weight long** — isolates selection, not the full
   execution/risk engine; comparable to, never a replacement for, the live path.
4. **Gold universe-selection reward** uses per-symbol average attribution; a
   stricter ASOF forward-window join is a refinement.
5. **Scanner-thread manager** reads positions/pending from the scanner's cached
   snapshot; `universe_sync` independently re-protects on the engine side
   (defense in depth), so protection holds even if the two momentarily diverge.

## 8. Deployment readiness

- **Safe to merge behind the flag:** default OFF ⇒ legacy behavior unchanged;
  full suite green; no regressions; fail-closed on manager error/stall.
- **Before enabling in paper:** run a paper-trading smoke test with
  `CENGINE_DYNAMIC_UNIVERSE_ROTATION=1` and confirm (a) the
  `[DYNAMIC UNIVERSE MANAGER]` block populates, (b) `universe.txt` tracks the
  active window, (c) positions/pending are never unsubscribed, (d) entries still
  reach the broker. (Not run here — no live credentials/data in this worktree.)
- **Before enabling in live:** the learned policy must remain SHADOW; promotion
  past SHADOW requires explicit operator approval via the existing governance.

## 9. Pull request summary

Adds autonomous baseline-permanent universe rotation as a gap-driven extension:
a pure decision core (config/baseline/layers/lifecycle/rotation), a selection-
authority manager feeding the existing publisher/subscription owner, universe-
selection decision persistence through the lakehouse, a terminal observability
block, a shadow-only learned contextual-bandit challenger with honest data-
sufficiency deferral, and a reproducible static-vs-dynamic replay comparison
harness. Gated OFF by default; 1031 tests pass; no regressions; no unsupported
performance claims.
