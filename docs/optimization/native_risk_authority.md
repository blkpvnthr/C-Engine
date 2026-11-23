# Native Risk Authority & the Inactive Candidate Emergency Stop (Milestone 6B)

Milestone 6B adds an autonomous optimization layer **on top of** C-Engine's existing
financial controls. It changes none of them. This document records the single, invariant
rule that makes that safe: **the native C++ `RiskEngine` is the sole financial authority,
and everything in `cengine/optimization/` is recommendation-only.**

## The sole financial authority: `RiskEngine`

`risk/risk_engine.hpp` / `risk/risk_engine.cpp` define the pre-trade authority boundary.
It is the only component permitted to accept or reject an order.

```
      Python TradingStrategy
              |
              v
      Python/C++ OrderManager
              |
              v
          RiskEngine        <-- sole pre-trade authority
          /       \
       reject     accept
                    |
                    v
             LimitOrderBook
```

Its surface (as bound in `bindings/python_bindings.cpp` on the `_cengine_native` module):

- `RiskEngine.evaluate(request) -> RiskResult` — the permit/deny decision for a new order.
- `RiskEngine.evaluate_amendment(request) -> RiskResult` — the same for an amendment.
- `activate_kill_switch()` / `deactivate_kill_switch()` / `kill_switch_active`.

Per its own header contract, the `RiskEngine` **does NOT** generate trading signals, match
orders, mutate the order book, calculate indicators, own portfolio positions, or send broker
orders. It only evaluates whether a proposed order or amendment is permitted. State-mutating
and evaluation operations are synchronized (thread-safe).

Nothing the optimizer produces can bypass, widen, or substitute for this boundary. Hard
risk limits are **not** members of any search space (`search.py`): "Hard risk limits and the
mandatory emergency protection are NOT in any search space."

## The candidate native emergency stop — INACTIVE

`risk/protective_exit_candidate.hpp` defines a **candidate** emergency liquidation trigger.
It is deliberately a pure, stateless, crash-independent native function so it could remain
operational if Python / MLX / GPU / model-loading were to fail. It is **not active**.

```cpp
// Default candidate emergency threshold (signed fraction). Research value, not production.
inline constexpr double kCandidateEmergencyThreshold = -0.03;   // -3%

struct EmergencyExitDecision {
    bool   triggered{false};
    double unrealized_return{0.0};
    double threshold{kCandidateEmergencyThreshold};
    bool   overrides_tighter_stop{false};   // never true
};

EmergencyExitDecision evaluate_emergency_exit(double unrealized_return,
        double emergency_threshold = kCandidateEmergencyThreshold) noexcept;

double clamp_emergency_threshold(double proposed, double current) noexcept;
```

Semantics and guarantees, straight from the source:

- **Not wired in.** It is **not** called by `RiskEngine::evaluate`, `evaluate_amendment`, or
  any live path. It stays inactive in production until independent Milestone 7 certification
  authorizes it. (The header's own words: *"NOT wired into RiskEngine::evaluate or any live
  path … must stay inactive in production until independent Milestone 7 certification."*)
- **Pure & crash-independent.** `evaluate_emergency_exit` is deterministic and
  side-effect-free — safe to call from any context, including degraded/fallback paths.
  `triggered` is true when `unrealized_return <= emergency_threshold`.
- **Never overrides a tighter stop.** `overrides_tighter_stop` is always `false`: a tighter
  active stop, if present, always takes precedence. No ML/RL recommendation may override an
  activated trigger.
- **Never loosened.** `clamp_emergency_threshold(proposed, current)` returns the **more
  protective** (tighter) of the two — a more-negative (wider/looser) threshold is rejected.
  The optimizer may only ever *tighten* it, never widen it. The governed config enforces the
  same bound: `candidate_emergency_pct` may be tightened but may not go looser than −3%
  (`OptimizationConfig.validate`).
- **A liquidation level, not a guaranteed max loss.** Gaps / halts / partial fills can
  produce worse fills; the threshold is the level at which liquidation is intended, not a
  realized loss guarantee.

Python bindings (in `bindings/python_bindings.cpp`) expose `EmergencyExitDecision`,
`evaluate_emergency_exit`, and `clamp_emergency_threshold` on `_cengine_native`, and the CLI
exposes a read-only check:

```
python -m cengine.optimization emergency-check --unrealized-return -0.031
```

This prints `triggered`, `threshold`, `overrides_tighter_stop`, and a note that it is the
**native candidate emergency exit (INACTIVE; M7 certifies)**. It computes a value; it takes
no action.

## The production −0.5% protective exit is preserved

The production rule — unconditional liquidation at **−0.5%** — is untouched by this
milestone. It is modeled in `cengine/optimization/exits/` as the preserved incumbent:

- `PRODUCTION_EXIT_PCT = -0.005` (`exits/adaptive.py`) is the control / fallback / rollback.
- `OptimizationConfig.production_exit_pct` is fixed at `-0.005` and `validate()` raises a
  `ConfigError` if anything attempts to change it: *"the production −0.5% exit must be
  preserved unchanged."*
- In the EXIT-00..15 matrix, **EXIT-00** is the production incumbent baseline every candidate
  is scored against.

## The recommendation-only boundary

All protective-exit "policy" in `cengine/optimization/exits/` emits **intended actions, never
orders**:

- `AdaptiveExitPolicy.decide(...)` consumes plain dicts and returns an `ExitDecision`
  (`HOLD` / `REDUCE_POSITION` / `EXIT_POSITION` / `EXIT_AND_REPLACE`). It never touches an
  order. On invalid or unavailable intelligence it **falls back to the incumbent −0.5%
  behavior** (`fallback_incumbent_exit`). Its independent emergency trigger (−3%) is never
  bypassed, and it never authorizes holding beyond the active adaptive stop.
- `ProtectiveExitStateMachine` is explicitly *"a RECOMMENDATION state machine — it emits
  intended actions, never orders; native C++ retains authority."* A triggered
  mandatory/emergency boundary always wins from any state; stale data or invalid predictions
  route to the incumbent exit.
- Wider candidate stops never increase capital at risk — `risk_normalized_notional(...)`
  holds *risk*, not notional, constant, so a wider stop yields a *smaller* position.

### Governed invariants (enforced in code)

`OptimizationConfig.validate()` (`config.py`) rejects any configuration that would cross a
hard line, so the authority boundary cannot be loosened by configuration:

| Invariant | Enforcement |
| --- | --- |
| No live / production activation | `production_activation_enabled` and `autonomous_live_activation` must be `False` |
| No autonomous risk expansion | `autonomous_risk_expansion` must be `False` |
| No hard-stop / mandatory-protection tuning | `hard_stop_optimization` must be `False` |
| No live exploration | `live_exploration` must be `False` |
| Production −0.5% exit preserved | `production_exit_pct` must equal `-0.005` |
| Emergency never loosened | `candidate_emergency_pct` may not be looser than `-0.030` |
| Adaptive stop ordering sane | `initial_stop_pct` not wider than `max_stop_pct` |
| M7 certification required | `milestone_7_certification_required = True` |

## Summary of the boundary

- The native C++ `RiskEngine` is the **sole** pre-trade financial authority.
- The candidate native emergency stop (−3%, `protective_exit_candidate.hpp`) exists but is
  **INACTIVE** — not wired into `evaluate`, never overriding a tighter stop, never loosened;
  Milestone 7 owns its certification and any activation.
- The production −0.5% exit is **preserved untouched**.
- Everything in `cengine/optimization/` is **recommendation-only**: no orders, no risk
  expansion, no self-code-modification, no live activation.

See `milestone7_boundary.md` for what Milestone 7 owns, and `daily_workflow.md` for how the
off-critical-path cycle runs.
