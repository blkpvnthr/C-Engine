# Adaptive Position Management & Exit Arbitration (Milestone 4, §12/§13/§14)

`cengine/position_management.py` is a unified layer over **actual** positions and
validated observations. It cleanly separates MANDATORY protection (the independent
−0.5% exit in `protective_exit.py`, never produced here) from DISCRETIONARY exits.

## Inputs — `PositionView`

Validated, no fabricated values: unrealized return, MFE/MAE, ATR %, realized vol,
momentum + acceleration, buyer/seller exhaustion, VWAP/BB relationships, holding
duration, strategy-invalidation flag, effective-exposure fraction. Unknown → `None`.

## Discretionary profit-protection policies (§13)

All default `MaturityState.RESEARCH` (shadow) and are **maturity-gated** — each emits
a `PositionAction(shadow_only=True)` unless explicitly promoted + authorized for the
run. Never auto-enabled for live.

| Policy | Trigger |
|---|---|
| `atr_trailing` | giveback from peak ≥ k × ATR |
| `break_even` | after arming in profit, falls to cost-adjusted break-even |
| `momentum_deterioration` | profitable long + momentum down + buyer exhaustion |
| `time_based` | stagnant beyond max holding duration |
| `technical_invalidation` | strategy thesis invalidated |

`AdaptivePositionManager.evaluate(view)` runs enabled policies (each wrapped so one
policy error never crashes the loop) and returns candidate `PositionAction`s.

## Exit arbitration (§14)

`ExitArbiter.arbitrate(actions)` enforces the deterministic priority hierarchy
(lower = higher): `EMERGENCY(1) < MANDATORY_PROTECTIVE(2) < PORTFOLIO_RISK(3) <
PROFIT_PROTECTION(4) < TECHNICAL_EXIT(5) < POSITION_ADJUST(6) < NEW_ENTRY(7) <
REALLOCATION(8)`. Among equal priority, a full exit outranks partial/adjust/entry and
a routable (non-shadow) action is preferred. `conflicts(...)` detects an incompatible
simultaneous exit+entry for a symbol so the arbiter collapses it to one action.

The **mandatory protective exit always outranks** every discretionary portfolio/
strategy decision; it is never emitted or suppressed here. Discretionary actions are
shadow-only by default; they reach the existing execution path only under explicit
authorization. See `reports/position_management_validation.json`.
