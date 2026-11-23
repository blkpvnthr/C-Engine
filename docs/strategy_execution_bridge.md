# Strategy Execution Bridge & Mock-Execution Certification

This document covers the M2 execution-certification package: how the advisory
five-family regime intelligence is connected to the existing execution authority,
the maturity/authorization model that gates it, and how mock execution is
certified without granting paper/live eligibility.

## Why a bridge

The merged Milestone 2 layer (regime engine, five families, arbitration, economic
ranking, decision events) is **advisory**: it emits recommendations and decision
evidence but routes nothing to execution. The bridge closes that gap **without**
creating a parallel engine or bypassing any financial control. Everything still
flows through the existing authority:

```
StrategyRecommendation
  → StrategyExecutionBridge          (cengine/strategy_execution_bridge.py)
  → CandidateRouter                  (trade.py; single-pending guard)
  → Exposure authorization gate      (cengine/exposure_registry.py)
  → native C++ RiskEngine            (native_adapters.NativeRiskAdapter → _cengine_native.RiskEngine)
  → OrderManager                     (order_manager.py)
  → ExecutionVenue                   (AlpacaExecutionVenue in prod; SimulatedBroker for mock)
  → fill reconciliation + decision/execution evidence (DecisionEventSink)
```

## Two gates, both required

A recommendation routes only when **both** hold:

1. **`conditions_passed`** — the strategy/entry/regime conditions passed. This is
   set by the real family evaluators (`_finalize` in `strategy_arbitration.py`),
   independent of advisory eligibility. The strict elective-long contract is:
   `DI+5m > DI−5m AND (price > VWAP OR price > SMA20_5m OR price > BB_middle_5m)
   AND developing_1m > previous_1m_close` — identical to the native `screen_market`
   contract, and applied to authorized inverse ETFs too (own price action).
2. **`authorization_permits(maturity, authorization)`** — the family's maturity
   permits the run's authorization.

## Maturity ladder (independent of advisory eligibility)

`cengine/strategy_maturity.py` defines `MaturityState`:

| State | Meaning | Routable under |
|---|---|---|
| `BLOCKED` | hard stop | nothing |
| `DISABLED` | administratively off | nothing |
| `RESEARCH` *(default)* | evaluated + logged | nothing |
| `SHADOW` | recommendations surfaced | nothing |
| `REPLAY_VALIDATED` | validated on historical replay | nothing |
| `MOCK_EXECUTION` | isolated test authorization | `MOCK` only |
| `PAPER_ELIGIBLE` | eligible for paper | `MOCK`, `PAPER` |
| `LIVE_ELIGIBLE` | eligible for live | `MOCK`, `PAPER`, `LIVE` |

`ExecutionAuthorization` is `NONE` (default — routes nothing), `MOCK`, `PAPER`, `LIVE`.

**Safety invariants** (`tests/test_strategy_maturity.py`):

* A `MOCK_EXECUTION` family is routable **only** under `MOCK` — never paper/live.
  Exercising a family through the mock broker grants **no** production eligibility.
* `BLOCKED` overrides every authorization and cannot be promoted.
* Promotion requires regression tests **and** an explicit operator assertion;
  nothing is auto-promoted.
* This maturity dimension is **separate** from the advisory `EligibilityState`
  (`EXECUTION` = authorized live elective entry), which remains default-off and
  is untouched by mock promotion.

## trade.py wiring (default-off)

The bridge is constructed only when `CENGINE_STRATEGY_EXECUTION_AUTH` is set to
`mock`/`paper`/`live` (default `none`). Even then, nothing routes until a family
is promoted to the matching maturity. The zero-argument engine therefore remains
advisory-only. Bridge orders emit ack/fill/risk decision events via a handler
composed onto the OrderManager audit chain (`make_decision_audit_handler`),
alongside the existing `StrategyStateBook` and journal handlers.

## Isolated mock broker

`cengine/mock_broker.py` `SimulatedBroker` implements `ExecutionVenue` with no
network and no wall-clock (monotonic counters only). Fill plans: `FULL`,
`PARTIAL_THEN_FULL`, `REJECT`, `NOFILL`. `cengine/mock_adapters.py` pairs the
**real** native RiskEngine with the simulated broker for a full-engine mock run
via `ENGINE_NATIVE_ADAPTER_MODULE=cengine.mock_adapters`.

## What "mock-certified" means (and does not)

Certified: the complete production decision path executes end-to-end for all five
families across instrument classes and adverse scenarios, with traceable evidence
(`tests/test_mock_execution_fullpath.py`, 19 cases; `tests/test_strategy_execution_bridge.py`,
10 cases). The **real** native C++ RiskEngine approves/rejects (including
insufficient-capital rejection).

Not implied: economic edge (out-of-sample), paper certification, or live
eligibility. Those are separate, gated milestones. Mandatory −0.5% protective
exits remain fully independent (`tests/test_protective_exit_independence.py`).

## Evidence

`reports/strategy_execution_integration.json`, `reports/strategy_activation_validation.json`,
`reports/strategy_mock_execution_validation.json`, `reports/decision_event_integrity.json`,
`reports/native_strategy_architecture.json`, `reports/protective_exit_validation.json`,
`reports/m2_independent_verification.json`, `reports/implementation_progress.json`.
