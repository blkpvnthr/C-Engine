# Intelligent Portfolio Replacement (Milestone 3, §13)

`cengine/portfolio_replacement.py` (`ReplacementCoordinator`) evaluates new
qualified opportunities for freed capital **only after** a mandatory protective
exit has fully completed and broker state is reconciled. It places no orders itself
— a replacement decision is handed to the existing bridge / router / native
authority. It reuses the existing decision logic in
`market_scan/replacement.py` (`select_replacement`, `ReplacementEligibility`).

## State machine

```
MANDATORY_EXIT_TRIGGERED
  → EXIT_ORDER_SUBMITTED
  → EXIT_PARTIALLY_FILLED
  → EXIT_FULLY_FILLED
  → BROKER_RECONCILED
  → CAPITAL_AVAILABLE
  → REPLACEMENT_EVALUATION
  → REPLACEMENT_AUTHORIZED      (or HELD_CASH)
  → REPLACEMENT_ORDER_SUBMITTED
```

`ready_for_replacement()` is True only after `BROKER_RECONCILED`;
`evaluate_replacement()` before reconciliation returns `exit_not_reconciled`.

## Guarantees

- **Exit never gated.** Replacement reacts strictly after the exit has fully filled
  and reconciled; it never gates or delays the mandatory exit (which runs on its own
  independent path).
- **No forced reinvestment.** No eligible candidate → hold cash (`HELD_CASH`);
  `capital_available ≤ 0` → hold cash.
- **No immediate re-entry.** A no-reentry cooldown (default 15 min, injectable)
  starts the moment a stop-out completes; the just-exited symbol — and any symbol
  still within its cooldown window — is excluded from replacement selection even if
  otherwise eligible. Held and pending symbols are also excluded (no pyramiding).
- **Qualification.** Candidates must pass the existing `ReplacementEligibility`
  gates (confirmed seller exhaustion, positive momentum recovery, DI+/BB-mid/VWAP,
  liquidity, regime alignment) and the downstream native financial authorization.

Evidence: `reports/portfolio_replacement_validation.json`.
