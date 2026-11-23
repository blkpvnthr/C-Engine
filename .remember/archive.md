# Archive

## Week of 2026-09-15
Investigated trader component reuse; mandated standalone direction-engine (574-file codebase: direction engine, adapters, exec layer on Alpaca). Selected Alpha Vantage API. Direction-engine Phase 1 complete: 27 modules, 53✓ tests, hierarchical multi-TF fusion (360D/1H + 180D/5M + L2), provider-neutral, defense-in-depth security. E2E paper engine live: 1575 quotes + 285 trades/12s, zero drops, SIP reconnect fixed, journaling/HDF5 resolved; metrics flowing.

## Week of 2026-09-22
Wired paper engine live: decision/metric streaming via MarketEngine hooks (bar_observer, on_decision, on_metrics, on_order_event), DecisionReport added, vol-scaled sizing/exit. Verified Alpaca equity metrics integration (account sync, 15s reconciliation) and intraday P&L tracking from last_equity. Trading signals flowing; live test with SQQQ breakout order; 29 tests ✓.

## Week of 2026-10-05
Fixed paused Atlas cluster (sync Mongo blocking journal); mapped journal/engine/sizing paths. Refactored run.py with MarketEngine observer hooks (bar_observer, decision, metrics, order events); built component-root with vol-scaled sizing, ATRTracker, DecisionReport. Live decision/metrics streaming validated end-to-end.
```