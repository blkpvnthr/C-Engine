# Instrument Exposure Registry (§5–8)

Classifies the discovery universe by **economic exposure**, not ticker name, so the
portfolio/risk layers can reason about leverage-adjusted and inverse exposure.

Module: `cengine/exposure_registry.py`. Tests: `tests/test_exposure_registry.py`.
Evidence: `reports/instrument_exposure_validation.json`,
`reports/inverse_etf_universe_validation.json`.

## Classification

`InstrumentType`: COMMON_EQUITY, BROAD_MARKET_ETF, SECTOR_ETF, INDUSTRY_ETF,
THEMATIC_ETF, LEVERAGED_LONG_ETF, INVERSE_ETF, LEVERAGED_INVERSE_ETF,
REFERENCE_ONLY, UNCLASSIFIED.

Each instrument carries: `underlying_benchmark`, `underlying_exposure_group`
(clustering key, e.g. `NASDAQ100`), `exposure_direction` (LONG/INVERSE/NEUTRAL),
signed `target_daily_leverage` (negative for inverse), `daily_reset`, `inverse`,
`leveraged`, `metadata_source`, plus reconciled `tradable` / `asset_status`.

The 56-symbol seed table is curated from documented issuer **fund objectives**
(directive §5 lists + categories), **never inferred from ticker naming** (§6). All
56 reconcile as present + tradable against the live Alpaca asset universe.

## Exposure accounting (§8)

`effective_exposure(market_value, target_daily_leverage)` → signed leverage-adjusted
underlying exposure (negative for inverse). It is a **one-day-reset estimate**, not
a guaranteed realized or multi-day exposure. `detect_opposing()` flags economically
redundant holdings (same exposure group, opposite signed leverage, e.g. TQQQ+SQQQ);
`detect_overlap()` groups same-benchmark holdings for concentration accounting.

## Safety invariant (§7)

`execution_authorized` defaults **True only for common equity and unlevered (1×)
long ETFs** (25 of 56 seeds). Every leveraged and inverse product defaults **False**
and requires explicit, user-authorized enablement (`registry.authorize(sym)`). An
unclassified or incorrectly classified geared product therefore can never reach
execution automatically.

## Status

Registry + accounting + safety gate are implemented and tested, reconciled against
real Alpaca data. Wiring into portfolio accounting, risk concentration limits, and
the strategy/execution layers is tracked under later milestones. Production should
reconcile leverage multipliers and benchmarks against issuer prospectuses and
discover additional Alpaca-supported ETFs dynamically (§6).
