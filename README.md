# C-Engine

C-Engine is a paper-first trading engine joining normalized Alpaca SIP market
data, HDF5 persistence, bar strategies, authoritative portfolio state, native
C++ risk, an internal limit-order book, and an Alpaca execution adapter.

Live execution is off by default. The standard endpoint is
`https://paper-api.alpaca.markets`; the live endpoint is rejected unless two
exact, separate confirmation variables are present. Tests never submit orders.

## Architecture

```text
Alpaca SIP -> normalization -> independent event bus
  |-> HDF5                 |-> order book / market depth
  |-> strategies -> sizing |-> liquidity / volatility / stat-arb
  |                        `-> optional Metal analytics
  `-> OrderManager -> native RiskEngine -> Alpaca paper venue
       ^                                      |
       `---- fills -> PositionBook <----------'
```

The engine does not invent quantities, strategy thresholds, market prices,
fills, or account state. Entry sizing and exit quantities are injected by the
application. Risk rejects when authoritative quote/account state is absent.

## macOS build and test

Requirements: Python 3.11+, CMake 3.24+, a C++20 compiler, and (for Metal)
full Xcode command-line Metal tools.

```bash
python3 -m venv .venv
.venv/bin/pip install -e '.[dev]'
.venv/bin/pytest -q
.venv/bin/ruff check .
cmake -S . -B build -DCENGINE_BUILD_PYTHON=OFF -DCENGINE_COMPILE_METAL=OFF
cmake --build build -j
```

Enable kernel compilation only on a supported Apple Silicon Mac with the
Metal compiler installed:

```bash
cmake -S . -B build-metal -DCENGINE_COMPILE_METAL=ON
cmake --build build-metal -j
```

`MetalRuntime` fails explicitly on non-Apple-Silicon hosts, missing PyObjC
Metal bindings, missing devices, libraries, functions, or command errors.

## Paper configuration

Copy `.env.example` to `.env` and supply paper credentials locally. `.env` is
ignored. Export the values into the process environment before startup.
Alpaca SIP access and the appropriate market-data entitlement are external
prerequisites.

The CLI intentionally refuses to invent a sizing policy. Compose `run_engine`
with explicit `QuantityPolicy` and `ExitQuantityProvider` implementations.
`ENGINE_NATIVE_ADAPTER_MODULE=native_adapters` selects the native risk and
paper execution adapters.

Engine composition additionally requires an explicit `ExecutionPolicy` and
the following operational settings; none receive trading defaults:

```text
--journal-path ./state/audit.jsonl
--max-feed-age-ns <provider-appropriate limit>
--max-bar-gap-ns <configured bar interval tolerance>
--reconciliation-interval-seconds <broker polling interval>
--session-timezone America/New_York
--session-open-minute 570
--session-close-minute 960
--session-weekdays 0,1,2,3,4
```

Startup fetches Alpaca account, position, and open-order state. Unknown broker
orders, missing internal positions, mismatched venue identifiers, stale
snapshots, reversed cumulative fills, or runtime reconciliation errors trip
the engine kill switch and prevent further order creation.

Every reconciliation cycle writes checksummed `portfolio_metrics` records to
the audit journal, including cash, equity, buying power, realized P/L, gross
and net exposure, pending reserved notional, open-order and position counts,
and high-water-mark drawdown.

## Replay

`cengine.replay.replay()` republishes already-normalized events in stable input
order without sleeping or synthesizing time. It rejects non-monotonic symbol
sequences. Feed events decoded from an HDF5 day file to the same event bus used
by live ingestion for deterministic strategy and analytics validation.

## Live execution gate

Live trading requires all three settings below; any missing or altered value
blocks construction of the execution adapter:

```text
ALPACA_TRADING_BASE_URL=https://api.alpaca.markets
CENGINE_ENABLE_LIVE=YES_I_ACCEPT_LIVE_TRADING_RISK
CENGINE_LIVE_CONFIRMATION=LIVE_ORDERS_MAY_LOSE_MONEY
CENGINE_LIVE_RECONCILIATION_CERTIFIED=PAPER_REPLAY_AND_BROKER_STATE_AGREE
```

Keep live values out of shell profiles and deployment defaults. Paper remains
the safe default.

## Advisory portfolio research

Install optional research dependencies with `pip install -e '.[research]'`.
`RegimeMarkowitzAllocator` uses CVXPY for constrained regime-weighted
mean/variance advice, while `NextSessionTideModel` supports explicit KNN or SVM
training. `PortfolioRiskManager` validates resulting weights and predicted
volatility. These components return data only: they have no `OrderManager`,
execution-venue, or risk-approval capability and are not connected to runtime
execution.

## Credential notice

An `.env` containing credential-like values existed in Git history. It has
been removed from the current tree, but history is immutable; rotate those
keys even if they were paper-only.
