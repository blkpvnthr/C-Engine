# Sustainance


This C-Engine is a paper-first trading engine joining normalized **Alpaca IEX**
market data, HDF5 persistence, bar strategies, authoritative portfolio state,
native C++ risk, an internal limit-order book, and an Alpaca execution adapter.

The real-time market-data feed defaults to **Alpaca IEX**
(`wss://stream.data.alpaca.markets/v2/iex`). SIP remains supported for entitled
accounts but is selected explicitly and is never used as an automatic fallback.

Live execution is off by default. The standard endpoint is
`https://paper-api.alpaca.markets`; the live endpoint is rejected unless three
exact, separate confirmation variables are present. Selecting the market-data
feed does **not** change paper vs. live trading — they are independent axes.
Tests never submit orders.

## Architecture

```text
Alpaca IEX (default feed; SIP opt-in)
  -> Normalization (QuoteEvent / TradeEvent / BarEvent)
  -> MarketEventBus (independent fan-out)
       |-> HDF5 archive (provenance: provider + feed)
       |-> order book / market depth   (top-of-book only; no fabricated L2)
       |-> liquidity / volatility / stat-arb
       |-> optional Metal analytics
       `-> Strategies -> sizing
             -> OrderManager -> native C++ RiskEngine -> Alpaca execution venue
                  ^                                           |
                  `---- fills -> PositionBook <---------------'
```

The engine does not invent quantities, strategy thresholds, market prices,
fills, or account state. Entry sizing and exit quantities are injected by the
application. Risk rejects when authoritative quote/account state is absent.

## Market-data feed (IEX default)

A single authoritative resolver (`market_data/feed.py`, used by `run.py`,
`start_engine.py`, and programmatic construction) decides the feed, the live
WebSocket endpoint, and the archive provenance label so they can never disagree.

* **Default:** IEX. An omitted feed resolves to IEX, never SIP.
* **Provenance:** every daily HDF5 file records both `provider` (e.g.
  `alpaca-iex`) and `feed` (e.g. `iex`); replay refuses to combine files from
  incompatible feeds (see `assert_feed_consistency`).
* **No automatic fallback:** a failed IEX connection is surfaced, not silently
  downgraded to SIP.
* **Startup verification:** the feed is derived from the endpoint actually
  connected to and verified against the configured feed before trading; a
  mismatch fails closed.

### IEX limitations (read before trusting a signal)

IEX is a **single exchange**, not consolidated SIP data:

* IEX quotes are IEX's own best bid/offer (exchange top-of-book), **not** the
  National Best Bid and Offer (NBBO).
* IEX quote sizes are IEX's resting size, **not** consolidated depth or
  market-wide buy/sell pressure. Quote-size imbalance computed from IEX is
  single-exchange and must not be presented as consolidated order-book
  imbalance.
* IEX trade prints are IEX volume, **not** consolidated market volume.
* No Alpaca feed provides exchange-by-exchange Level-II depth; the engine never
  fabricates L2 depth from a top-of-book quote.

Any signal that genuinely requires consolidated market coverage (full-market
volume, NBBO, market-wide imbalance) is ineligible on IEX and must be driven by
an entitled consolidated source (SIP) or an explicit provider-depth feed.
Thresholds originally calibrated on SIP data should be re-validated on IEX
before deployment.

### Switching to SIP (entitled accounts only)

SIP requires an Alpaca market-data subscription with SIP entitlement. Select it
explicitly (data feed only — unrelated to paper vs. live execution):

```text
CENGINE_ALPACA_FEED=sip
# provenance defaults to alpaca-sip; override with CENGINE_ARCHIVE_PROVIDER_LABEL
```

Or pass a validated explicit endpoint:

```text
CENGINE_ALPACA_URL=wss://stream.data.alpaca.markets/v2/sip
```

The URL's scheme, host, and feed path are validated and the feed name is
derived from the path, so provenance always matches the endpoint. A feed the
account is not entitled to is rejected by Alpaca at authentication; the engine
does not fall back to another feed.

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
ignored. Export the values into the process environment before startup. The
default IEX feed works with a standard Alpaca market-data subscription; SIP is
an opt-in upgrade requiring the SIP entitlement (see "Switching to SIP").

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

## Daily NoSQL journal storage

Install MongoDB support with `pip install -e '.[nosql]'`. Configure the URI as
an environment variable so credentials do not enter shell history:

```text
CENGINE_MONGODB_URI=mongodb://127.0.0.1:27017/?replicaSet=rs0
CENGINE_MONGODB_DATABASE=cengine
```

Also supply `--mongodb-timeout-ms` explicitly. Each verified journal record is
written to a timezone-aware daily collection such as `journal_20260921`.
Collections have a unique global-sequence index and a timestamp/kind index.
Repeated replay is idempotent; reuse of a sequence with another checksum fails
closed. The local checksummed JSONL file remains the write-ahead and recovery
source. At startup it is backfilled into MongoDB before trading proceeds.

For production, use a MongoDB replica set with authentication, encryption in
transit, backups, and majority write concern configured in the connection URI.

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

`.env.example` is a template and must contain only empty placeholders. Earlier
commits of `.env.example` (and an `.env`) contained credential-shaped values —
an Alpaca paper API key/secret and a MongoDB Atlas connection URI. The current
tree has been scrubbed to empty placeholders, but **Git history is immutable**:
rotate the previously committed Alpaca key/secret and the Atlas database
password, even though they were paper-only. History is intentionally **not**
rewritten here; perform rotation (and any history purge) through your own
operational process.
