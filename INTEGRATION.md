# QQ-Trader → Sustenance Integration

Status of integrating QQ-Trader's trading intelligence + dynamic market scanner
into Sustenance's native engine. Work proceeds incrementally on the branch
`feature/qq-trader-integration`; validated milestones are committed.

The original QQ-Trader repo (`/Users/blkpvnthr/qq-trader`) is the authoritative
source and is **never modified**. All work happens in Sustenance. At runtime the
application has **no dependency on the QQ-Trader repository**.

---

## 1. Audit findings (Phase 0)

### Destination (sustenance) — native engine, verified
- **Native layer is real and compiled** (`_cengine_native` pybind11 module, built
  from `analytics/`, `risk/risk_engine.cpp`, `orderbook/`, etc.). There is **no
  pure-Python risk fallback** by design. It must run under the venv Python
  (3.14) where the `.so` is installed.
- **Single authoritative broker path:** `Strategy → run.CandidateRouter →
  order_manager.OrderManager.submit → native RiskEngine.evaluate →
  cengine.execution.AlpacaExecutionVenue.submit`. `liquidity.py` is a separate
  internal *simulation* order book, not a broker writer.
- Entry point today is `run.py` (no-arg composition root) + `start_engine.py`.
  Lifecycle state machine (`cengine/engine_state.py`) + SIGINT/SIGTERM shutdown +
  kill switch all exist.
- **Universe today is a static tuple** (`DEFAULT_SYMBOLS=("QQQ","SQQQ")` /
  `CENGINE_SYMBOLS`). No universe manager, no scanner.
- `cengine.event_bus.MarketEventBus` already has unused subscriber channels a
  consumer can attach to without touching the order path.
- Live trading is **triple-gated** by explicit `.env` confirmations.
- **Env:** trading reads `ALPACA_API_KEY`/`ALPACA_SECRET_KEY`; market data reads
  `APCA_API_KEY_ID`/`APCA_API_SECRET_KEY` (mirrored at startup);
  `ALPACA_TRADING_BASE_URL`; feed `CENGINE_ALPACA_FEED` (IEX default).

> **Environment fix applied during the audit:** the venv's scikit-build editable
> install pointed `cengine`/`market_data` at a dead path
> (`/Users/blkpvnthr/Desktop/sustainance`), so the engine could not import under
> the venv at all. Reinstalling editable (`pip install -e .`) repointed it here
> and rebuilt the native lib. The full test suite now collects and passes.

### Source (qq-trader) — trading intelligence
- `trade.py` is a 370 KB monolith (`ProductionOrderExecutor` + `AlpacaBroker`).
  It holds the **execution-side** halves of loss-removal, replacement, DI/BB +
  VWAP confirmation, and sizing/risk.
- **The market scanner is a real, self-contained `market_scan/` package** (14
  runtime modules, unit-tested) that was merged into qq-trader. It is already
  architected the way this integration needs: pure ranking/discovery/replacement
  logic with **injectable I/O and zero broker order submission** — it discovers,
  ranks, and publishes `universe.txt` + `live_context_<mode>.json`, and delegates
  all execution.

---

## 2. Source → destination migration map

| Source | Destination | Treatment |
|---|---|---|
| `market_scan/` (14 runtime modules) | `market_scan/` | **Migrated verbatim** (pure logic, injectable I/O) |
| `universe.py` | `universe.py` | Migrated (stdlib-only; canonical load/atomic-write) |
| `order_flow_engine.py` | `order_flow_engine.py` | Migrated (order-flow / seller-exhaustion brain) |
| `amps_v4_regime.py` | `amps_v4_regime.py` | Migrated (regime / directional scoring) |
| root `scanner.py` (heavy: matplotlib/sklearn) | *(not migrated)* | Only `get_sp500_symbols`/`get_nasdaq_symbols` are used, lazily + injectable; a lightweight shim is the planned follow-up. Discovery is fail-safe meanwhile. |
| `universe.txt` | `universe.txt` | **Copied verbatim** — the canonical persisted universe |
| `trade.py` monolith execution halves | `trade.py` (new, thin) + `run.py` engine | **Re-expressed**, not copied: orchestration in `trade.py`, execution stays on the native OrderManager path |
| scanner's host `executor` (`ProductionOrderExecutor`) | `trade.ScannerHost` | New thin credential/state adapter (`.broker`, `.config.paper`) — **no order path** |
| legacy CLI analyzers (`*_analyzer.py`) | carried in `market_scan/` | Standalone CLIs, not on the runtime path |
| qq-trader `tests/test_market_scan_*` (12) | `tests/` | Migrated (100 tests, all pass). `test_market_scan_trade_integration` is source-specific → superseded by `tests/test_trade_entry.py`. |

**Runtime deps added** (`pyproject.toml` `scan` extra): `alpaca-py`, `pandas`,
`requests`, `lxml`.

---

## 3. What this milestone delivers (Phase 1/2 core)

- **`trade.py` — the single unified entry point.** Loads `.env`, resolves mode
  (`--paper` default / `--live` / `--shadow` / `--validate` / `--status`), seeds
  the canonical universe into the engine, auto-starts the scanner daemon + the
  independent protective monitor, and delegates the engine lifecycle to
  `run.main()`. Historical modes exit with a clear not-yet-implemented message.
- **`canonical_universe.CanonicalUniverseManager`** — immutable, versioned
  `UniverseSnapshot`s loaded from `universe.txt`; version bumps only on genuine
  membership/order change; fail-closed on load, protective (keep-last-good) on
  refresh; thread-safe. The scanner remains the sole *file* publisher.
- **`ScannerHost`** — the minimal host adapter the scanner needs, deliberately
  exposing no order path (execution stays on the native OrderManager).
- **`protective_exit.ProtectiveExitCoordinator`** (milestone 2) — the single
  authoritative, scanner-/strategy-independent −0.5% protective-exit path. Runs
  as a periodic (5 s backstop) task **on the engine loop**, reads the
  authoritative in-engine position book + live quote price (no broker REST),
  and **executes** the mandatory close through the engine's single
  `OrderManager.submit → native risk → AlpacaExecutionVenue` path. Duplicate
  closes are prevented via a dedicated `strategy_id` + the existing single-pending
  guard; oversell is prevented by sizing off the projected position (native risk
  re-checks). Wired into `run.MarketEngine` (`__init__` + supervised task in
  `run()`). `trade.py`'s thread `ProtectiveExitMonitor` is now used only in
  `--shadow` (no engine) as detection/recommendation-only.
- **Config** — `.env.example` documents the scanner `MARKET_SCAN_*` vars and the
  spec-alias mapping; scanner runtime artifacts gitignored.

### Validation (all under the venv Python 3.14)
- `python3 trade.py --validate` → **READY** (native module, scanner graph, deps,
  universe, config, credentials all PASS).
- `python3 trade.py --status` → loads universe + detects paper/live; degrades
  cleanly when the broker rejects credentials.
- **Full test suite: 298 passed, 0 failed**, including tests proving a
  coordinator-built close flows through the *real* `OrderManager.submit`, a
  *real* `MarketScanner` OFE fed end-to-end through the single-pipeline bridge,
  DI/BB entry/inverse-exit behaviour, the entry-point consolidation, and
  **genuine on-device Metal kernel execution** (45/45 pipeline-create; indicator +
  covariance GPU/CPU parity; optimizer consuming GPU covariance).

### Milestone 2 (done): protective-exit execution
The mandatory −0.5% protective exit now **executes** through the engine's single
OrderManager path (see `protective_exit.ProtectiveExitCoordinator` above). No
parallel broker writer; duplicate-close and oversell guarded.

### Milestone 5B (in progress): Apple Metal GPU integration
Reuses the existing `metal/*.metal` shaders + `cengine.metallib` (no competing GPU
framework). Validated on **Apple M5 Max** with pyobjc Metal.

- **`cengine/metal_backend.py`** — namespace-aware wrapper over the existing
  `MetalRuntime` (the shaders live in a `gme::` C++ namespace; only the 4 indicator
  kernels are global). Discovers `cengine.metallib` across build dirs; CPU-fallback
  safe (GPU never gates native risk / protective exits). A **45-kernel registry**
  (`MetalKernelRegistration`) maps each kernel → host interface, consumer, CPU
  oracle; coverage is computed from live evidence, not asserted.
- **Host interfaces + CPU parity** (`metal_covariance.py`, `metal_cpu_ref.py`,
  extended `metal_buffers.py` with float32/struct helpers): the 4 indicator kernels
  (via `MetalIndicators`) and the covariance core (`column_means`,
  `covariance_matrix`, `ewma_covariance_matrix`).
- **Consumer**: `metal_portfolio.covariance_from_returns` feeds Metal covariance
  into `RegimeMarkowitzAllocator` end-to-end (test proves the optimizer consumes it).
- **CLI**: `python3 trade.py --validate-metal` (integration report + `reports/*.json`),
  `--benchmark-metal` (GPU vs CPU p50/p95); startup GPU console section added.

**Coverage (real, on-device):** 45/45 kernels compiled + pipeline-callable
(**100% pipeline coverage**); **7 hardware-executed + CPU-parity-validated**
(indicators exact; covariance <1e-6); 21 consumer-connected (declared); **38
blocked** with specific reasons (structured/`ulong` ABIs, two-stage reduction
pipelines, and multi-level depth kernels that need genuine L2 data — IEX is L1).
Benchmarks show covariance GPU wins at scale (32 assets: 0.63ms GPU vs 2.46ms CPU)
while small single-series indicators stay on CPU (dispatch overhead).

Deliverables: `docs/metal_kernel_inventory.md`, `reports/metal_integration_coverage.json`,
`reports/metal_hardware_validation.json`, `reports/metal_cpu_gpu_parity.json`,
`reports/metal_performance_benchmarks.json`. Full-coverage (all 45 consumer-wired)
remains a tracked gate; the blocked kernels are reported, not hidden.

### Milestone 6 (done): entry-point consolidation — run.py / start_engine.py removed
`trade.py` is now the **single** operational entry point and directly owns the
engine. `run.py` (1726 lines) and `start_engine.py` (208 lines) are **deleted**;
nothing imports them (enforced by `tests/test_entrypoint_consolidation.py`).

**Function-level migration report:**

| From | To (now in `trade.py`) | Notes |
|---|---|---|
| `run.MarketEngine` (+ all methods) | `trade.MarketEngine` | moved verbatim |
| `run.CandidateRouter`, `EngineConfig`, policies, protocols | `trade.*` | moved verbatim |
| `run.config_from_env` / `config_from_args` | `trade.config_from_env` / `config_from_args` | moved verbatim |
| `run.load_runtime_env` / `_load_env_file` | `trade.load_runtime_env` / `_load_env_file` | moved verbatim |
| `run.run_engine` | `trade.run_engine` | moved verbatim |
| `run.main` (composition root) | `trade._engine_composition_root` | renamed; called by `run_autonomous` |
| `run` `__main__` guard | removed | `trade.main()` is the sole guard |
| `start_engine.ConfiguredQuantityPolicy` | `trade.ConfiguredQuantityPolicy` | migrated; selected by `CENGINE_ENTRY_QUANTITY` |
| `start_engine.build_config` / `required_env*` / `StrategyPositionExitProvider` | — | superseded by `config_from_env` + `StrategyOwnedExitProvider` (documented) |
| importers: `tests/test_source_liveness`, `test_engine_feed_config`, `test_candidate_router` | updated `run` → `trade` | |
| docs: `README.md`, `market_data/feed.py` docstring | updated `run.py`/`start_engine.py` → `trade.py` | |

Acceptance (`test_entrypoint_consolidation.py`): run.py removed, start_engine.py
removed, `trade` owns MarketEngine/config_from_env/run_engine/..., no module
imports the legacy files, and the unified CLI still dispatches `--validate`.

### Milestone 5 (in progress): canonical strategy integration
- **Shared indicators** (`indicators.py`): Wilder DMI (DI+/DI-/ADX), SMA
  (Bollinger middle band), session VWAP accumulator, momentum — pure, look-ahead-
  free, unit-tested (directional invariants + exact SMA/VWAP).
- **`strategies.DIBBStrategy`** (spec §14/§15): long-only DI/BB entry
  (`DI+ > DI-` and `price > BB-mid`, with configurable mandatory VWAP + momentum
  confirmation that fail **closed**) and inverse exit (`DI- > DI+` and
  `price < BB-mid`). Added to `StrategyName`, `StrategySetConfig`, and
  `build_strategy_coordinator`. **Opt-in** via `CENGINE_DIBB_ENABLED` so an
  unvalidated strategy never silently gains execution eligibility (spec §16);
  entries still flow through the canonical-universe-seeded engine + native risk.
- Remaining M5 (pending): engine-side portfolio-optimizer allocation layer,
  regime-driven sizing, and AMPS/ML-RL *execution* eligibility (the
  `order_flow_engine`, `amps_v4_regime`, and `portfolio_optimizer` modules are
  already migrated and consumed by the scanner's ranking). **Empirical strategy
  validation is deferred to the backtest/replay work (Milestone 7).**

### Milestone 4 (done): single canonical IEX market-data pipeline
The migrated scanner no longer opens its own IEX websocket; the engine owns the
one socket and the scanner consumes the **same** normalized events:
- `scanner_bridge.ScannerEventBridge` — bounded (drop-oldest), thread-safe
  conduit. `MarketEngine._publish_market_event` offers every quote/trade to it;
  a slow scanner can never back-pressure the feed.
- `scanner_bridge.BridgeStreamClient` — drop-in replacement for the scanner's
  `StreamClient` (`run`/`stop`/`apply_subscriptions`/`is_live`) sourced from the
  bridge; converts engine ticks/ns → OFE dollars/epoch-seconds. The scanner keeps
  running on its own thread so its ranking/discovery compute never blocks the
  execution loop.
- `MarketEngine` now **owns** the scanner (starts it in `run()`, stops it in
  `stop()`) via an engine-state-backed host (positions from the lock-guarded
  `PositionBook`; credentials from env; no second broker REST for pending).
- `UniverseSubscriptionSync` folds the scanner's desired **discovery candidates**
  into the single socket's subscription, bounded by `MARKET_SCAN_MAX_SUBSCRIPTIONS`
  (must-have symbols — canonical/positions/pending/refs — are never trimmed).
- `trade.py` no longer starts a separate scanner thread for paper/live (the
  engine owns it); `--shadow` still runs its own scanner (no engine there).

### Milestone 3 (done): dynamic universe resubscription
The engine now keeps its live market-data subscription in step with the
scanner-republished `universe.txt` **without a restart**:
- `AlpacaMarketDataStream.update_symbols(add, remove)` sends live
  subscribe/unsubscribe frames on the open socket (and updates the `Subscription`
  so a reconnect re-subscribes to the new set); it refuses to drop the last symbol.
- `universe_sync.UniverseSubscriptionSync` runs as a periodic engine-loop task:
  required set = canonical universe ∪ open positions ∪ pending orders ∪ refs;
  it diffs against the live subscription and applies the delta. **Held positions
  and pending-order symbols are never unsubscribed** on a universe shrink (spec
  §12). Wired into `run.MarketEngine`; gated by `CENGINE_DYNAMIC_UNIVERSE`
  (an explicit `CENGINE_SYMBOLS` override pins symbols and disables it).

---

## 4. Known limitations / honest status

These are **not yet done** and are the next milestones (do not assume otherwise):

1. **Remaining Milestone 5**: DI/BB entry+inverse-exit is ported (opt-in); still
   pending are the engine-side portfolio-optimizer allocation layer, regime-driven
   sizing, and AMPS/ML-RL execution eligibility.
2. **Backtest / replay / research** modes are stubs. (Milestone 7.)
3. **Full validation** (dynamic equity scaling across account sizes, comparative
   backtests, performance benchmarks) is Milestone 8.
4. **Credentials.** The current `.env` keys are rejected by Alpaca
   (`unauthorized`) — they need rotation before any paper/live run.

No live orders are placed by any part of this work. Protective-exit execution is
validated by unit + real-OrderManager-path tests; it has not been exercised
against a live broker session (requires valid rotated credentials + Mongo).

---

## 5. Run it

```bash
cd /Users/blkpvnthr/sustenance
.venv/bin/python -m pip install -e '.[scan]'   # scanner runtime deps
.venv/bin/python trade.py --validate            # offline readiness
.venv/bin/python trade.py --status              # universe + account snapshot
.venv/bin/python trade.py --paper               # paper autonomous (needs valid creds + Mongo)
```
