# Sustenance — Master Implementation Spec (living status)

This tracks the **Native Metal Architectural Correction**: making Apple Metal the
operational market-wide computational engine for opportunity discovery, backed by a
native C++ / Objective-C++ core, with Python orchestrating (not screening). It is a
status mirror of the user's correction directive — the authoritative requirements
live there; this file records **what is done, how it was verified, and what remains**.

Single entry point (unchanged): `python3 trade.py`. Canonical interpreter: **pyenv
Python 3.11.8**. Free Alpaca IEX only; ≤30 WebSocket subscriptions; no live orders
for testing; mandatory −0.5% native protection stays independent of GPU/Metal.

## Baseline the correction fixes (measured in the audit)
- Metal PyObjC-hosted, synchronous one-shot dispatch, **no pipeline cache**.
- **7 of 45** kernels wired (15.56%); **zero GPU on the live screening path**.
- Live screening = Python per-symbol loops, capped ~30 symbols; full-universe
  screening only in an offline yfinance EOD scanner.
- Universe = 31-symbol file; NASDAQ/SP500 fetchers broken; DJIA missing; no
  batched IEX REST sweep on the runtime path; no 1-second dashboard;
  `--backtest/--replay/--research` stubs.

## Phase status
| Phase | Scope | Status |
|---|---|---|
| **A** | Native Obj-C++ Metal runtime: device/queue/library, **pipeline cache**, **persistent buffers**, async + completion handlers, telemetry, CPU fallback, native kernel registry, batched pybind; 7 wired kernels native + CPU parity | **COMPLETE** |
| B | New discovery kernels (DMI/DI/ADX/SMA20/Bollinger/VWAP/ATR/rel-vol/1m-momentum/accel/entry/exit/compaction/scoring/ranking/breadth); live IEX→native→GPU→scanner pathway on the ~31-symbol set | NEXT |
| C | Full NASDAQ∪SP500∪DJIA master universe + daily refresh; batched IEX REST sweeps; per-symbol native state; market-wide GPU screening/ranking; dynamic 30-sym WS promotion; scale to full universe | PENDING |
| D | GPU covariance/EWMA/correlation/vol → optimizer → native sizing → native risk → OrderManager; portfolio replacement | PENDING |
| E | Remaining 38 kernels (reductions/MC/statarb; L1-imbalance research; L2 blocked) with legitimate consumers | PENDING |
| F | 1-second dashboard; `--benchmark-discovery`; backtest/replay/research; fault injection; benchmark ladder; OOS; certification | PENDING |

## Phase A evidence (Apple M5 Max, pyenv 3.11.8)
- 45/45 kernels pipeline-callable on device; 7 hardware-executed + parity-validated
  (indicators exact; covariance/EWMA `< 1e-6`).
- Pipeline cache + persistent buffer reuse confirmed via telemetry.
- Native-vs-legacy-PyObjC numerical parity proven (migration regression gate).
- Tests: **313 passed** (298 baseline + 15 new); single entry point `--validate` READY.
- Reports: `reports/architecture_correction.json`, `metal_kernel_inventory.json`,
  `metal_kernel_coverage.json`, `metal_hardware_validation.json`,
  `metal_cpu_parity.json`, `operational_consumer_coverage.json`,
  `build_environment.json`, `native_library_identity.json`,
  `shader_library_identity.json`, `implementation_progress.json`.

## Acceptance (not yet met — tracked for later phases)
Genuine IEX observations reaching hardware-executed kernels whose results the
scanner consumes; discovery evaluating far more than 30 symbols; all market groups
represented; operational covariance influencing sizing; Python no longer the primary
market-wide screener; dashboard metrics proving live GPU use. **No profitability or
live-readiness claim is made from compilation or paper execution alone.**

See `INTEGRATION.md` §6 for the detailed Phase A changelog and `docs/` for
architecture notes.
