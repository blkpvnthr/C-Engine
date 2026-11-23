# C-Engine — M7B Historical Data Integrity Certification

_Domain: `data_integrity`. Verdict: **CERTIFIED**. Scope: the local Parquet + DuckDB lakehouse backing all M7 certification runs._

> **Provenance.** The checks below certify that the historical bar data is genuine, point-in-time, and leak-free. They do **not** certify any economic edge. All economic results elsewhere in the M7 certification are historical-replay simulations on these genuine bars — not live performance, not a profitability claim. No live orders were placed.

---

## Why this domain exists

Every downstream verdict — strategy economics, sealed-holdout scoring, recovery reproduction, stress windows — is only as trustworthy as the bars it runs on. Per the falsification-first mandate, the data layer is the first thing that must survive scrutiny: a single look-ahead leak or a mis-scaled price would silently manufacture a fake edge. M7B therefore treats the dataset as guilty until proven clean, and runs eight independent checks before any economic question is allowed to proceed.

If data integrity fails, nothing above it can be believed. It passed.

---

## Dataset inventory (at certification time)

| Property | Value |
|---|---|
| Source | Alpaca SIP (genuine market bars) |
| Storage | Local-first Parquet + DuckDB lakehouse |
| Total bars | 12.2M |
| 1Min bars | 6.76M |
| 1Min symbols | 5 canonical — AAPL, MSFT, NVDA, SPY, QQQ |
| 1Min span | 2020-01-01 .. 2026-11-21 |
| 5Min bars | 5.45M |
| 5Min symbols | 31 |
| Price encoding | Integer ticks, `price_scale = 10000` |
| Distinct series | 36 (5 × 1Min + 31 × 5Min) |

Series = one (symbol, resolution) pair. The 36-series figure is the unit the coverage and structural checks are applied across.

---

## The eight checks

Each check is independent and mechanical. A series must pass **all** eight to be certified; the domain is certified only when every series passes.

### 1. Inventory
**Question:** Is the dataset present, complete, and exactly as declared (symbol counts, resolutions, bar counts, span)?
**Result:** 12.2M bars confirmed — 6.76M × 5 symbols at 1Min over 2020-01-01..2026-11-21, and 5.45M across 31 symbols at 5Min. Inventory matches the declared manifest.
**Status:** CERTIFIED.

### 2. Duplicates
**Question:** Does any (series, timestamp) key appear more than once?
**Result:** **0 duplicate bars** across all 36 series. Every timestamp within a series is unique.
**Status:** CERTIFIED.

### 3. Monotonicity
**Question:** Are timestamps strictly increasing within each series (no backward jumps, no repeats)?
**Result:** **Strictly monotonic** timestamps on every series. Combined with check 2, this guarantees a well-ordered, gap-auditable time index.
**Status:** CERTIFIED.

### 4. OHLC validity
**Question:** Are the bar invariants respected — `low <= open,close <= high`, `high >= low`, and `volume >= 0`?
**Result:** **Valid OHLC** on all bars, with **non-negative volume** throughout. No inverted ranges, no negative volumes.
**Status:** CERTIFIED.

### 5. Point-in-time / no look-ahead
**Question:** Could any bar be consumed before it was actually available? (The single most dangerous failure mode for a backtest.)
**Result:** **No look-ahead** — for every record `availability_ts >= event_ts`. Nothing is readable before it existed, so a strategy replaying these bars cannot peek at the future.
**Status:** CERTIFIED.

### 6. Price scale
**Question:** Is price encoding uniform and correctly scaled across the lakehouse?
**Result:** **Uniform `price_scale = 10000`** everywhere. Prices are stored as integer ticks (1 tick = 1/10000 of a unit), eliminating float drift and preventing a per-series scale mismatch from distorting returns.
**Status:** CERTIFIED.

### 7. Session coverage
**Question:** Does each series cover every NYSE trading day in its span, with no missing sessions?
**Result:** **100% NYSE trading-day coverage** on all 36 series. No session holes relative to the exchange calendar.
**Status:** CERTIFIED.

### 8. 1m / 5m alignment (reconciliation)
**Question:** When 1Min bars are aggregated into 5-minute buckets, do they reconcile with the stored 5Min bars?
**Result:** **99.99% reconciliation** on the AAPL spot-check across **8,419 buckets**. The two resolutions are mutually consistent, confirming the aggregation path and the two feeds agree.
**Status:** CERTIFIED.

---

## Integrity numbers

| Check | Metric | Result | Status |
|---|---|---|---|
| 1. Inventory | Total bars | 12.2M (6.76M 1Min / 5.45M 5Min) | CERTIFIED |
| 1. Inventory | Series | 36 (5 × 1Min, 31 × 5Min) | CERTIFIED |
| 2. Duplicates | Duplicate bars | 0 | CERTIFIED |
| 3. Monotonicity | Timestamp ordering | Strictly monotonic (all series) | CERTIFIED |
| 4. OHLC validity | Bar invariants + volume | Valid OHLC, volume >= 0 | CERTIFIED |
| 5. Point-in-time | `availability_ts >= event_ts` | Holds on all records (no look-ahead) | CERTIFIED |
| 6. Price scale | `price_scale` | 10000, uniform | CERTIFIED |
| 7. Session coverage | NYSE trading-day coverage | 100% on all 36 series | CERTIFIED |
| 8. 1m/5m alignment | Bucket reconciliation | 99.99% (AAPL, 8,419 buckets) | CERTIFIED |

**Domain verdict: `data_integrity` = CERTIFIED.**

---

## How to reproduce

```
python -m cengine.certification data
```

The data-integrity suite runs the eight checks above and emits the per-series results and the domain verdict. It is also executed as part of:

```
python -m cengine.certification full
python -m cengine.certification report
```

---

## What this does and does not establish

**Establishes:**
- The bars are genuine Alpaca SIP data, not synthetic or placeholder.
- The time index is clean: unique, strictly ordered, fully covering the exchange calendar.
- Prices are uniformly scaled and OHLC-valid, so computed returns are trustworthy.
- There is **no look-ahead**: availability never precedes the event, so replays are causally honest.
- 1Min and 5Min feeds are internally consistent.

**Does NOT establish:**
- Any trading edge, expectancy, or profitability. Clean data is a precondition for honest research, not evidence of an edge.
- Live or paper readiness. Those verdicts are reported in their own domains and are **NOT_CERTIFIED** / not live-eligible.
- True market-impact capacity, which remains INSUFFICIENT_DATA and is not fabricated.

Clean, leak-free data is exactly why the economic findings downstream can be stated as honest negatives rather than being dismissed as data artifacts.

---

> **Provenance.** Results are historical-replay simulations on genuine bars, not live performance, not a profitability claim; no live orders were placed.
