# Operations

## Standalone CLI

Run any stage with zero infrastructure:

```bash
python -m cengine.lakehouse <command> [options]
```

| Command | What it does |
|---------|--------------|
| `pipeline` | Run the full batch pipeline (silver→gold→quality→attribution→maintenance→storage→views). `--no-compact` to skip maintenance. Idempotent. |
| `historical` | Bounded, resumable SIP historical ingestion into bronze. |
| `storage` | Print the storage/footprint/health report. |
| `compact` | Run compaction + bronze retention. |
| `quality` | Print the data-quality profile. |
| `warehouse` | Register DuckDB views + print table counts. |

`--root PATH` overrides `CENGINE_LAKEHOUSE_ROOT` for any command.

### Historical ingestion

```bash
python -m cengine.lakehouse historical \
  --symbols AAPL,MSFT,SPY --timeframe 5Min --start 2026-10-01 \
  --feed sip --batch 100 --max-symbols 500
```

- Reuses the existing Alpaca SIP REST fetcher (`make_sip_fetch_fn`) and the engine's
  credential loading (`load_runtime_env`: `ALPACA_API_KEY`/`ALPACA_SECRET_KEY` or
  `APCA_*`). **No new websocket.**
- **No silent SIP→IEX fallback**: `--feed iex` raises; a SIP API error (e.g. 403
  unentitled) is recorded as an explicit batch failure, never downgraded.
- **Resumable**: a per-`(feed, timeframe, symbol)` high-water checkpoint means a re-run
  skips already-ingested bars. (M5 evidence: 15,480 AAPL/MSFT/SPY bars written, re-run
  skipped all 15,480.)
- **Bounded**: `--max-symbols` caps per-run work; symbols are deduped across tiers.

## Live wiring (`trade.py`)

When `CENGINE_LAKEHOUSE != 0` (default on), the engine:

1. Hoists the `DecisionEventSink` to `self._decision_sink` and builds an
   `AnalyticalIngestionSink` + a bounded, drop-oldest market buffer
   (`CENGINE_LAKEHOUSE_MARKET_BUF`, default 200000).
2. **Taps raw observations** in `_publish_market_event` (bar/quote/trade; index factors
   skipped) — buffered, loss-accounted, never raises.
3. Runs a **detached drain task** (`CENGINE_LAKEHOUSE_DRAIN_S`, default 30s) that flushes
   the buffer + drains the decision sink to bronze **via a thread executor** (Parquet IO
   off the event loop). It is not in the `FIRST_COMPLETED` set, so it can neither trigger
   nor block shutdown.
4. Does a **final post-session flush** in `run()`'s `finally`, after the authoritative
   journal is durably closed.
5. Shows a **`LAKEHOUSE` dashboard panel** (feed / written / buffer / dropped /
   quarantined).

**Invariant:** analytical capture is never on the order path, native `RiskEngine`, broker
reconciliation, or the mandatory -0.5% protective-exit task. A failure disables capture
only; trading is unaffected.

## Maintenance (`maintenance.py`)

- **Compaction** merges many small part files per partition into one, deduping by
  `event_id`. Safe-by-construction: the compacted part is written atomically first, then
  originals are removed. Triggered when a partition has ≥ `compaction_min_files` (16)
  parts.
- **Retention** (`prune_bronze`) deletes bronze `trade_date=` partitions older than
  `retention_days` (default 0 = keep forever), logging every prune to
  `_manifests/retention_prune.jsonl`. It never touches silver/gold or the journal.

## Storage monitoring (`storage_monitor.py`)

`StorageMonitor.report()` returns a read-only footprint: free disk, lakehouse size,
per-dataset file/size counts, compaction candidates, a health verdict
(`OK`/`WARN`/`CRITICAL`), and recommendations (external-SSD hint, compaction, retention).
`CRITICAL` when `accepts_writes()` is false.

## Typical daily flow

```bash
# during the session: live capture happens automatically (trade.py)
# after the close:
python -m cengine.lakehouse historical --symbols-file universe.txt --start 2026-11-01
python -m cengine.lakehouse pipeline
python -m cengine.lakehouse storage
```
