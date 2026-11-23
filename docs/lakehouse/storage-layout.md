# Storage Layout & Configuration

## On-disk layout

```
<root>/                        # default: data/lakehouse/ (CENGINE_LAKEHOUSE_ROOT)
├── bronze/
│   └── <dataset>/feed=<feed>/trade_date=YYYY-MM-DD/part-<hash>.parquet
├── silver/
│   └── <dataset>/feed=<feed>/trade_date=YYYY-MM-DD/part-<hash>.parquet
├── gold/
│   └── <dataset>/.../part-<hash>.parquet
├── quarantine/
│   └── <dataset>/reason=<reason>/part-<hash>.parquet
├── _manifests/               # append-only JSONL, one file per <layer>.<dataset>
├── _checkpoints/             # silver / historical high-water marks, pipeline runs
├── _model_registry/          # governance models.jsonl
└── cengine.duckdb            # warehouse catalog (views only; no bulk copy)
```

Partitioning is `feed=<feed>/trade_date=YYYY-MM-DD` (derived in `_default_partition`
from `event_timestamp_ns` + `source_feed`). Partition keys are written in sorted order,
so `feed=` precedes `trade_date=` on disk.

## Atomic, idempotent publication (`parquet_writer.py`)

- **Atomic**: write to a `*.parquet.tmp` in the destination dir, `flush` + `os.fsync`,
  then `os.replace` into the final name. A reader never sees a partial file.
- **Idempotent**: the part file is named `part-<sha256(content)[:16]>.parquet`. Writing
  identical bytes again is a no-op (`reason="idempotent"`) — safe across restarts and
  re-ingestion.
- **Recoverable**: `recover_incomplete()` deletes orphan `*.parquet.tmp` left by an
  interrupted write (run at sink startup).
- **Manifested**: every published part appends `{path, rows, checksum, partition,
  written_ns}` to `_manifests/<layer>.<dataset>.jsonl`.
- **Quarantine**: invalid records are persisted under `quarantine/…/reason=…` with the
  reason, never silently discarded.

Compression is zstd by default (snappy fallback if the build lacks zstd).

## Configuration (`config.py`)

All knobs are env-overridable so an external SSD or different policy needs **no code
change**. `LakehouseConfig.from_env()`:

| Env var | Field | Default |
|---------|-------|---------|
| `CENGINE_LAKEHOUSE_ROOT` | `root` | `data/lakehouse/` (under cwd) |
| `CENGINE_LAKEHOUSE_COMPRESSION` | `compression` | `zstd` |
| `CENGINE_LAKEHOUSE_ZSTD_LEVEL` | `compression_level` | `3` |
| `CENGINE_LAKEHOUSE_ROW_GROUP` | `row_group_size` | `128000` |
| `CENGINE_LAKEHOUSE_BATCH` | `write_batch_size` | `10000` |
| `CENGINE_LAKEHOUSE_MAX_GIB` | `max_storage_gib` | `0` (unlimited) |
| `CENGINE_LAKEHOUSE_MIN_FREE_GIB` | `min_free_disk_gib` | `5.0` |
| `CENGINE_LAKEHOUSE_RETENTION_DAYS` | `retention_days` | `0` (keep forever) |

Live-wiring gates (used by `trade.py`): `CENGINE_LAKEHOUSE` (default on),
`CENGINE_LAKEHOUSE_MARKET_BUF` (buffer size, default 200000),
`CENGINE_LAKEHOUSE_DRAIN_S` (drain interval, default 30s).

### External SSD
```bash
export CENGINE_LAKEHOUSE_ROOT=/Volumes/FastSSD/cengine-lakehouse
```

## Disk safety (never affects trading)

`accepts_writes()` returns `(ok, reason)` and **fails closed** below `min_free_disk_gib`
or at/over `max_storage_gib`. It is called only by the analytical writers; when it
returns `False`, analytical writes stop gracefully and the loss is accounted. This gate
is never on the order or protective-exit path.

## Git hygiene

`.gitignore` excludes `data/lakehouse/`, `*.parquet`, `*.duckdb`, `*.duckdb.wal`,
`models/`, and `data/model_registry/`. Datasets, financial records, credentials, and
models are never committed. The JSON reports under `reports/` and these docs are the
only lakehouse artifacts tracked in git.
