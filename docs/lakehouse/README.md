# C-Engine Local Adaptive ML/RL Data Platform (Milestone 5)

An entirely **local** lakehouse — Parquet + PyArrow + DuckDB, no cloud, no MongoDB
Atlas, no Supabase — that continuously accumulates market observations, engine
decisions, execution experience, and realized outcomes so a **future** milestone (M6)
can train supervised / contextual-bandit / offline-RL models on reproducible,
point-in-time-correct datasets.

## What it is (and is not)

- **Analytical only.** The authoritative transactional financial record stays in the
  separate audit journal (`cengine/journal.py`). Parquet is **never** the
  order-management database.
- **Never on the critical path.** Capture is nonblocking, bounded, and loss-accounted.
  It can never block or delay the order path, the native `RiskEngine`, broker
  reconciliation, or the mandatory **-0.5% protective exit**.
- **Point-in-time honest.** Every record distinguishes *market-event time* from
  *availability time*, so training data can be reconstructed exactly as the engine saw
  it — no look-ahead, no leakage, no survivorship bias.
- **Does not train.** M5 builds and governs the datasets; it records model/dataset
  metadata but trains and promotes nothing. Training is M6.

## The three layers

| Layer | Contents | Module |
|-------|----------|--------|
| **Bronze** | Immutable raw source observations + engine/execution events, wrapped in the canonical envelope | `bronze.py`, `envelope.py`, `parquet_writer.py` |
| **Silver** | Validated, normalized, typed, deduplicated, versioned datasets | `silver.py` |
| **Gold** | Point-in-time features, FIFO position outcomes, supervised + offline-RL datasets | `gold.py`, `feature_store.py` |

Everything is queryable through a local DuckDB warehouse of views over the Parquet
files (`warehouse.py`).

## Stack

Parquet (zstd) · PyArrow · DuckDB · `DEFAULT_PRICE_SCALE = 10000` ticks end-to-end.

## Documentation map

1. [architecture.md](architecture.md) — the three layers and data flow
2. [storage-layout.md](storage-layout.md) — on-disk layout, partitioning, config
3. [event-envelope.md](event-envelope.md) — the canonical versioned envelope
4. [point-in-time.md](point-in-time.md) — leakage/look-ahead/survivorship prevention
5. [datasets.md](datasets.md) — the bronze/silver/gold dataset + view catalog
6. [supervised-and-rl.md](supervised-and-rl.md) — the learning datasets
7. [operations.md](operations.md) — CLI, live wiring, maintenance
8. [data-quality.md](data-quality.md) — quality framework + attribution
9. [governance.md](governance.md) — model governance metadata
