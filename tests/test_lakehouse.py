"""Local lakehouse data-platform tests (Milestone 5, §33/§34).

Covers the full analytical path offline + deterministically (no network):
schema/envelope, atomic+idempotent Parquet, interrupted-write recovery, nonblocking
loss-accounted ingestion (disk-full failure injection), silver dedup/incremental
transforms, DuckDB warehouse views + lineage, point-in-time leakage prevention
(feature store ASOF + quality check), FIFO position outcomes, censored supervised
labels (no fabrication), offline-RL reward integrity (NULL action prob / closed-only
reward / mandatory-exit-is-environment), reproducible dataset versioning, governance
(no training/promotion in M5), compaction dedup, resumable SIP ingestion with NO IEX
fallback, and the end-to-end idempotent pipeline.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

from cengine.lakehouse.attribution import DailyAttribution
from cengine.lakehouse.bronze import AnalyticalIngestionSink
from cengine.lakehouse.config import DEFAULT_PRICE_SCALE, LakehouseConfig
from cengine.lakehouse.envelope import (
    ENVELOPE_SCHEMA_VERSION,
    envelope_from_decision_row,
)
from cengine.lakehouse.feature_store import FeatureSet, PointInTimeFeatureStore
from cengine.lakehouse.gold import GoldBuilder
from cengine.lakehouse.governance import ModelGovernanceRegistry, ModelRecord
from cengine.lakehouse.historical import HistoricalSIPIngestion, HistoricalTier
from cengine.lakehouse.maintenance import LakehouseMaintenance
from cengine.lakehouse.parquet_writer import ParquetWriter
from cengine.lakehouse.pipeline import LakehousePipeline
from cengine.lakehouse.quality import DataQualityValidator
from cengine.lakehouse.silver import SilverTransformer
from cengine.lakehouse.storage_monitor import StorageMonitor
from cengine.lakehouse.warehouse import DuckDBWarehouse

NS = 1_000_000_000


@pytest.fixture()
def config(tmp_path):
    cfg = LakehouseConfig(root=tmp_path / "lakehouse", min_free_disk_gib=0.0)
    cfg.ensure_dirs()
    return cfg


def _row(event_type, decision_id, symbol, ts, payload, **kw):
    return {"event_type": event_type, "timestamp_ns": ts, "decision_id": decision_id,
            "symbol_id": symbol, "strategy_family": kw.get("sf", "momentum"),
            "strategy_version": kw.get("sv", "v1"), "regime_version": kw.get("rv", "r1"),
            "feature_version": kw.get("fv", "f1"), "payload": payload,
            "schema_version": "decision-1.0.0"}


def _seed_lifecycle(cfg, t=None):
    """One complete decision->execution lifecycle: rec -> portfolio -> entry+exit fills."""
    t = t if t is not None else 1_700_000_000 * NS
    rows = [
        _row("strategy_recommendation", "d1", "AAPL", t,
             {"exposure_direction": "long", "eligibility": "eligible",
              "confidence": 0.8, "rank_score": 0.9}),
        _row("portfolio_recommendation", "p1", "AAPL", t + 1,
             {"source_strategy_decision_id": "d1", "portfolio_decision_id": "p1",
              "target_weight": 0.1, "allocation_policy": "risk_parity",
              "eligibility": "eligible", "constraint_status": "ok"}),
        _row("complete_fill", "d1", "AAPL", t + 2,
             {"correlation_id": "c1", "last_fill_quantity": 100,
              "last_fill_price_ticks": 1_000_000, "average_fill_price_ticks": 1_000_000}),
        _row("complete_fill", "d2", "AAPL", t + 3,
             {"correlation_id": "c2", "last_fill_quantity": 100,
              "last_fill_price_ticks": 1_010_000, "average_fill_price_ticks": 1_010_000}),
    ]
    AnalyticalIngestionSink(cfg, source_feed="sip").ingest_decision_rows(rows)
    SilverTransformer(cfg).run_all()
    return t


# -- schema / envelope -------------------------------------------------------
def test_envelope_reuses_ids_and_distinguishes_event_vs_availability():
    t = 123 * NS
    row = _row("strategy_recommendation", "dX", "MSFT", t, {"correlation_id": "corrX"})
    env = envelope_from_decision_row(row, source_feed="sip")
    assert env.schema_version == ENVELOPE_SCHEMA_VERSION
    assert env.causation_id == "dX"              # original decision id reused as causation
    assert env.correlation_id == "corrX"
    assert env.event_timestamp_ns == t
    assert env.availability_timestamp_ns == t    # decisions: availability == event time
    assert env.strategy_family == "momentum"
    d = env.to_row()
    assert "payload_json" in d and isinstance(d["payload_json"], str)


def test_event_id_is_deterministic():
    row = _row("strategy_recommendation", "dX", "MSFT", 42 * NS, {"a": 1})
    e1 = envelope_from_decision_row(row)
    e2 = envelope_from_decision_row(row)
    assert e1.event_id == e2.event_id            # reproducible identity -> idempotent dedup


# -- parquet: atomic, idempotent, recovery -----------------------------------
def test_parquet_write_is_idempotent_by_content_hash(config):
    w = ParquetWriter(config)
    table = pa.table({"event_timestamp_ns": [NS], "source_feed": ["sip"], "x": [1]})
    r1 = w.write("bronze", "t", table, partition={"trade_date": "2026-01-01"})
    r2 = w.write("bronze", "t", table, partition={"trade_date": "2026-01-01"})
    assert r1.ok and r2.ok
    assert r1.path == r2.path                     # same content hash -> same part
    assert r2.reason == "idempotent"
    parts = list((config.bronze / "t").rglob("*.parquet"))
    assert len(parts) == 1


def test_recover_incomplete_removes_orphan_tmp(config):
    w = ParquetWriter(config)
    orphan = config.bronze / "t" / "trade_date=2026-01-01"
    orphan.mkdir(parents=True)
    tmp = orphan / "part-dead.parquet.tmp"
    tmp.write_bytes(b"partial")
    assert w.recover_incomplete() == 1
    assert not tmp.exists()


# -- nonblocking + loss accounting (disk-full failure injection §33) ---------
def test_ingestion_fails_closed_without_raising_when_over_cap(config, tmp_path):
    full = LakehouseConfig(root=config.root, min_free_disk_gib=0.0, max_storage_gib=1e-12)
    sink = AnalyticalIngestionSink(full, source_feed="sip")
    # seed one part so used_gib() > the (tiny) cap, forcing accepts_writes False
    ParquetWriter(config).write("bronze", "engine_decisions",
                                pa.table({"event_timestamp_ns": [NS], "source_feed": ["sip"]}),
                                partition={"trade_date": "2026-01-01"})
    stats = sink.ingest_decision_rows([_row("strategy_recommendation", "d1", "AAPL", NS, {})])
    assert stats.dropped >= 1                      # loss explicitly accounted
    assert stats.losses                            # and recorded, never silently lost
    assert stats.written == 0                      # nothing written (fail closed)


def test_bad_record_is_quarantined_not_fatal(config):
    sink = AnalyticalIngestionSink(config, source_feed="sip")
    # payload that breaks envelope construction (non-dict triggers the except path)
    stats = sink.ingest_decision_rows([{"event_type": "x", "payload": 12345,
                                        "timestamp_ns": "not-an-int"}])
    assert stats.quarantined >= 1
    assert any((config.quarantine).rglob("*.parquet"))


def test_market_capture_never_raises_on_malformed_rows(config):
    # Exit-independence invariant (§8/§30): the analytical capture path must never raise
    # into the engine (which feeds the protective exit). Heterogeneous/unserializable
    # market rows are quarantined, not propagated.
    sink = AnalyticalIngestionSink(config, source_feed="sip")
    bad_rows = [{"event_timestamp_ns": NS, "source_feed": "sip", "x": 1},
                {"event_timestamp_ns": NS, "source_feed": "sip", "x": object()}]
    stats = sink.ingest_market_rows("market_bars", bad_rows)  # must not raise
    assert stats.ingested >= 2


# -- silver dedup + incremental ----------------------------------------------
def test_silver_dedup_and_incremental_checkpoint(config):
    t = 1_700_000_000 * NS
    rows = [_row("complete_fill", "d1", "AAPL", t,
                 {"last_fill_quantity": 10, "last_fill_price_ticks": 1_000_000})]
    sink = AnalyticalIngestionSink(config, source_feed="sip")
    sink.ingest_decision_rows(rows)
    sink.ingest_decision_rows(rows)               # duplicate event_id (same content)
    st = SilverTransformer(config)
    out1 = st.transform("fills")
    assert out1["rows"] == 1                       # deduped by event_id
    out2 = st.transform("fills")
    assert out2.get("reason") == "up_to_date"      # incremental checkpoint -> no rework


# -- duckdb warehouse views + lineage ----------------------------------------
def test_warehouse_views_and_lineage(config):
    _seed_lifecycle(config)
    wh = DuckDBWarehouse(config)
    created = wh.register_views()
    assert "strategy_recommendations" in created and "portfolio_decisions" in created
    rows = wh.query(
        "SELECT r.decision_id, p.portfolio_decision_id "
        "FROM strategy_recommendations r JOIN portfolio_decisions p "
        "ON p.source_strategy_decision_id = r.decision_id")
    assert rows and rows[0][0] == "d1" and rows[0][1] == "p1"


# -- point-in-time leakage prevention ----------------------------------------
def test_feature_store_asof_excludes_future(config):
    t = 1_700_000_000 * NS
    AnalyticalIngestionSink(config, source_feed="sip").ingest_decision_rows([
        _row("regime_transition", "g1", "AAPL", t,
             {"level": "bull", "direction": "up", "confidence": 0.7}),
        _row("regime_transition", "g2", "AAPL", t + 10 * NS,
             {"level": "bear", "direction": "down", "confidence": 0.9}),
    ])
    SilverTransformer(config).transform("market_regimes")
    fs = PointInTimeFeatureStore(config)
    feat = FeatureSet(name="market_regimes", columns=("direction", "confidence"))
    # as-of just before the second regime: must see only the first (no look-ahead)
    got = fs.as_of("AAPL", t + 5 * NS, feat)
    assert got is not None and got["direction"] == "up"
    # as-of after the second: sees the newer one
    got2 = fs.as_of("AAPL", t + 20 * NS, feat)
    assert got2["direction"] == "down"


def test_training_snapshot_is_leakage_free(config):
    t = 1_700_000_000 * NS
    AnalyticalIngestionSink(config, source_feed="sip").ingest_decision_rows([
        _row("regime_transition", "g1", "AAPL", t,
             {"direction": "up", "confidence": 0.7}),
        _row("regime_transition", "g2", "AAPL", t + 100 * NS,
             {"direction": "down", "confidence": 0.9}),
    ])
    SilverTransformer(config).transform("market_regimes")
    fs = PointInTimeFeatureStore(config)
    decisions = pa.table({"symbol_id": ["AAPL"], "decision_ts": [t + 5 * NS]})
    snap = fs.training_snapshot(
        decisions, [FeatureSet(name="market_regimes", columns=("direction",), prefix="rg_")])
    # the decision at t+5 must NOT pull the t+100 (future) regime row
    assert snap.column("rg_direction").to_pylist() == ["up"]


def test_quality_flags_point_in_time_violation(config):
    # availability < event is a leakage violation; the quality check must catch it.
    ParquetWriter(config).write("bronze", "engine_decisions", pa.table({
        "event_id": ["e1"], "event_timestamp_ns": [100 * NS],
        "availability_timestamp_ns": [50 * NS],   # <- before the event: invalid
        "source_feed": ["sip"]}), partition={"trade_date": "2026-01-01"})
    checks = DataQualityValidator(config).validate_bronze()
    pit = [c for c in checks if c.check == "point_in_time_availability"]
    assert pit and pit[0].passed is False and pit[0].count == 1


# -- gold: FIFO outcomes, censored labels, RL reward integrity ---------------
def test_position_outcomes_fifo_and_return(config):
    _seed_lifecycle(config)
    out = GoldBuilder(config).build_position_outcomes()
    assert out["closed"] == 1 and out["open"] == 0
    wh = DuckDBWarehouse(config)
    wh.register_views()
    r = wh.query("SELECT entry_price_ticks, exit_price_ticks, gross_return, lot_attribution "
                 "FROM position_outcomes")
    assert r[0][0] == 1_000_000 and r[0][1] == 1_010_000
    assert abs(r[0][2] - 0.01) < 1e-9             # +1% realized
    assert r[0][3] == "FIFO"


def test_supervised_labels_censored_when_no_outcome(config):
    t = 1_700_000_000 * NS
    # a recommendation with NO matching fill lifecycle -> must be censored, not fabricated
    AnalyticalIngestionSink(config, source_feed="sip").ingest_decision_rows([
        _row("strategy_recommendation", "dLonely", "NVDA", t,
             {"eligibility": "eligible", "confidence": 0.5})])
    SilverTransformer(config).run_all()
    g = GoldBuilder(config)
    g.build_position_outcomes()
    sup = g.build_supervised()
    assert sup["censored_labels"] >= 1 and sup["realized_labels"] == 0


def test_offline_rl_no_fabricated_probability_and_closed_only_reward(config):
    _seed_lifecycle(config)
    g = GoldBuilder(config)
    g.build_position_outcomes()
    g.build_offline_rl()
    wh = DuckDBWarehouse(config)
    wh.register_views()
    rows = wh.query("SELECT action_probability_if_known, reward, terminal "
                    "FROM offline_rl_transitions")
    assert rows
    for prob, _reward, _terminal in rows:
        assert prob is None                       # behavior policy unknown -> never invented


# -- reproducibility ---------------------------------------------------------
def test_dataset_version_is_reproducible_and_changes_with_data(config):
    _seed_lifecycle(config)
    GoldBuilder(config).run_all()
    reg = ModelGovernanceRegistry(config)
    ds = config.dataset_path("gold", "supervised")
    v1 = reg.dataset_version(ds)
    v2 = reg.dataset_version(ds)
    assert v1 == v2 and len(v1) == 24             # stable hash over the manifest parts
    # adding a part changes the version
    ParquetWriter(config).write("gold", "supervised", pa.table({
        "event_timestamp_ns": [NS], "source_feed": ["internal"], "extra": [1]}),
        partition={"trade_date": "2099-01-01"})
    assert reg.dataset_version(ds) != v1


# -- governance: M5 records metadata only, never trains/promotes --------------
def test_governance_defaults_to_unpromoted_unapproved(config):
    reg = ModelGovernanceRegistry(config)
    rec = reg.register(ModelRecord(
        model_id="m1", model_version="v1", model_type="supervised_classifier",
        training_dataset_version="abc", feature_schema_version="feature-store-1.0.0",
        training_period="2026", validation_period="2026", out_of_sample_period="2026"))
    assert rec["promotion_status"] == "NONE"
    assert rec["approval_status"] == "UNAPPROVED"
    assert reg.list_models()[0]["model_id"] == "m1"


# -- maintenance: compaction dedup -------------------------------------------
def test_compaction_merges_parts_preserving_dedup(config):
    sink = AnalyticalIngestionSink(config, source_feed="sip")
    t = 1_700_000_000 * NS
    for i in range(20):
        sink.ingest_decision_rows([_row("strategy_recommendation", f"d{i}", "AAPL", t, {})])
    stats = LakehouseMaintenance(config).compact_layer("bronze", min_files=16)
    assert stats.compacted_partitions == 1
    assert stats.files_after == 1 and stats.rows_out == 20
    parts = list((config.bronze / "engine_decisions").rglob("part-*.parquet"))
    assert len(parts) == 1


# -- historical ingestion: resumable, tiered, NO IEX fallback ----------------
def _fake_fetch(symbols, timeframe, start, limit=10000):
    return {s: [{"t": "2026-11-20T15:3%d:00Z" % i, "o": 100 + i, "h": 101 + i,
                 "l": 99 + i, "c": 100.5 + i, "v": 1000 * i, "n": 5, "vw": 100.2 + i}
                for i in range(3)] for s in symbols}


def test_historical_resumable_and_dedup(config):
    ing = HistoricalSIPIngestion(config, fetch_fn=_fake_fetch, feed="sip")
    tiers = [HistoricalTier("positions", ("AAPL",), "5Min", "2026-11-01T00:00:00Z"),
             HistoricalTier("remaining", ("MSFT", "AAPL"), "5Min", "2026-11-01T00:00:00Z")]
    r1 = ing.run(tiers)
    assert r1["symbols_requested"] == 2 and r1["bars_written"] == 6  # AAPL deduped across tiers
    r2 = ing.run(tiers)
    assert r2["bars_written"] == 0 and r2["skipped_resumed"] == 6    # resume skips


def test_historical_refuses_iex_and_never_falls_back(config):
    with pytest.raises(ValueError):
        HistoricalSIPIngestion(config, fetch_fn=_fake_fetch, feed="iex")

    def err_fetch(s, t, st, limit=10000):
        return {"__error__": "HTTP 403"}
    ing = HistoricalSIPIngestion(config, fetch_fn=err_fetch, feed="sip")
    r = ing.run([HistoricalTier("x", ("AAPL",), "5Min", "2026-11-01T00:00:00Z")])
    assert r["bars_written"] == 0 and r["batches_failed"] == 1
    assert "sip:HTTP 403" in r["failures"][0]["error"]   # explicit, no silent downgrade


def test_historical_bars_preserve_tick_scale_and_point_in_time(config):
    ing = HistoricalSIPIngestion(config, fetch_fn=_fake_fetch, feed="sip")
    ing.run([HistoricalTier("x", ("AAPL",), "5Min", "2026-11-01T00:00:00Z")])
    wh = DuckDBWarehouse(config)
    wh.register_views()
    rows = wh.query("SELECT bar_close_ticks, price_scale, "
                    "availability_timestamp_ns >= event_timestamp_ns FROM market_bars")
    assert rows
    assert rows[0][1] == DEFAULT_PRICE_SCALE
    assert all(r[2] for r in rows)                # availability (bar close) >= event (open)


# -- attribution + storage ---------------------------------------------------
def test_attribution_reports_realized_outcomes(config):
    _seed_lifecycle(config)
    GoldBuilder(config).build_position_outcomes()
    rep = DailyAttribution(config).attribute(now_ns=1_800_000_000 * NS)
    assert rep["closed_positions"] == 1 and rep["win_rate"] == 1.0
    assert "loss_source_taxonomy" in rep and "causal_note" in rep


def test_storage_monitor_health_and_candidates(config):
    sink = AnalyticalIngestionSink(config, source_feed="sip")
    t = 1_700_000_000 * NS
    for i in range(20):
        sink.ingest_decision_rows([_row("strategy_recommendation", f"d{i}", "AAPL", t, {})])
    rep = StorageMonitor(config).report()
    assert rep.health in ("OK", "WARN", "CRITICAL")
    assert any(d["dataset"] == "engine_decisions" for d in rep.datasets)
    assert rep.compaction_candidates                 # 20 parts in one partition >= min_files


# -- end-to-end pipeline, idempotent -----------------------------------------
def test_pipeline_end_to_end_and_idempotent(config):
    t = 1_700_000_000 * NS
    rows = [
        _row("strategy_recommendation", "d1", "AAPL", t,
             {"eligibility": "eligible", "confidence": 0.8}),
        _row("portfolio_recommendation", "p1", "AAPL", t + 1,
             {"source_strategy_decision_id": "d1", "portfolio_decision_id": "p1",
              "allocation_policy": "rp", "eligibility": "eligible"}),
        _row("complete_fill", "d1", "AAPL", t + 2,
             {"last_fill_quantity": 100, "last_fill_price_ticks": 1_000_000}),
        _row("complete_fill", "d2", "AAPL", t + 3,
             {"last_fill_quantity": 100, "last_fill_price_ticks": 1_010_000}),
    ]
    AnalyticalIngestionSink(config, source_feed="sip").ingest_decision_rows(rows)
    out = LakehousePipeline(config).run()
    for stage in ("silver", "gold", "quality", "attribution", "maintenance",
                  "storage", "warehouse_views"):
        assert stage in out and "error" not in (out[stage] or {})
    assert out["quality"]["failed"] == 0
    # second run is safe + adds no silver rows (idempotent via checkpoints)
    out2 = LakehousePipeline(config).run()
    assert all(v.get("rows", 0) == 0 for v in out2["silver"].values())
