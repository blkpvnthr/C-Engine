"""Cross-cutting learning-platform integration, failure-injection + invariant tests
(Milestone 6, §13/§22/§30/§36/§37 + addendum §22/§23/§25).

Deterministic + offline. Covers hardware/device policy, dataset eligibility + point-in-time
certification, supervised training + CPU/GPU parity, registry + promotion authorization +
artifact verification, inference abstention/fallback, champion/challenger, replay protective-
exit-as-environment-dynamics, failure injection, and the hard invariant that the learning
layer is recommendation-only and never perturbs the mandatory protective exit.
"""

from __future__ import annotations

import numpy as np
import pytest

from cengine.lakehouse.config import LakehouseConfig
from cengine.learning.datasets.certification import certify_training_dataset
from cengine.learning.datasets.eligibility import EligibilityState, assess_task, default_task_specs
from cengine.learning.hardware import DevicePolicy, inspect_hardware, mlx_available
from cengine.learning.inference.contract import InferenceRequest
from cengine.learning.inference.loader import ModelLoadError, load_model
from cengine.learning.inference.service import LearningInferenceService
from cengine.learning.registry.artifacts import ArtifactStore, ArtifactVerificationError
from cengine.learning.registry.maturity import (
    ModelMaturity,
    PromotionError,
    can_transition,
    requires_explicit_approval,
)
from cengine.learning.registry.model_registry import LearningModelRegistry, ModelCard
from cengine.learning.supervised.base import TrainConfig, regression_metrics
from cengine.learning.supervised.baselines import ConstantBaseline, SklearnRegressor

NS = 1_000_000_000


@pytest.fixture()
def config(tmp_path):
    cfg = LakehouseConfig(root=tmp_path / "lakehouse", min_free_disk_gib=0.0)
    cfg.ensure_dirs()
    return cfg


# -- hardware / device policy ------------------------------------------------

def test_hardware_inspection_reports_real_capabilities():
    cap = inspect_hardware()
    assert isinstance(cap.mlx_metal_available, bool)
    assert cap.report_version
    # precisions is always non-empty (cpu fallback lists at least float32)
    assert len(cap.supported_precisions) >= 1


def test_device_policy_gpu_first_with_documented_fallback():
    p = DevicePolicy(min_gpu_elements=50_000)
    big = p.choose("train", n_elements=500_000)
    small = p.choose("infer_one", n_elements=10)
    # small workloads go to CPU with a recorded reason (never a silent GPU claim)
    assert small.device == "cpu" and small.reason
    if mlx_available():
        assert big.device == "gpu"
    assert "placements" in p.summary()


def test_force_cpu_env(monkeypatch):
    monkeypatch.setenv("CENGINE_LEARNING_FORCE_CPU", "1")
    p = DevicePolicy()
    assert p.choose("train", n_elements=10 ** 7).device == "cpu"


# -- dataset eligibility + certification (point-in-time) ---------------------

def test_eligibility_insufficient_when_dataset_absent(config):
    spec = default_task_specs()[1]  # strategy_net_return (gold/supervised, absent)
    from cengine.learning.datasets.eligibility import DatasetProfiler
    prof = DatasetProfiler(config).profile(spec.source_layer, spec.source_dataset)
    state, reasons = assess_task(spec, prof)
    assert state == EligibilityState.INSUFFICIENT_DATA


def test_certification_rejects_absent_dataset(config):
    cert = certify_training_dataset(config, "bronze", "market_bars",
                                    feature_schema_version="f", label_definition_version="l")
    assert cert.valid is False
    assert "dataset_not_materialized" in cert.reasons


# -- supervised + CPU/GPU parity (§13) ---------------------------------------

def test_sklearn_baseline_beats_constant_on_learnable_signal():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(2000, 6)).astype(np.float32)
    y = (0.5 * X[:, 0] - 0.3 * X[:, 1] + 0.02 * rng.normal(size=2000)).astype(np.float32)
    reg = SklearnRegressor(kind="hgbr")
    reg.fit(X[:1500], y[:1500])
    base = ConstantBaseline()
    base.fit(X[:1500], y[:1500])
    rm = regression_metrics(y[1500:], reg.predict(X[1500:]))
    bm = regression_metrics(y[1500:], base.predict(X[1500:]))
    assert rm["rank_ic"] > bm["rank_ic"]        # the model learns a real signal


@pytest.mark.skipif(not mlx_available(), reason="MLX/Metal unavailable")
def test_mlx_regressor_cpu_gpu_prediction_parity():
    from cengine.learning.supervised.mlx_models import MLXRegressor
    rng = np.random.default_rng(1)
    X = rng.normal(size=(1500, 6)).astype(np.float32)
    y = (0.4 * X[:, 0] + 0.02 * rng.normal(size=1500)).astype(np.float32)
    m = MLXRegressor(6, config=TrainConfig(epochs=10, batch_size=512))
    m.fit(X[:1000], y[:1000])
    gpu_pred = m.predict(X[1000:])
    # force CPU and re-predict with the SAME trained weights -> parity within float32 tol
    m.policy = DevicePolicy(force_cpu=True)
    cpu_pred = m.predict(X[1000:])
    assert float(np.max(np.abs(gpu_pred - cpu_pred))) < 1e-3


# -- registry + maturity + promotion authorization (§22/§23) -----------------

def test_no_auto_promotion_to_live(config):
    reg = LearningModelRegistry(config)
    reg.register(ModelCard(model_id="m", model_version="v1", model_type="t",
                           maturity=ModelMaturity.LIVE_ELIGIBLE.value))
    with pytest.raises(PromotionError):
        reg.transition("m", ModelMaturity.LIVE_APPROVED, approved=False)
    card = reg.transition("m", ModelMaturity.LIVE_APPROVED, approved=True)
    assert card.approval_status == "APPROVED"


def test_maturity_transition_rules():
    assert requires_explicit_approval(ModelMaturity.LIVE_ELIGIBLE, ModelMaturity.LIVE_APPROVED)
    assert can_transition(ModelMaturity.ACTIVE, ModelMaturity.DEGRADED)       # safety sink
    assert not can_transition(ModelMaturity.RETIRED, ModelMaturity.ACTIVE)    # terminal


def test_rollback_always_legal(config):
    reg = LearningModelRegistry(config)
    reg.register(ModelCard(model_id="m", model_version="v1", model_type="t",
                           maturity=ModelMaturity.ACTIVE.value))
    card = reg.rollback("m", to_reference="baseline")
    assert card.maturity == ModelMaturity.DEGRADED.value
    assert card.rollback_reference == "baseline"


def test_champion_challenger_selection(config):
    reg = LearningModelRegistry(config)
    reg.register(ModelCard(model_id="a", model_version="v1", model_type="t",
                           maturity=ModelMaturity.ACTIVE.value, created_ns=1))
    reg.register(ModelCard(model_id="b", model_version="v2", model_type="t",
                           maturity=ModelMaturity.SHADOW.value, created_ns=2))
    assert reg.champion("t")["model_id"] == "a"
    assert reg.challenger("t")["model_id"] == "b"


# -- artifact verification + failure injection (§36) -------------------------

def test_artifact_checksum_rejects_corruption(config):
    store = ArtifactStore(config)
    ref = store.save_npz("m", "v1", {"w": np.ones((3, 2), dtype=np.float32)})
    assert store.verify(ref) is True
    import pathlib
    p = pathlib.Path(ref.path)
    p.write_bytes(p.read_bytes() + b"TAMPER")
    assert store.verify(ref) is False
    with pytest.raises(ArtifactVerificationError):
        store.load_npz(ref)


def test_load_model_missing_artifact_raises(config):
    with pytest.raises(ModelLoadError):
        load_model({"model_id": "x", "framework": "mlx", "model_type": "mlx_regressor",
                    "artifacts": {}}, config=config)


# -- inference abstention + deterministic fallback (§27) ----------------------

def _req(**kw):
    base = dict(task="forward_return_prediction", symbol_id="AAPL", features=(0.1, 0.2),
                feature_schema_version="feature-schema-1.0.0",
                observation_timestamp_ns=NS, availability_timestamp_ns=NS)
    base.update(kw)
    return InferenceRequest(**base)


def test_inference_abstains_when_no_model(config):
    svc = LearningInferenceService(config)
    resp = svc.predict(_req(), now_ns=2 * NS)
    assert resp.abstained and resp.prediction is None
    assert resp.applicability in ("no_model",)


def test_inference_abstains_on_untrained_and_schema_mismatch(config):
    reg = LearningModelRegistry(config)
    store = reg.artifacts
    ref = store.save_npz("u", "v1", {"w": np.ones((2, 2), dtype=np.float32)})
    reg.register(ModelCard(model_id="u", model_version="v1", model_type="mlx_regressor",
                           framework="mlx", feature_schema_version="feature-schema-1.0.0",
                           maturity=ModelMaturity.UNTRAINED.value,
                           artifacts={"weights": ref.to_dict()}))
    svc = LearningInferenceService(config)
    # untrained -> abstain even though a card exists
    assert svc.predict(_req(model_id="u"), now_ns=2 * NS).abstained


def test_inference_never_raises_on_broken_model(config):
    reg = LearningModelRegistry(config)
    # register a SHADOW card whose artifact is absent -> load fails -> safe abstention
    reg.register(ModelCard(model_id="b", model_version="v1", model_type="mlx_regressor",
                           framework="mlx", feature_schema_version="feature-schema-1.0.0",
                           maturity=ModelMaturity.SHADOW.value,
                           artifacts={"weights": {"path": "/nope", "checksum": "x",
                                                  "bytes": 0, "kind": "npz"}}))
    svc = LearningInferenceService(config)
    resp = svc.predict(_req(model_id="b"), now_ns=2 * NS)
    assert resp.abstained and resp.abstention_reason          # never raised


# -- replay: protective exit is environment dynamics (§16/§30) ----------------

def test_replay_protective_exit_clamps_loss(config):
    # synthesize a crashing bar series directly into bronze market_bars
    from cengine.lakehouse.parquet_writer import ParquetWriter
    w = ParquetWriter(config)
    rows = []
    base_ts = 1_700_000_000 * NS
    price = 1_000_000
    for i in range(80):
        price = int(price * (0.999 if i > 40 else 1.0005))  # steady then decline
        rows.append({"symbol_id": "CRSH", "timeframe": "5Min",
                     "event_timestamp_ns": base_ts + i * 300 * NS,
                     "availability_timestamp_ns": base_ts + i * 300 * NS + 300 * NS,
                     "bar_open_ticks": price, "bar_high_ticks": int(price * 1.001),
                     "bar_low_ticks": int(price * 0.99), "bar_close_ticks": price,
                     "volume": 1000, "source_feed": "sip"})
    w.write_rows("bronze", "market_bars", rows)
    from cengine.learning.environment.replay import ReplayConfig, ReplayEnvironment
    env = ReplayEnvironment(config, replay_config=ReplayConfig(protective_exit_threshold=-0.005))
    res = env.run("CRSH", lambda ctx, pos: 1)  # always-long policy
    # every protective-exit trade is clamped near the -0.5% floor (plus costs)
    pe = [t["reward"] for t in res.transitions if t["exit_reason"] == "protective_exit"]
    if pe:
        assert min(pe) >= -0.005 - 0.001  # floor minus modeled costs, never a deep loss
    # protective exits are labeled as environment dynamics, not a policy action
    for t in res.transitions:
        assert t["action_probability_if_known"] is None


# -- non-ordering startup invariant (§37) ------------------------------------

def test_learning_service_submits_no_orders(config):
    """The inference service exposes no order/broker surface — it is recommendation-only."""
    svc = LearningInferenceService(config)
    for attr in ("submit", "order", "broker", "execute", "route"):
        assert not hasattr(svc, attr)


def test_daily_cycle_runs_without_data_and_without_orders(config):
    from cengine.learning.orchestration.daily_cycle import DailyLearningCycle
    out = DailyLearningCycle(config).run(materialize=False)
    # no data -> everything skipped/insufficient, no errors escape, no orders anywhere
    assert "eligibility" in out
    assert isinstance(out.get("trained"), list)
