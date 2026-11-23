"""Unit tests for model drift monitoring (Milestone 6, §26).

All fixtures are EXPLICITLY SYNTHETIC (generated with a seeded numpy RNG) — they exercise the
detector/state logic, not any real market data. The core contract checks:
  * identical distribution            -> HEALTHY
  * shifted distribution              -> elevated PSI and a DEGRADED/ROLLBACK_REQUIRED state
  * missing-feature spike             -> flagged (missing detector + overall non-HEALTHY)
plus prediction drift, calibration deterioration, insufficient-data ABSTAIN, schema mismatch,
the no-trading-authority invariant, round-trip serialization, and the observable CPU device.
"""

from __future__ import annotations

import numpy as np
import pytest

from cengine.learning.hardware import mlx_available
from cengine.learning.monitoring import (
    DriftMonitor,
    DriftState,
    DriftThresholds,
    ReferenceDistribution,
    expected_calibration_error,
)

_NAMES = ("f0", "f1", "f2", "f3", "f4")


def _synthetic_reference(n: int = 3000, seed: int = 0) -> tuple[np.ndarray, ReferenceDistribution]:
    """SYNTHETIC standard-normal training matrix + its captured reference distribution."""
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, 1.0, size=(n, len(_NAMES)))
    return x, ReferenceDistribution.from_training(x, _NAMES)


def test_identical_distribution_is_healthy():
    x_ref, ref = _synthetic_reference()
    monitor = DriftMonitor(ref)
    # Comparing the reference batch against itself: PSI is exactly 0 per feature.
    report = monitor.evaluate(x_ref)
    assert report.overall_state == DriftState.HEALTHY
    assert report.healthy is True
    assert max(report.feature_psi.values()) < ref.n_bins  # ~0, well under any band
    assert all(v < DriftThresholds().psi_warn for v in report.feature_psi.values())


def test_resampled_same_distribution_is_healthy():
    x_ref, ref = _synthetic_reference(seed=1)
    monitor = DriftMonitor(ref)
    rng = np.random.default_rng(99)
    x_live = rng.normal(0.0, 1.0, size=(3000, len(_NAMES)))  # SYNTHETIC same dist
    report = monitor.evaluate(x_live)
    assert report.overall_state == DriftState.HEALTHY
    assert max(report.feature_psi.values()) < DriftThresholds().psi_degrade


def test_shifted_distribution_elevates_psi_and_degrades():
    _, ref = _synthetic_reference(seed=2)
    monitor = DriftMonitor(ref)
    rng = np.random.default_rng(3)
    x_live = rng.normal(3.0, 1.0, size=(3000, len(_NAMES)))  # SYNTHETIC +3 sigma shift
    report = monitor.evaluate(x_live)
    # PSI must be elevated well past the "significant shift" band on every shifted feature.
    assert min(report.feature_psi.values()) > DriftThresholds().psi_degrade
    # With all five features severely shifted, the aggregate escalates to a severe state.
    assert report.overall_state in (DriftState.DEGRADED, DriftState.ROLLBACK_REQUIRED)
    # Many simultaneous severe detectors -> rollback recommendation.
    assert report.overall_state == DriftState.ROLLBACK_REQUIRED
    assert "revert" in report.recommended_action.lower()


def test_single_feature_moderate_shift_is_degraded_not_rollback():
    _, ref = _synthetic_reference(seed=7)
    monitor = DriftMonitor(ref)
    rng = np.random.default_rng(8)
    x_live = rng.normal(0.0, 1.0, size=(3000, len(_NAMES)))
    x_live[:, 2] += 0.6  # SYNTHETIC: one feature moderately shifted -> PSI in the DEGRADED band
    report = monitor.evaluate(x_live)
    t = DriftThresholds()
    assert t.psi_degrade <= report.feature_psi["f2"] < t.psi_rollback
    # Exactly one severe detector (< rollback_min_detectors) and no extreme feature ->
    # DEGRADED, not a whole-model rollback.
    assert report.overall_state == DriftState.DEGRADED


def test_missing_feature_spike_is_flagged():
    _, ref = _synthetic_reference(seed=4)
    monitor = DriftMonitor(ref)
    rng = np.random.default_rng(5)
    x_live = rng.normal(0.0, 1.0, size=(2000, len(_NAMES)))
    x_live[:600, 1] = np.nan  # 30% missing in one feature (SYNTHETIC spike)
    report = monitor.evaluate(x_live)
    assert report.missing_rates["f1"] == pytest.approx(0.3, abs=1e-6)
    missing_det = next(d for d in report.detectors if d.name == "missing_feature")
    assert missing_det.state != DriftState.HEALTHY
    assert report.overall_state != DriftState.HEALTHY


def test_full_missing_feature_blocks():
    _, ref = _synthetic_reference(seed=10)
    monitor = DriftMonitor(ref)
    rng = np.random.default_rng(11)
    x_live = rng.normal(0.0, 1.0, size=(1500, len(_NAMES)))
    x_live[:, 3] = np.nan  # 100% missing -> past block band
    report = monitor.evaluate(x_live)
    missing_det = next(d for d in report.detectors if d.name == "missing_feature")
    assert missing_det.state == DriftState.BLOCKED
    assert report.overall_state in (DriftState.BLOCKED, DriftState.ROLLBACK_REQUIRED)


def test_out_of_distribution_rate_detects_off_manifold_inputs():
    _, ref = _synthetic_reference(seed=12)
    monitor = DriftMonitor(ref)
    rng = np.random.default_rng(13)
    # SYNTHETIC: all rows are ~10-sigma off the reference manifold.
    x_live = rng.normal(10.0, 1.0, size=(1500, len(_NAMES)))
    report = monitor.evaluate(x_live)
    ood_det = next(d for d in report.detectors if d.name == "out_of_distribution")
    assert report.ood_rate > DriftThresholds().ood_rate_degrade
    assert ood_det.state in (DriftState.DEGRADED, DriftState.BLOCKED)


def test_prediction_distribution_drift():
    rng = np.random.default_rng(20)
    x = rng.normal(0.0, 1.0, size=(3000, len(_NAMES)))
    preds = rng.normal(0.0, 1.0, size=3000)  # SYNTHETIC reference prediction dist
    ref = ReferenceDistribution.from_training(x, _NAMES, predictions=preds)
    monitor = DriftMonitor(ref)
    # Live features in-distribution, but predictions shifted.
    x_live = rng.normal(0.0, 1.0, size=(3000, len(_NAMES)))
    preds_live = rng.normal(2.5, 1.0, size=3000)
    report = monitor.evaluate(x_live, predictions=preds_live)
    assert report.prediction_psi is not None
    assert report.prediction_psi > DriftThresholds().pred_psi_degrade
    pred_det = next(d for d in report.detectors if d.name == "prediction_distribution")
    assert pred_det.evaluated is True
    assert pred_det.state in (DriftState.DEGRADED, DriftState.ROLLBACK_REQUIRED)


def test_prediction_detector_not_evaluated_without_reference_predictions():
    _, ref = _synthetic_reference(seed=21)
    monitor = DriftMonitor(ref)
    rng = np.random.default_rng(22)
    report = monitor.evaluate(rng.normal(0, 1, size=(500, len(_NAMES))),
                              predictions=rng.normal(0, 1, size=500))
    pred_det = next(d for d in report.detectors if d.name == "prediction_distribution")
    assert pred_det.evaluated is False
    assert report.prediction_psi is None


def test_calibration_deterioration_detected():
    rng = np.random.default_rng(30)
    x = rng.normal(0.0, 1.0, size=(3000, len(_NAMES)))
    # SYNTHETIC well-calibrated reference: labels drawn from the predicted probability.
    ref_p = rng.uniform(0.0, 1.0, size=3000)
    ref_y = (rng.uniform(0.0, 1.0, size=3000) < ref_p).astype(float)
    ref = ReferenceDistribution.from_training(x, _NAMES, predictions=ref_p, y_true=ref_y)
    assert ref.ref_ece is not None and ref.ref_ece < 0.1  # reference is well-calibrated
    monitor = DriftMonitor(ref)
    x_live = rng.normal(0.0, 1.0, size=(3000, len(_NAMES)))
    # SYNTHETIC miscalibration: model says ~0.9 everywhere but is right only ~30% of the time.
    live_p = np.clip(rng.normal(0.9, 0.03, size=3000), 0.0, 1.0)
    live_y = (rng.uniform(0.0, 1.0, size=3000) < 0.3).astype(float)
    report = monitor.evaluate(x_live, predictions=live_p, y_live=live_y)
    assert report.calibration_delta is not None and report.calibration_delta > 0.0
    cal_det = next(d for d in report.detectors if d.name == "calibration")
    assert cal_det.evaluated is True
    assert cal_det.state in (DriftState.WARNING, DriftState.DEGRADED)


def test_calibration_never_evaluated_without_live_labels():
    rng = np.random.default_rng(31)
    x = rng.normal(0.0, 1.0, size=(2000, len(_NAMES)))
    ref_p = rng.uniform(0.0, 1.0, size=2000)
    ref_y = (rng.uniform(0.0, 1.0, size=2000) < ref_p).astype(float)
    ref = ReferenceDistribution.from_training(x, _NAMES, predictions=ref_p, y_true=ref_y)
    monitor = DriftMonitor(ref)
    # No y_live supplied -> calibration must NOT be evaluated (labels are never fabricated).
    report = monitor.evaluate(rng.normal(0, 1, size=(500, len(_NAMES))),
                              predictions=rng.uniform(0, 1, size=500))
    cal_det = next(d for d in report.detectors if d.name == "calibration")
    assert cal_det.evaluated is False
    assert report.calibration_delta is None


def test_insufficient_data_abstains_never_certifies_health():
    x_ref, ref = _synthetic_reference(seed=40)
    monitor = DriftMonitor(ref)
    # A tiny batch of in-distribution rows: cannot certify HEALTHY -> ABSTAIN.
    report = monitor.evaluate(x_ref[:20])
    assert report.insufficient_data is True
    assert report.overall_state == DriftState.ABSTAIN


def test_small_batch_in_distribution_does_not_false_rollback():
    # Regression guard: a TINY in-distribution batch must NOT be called severe. Small-sample
    # PSI noise (here ~3.0 on identical data) previously faked ROLLBACK_REQUIRED; it must now
    # be suppressed to ABSTAIN because the distribution detectors are unreliable at this size.
    x_ref, ref = _synthetic_reference(seed=41)
    report = DriftMonitor(ref).evaluate(x_ref[:20])
    assert report.insufficient_data is True
    assert max(report.feature_psi.values()) > 0.5  # the noise IS there...
    assert report.overall_state == DriftState.ABSTAIN  # ...but it must not win


def test_insufficient_data_does_not_mask_a_data_integrity_signal():
    # A data-integrity failure (a fully-missing feature) is reliable at ANY batch size and
    # must still surface even when the batch is too small for distribution drift.
    _, ref = _synthetic_reference(seed=42)
    monitor = DriftMonitor(ref)
    rng = np.random.default_rng(43)
    x_live = rng.normal(0.0, 1.0, size=(20, len(_NAMES)))  # too few rows for PSI...
    x_live[:, 0] = np.nan  # ...but feature 0 is 100% missing -> authoritative BLOCKED
    report = monitor.evaluate(x_live)
    assert report.insufficient_data is True
    missing_det = next(d for d in report.detectors if d.name == "missing_feature")
    assert missing_det.state == DriftState.BLOCKED
    assert report.overall_state == DriftState.BLOCKED  # integrity signal is not masked


def test_schema_mismatch_blocks():
    _, ref = _synthetic_reference(seed=50)
    monitor = DriftMonitor(ref)
    rng = np.random.default_rng(51)
    x_live = rng.normal(0.0, 1.0, size=(1000, len(_NAMES) - 1))  # wrong feature count
    report = monitor.evaluate(x_live)
    assert report.overall_state == DriftState.BLOCKED
    assert report.detectors[0].name == "feature_schema"


def test_report_never_grants_trading_authority():
    _, ref = _synthetic_reference(seed=60)
    monitor = DriftMonitor(ref)
    rng = np.random.default_rng(61)
    for shift in (0.0, 3.0):
        report = monitor.evaluate(rng.normal(shift, 1.0, size=(1500, len(_NAMES))))
        assert report.grants_trading_authority is False
        assert "advisory" in report.recommended_action.lower() or \
               "no action" in report.recommended_action.lower()
        d = report.to_dict()
        assert d["grants_trading_authority"] is False


def test_reference_distribution_round_trips():
    rng = np.random.default_rng(70)
    x = rng.normal(0.0, 1.0, size=(1200, len(_NAMES)))
    preds = rng.normal(0.0, 1.0, size=1200)
    y = (preds > 0).astype(float)
    ref = ReferenceDistribution.from_training(x, _NAMES, predictions=preds, y_true=y)
    restored = ReferenceDistribution.from_dict(ref.to_dict())
    monitor_a = DriftMonitor(ref)
    monitor_b = DriftMonitor(restored)
    x_live = rng.normal(0.0, 1.0, size=(1500, len(_NAMES)))
    rep_a = monitor_a.evaluate(x_live, predictions=rng.normal(0, 1, size=1500))
    rep_b = monitor_b.evaluate(x_live, predictions=rng.normal(0, 1, size=1500))
    # Restored reference yields the same per-feature PSI on the same live batch.
    for name in _NAMES:
        assert rep_a.feature_psi[name] == pytest.approx(rep_b.feature_psi[name], abs=1e-9)


def test_expected_calibration_error_bounds():
    # Perfectly calibrated constant predictor at the base rate -> ~0 ECE.
    y = np.array([1.0, 0.0, 1.0, 0.0])
    p = np.full(4, 0.5)
    assert expected_calibration_error(y, p, n_bins=10) == pytest.approx(0.0, abs=1e-9)
    # Confidently wrong -> ECE near 1.
    assert expected_calibration_error(np.zeros(100), np.ones(100), n_bins=10) > 0.9


def test_device_is_cpu_by_design():
    _, ref = _synthetic_reference(seed=80)
    monitor = DriftMonitor(ref)
    rng = np.random.default_rng(81)
    report = monitor.evaluate(rng.normal(0, 1, size=(1000, len(_NAMES))))
    # Drift monitoring is histogram/scipy based: CPU by design, even when a GPU is present.
    assert report.device == "cpu"
    assert isinstance(report.device_reason, str) and report.device_reason


@pytest.mark.skipif(not mlx_available(),
                    reason="GPU-present assertion: requires an actual MLX Metal device")
def test_cpu_choice_is_observable_when_gpu_present():
    # Even on a machine WITH a usable Metal GPU, drift monitoring stays on CPU and that
    # decision is recorded observably (never a silent GPU claim).
    _, ref = _synthetic_reference(seed=90)
    monitor = DriftMonitor(ref)
    rng = np.random.default_rng(91)
    report = monitor.evaluate(rng.normal(0, 1, size=(1000, len(_NAMES))))
    assert report.device == "cpu"
    assert monitor.policy.gpu_available is True  # GPU exists...
    assert report.device_reason  # ...but a documented CPU reason is recorded
