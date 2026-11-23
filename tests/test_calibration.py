"""Tests for the probability-calibration + uncertainty-evaluation module (§20).

All fixtures are EXPLICITLY SYNTHETIC — no market data, no fabricated labels standing in
for real outcomes. We draw labels from a known latent probability so the "true" calibration
is defined, then verify: a miscalibrated set scores a high ECE; isotonic and temperature
calibrators FIT ON A VALIDATION SPLIT lower ECE on a disjoint held-out split; quantile /
prediction-interval coverage matches nominal on correctly-specified synthetic quantiles and
is off for a too-narrow interval; and confidence-gated abstention is flagged justified only
when it measurably improves selective accuracy.

These run on the CPU (pure numpy / scikit-learn), so there are no GPU assertions to skip.
"""

from __future__ import annotations

import numpy as np
import pytest

from cengine.learning.calibration import (
    CALIBRATION_DEVICE,
    IsotonicCalibrator,
    TemperatureScaler,
    apply_abstention,
    brier_score,
    calibration_report,
    confidence_from_proba,
    expected_calibration_error,
    log_loss,
    maximum_calibration_error,
    prediction_interval_coverage,
    quantile_coverage,
    reliability_curve,
)
from cengine.learning.supervised.base import classification_metrics


def _overconfident_split(seed: int = 7, n: int = 4000, sharpen: float = 2.5):
    """Draw labels from a latent prob, report OVERCONFIDENT probs; return (val, test).

    reported_logit = sharpen * true_logit  ->  reported probs are too extreme. Temperature
    scaling with T=sharpen (or isotonic) recovers calibration.
    """
    rng = np.random.default_rng(seed)
    true_logit = rng.normal(0.0, 1.3, size=n)
    true_p = 1.0 / (1.0 + np.exp(-true_logit))
    y = (rng.uniform(size=n) < true_p).astype(np.float64)
    reported = 1.0 / (1.0 + np.exp(-sharpen * true_logit))
    half = n // 2
    return (reported[:half], y[:half]), (reported[half:], y[half:])


# ---------------------------------------------------------------- scalar scores reuse base
def test_scalar_scores_match_shared_base_metrics():
    rng = np.random.default_rng(1)
    y = (rng.uniform(size=200) < 0.5).astype(float)
    p = rng.uniform(0.05, 0.95, size=200)
    base = classification_metrics(y, p)
    assert brier_score(y, p) == pytest.approx(base["brier"])
    assert log_loss(y, p) == pytest.approx(base["log_loss"])


# ---------------------------------------------------------------- reliability + ECE
def test_perfectly_calibrated_has_low_ece():
    # labels drawn directly from the reported probability -> well calibrated
    rng = np.random.default_rng(3)
    p = rng.uniform(0.0, 1.0, size=8000)
    y = (rng.uniform(size=p.size) < p).astype(float)
    ece = expected_calibration_error(y, p, n_bins=10)
    assert ece < 0.03
    curve = reliability_curve(y, p, n_bins=10)
    assert int(curve.bin_counts.sum()) == p.size
    assert curve.mce() >= curve.ece()  # max gap >= mean gap


def test_miscalibrated_set_has_high_ece():
    (pv, yv), (pt, yt) = _overconfident_split()
    all_p = np.concatenate([pv, pt])
    all_y = np.concatenate([yv, yt])
    ece = expected_calibration_error(all_y, all_p, n_bins=15)
    assert ece > 0.07, f"expected a clearly miscalibrated ECE, got {ece}"
    # MCE is always >= ECE
    assert maximum_calibration_error(all_y, all_p, n_bins=15) >= ece


def test_quantile_binning_handles_clustered_predictions():
    # predictions clustered near 0 and 1 -> uniform bins leave many empty; quantile does not
    rng = np.random.default_rng(5)
    p = np.concatenate([rng.uniform(0.0, 0.05, 500), rng.uniform(0.95, 1.0, 500)])
    y = (rng.uniform(size=p.size) < p).astype(float)
    curve = reliability_curve(y, p, n_bins=10, strategy="quantile")
    assert int(curve.bin_counts.sum()) == p.size
    # no NaNs in populated bins; ECE is a finite number
    assert np.isfinite(curve.ece())


# ---------------------------------------------------- calibrators fit on validation only
def test_isotonic_fit_on_val_lowers_ece_on_heldout():
    (pv, yv), (pt, yt) = _overconfident_split(seed=11)
    uncal = expected_calibration_error(yt, pt, n_bins=15)
    cal = IsotonicCalibrator(min_samples=50).fit(pv, yv)  # VALIDATION split only
    assert cal.fitted and not cal.research_only
    pt_cal = cal.transform(pt)  # applied to the disjoint held-out split
    cal_ece = expected_calibration_error(yt, pt_cal, n_bins=15)
    assert cal_ece < uncal, f"isotonic did not improve ECE: {cal_ece} !< {uncal}"
    assert cal_ece < 0.06


def test_temperature_fit_on_val_lowers_ece_and_recovers_temperature():
    sharpen = 2.5
    (pv, yv), (pt, yt) = _overconfident_split(seed=13, sharpen=sharpen)
    uncal = expected_calibration_error(yt, pt, n_bins=15)
    scaler = TemperatureScaler(min_samples=50).fit(pv, yv)  # VALIDATION split only
    assert scaler.fitted and not scaler.research_only
    # reported_logit = sharpen * true_logit, so the recovering temperature ~ sharpen
    assert scaler.temperature == pytest.approx(sharpen, rel=0.35)
    pt_cal = scaler.transform(pt)
    cal_ece = expected_calibration_error(yt, pt_cal, n_bins=15)
    assert cal_ece < uncal, f"temperature did not improve ECE: {cal_ece} !< {uncal}"
    assert cal_ece < 0.06
    # log loss on the held-out split also improves
    assert log_loss(yt, pt_cal) < log_loss(yt, pt)


def test_calibrator_insufficient_data_is_research_only_not_crash():
    # single-class validation -> identity fallback, flagged non-promotable
    iso = IsotonicCalibrator(min_samples=50).fit(np.array([0.2, 0.8, 0.6]),
                                                 np.array([1.0, 1.0, 1.0]))
    assert iso.fitted and iso.research_only
    probe = np.array([0.1, 0.5, 0.9])
    assert np.allclose(iso.transform(probe), probe)  # pass-through identity

    ts = TemperatureScaler(min_samples=50).fit(np.array([0.3, 0.7]),
                                               np.array([0.0, 0.0]))
    assert ts.fitted and ts.research_only and ts.temperature == 1.0

    # a valid but tiny fit is also flagged research_only
    rng = np.random.default_rng(9)
    p = rng.uniform(0.1, 0.9, 10)
    y = (rng.uniform(size=10) < p).astype(float)
    small = IsotonicCalibrator(min_samples=50).fit(p, y)
    assert small.fitted and small.research_only


def test_calibrator_requires_fit_before_transform():
    with pytest.raises(RuntimeError):
        IsotonicCalibrator().transform(np.array([0.5]))
    with pytest.raises(RuntimeError):
        TemperatureScaler().transform(np.array([0.5]))


# ------------------------------------------------------------- coverage checks
def test_quantile_coverage_matches_nominal_for_correct_quantiles():
    from statistics import NormalDist

    rng = np.random.default_rng(21)
    n = 20000
    sigma = 2.0
    y = rng.normal(0.0, sigma, size=n)
    levels = np.array([0.1, 0.5, 0.9])
    # correctly specified per-sample quantiles of the SAME N(0, sigma) the data came from
    zq = np.array([NormalDist().inv_cdf(q) for q in levels])
    q_pred = np.tile(sigma * zq, (n, 1))
    rep = quantile_coverage(y, q_pred, levels)
    assert rep.monotonic
    assert rep.max_abs_error < 0.02, rep.to_dict()


def test_quantile_coverage_detects_crossing_and_miscoverage():
    rng = np.random.default_rng(22)
    y = rng.normal(0.0, 1.0, size=5000)
    levels = np.array([0.1, 0.9])
    # deliberately crossed / wrong quantiles (upper below lower)
    q_pred = np.tile(np.array([1.0, -1.0]), (y.size, 1))
    rep = quantile_coverage(y, q_pred, levels)
    assert not rep.monotonic
    assert rep.max_abs_error > 0.1


def test_prediction_interval_coverage():
    rng = np.random.default_rng(23)
    from statistics import NormalDist

    y = rng.normal(0.0, 1.0, size=20000)
    z = NormalDist().inv_cdf(0.9)  # 80% central interval -> [-z, z]
    good = prediction_interval_coverage(y, np.full(y.size, -z), np.full(y.size, z),
                                        nominal=0.8)
    assert good.picp == pytest.approx(0.8, abs=0.02)
    assert abs(good.coverage_error) < 0.02
    assert good.mpiw == pytest.approx(2 * z, rel=1e-6)

    # a too-narrow interval under-covers
    narrow = prediction_interval_coverage(y, np.full(y.size, -0.2), np.full(y.size, 0.2),
                                          nominal=0.8)
    assert narrow.picp < 0.3
    assert narrow.coverage_error < -0.4


# ------------------------------------------------------------------- abstention
def test_abstention_is_justified_when_it_improves_selective_accuracy():
    # low-confidence region (p near 0.5) is pure noise; high-confidence region is correct.
    rng = np.random.default_rng(31)
    n = 4000
    # confident, correct predictions
    conf_p = np.where(rng.uniform(size=n) < 0.5, 0.95, 0.05)
    conf_y = (conf_p > 0.5).astype(float)  # always right when confident
    # unconfident, coin-flip-correct predictions near the boundary
    unc_p = rng.uniform(0.45, 0.55, size=n)
    unc_y = (rng.uniform(size=n) < 0.5).astype(float)
    p = np.concatenate([conf_p, unc_p])
    y = np.concatenate([conf_y, unc_y])

    res = apply_abstention(y, p, confidence_threshold=0.6, min_coverage=0.1)
    assert res.justified
    assert res.selective_accuracy > res.full_accuracy
    assert 0.0 < res.coverage < 1.0
    assert res.abstention_rate == pytest.approx(1.0 - res.coverage)
    # everything in the noisy band should have been abstained
    assert res.abstained[n:].all()
    assert not res.abstained[:n].any()


def test_abstention_not_justified_when_it_removes_the_good_predictions():
    # INVERTED signal: high-confidence predictions are WRONG, low-confidence ones are RIGHT.
    # Gating away low confidence therefore keeps the wrong ones -> selective accuracy drops,
    # so abstention must NOT be reported as justified.
    rng = np.random.default_rng(32)
    n = 4000
    # confident but wrong
    conf_p = np.where(rng.uniform(size=n) < 0.5, 0.95, 0.05)
    conf_y = (conf_p <= 0.5).astype(float)  # opposite of the hard prediction
    # unconfident but correct
    unc_p = np.where(rng.uniform(size=n) < 0.5, 0.52, 0.48)
    unc_y = (unc_p > 0.5).astype(float)
    p = np.concatenate([conf_p, unc_p])
    y = np.concatenate([conf_y, unc_y])
    res = apply_abstention(y, p, confidence_threshold=0.6, min_coverage=0.1)
    assert not res.justified
    assert res.selective_accuracy < res.full_accuracy


def test_confidence_from_proba_and_distinctness_contract():
    p = np.array([0.0, 0.1, 0.5, 0.9, 1.0])
    conf = confidence_from_proba(p)
    assert np.allclose(conf, [1.0, 0.9, 0.5, 0.9, 1.0])
    assert conf.min() >= 0.5 and conf.max() <= 1.0


# ------------------------------------------------------------------- report bundle
def test_calibration_report_bundles_metrics_and_records_cpu_device():
    (pv, yv), (pt, yt) = _overconfident_split(seed=41)
    rep = calibration_report(yt, pt, n_bins=10)
    for key in ("brier", "log_loss", "accuracy", "base_rate", "ece", "mce", "device"):
        assert key in rep
    assert rep["device"] == CALIBRATION_DEVICE == "cpu"
    assert rep["mce"] >= rep["ece"]
