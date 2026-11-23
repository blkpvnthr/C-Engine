"""Probability calibration + uncertainty evaluation (Milestone 6, §20).

CPU-only (pure numpy / scikit-learn) post-processing that scores and recalibrates the
probabilistic outputs of the supervised models: Brier / log loss, reliability curves,
ECE / MCE, isotonic + temperature-scaling calibrators (FIT ON VALIDATION ONLY), prediction-
interval / quantile coverage checks, and a justified-abstention helper. Model confidence
(a calibratable probability) is kept DISTINCT from the engine's heuristic ranking
confidence — see ``calibration`` module docstring. Recommendation-only; never touches the
mandatory -0.5% protective exit.
"""

from cengine.learning.calibration.calibration import (
    CALIBRATION_DEVICE,
    AbstentionResult,
    IntervalCoverageReport,
    IsotonicCalibrator,
    QuantileCoverageReport,
    ReliabilityCurve,
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

__all__ = [
    "CALIBRATION_DEVICE",
    "AbstentionResult",
    "IntervalCoverageReport",
    "IsotonicCalibrator",
    "QuantileCoverageReport",
    "ReliabilityCurve",
    "TemperatureScaler",
    "apply_abstention",
    "brier_score",
    "calibration_report",
    "confidence_from_proba",
    "expected_calibration_error",
    "log_loss",
    "maximum_calibration_error",
    "prediction_interval_coverage",
    "quantile_coverage",
    "reliability_curve",
]
