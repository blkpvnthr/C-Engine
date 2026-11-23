"""Model drift monitoring (Milestone 6, §26).

Advisory, recommendation-only drift detection: a live inference batch is compared against a
train-time reference distribution and mapped onto explicit health states. It grants NO trading
authority to any model and never touches the mandatory protective-exit path. Pure numpy/scipy
on the CPU by design; the CPU placement is recorded observably via ``DevicePolicy``.
"""

from cengine.learning.monitoring.drift import (
    DRIFT_MONITOR_VERSION,
    DetectorResult,
    DriftMonitor,
    DriftReport,
    DriftState,
    DriftThresholds,
    ReferenceDistribution,
    expected_calibration_error,
)

__all__ = [
    "DRIFT_MONITOR_VERSION",
    "DetectorResult",
    "DriftMonitor",
    "DriftReport",
    "DriftState",
    "DriftThresholds",
    "ReferenceDistribution",
    "expected_calibration_error",
]
