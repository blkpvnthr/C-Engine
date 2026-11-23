"""Learning dataset profiling, eligibility gating and point-in-time certification (§3/§5)."""

from cengine.learning.datasets.certification import (
    DATASET_CERT_VERSION,
    DatasetCertificate,
    certify_training_dataset,
)
from cengine.learning.datasets.eligibility import (
    ELIGIBILITY_VERSION,
    DatasetProfile,
    DatasetProfiler,
    EligibilityState,
    LearningTaskSpec,
    assess_task,
    default_task_specs,
)

__all__ = [
    "DATASET_CERT_VERSION",
    "ELIGIBILITY_VERSION",
    "DatasetCertificate",
    "DatasetProfile",
    "DatasetProfiler",
    "EligibilityState",
    "LearningTaskSpec",
    "assess_task",
    "certify_training_dataset",
    "default_task_specs",
]
