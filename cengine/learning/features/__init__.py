"""Point-in-time feature engineering for the learning layer (§5 / addendum §5)."""

from cengine.learning.features.builder import (
    BarFeatureMatrix,
    build_bar_features,
    gold_training_frame,
)
from cengine.learning.features.schema import (
    BAR_FEATURE_COLUMNS,
    FEATURE_SCHEMA_VERSION,
    FeatureSchema,
    bar_feature_schema,
)

__all__ = [
    "BAR_FEATURE_COLUMNS",
    "FEATURE_SCHEMA_VERSION",
    "BarFeatureMatrix",
    "FeatureSchema",
    "bar_feature_schema",
    "build_bar_features",
    "gold_training_frame",
]
