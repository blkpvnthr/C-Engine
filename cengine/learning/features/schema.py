"""Versioned feature schema (Milestone 6, §5/§18).

A FeatureSchema pins the exact ordered feature columns a model was trained on so that
inference-time tensors are built identically and an incompatible schema is rejected
rather than silently mis-fed. Feature names mirror the engine's existing technical
vocabulary (DI/ADX/ATR/VWAP/SMA/BB + returns/momentum/volatility/volume) so lakehouse
features and learning features stay aligned.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

FEATURE_SCHEMA_VERSION = "feature-schema-1.0.0"

# Ordered, look-ahead-free technical features derived from completed bars (§5). Every
# feature at bar t uses ONLY bars <= t; the forward-return label uses bars > t and is a
# target, never an input.
BAR_FEATURE_COLUMNS = (
    "ret_1", "ret_5", "ret_15",
    "mom_10", "mom_20",
    "vol_10", "vol_20",
    "sma_ratio_10", "sma_ratio_20",
    "range_10",
    "rsi_14",
    "volume_ratio_10",
    "vwap_dist",
)


@dataclass(frozen=True)
class FeatureSchema:
    version: str
    columns: tuple
    description: str = ""

    @property
    def n_features(self) -> int:
        return len(self.columns)

    def fingerprint(self) -> str:
        """Stable hash over (version, ordered columns) — the compatibility key."""
        basis = self.version + "|" + ",".join(self.columns)
        return hashlib.sha256(basis.encode()).hexdigest()[:16]

    def is_compatible(self, other: "FeatureSchema") -> bool:
        return self.fingerprint() == other.fingerprint()

    def to_dict(self) -> dict:
        return {"version": self.version, "columns": list(self.columns),
                "n_features": self.n_features, "fingerprint": self.fingerprint(),
                "description": self.description}


def bar_feature_schema() -> FeatureSchema:
    return FeatureSchema(
        version=FEATURE_SCHEMA_VERSION, columns=BAR_FEATURE_COLUMNS,
        description="look-ahead-free intraday bar technical features (returns/momentum/"
                    "volatility/SMA ratios/range/RSI/volume/VWAP distance)")
