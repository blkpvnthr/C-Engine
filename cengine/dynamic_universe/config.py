from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class UniverseRotationConfig:
    """Allocation targets and cadences for dynamic universe rotation.

    Slot counts are *targets*, not rigid quotas: unused opportunity slots are
    filled with eligible baseline securities, and protected positions/orders
    always take precedence (see the rotation scheduler).
    """

    active_universe_size: int = 30
    baseline_target_slots: int = 20
    opportunity_target_slots: int = 10
    refresh_interval_seconds: float = 60.0
    opportunity_refresh_seconds: float = 300.0
    baseline_max_revisit_seconds: float = 300.0
    candidate_min_residence_seconds: float = 180.0
    protect_open_positions: bool = True
    protect_pending_orders: bool = True
    hysteresis_margin: float = 0.05

    @classmethod
    def from_env(
        cls, getenv: Callable[[str, str | None], str | None] = os.getenv
    ) -> "UniverseRotationConfig":
        def _int(key: str, default: int) -> int:
            raw = getenv(key, None)
            return int(raw) if raw not in (None, "") else default

        def _float(key: str, default: float) -> float:
            raw = getenv(key, None)
            return float(raw) if raw not in (None, "") else default

        def _bool(key: str, default: bool) -> bool:
            raw = getenv(key, None)
            if raw in (None, ""):
                return default
            return str(raw).strip().lower() not in ("0", "false", "no")

        cfg = cls(
            active_universe_size=_int("CENGINE_UNIVERSE_ACTIVE_SIZE", 30),
            baseline_target_slots=_int("CENGINE_UNIVERSE_BASELINE_SLOTS", 20),
            opportunity_target_slots=_int("CENGINE_UNIVERSE_OPPORTUNITY_SLOTS", 10),
            refresh_interval_seconds=_float("CENGINE_UNIVERSE_REFRESH_S", 60.0),
            opportunity_refresh_seconds=_float("CENGINE_UNIVERSE_OPP_REFRESH_S", 300.0),
            baseline_max_revisit_seconds=_float("CENGINE_UNIVERSE_REVISIT_S", 300.0),
            candidate_min_residence_seconds=_float("CENGINE_UNIVERSE_MIN_RESIDENCE_S", 180.0),
            protect_open_positions=_bool("CENGINE_UNIVERSE_PROTECT_POSITIONS", True),
            protect_pending_orders=_bool("CENGINE_UNIVERSE_PROTECT_ORDERS", True),
            hysteresis_margin=_float("CENGINE_UNIVERSE_HYSTERESIS", 0.05),
        )
        cfg.validate()
        return cfg

    def validate(self) -> None:
        if self.active_universe_size < 1:
            raise ValueError("active_universe_size must be >= 1")
        for name in (
            "refresh_interval_seconds",
            "opportunity_refresh_seconds",
            "baseline_max_revisit_seconds",
            "candidate_min_residence_seconds",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be >= 0")
        if self.baseline_target_slots < 0 or self.opportunity_target_slots < 0:
            raise ValueError("slot targets must be >= 0")
        if not (0.0 <= self.hysteresis_margin < 1.0):
            raise ValueError("hysteresis_margin must be in [0, 1)")
