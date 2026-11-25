"""DynamicUniverseManager — orchestrates one rotation cycle and the periodic loop.

This is the selection/rotation *owner*. It does NOT own a market-data socket and
never opens a second connection. Each cycle it:

  1. gathers protected symbols (open positions + pending orders + incomplete
     executions + unresolved reconciliation) from an injected source,
  2. pulls the current opportunity ranking (injected; wired to the scanner),
  3. runs the pure :class:`UniverseRotationScheduler`,
  4. writes the resulting active window (<=30, protection-inclusive) to
     ``universe.txt`` as the SOLE writer — the existing subscription owner
     (``UniverseSubscriptionSync`` / ``AdaptiveSubscriptionController``) then
     reconciles the live stream to it without a restart,
  5. persists lifecycle/active state for restart recovery,
  6. emits universe-selection decision events (optional; wired in a later phase),
  7. exposes dashboard stats.

All dependencies are injected so the orchestration is unit-testable without the
live engine. ``now_s`` defaults to wall-clock (epoch seconds) so residence /
revisit durations survive restarts via the persisted state.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Protocol, Sequence

import universe

from .baseline import BaselineRegistry
from .config import UniverseRotationConfig
from .layers import UniverseLayers
from .rotation import RankedSymbol, RotationResult, UniverseRotationScheduler
from .state import load_state, save_state

logger = logging.getLogger("dynamic_universe")

RankingSource = Callable[[], Sequence[RankedSymbol]]
ProtectionSource = Callable[[], Sequence[str]]


@dataclass(frozen=True)
class ActiveUniverseSnapshot:
    """Immutable active-universe membership with a monotonic generation id.

    The ``DynamicUniverseManager`` is the sole authority that produces these; the
    existing ``UniversePublisher`` is the sole writer that consumes ``.active``
    and publishes it to universe.txt + live_context.json from THIS one snapshot,
    so both artifacts always share identical membership (spec §6 step 12,
    integration authority-separation).
    """

    active: tuple[str, ...]
    generation: int
    created_s: float
    degraded: bool
    protected: tuple[str, ...]
    active_baseline: tuple[str, ...]
    active_opportunity: tuple[str, ...]
    stale_baseline: tuple[str, ...]
    reasons: dict[str, str]


class DecisionEmitter(Protocol):
    def __call__(
        self,
        result: RotationResult,
        layers: UniverseLayers,
        scores: dict,
        now_s: float,
    ) -> None: ...


class DynamicUniverseManager:
    def __init__(
        self,
        *,
        baseline: BaselineRegistry,
        config: UniverseRotationConfig,
        ranking_source: RankingSource,
        protection_source: ProtectionSource,
        references: Sequence[str] = (),
        event_emitter: Optional[DecisionEmitter] = None,
        state_path: Optional[str | Path] = None,
        now_fn: Callable[[], float] = time.time,
        log_fn: Optional[Callable[[str], None]] = None,
    ) -> None:
        self._baseline = baseline
        self._cfg = config
        self._ranking_source = ranking_source
        self._protection_source = protection_source
        self._references = tuple(universe.normalize_symbol(s) for s in references if universe.normalize_symbol(s))
        self._emit = event_emitter
        self._state_path = Path(state_path) if state_path else None
        self._now_fn = now_fn
        self._log = log_fn or (lambda m: logger.info(m))
        self._scheduler = UniverseRotationScheduler(config)
        self._lock = threading.RLock()

        if self._state_path is not None:
            self._tracker, self._previous_active, self._version = load_state(
                self._state_path, config
            )
        else:
            from .lifecycle import LifecycleTracker

            self._tracker = LifecycleTracker(config)
            self._previous_active = ()
            self._version = 0

        self._last_result: Optional[RotationResult] = None
        self._last_rotation_s: float = 0.0
        self._layers: Optional[UniverseLayers] = None
        self._snapshot: Optional[ActiveUniverseSnapshot] = None

    # ------------------------------------------------------------------ one pass
    def rotate_once(self, now_s: Optional[float] = None) -> RotationResult:
        with self._lock:
            return self._rotate_locked(now_s)

    def maybe_rotate(self, now_s: Optional[float] = None) -> Optional[ActiveUniverseSnapshot]:
        """Rotate only if ``refresh_interval_seconds`` has elapsed; else return the
        cached snapshot. Lets the ~3s publish heartbeat pull the current membership
        without forcing a rotation every tick (spec §6 / heartbeat contract)."""
        with self._lock:
            now = float(now_s) if now_s is not None else float(self._now_fn())
            due = (
                self._snapshot is None
                or (now - self._last_rotation_s) >= self._cfg.refresh_interval_seconds
            )
            if due:
                self._rotate_locked(now)
            return self._snapshot

    def current_active(self) -> Optional[ActiveUniverseSnapshot]:
        """The latest immutable active snapshot, or None before the first rotation
        (publishers must treat None as fail-closed: do not publish membership)."""
        with self._lock:
            return self._snapshot

    def _rotate_locked(self, now_s: Optional[float] = None) -> RotationResult:
        now = float(now_s) if now_s is not None else float(self._now_fn())

        baseline_syms = self._baseline.symbols()
        baseline_set = {universe.normalize_symbol(s) for s in baseline_syms}

        try:
            ranked = list(self._ranking_source() or [])
        except Exception as exc:  # noqa: BLE001 - fail-safe: no ranking -> baseline only
            logger.warning("dynamic-universe: ranking source failed: %s", exc)
            ranked = []
        ranked_by = {universe.normalize_symbol(r.symbol): r for r in ranked}

        baseline_ranked = [
            ranked_by.get(
                universe.normalize_symbol(s),
                RankedSymbol(universe.normalize_symbol(s), 0.0, False),
            )
            for s in baseline_syms
        ]
        opportunities = [
            r for r in ranked if universe.normalize_symbol(r.symbol) not in baseline_set
        ]

        try:
            protected = list(self._protection_source() or [])
        except Exception as exc:  # noqa: BLE001 - fail-safe: never drop protection by crashing
            logger.warning("dynamic-universe: protection source failed: %s", exc)
            protected = []

        result = self._scheduler.rotate(
            baseline=baseline_ranked,
            opportunities=opportunities,
            protected=protected,
            references=list(self._references),
            previous_active=self._previous_active,
            tracker=self._tracker,
            now_s=now,
        )

        # The manager is the SELECTION authority only; it never writes files.
        # It publishes an immutable snapshot that the existing UniversePublisher
        # (sole writer) consumes. An empty active window yields no snapshot update
        # (fail-closed: the publisher keeps the last-good file importable).
        self._previous_active = result.active
        self._version += 1
        self._last_result = result
        self._last_rotation_s = now
        self._layers = UniverseLayers.build(
            baseline=baseline_syms,
            opportunity=[r.symbol for r in opportunities],
            held=protected,
            pending=(),
            active=result.active,
            version=self._version,
        )
        if result.active:
            self._snapshot = ActiveUniverseSnapshot(
                active=result.active,
                generation=self._version,
                created_s=now,
                degraded=result.degraded,
                protected=result.protected,
                active_baseline=result.active_baseline,
                active_opportunity=result.active_opportunity,
                stale_baseline=result.stale_baseline,
                reasons=dict(result.reasons),
            )

        if result.added or result.removed or result.degraded:
            self._log(
                f"[dynamic-universe] rotate v{self._version}: "
                f"+{list(result.added)} -{list(result.removed)} "
                f"active={len(result.active)}/{self._cfg.active_universe_size}"
                + (" DEGRADED" if result.degraded else "")
            )
        # Coverage reporting (spec §10): baseline symbols that could not be
        # evaluated within the configured revisit interval this cycle.
        if result.stale_baseline:
            self._log(
                f"[dynamic-universe] COVERAGE: {len(result.stale_baseline)} baseline "
                f"symbol(s) past revisit deadline not placed this cycle: "
                f"{list(result.stale_baseline)}"
            )

        if self._state_path is not None:
            try:
                save_state(
                    self._state_path,
                    tracker=self._tracker,
                    previous_active=self._previous_active,
                    version=self._version,
                )
            except Exception as exc:  # noqa: BLE001 - persistence is best-effort
                logger.warning("dynamic-universe: state persist failed: %s", exc)

        if self._emit is not None and self._layers is not None:
            try:
                scores = {
                    universe.normalize_symbol(r.symbol): r.score
                    for r in (baseline_ranked + opportunities)
                }
                self._emit(result, self._layers, scores, now)
            except Exception as exc:  # noqa: BLE001 - telemetry must never break rotation
                logger.warning("dynamic-universe: event emit failed: %s", exc)

        return result

    # ------------------------------------------------------------------- loop
    async def run(self, stop: asyncio.Event) -> None:
        self._log(
            f"[dynamic-universe] active: rotating the universe every "
            f"{self._cfg.refresh_interval_seconds:.0f}s"
        )
        while not stop.is_set():
            try:
                self.rotate_once()
            except Exception as exc:  # noqa: BLE001 - the loop must never die
                logger.error("dynamic-universe rotate iteration error: %s", exc)
            try:
                await asyncio.wait_for(
                    stop.wait(), timeout=self._cfg.refresh_interval_seconds
                )
            except asyncio.TimeoutError:
                pass

    # ------------------------------------------------------------- observability
    @property
    def layers(self) -> Optional[UniverseLayers]:
        return self._layers

    @property
    def last_result(self) -> Optional[RotationResult]:
        return self._last_result

    def dashboard_stats(self) -> dict[str, Any]:
        """Return the [DYNAMIC UNIVERSE MANAGER] counters (never raises)."""
        res = self._last_result
        layers = self._layers
        baseline_n = len(self._baseline.symbols())
        if res is None or layers is None:
            return {
                "baseline": baseline_n,
                "dynamic_candidates": 0,
                "master": baseline_n,
                "active": 0,
                "active_cap": self._cfg.active_universe_size,
                "protected": 0,
                "active_baseline": 0,
                "active_opportunity": 0,
                "last_rotation_s": 0.0,
                "coverage": "PENDING",
            }
        coverage = "HEALTHY"
        if res.degraded:
            coverage = "DEGRADED"
        elif res.stale_baseline:
            coverage = "STALE_BASELINE"
        return {
            "baseline": baseline_n,
            "dynamic_candidates": len(layers.opportunity),
            "master": len(layers.master),
            "active": len(res.active),
            "active_cap": self._cfg.active_universe_size,
            "protected": len(res.protected),
            "active_baseline": len(res.active_baseline),
            "active_opportunity": len(res.active_opportunity),
            "pending_rotation": len(res.added) + len(res.removed),
            "stale_baseline": len(res.stale_baseline),
            "stale_baseline_symbols": list(res.stale_baseline),
            "last_rotation_s": self._last_rotation_s,
            "next_rotation_s": self._last_rotation_s + self._cfg.refresh_interval_seconds,
            "generation": self._version,
            "coverage": coverage,
        }
