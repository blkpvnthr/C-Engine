"""Persistent rotation state for restart recovery (spec §11).

Serializes the lifecycle tracker + the last active window to an atomically-written
JSON file so that, after an engine restart, residence/revisit clocks and the
previous-active set are restored rather than reset. Fail-safe: a missing or
unreadable state file yields a fresh tracker (never raises on load).
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from dataclasses import asdict
from pathlib import Path

from .config import UniverseRotationConfig
from .lifecycle import CandidateState, LifecycleState, LifecycleTracker

logger = logging.getLogger(__name__)

STATE_VERSION = 1


def save_state(
    path: str | Path,
    *,
    tracker: LifecycleTracker,
    previous_active: tuple[str, ...],
    version: int,
) -> None:
    """Atomically persist the tracker + previous active window."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_version": STATE_VERSION,
        "universe_version": int(version),
        "previous_active": list(previous_active),
        "candidates": [
            {**asdict(cs), "state": cs.state.value}
            for cs in tracker.all().values()
        ],
    }
    fd, tmp = tempfile.mkstemp(dir=str(p.parent), prefix=".du_state.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, p)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def load_state(
    path: str | Path, config: UniverseRotationConfig
) -> tuple[LifecycleTracker, tuple[str, ...], int]:
    """Restore ``(tracker, previous_active, version)``; fresh tracker if absent/bad."""
    tracker = LifecycleTracker(config)
    p = Path(path)
    if not p.exists():
        return tracker, (), 0
    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
        states = [
            CandidateState(
                symbol=c["symbol"],
                state=LifecycleState(c["state"]),
                is_baseline=bool(c["is_baseline"]),
                last_eval_s=float(c["last_eval_s"]),
                entered_active_s=(
                    None if c.get("entered_active_s") is None else float(c["entered_active_s"])
                ),
                rotated_out_s=(
                    None if c.get("rotated_out_s") is None else float(c["rotated_out_s"])
                ),
                last_score=float(c.get("last_score", 0.0)),
                reason=str(c.get("reason", "")),
            )
            for c in payload.get("candidates", [])
        ]
        tracker.restore(states)
        prev = tuple(str(s) for s in payload.get("previous_active", []))
        version = int(payload.get("universe_version", 0))
        return tracker, prev, version
    except Exception as exc:  # noqa: BLE001 - fail-safe: fresh state on any error
        logger.warning("dynamic-universe state load failed (fresh start): %s", exc)
        return LifecycleTracker(config), (), 0
