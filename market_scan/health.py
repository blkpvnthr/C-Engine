"""Scanner health + the resilient dynamic-universe entry gate.

``live_context.json`` is the scanner's published health/metadata record.
:func:`load_dynamic_universe` is what trade.py consults before authorizing new
entries. It is deliberately paranoid and fail-closed: it returns an empty entry
universe (block new entries) unless the scanner is enabled AND healthy AND the
published ``universe.txt`` is fresh, parseable, and mutually consistent with the
metadata. It never raises, never falls back to a static universe, and never
gates exits or risk management — callers use it only for *new* entries/
replacements.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import universe

# Published scanner states (string-valued for JSON round-tripping).
STATUS_HEALTHY = "healthy"
STATUS_UNHEALTHY = "unhealthy"
STATUS_DISABLED = "disabled"
STATUS_STARTING = "starting"


@dataclass
class DynamicUniverseResult:
    """Outcome of a dynamic-universe read.

    ``symbols`` is the entry-eligible universe in descending rank order, or an
    empty list when new entries must be blocked. ``state`` is one of the
    ``STATUS_*`` constants; ``reason`` is a short human-readable explanation for
    logs/dashboard.
    """

    symbols: list[str] = field(default_factory=list)
    state: str = STATUS_UNHEALTHY
    reason: str = ""
    universe_age_seconds: float | None = None
    context_age_seconds: float | None = None

    @property
    def new_entries_permitted(self) -> bool:
        return self.state == STATUS_HEALTHY and bool(self.symbols)


def read_live_context(path: str | Path) -> dict[str, Any] | None:
    """Read live_context.json, returning ``None`` on any problem (never raises)."""
    try:
        text = Path(path).read_text(encoding="utf-8")
        data = json.loads(text)
    except (OSError, ValueError, TypeError):
        return None
    return data if isinstance(data, dict) else None


def write_live_context(path: str | Path, payload: dict[str, Any]) -> None:
    """Atomically write live_context.json (temp + os.replace); never raises on
    serialization of the common payload shapes used by the scanner."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps(payload, default=str, sort_keys=True, indent=2)
    fd, tmp_name = tempfile.mkstemp(
        dir=str(p.parent), prefix=".live_context.", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(body)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, p)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def _file_mtime(path: Path) -> float | None:
    try:
        return os.stat(path).st_mtime
    except OSError:
        return None


_MISMATCH_REASON = "universe.txt/metadata membership mismatch"


def load_dynamic_universe(
    *,
    universe_path: str | Path,
    context_path: str | Path,
    enabled: bool,
    now: float,
    max_age_seconds: float = 15.0,
) -> DynamicUniverseResult:
    """Resolve the entry-eligible universe under the three operational states.

    Returns ``DynamicUniverseResult`` with ``symbols == []`` whenever new
    entries must be blocked (disabled, unhealthy, stale, inconsistent, or
    unreadable). Only returns a non-empty, descending-rank universe when the
    scanner is enabled AND the published metadata reports healthy AND the file
    and metadata are fresh and consistent.

    universe.txt and live_context.json are each written atomically but not
    jointly, so a read that races a publish can momentarily observe one updated
    and the other not (a membership mismatch). That always fails SAFE (blocks
    entries), but to avoid a spurious one-cycle block we re-read both files once
    on a mismatch to ride out the sub-millisecond interleave; a genuine,
    persistent inconsistency still blocks.
    """
    res = _eval_dynamic_universe(
        universe_path=universe_path, context_path=context_path,
        enabled=enabled, now=now, max_age_seconds=max_age_seconds)
    if res.reason == _MISMATCH_REASON:
        res = _eval_dynamic_universe(
            universe_path=universe_path, context_path=context_path,
            enabled=enabled, now=now, max_age_seconds=max_age_seconds)
    return res


def _eval_dynamic_universe(
    *,
    universe_path: str | Path,
    context_path: str | Path,
    enabled: bool,
    now: float,
    max_age_seconds: float = 15.0,
) -> DynamicUniverseResult:
    upath = Path(universe_path)
    cpath = Path(context_path)

    # State 3: disabled -> block entries, NO static fallback.
    if not enabled:
        return DynamicUniverseResult(
            [], STATUS_DISABLED, "MARKET_SCAN_ENABLED=0"
        )

    ctx = read_live_context(cpath)
    if ctx is None:
        return DynamicUniverseResult(
            [], STATUS_UNHEALTHY, f"live_context missing/unreadable: {cpath}"
        )

    if ctx.get("enabled") is False or ctx.get("scanner_status") == STATUS_DISABLED:
        return DynamicUniverseResult([], STATUS_DISABLED, "scanner reports disabled")

    if ctx.get("scanner_status") != STATUS_HEALTHY:
        return DynamicUniverseResult(
            [], STATUS_UNHEALTHY,
            f"scanner_status={ctx.get('scanner_status')!r}",
        )

    # Metadata freshness.
    updated_at = ctx.get("updated_at")
    ctx_age: float | None = None
    if not isinstance(updated_at, (int, float)):
        return DynamicUniverseResult([], STATUS_UNHEALTHY, "metadata missing updated_at")
    ctx_age = now - float(updated_at)
    if ctx_age < 0 or ctx_age > max_age_seconds:
        return DynamicUniverseResult(
            [], STATUS_UNHEALTHY, f"metadata stale (age={ctx_age:.1f}s)",
            context_age_seconds=ctx_age,
        )

    # Market-data freshness is a hard gate: a fresh file with stale ticks is
    # not sufficient evidence.
    if ctx.get("market_data_fresh") is False:
        return DynamicUniverseResult(
            [], STATUS_UNHEALTHY, "market data not fresh", context_age_seconds=ctx_age
        )

    # Universe file freshness (mtime).
    mtime = _file_mtime(upath)
    if mtime is None:
        return DynamicUniverseResult(
            [], STATUS_UNHEALTHY, f"universe.txt missing: {upath}",
            context_age_seconds=ctx_age,
        )
    file_age = now - mtime
    if file_age < 0 or file_age > max_age_seconds:
        return DynamicUniverseResult(
            [], STATUS_UNHEALTHY, f"universe.txt stale (age={file_age:.1f}s)",
            universe_age_seconds=file_age, context_age_seconds=ctx_age,
        )

    # Parse the published universe (fail-closed loader -> swallow, don't crash).
    try:
        symbols = universe.load_canonical_universe(upath)
    except RuntimeError as exc:
        return DynamicUniverseResult(
            [], STATUS_UNHEALTHY, f"universe.txt invalid: {exc}",
            universe_age_seconds=file_age, context_age_seconds=ctx_age,
        )

    # Consistency: file membership must match what the metadata advertises.
    ctx_universe = ctx.get("universe")
    if isinstance(ctx_universe, list):
        if {str(s).upper() for s in ctx_universe} != {s.upper() for s in symbols}:
            return DynamicUniverseResult(
                [], STATUS_UNHEALTHY, _MISMATCH_REASON,
                universe_age_seconds=file_age, context_age_seconds=ctx_age,
            )

    # Even a healthy scanner may have zero entry-eligible candidates (e.g. the
    # published file is only a floor / everything failed eligibility). Block new
    # entries in that case without flagging the scanner unhealthy.
    eligible = ctx.get("entry_eligible_count")
    if not isinstance(eligible, int):
        eligible = len(symbols)
    if eligible < 1:
        return DynamicUniverseResult(
            [], STATUS_HEALTHY, "no eligible entry candidates",
            universe_age_seconds=file_age, context_age_seconds=ctx_age,
        )

    return DynamicUniverseResult(
        symbols, STATUS_HEALTHY, "ok",
        universe_age_seconds=file_age, context_age_seconds=ctx_age,
    )
