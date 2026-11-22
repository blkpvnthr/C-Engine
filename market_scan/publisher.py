"""Publish the ranked universe to universe.txt + live_context.json.

The publisher is the single writer of both artifacts. It enforces the two
rules that keep the live engine stable:

  * **universe.txt** is rewritten only when ENTRY membership changes or a forced
    (discovery-cadence) tick asks for it — never on every rank tick — so the
    engine's deterministic scan rotation is not reset by constant reordering.
    On unchanged membership the file's mtime is still bumped (a heartbeat), so a
    live scanner keeps the file "fresh" while a dead one lets it go stale.
  * **live_context.json** is refreshed every tick with the live ranking, health,
    and ``entry_eligible_count`` so the hot-reload gate can verify freshness and
    block entries the instant nothing qualifies — without emptying the file.

Zero eligible candidates leaves the last-good universe.txt in place (so legacy
importers never hit the fail-closed loader) but reports
``entry_eligible_count == 0`` so :func:`market_scan.health.load_dynamic_universe`
blocks new entries.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import universe
from market_scan.config import MarketScanConfig
from market_scan.health import STATUS_HEALTHY, write_live_context


@dataclass
class PublishResult:
    symbols: list[str]          # universe.txt content after this publish
    eligible_count: int         # number of genuine entry candidates (0 blocks entries)
    rewrote_file: bool          # whether universe.txt content was rewritten
    published_at: float         # mtime of the last content rewrite
    added: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)


class UniversePublisher:
    def __init__(
        self,
        universe_path: str | Path,
        context_path: str | Path,
        config: MarketScanConfig,
        mode: str = "paper",
    ) -> None:
        self.universe_path = Path(universe_path)
        self.context_path = Path(context_path)
        self.config = config
        self.mode = mode
        self._last_written: list[str] | None = None
        self._published_at: float = 0.0

    def _seed_from_disk(self) -> None:
        if self._last_written is not None:
            return
        try:
            self._last_written = universe.load_canonical_universe(self.universe_path)
        except RuntimeError:
            self._last_written = []

    def publish(
        self,
        ranked_candidates,
        *,
        now: float,
        force_rewrite: bool = False,
        health: dict[str, Any] | None = None,
    ) -> PublishResult:
        """Publish ``ranked_candidates`` (descending rank). Returns a
        :class:`PublishResult`. Never raises on the normal paths."""
        self._seed_from_disk()
        prev = list(self._last_written or [])

        # Normalize + de-dupe + cap to the target entry-universe size.
        candidates = list(
            dict.fromkeys(
                universe.normalize_symbol(s)
                for s in (ranked_candidates or [])
                if universe.normalize_symbol(s)
            )
        )[: self.config.target_universe_size]
        eligible = len(candidates)

        rewrote = False
        added: list[str] = []
        removed: list[str] = []

        if candidates:
            membership_changed = set(candidates) != set(prev)
            if membership_changed or force_rewrite:
                universe.write_canonical_universe(candidates, self.universe_path)
                self._last_written = candidates
                self._published_at = now
                rewrote = True
                added = [s for s in candidates if s not in set(prev)]
                removed = [s for s in prev if s not in set(candidates)]
        # else: nothing qualifies -> keep the last-good file importable and let
        # entry_eligible_count=0 block entries.

        # Always bump the file mtime to `now` so (a) it stays consistent with the
        # logical clock used by the freshness gate and (b) a live scanner keeps
        # the file fresh while a dead one lets it go stale.
        self._touch(now)

        file_content = list(self._last_written or [])
        self._write_context(
            now=now,
            file_content=file_content,
            ranked=candidates,
            eligible=eligible,
            health=health or {},
            added=added,
            removed=removed,
        )
        return PublishResult(
            symbols=file_content,
            eligible_count=eligible,
            rewrote_file=rewrote,
            published_at=self._published_at,
            added=added,
            removed=removed,
        )

    def _touch(self, now: float) -> None:
        try:
            if self.universe_path.exists():
                os.utime(self.universe_path, (now, now))
        except OSError:
            pass

    def _write_context(
        self,
        *,
        now: float,
        file_content: list[str],
        ranked: list[str],
        eligible: int,
        health: dict[str, Any],
        added: list[str],
        removed: list[str],
    ) -> None:
        payload: dict[str, Any] = {
            "scanner_status": health.get("scanner_status", STATUS_HEALTHY),
            "enabled": True,
            "mode": self.mode,
            "pid": os.getpid(),
            "updated_at": now,
            "universe_published_at": self._published_at,
            "universe": file_content,
            "universe_size": len(file_content),
            "entry_eligible_count": eligible,
            "ranked": ranked,
            "added": added,
            "removed": removed,
            "market_data_fresh": health.get("market_data_fresh", True),
        }
        # Merge any extra health/telemetry fields (regime, subscriptions, etc.).
        for key, value in health.items():
            if key not in payload:
                payload[key] = value
        write_live_context(self.context_path, payload)
