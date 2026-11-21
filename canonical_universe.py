"""Canonical trading-universe manager.

The canonical ``universe.txt`` is the single authoritative persisted source of
new-entry eligibility. The migrated real-time market scanner is the sole *file*
publisher (it atomically rewrites ``universe.txt`` via
:func:`universe.write_canonical_universe` only when membership changes, and
heartbeats the file mtime otherwise). This manager is the single in-memory
*consumer/distributor*: it loads the file, parses it through the canonical
loader, and hands immutable, versioned snapshots to every engine component
(stream subscription, strategies, portfolio optimisation, replacement) so they
never re-read or re-parse the file on their hot paths and always observe a
consistent membership.

Design contract (see project spec §8, §11):

  * snapshots are immutable (``frozen`` dataclass, ``tuple`` symbols);
  * the ``version`` bumps *only* on a genuine membership change (an ordering
    change counts -- descending rank is meaningful); an unchanged republish or
    an mtime-only heartbeat never bumps it;
  * loading fails closed (a missing/empty file raises on construction, matching
    :func:`universe.load_canonical_universe`), but a *refresh* never raises -- a
    bad republish keeps the last good snapshot rather than blanking the live
    universe;
  * all access is thread-safe (the engine reads while a watcher refreshes).
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import universe

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class UniverseSnapshot:
    """An immutable, versioned view of the canonical trading universe.

    Fields mirror the project spec's ``UniverseSnapshot`` and are extended with
    ``source_path`` for provenance. ``symbols`` preserves the published
    (descending-rank) order.
    """

    version: int
    published_timestamp_ns: int
    symbols: tuple[str, ...]
    ranking_version: str
    source_feed: str
    source_path: str = ""
    _symbol_set: frozenset[str] = field(default_factory=frozenset, repr=False, compare=False)

    def contains(self, symbol: str) -> bool:
        return universe.normalize_symbol(symbol) in self._symbol_set

    def __len__(self) -> int:  # convenience
        return len(self.symbols)


class CanonicalUniverseManager:
    """Loads and distributes immutable, versioned universe snapshots."""

    def __init__(
        self,
        path: str | Path,
        *,
        source_feed: str = "iex",
        ranking_version: str = "0",
    ) -> None:
        self._path = Path(path)
        self._source_feed = source_feed
        self._lock = threading.RLock()
        # Initial load fails closed (raises RuntimeError on missing/empty file).
        symbols = tuple(universe.load_canonical_universe(self._path))
        self._snapshot = UniverseSnapshot(
            version=1,
            published_timestamp_ns=time.time_ns(),
            symbols=symbols,
            ranking_version=ranking_version,
            source_feed=source_feed,
            source_path=str(self._path),
            _symbol_set=frozenset(symbols),
        )

    # ------------------------------------------------------------------ readers
    def current(self) -> UniverseSnapshot:
        """Return the current immutable snapshot (thread-safe)."""
        with self._lock:
            return self._snapshot

    def symbols(self) -> tuple[str, ...]:
        return self.current().symbols

    def version(self) -> int:
        return self.current().version

    def contains(self, symbol: str) -> bool:
        return self.current().contains(symbol)

    # ------------------------------------------------------------------ refresh
    def refresh_from_file(self, *, ranking_version: str | None = None) -> UniverseSnapshot | None:
        """Re-read ``universe.txt``; publish a new snapshot iff membership changed.

        Returns the new snapshot on a genuine membership/order change, or
        ``None`` when nothing changed or the file could not be read as a valid
        universe (in which case the last good snapshot is retained). Never
        raises -- a scanner that writes a transiently-bad file must not take
        down the engine or blank the live entry universe.
        """
        try:
            symbols = tuple(universe.load_canonical_universe(self._path))
        except Exception as exc:  # noqa: BLE001 - protective: keep last good universe
            logger.warning("universe refresh failed (keeping last good): %s", exc)
            return None

        with self._lock:
            if symbols == self._snapshot.symbols:
                return None  # unchanged membership/order -> no version bump
            new = UniverseSnapshot(
                version=self._snapshot.version + 1,
                published_timestamp_ns=time.time_ns(),
                symbols=symbols,
                ranking_version=(
                    ranking_version if ranking_version is not None
                    else self._snapshot.ranking_version
                ),
                source_feed=self._source_feed,
                source_path=str(self._path),
                _symbol_set=frozenset(symbols),
            )
            self._snapshot = new
            logger.info(
                "canonical universe v%d: %d symbols", new.version, len(new.symbols)
            )
            return new
