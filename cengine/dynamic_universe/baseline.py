from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import universe

logger = logging.getLogger(__name__)

# Permanent baseline (operator-managed). Exactly the spec-30. UPRO is NOT here.
BASELINE_UNIVERSE: tuple[str, ...] = (
    "ANT", "GPN", "DVA", "HUM", "UNH", "HCA", "UHS", "GIS", "WMT", "COST",
    "NVDA", "AMD", "AVGO", "AAPL", "MU", "TSLA", "SPY", "QQQ", "DIA", "IWM",
    "XLP", "XLV", "GLD", "SQQQ", "SOXS", "SPXU", "SDOW", "TZA", "TQQQ", "SOXL",
)

BASELINE_TXT = universe.PROJECT_ROOT / "baseline_universe.txt"


@dataclass(frozen=True)
class BaselineChange:
    timestamp_s: float
    actor: str
    added: tuple[str, ...]
    removed: tuple[str, ...]
    resulting: tuple[str, ...]


def _clean(symbols: Iterable[str]) -> tuple[str, ...]:
    """Normalize/dedupe/validate via the shared pure universe pipeline.

    ``max_symbols=0`` means 'no cap' (baseline size is operator-chosen).
    """
    result = universe.build_universe(list(symbols), max_symbols=0)
    return tuple(result.symbols)


class BaselineRegistry:
    """Authoritative, permanent baseline membership.

    The ONLY writer of ``baseline_universe.txt``. The Opportunity Finder must
    never call :meth:`apply_operator_update`. Permanence applies to *membership*,
    not to continuous subscription: a baseline symbol may rotate out of the
    active window but is always eligible for reinsertion.
    """

    def __init__(self, path: str | Path = BASELINE_TXT) -> None:
        self._path = Path(path)
        self._lock = threading.RLock()
        self._audit: list[BaselineChange] = []
        if not self._path.exists():
            # First run: seed with the spec-30 constant.
            universe.write_canonical_universe(BASELINE_UNIVERSE, self._path)
        self._symbols = _clean(universe.load_canonical_universe(self._path))
        if not self._symbols:
            # Defensive: an existing-but-unusable file falls back to the constant.
            self._symbols = _clean(BASELINE_UNIVERSE)

    @property
    def path(self) -> Path:
        return self._path

    def symbols(self) -> tuple[str, ...]:
        with self._lock:
            return self._symbols

    def contains(self, symbol: str) -> bool:
        return universe.normalize_symbol(symbol) in set(self.symbols())

    is_baseline = contains

    def reload(self) -> bool:
        """Re-read the file; keep last-good on invalid. Returns True iff changed."""
        try:
            fresh = _clean(universe.load_canonical_universe(self._path))
        except Exception as exc:  # noqa: BLE001 - keep last good
            logger.warning("baseline reload failed (keeping last good): %s", exc)
            return False
        if not fresh:
            return False
        with self._lock:
            if fresh == self._symbols:
                return False
            self._symbols = fresh
            return True

    def apply_operator_update(
        self,
        symbols: Sequence[str],
        *,
        actor: str = "operator",
        now_s: float | None = None,
    ) -> BaselineChange:
        """Validate + atomically rewrite the baseline. Reject-invalid-keep-last-good."""
        cleaned = _clean(symbols)
        if not cleaned:
            raise ValueError(
                "refusing baseline update with 0 usable symbols "
                "(last-good baseline left untouched)"
            )
        ts = float(now_s) if now_s is not None else 0.0
        with self._lock:
            old = set(self._symbols)
            new = set(cleaned)
            # Atomic file write BEFORE mutating in-memory state; on failure the
            # exception propagates and in-memory state is unchanged.
            universe.write_canonical_universe(cleaned, self._path)
            change = BaselineChange(
                timestamp_s=ts,
                actor=actor,
                added=tuple(s for s in cleaned if s not in old),
                removed=tuple(s for s in self._symbols if s not in new),
                resulting=cleaned,
            )
            self._symbols = cleaned
            self._audit.append(change)
            logger.info(
                "baseline updated by %s: +%s -%s (now %d)",
                actor, change.added, change.removed, len(cleaned),
            )
            return change

    def audit_log(self) -> tuple[BaselineChange, ...]:
        with self._lock:
            return tuple(self._audit)
