from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import universe


def _norm_seq(symbols: Iterable[str]) -> tuple[str, ...]:
    out: list[str] = []
    seen: set[str] = set()
    for s in symbols:
        n = universe.normalize_symbol(s)
        if n and n not in seen:
            seen.add(n)
            out.append(n)
    return tuple(out)


@dataclass(frozen=True)
class UniverseLayers:
    """Immutable snapshot of the four universe layers (spec §3).

    * baseline    — permanent operator-managed membership
    * opportunity — dynamically discovered/ranked candidates
    * master      — dedup union of baseline + opportunity + held + pending
    * active      — current scanning window (<=30, enforced by the scheduler)
    """

    baseline: tuple[str, ...]
    opportunity: tuple[str, ...]
    held: tuple[str, ...]
    pending: tuple[str, ...]
    active: tuple[str, ...]
    version: int

    @classmethod
    def build(
        cls,
        baseline: Iterable[str],
        opportunity: Iterable[str],
        held: Iterable[str],
        pending: Iterable[str],
        active: Iterable[str],
        version: int,
    ) -> "UniverseLayers":
        return cls(
            baseline=_norm_seq(baseline),
            opportunity=_norm_seq(opportunity),
            held=_norm_seq(held),
            pending=_norm_seq(pending),
            active=_norm_seq(active),
            version=int(version),
        )

    @property
    def master(self) -> tuple[str, ...]:
        return _norm_seq(self.baseline + self.opportunity + self.held + self.pending)

    def counts(self) -> dict[str, int]:
        return {
            "baseline": len(self.baseline),
            "opportunity": len(self.opportunity),
            "master": len(self.master),
            "active": len(self.active),
            "held": len(self.held),
            "pending": len(self.pending),
        }
