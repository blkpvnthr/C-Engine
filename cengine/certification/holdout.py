"""Sealed certification holdout (§5 / §20).

The holdout is the most recent chronological slice of each symbol's genuine data, reserved and
**never used for tuning, model selection, or threshold fitting**. It is derived deterministically
from the data span via the existing :func:`oos_windows` (train 60% / validation 20% / holdout
20%, embargoed), so the boundary is a pure function of the data — it cannot be nudged to flatter
a result. Each holdout is fingerprinted (symbol + boundaries + policy) and the full set forms a
``HoldoutManifest`` snapshot taken at certification time.

A process guard (:class:`HoldoutSeal`) records every holdout access with a reason. Legitimate
certification touches the holdout exactly once per symbol, to SCORE a frozen candidate — never
to choose one. The access log is part of the evidence: it lets the final report prove the
holdout was scored, not tuned on.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from cengine.certification.contracts import safe_ticker, stable_id

HOLDOUT_VERSION = "certification-holdout-1.0.0"
_HOLDOUT_POLICY = {"train_frac": 0.6, "validation_frac": 0.2, "embargo_days": 3}


@dataclass
class SymbolHoldout:
    symbol: str
    train: tuple
    validation: tuple
    holdout: tuple
    embargo_days: int
    fingerprint: str

    def to_dict(self) -> dict:
        return {"symbol": self.symbol, "train": list(self.train),
                "validation": list(self.validation), "holdout": list(self.holdout),
                "embargo_days": self.embargo_days, "fingerprint": self.fingerprint}


def derive_symbol_holdout(symbol: str, *, config=None) -> Optional[SymbolHoldout]:
    """Deterministically derive the sealed windows for one symbol from its data span."""
    from cengine.optimization.validation import oos_windows
    sym = safe_ticker(symbol)
    w = oos_windows(config, sym, train_frac=_HOLDOUT_POLICY["train_frac"],
                    validation_frac=_HOLDOUT_POLICY["validation_frac"],
                    embargo_days=_HOLDOUT_POLICY["embargo_days"])
    if w is None:
        return None
    fp = stable_id(sym, w.train, w.validation, w.holdout, tuple(sorted(_HOLDOUT_POLICY.items())))
    return SymbolHoldout(symbol=sym, train=w.train, validation=w.validation,
                         holdout=w.holdout, embargo_days=w.embargo_days, fingerprint=fp)


@dataclass
class HoldoutManifest:
    """Snapshot of every sealed holdout at certification time."""

    symbols: list = field(default_factory=list)      # list[SymbolHoldout]
    policy: dict = field(default_factory=lambda: dict(_HOLDOUT_POLICY))
    version: str = HOLDOUT_VERSION

    def get(self, symbol: str) -> Optional[SymbolHoldout]:
        s = safe_ticker(symbol)
        return next((h for h in self.symbols if h.symbol == s), None)

    def fingerprint(self) -> str:
        return stable_id(tuple(h.fingerprint for h in self.symbols),
                         tuple(sorted(self.policy.items())))

    def to_dict(self) -> dict:
        return {"version": self.version, "policy": self.policy,
                "manifest_fingerprint": self.fingerprint(),
                "symbols": [h.to_dict() for h in self.symbols]}


def build_holdout_manifest(symbols: list, *, config=None) -> HoldoutManifest:
    out = []
    for sym in symbols:
        h = derive_symbol_holdout(sym, config=config)
        if h is not None:
            out.append(h)
    return HoldoutManifest(symbols=out)


class HoldoutSeal:
    """Process-level access log for the sealed holdout.

    Not a security boundary (a determined caller can read the data directly); it is an
    *evidence* mechanism. Certification code accesses the holdout only through :meth:`open`,
    which records the access, so the final report can demonstrate the holdout was scored once
    per symbol for certification and never used for selection/tuning.
    """

    def __init__(self) -> None:
        self._accesses: list = []

    def open(self, symbol: str, reason: str) -> tuple:
        """Record a holdout access and return the holdout window for ``symbol``."""
        h = derive_symbol_holdout(symbol)
        if h is None:
            self._accesses.append({"symbol": safe_ticker(symbol), "reason": reason,
                                   "granted": False})
            raise ValueError(f"no holdout window for {symbol} (insufficient data span)")
        self._accesses.append({"symbol": h.symbol, "reason": reason, "granted": True,
                               "window": list(h.holdout), "fingerprint": h.fingerprint})
        return h.holdout

    @property
    def accesses(self) -> list:
        return list(self._accesses)

    def audit(self) -> dict:
        granted = [a for a in self._accesses if a.get("granted")]
        per_symbol: dict = {}
        for a in granted:
            per_symbol[a["symbol"]] = per_symbol.get(a["symbol"], 0) + 1
        multi = {s: n for s, n in per_symbol.items() if n > 1}
        return {"total_accesses": len(self._accesses), "granted": len(granted),
                "symbols_touched": len(per_symbol),
                "symbols_touched_more_than_once": multi,
                "reasons": sorted({a["reason"] for a in self._accesses})}
