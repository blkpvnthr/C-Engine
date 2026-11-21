"""Monitor-based -0.5% loss removal + dynamic replacement selection.

Pure decision logic only — the live engine (trade.py) emits the actual closing
SELL through its un-vetoable ``exits`` funnel and the replacement BUY through its
``rebalance`` funnel + shared risk pipeline, so this module creates NO competing
optimizer and places NO orders.

Two responsibilities:
  * :func:`find_loss_exits` — detect every open position whose broker-reported
    ``unrealized_plpc`` (decimal fraction) is <= the threshold (-0.005 = -0.5%).
    Detection is un-overridable: no bullish signal, strategy conflict, or
    minimum-hold can suppress it (the caller routes the result straight to the
    exit funnel).
  * :func:`select_replacement` — walk the validated, descending-rank universe and
    pick the highest-ranked candidate that passes every replacement gate and is
    not excluded (held / pending / no-reentry / just-closed). No eligible
    candidate -> ``None`` (remain in cash; never force a replacement).

Designed to later serve as the *secondary* backstop once the broker-native
resting stop lands on ``amps-v6-loss-removal-optimizer``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Optional

from market_scan.config import MarketScanConfig
from market_scan.ranking import MarketContext, passes_liquidity

# Minimum normalized seller-exhaustion to accept a replacement entry.
SELLER_EXHAUSTION_MIN = 0.5


@dataclass
class LossExit:
    symbol: str
    unrealized_plpc: float
    reason: str = "loss_removal_-0.5pct"


@dataclass
class ReplacementEligibility:
    seller_exhaustion: bool = False
    positive_momentum: bool = False
    di_plus_gt_minus: bool = False
    above_bb_mid: bool = False
    above_vwap: bool = False
    liquidity_ok: bool = False
    regime_aligned: bool = False
    risk_ok: bool = True            # existing engine risk gates applied downstream
    require_seller_exhaustion: bool = True  # whether seller_exhaustion is MANDATORY
    reason: str = ""

    def _required_checks(self) -> dict[str, bool]:
        checks = {
            "positive_momentum": self.positive_momentum,
            "di_plus_gt_minus": self.di_plus_gt_minus,
            "above_bb_mid": self.above_bb_mid,
            "above_vwap": self.above_vwap,
            "liquidity_ok": self.liquidity_ok,
            "regime_aligned": self.regime_aligned,
            "risk_ok": self.risk_ok,
        }
        if self.require_seller_exhaustion:
            checks["seller_exhaustion"] = self.seller_exhaustion
        return checks

    @property
    def eligible(self) -> bool:
        return all(self._required_checks().values())

    @property
    def scanner_side_eligible(self) -> bool:
        """The subset of gates the SCANNER can evaluate from its features:
        momentum + liquidity + regime (+ seller exhaustion if required). The
        directional gates (DI+ > DI-, > BB-mid, > VWAP) are deferred to the engine
        (confirm_buy_direction) at order time, where fresh intraday bars exist --
        so they are never duplicated or approximated scanner-side."""
        checks = [self.positive_momentum, self.liquidity_ok, self.regime_aligned, self.risk_ok]
        if self.require_seller_exhaustion:
            checks.append(self.seller_exhaustion)
        return all(checks)

    def failing(self) -> list[str]:
        return [k for k, v in self._required_checks().items() if not v]


def _as_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        f = float(x)
        return None if math.isnan(f) else f
    except (TypeError, ValueError):
        return None


def find_loss_exits(positions: Iterable[Any], threshold: float = -0.005) -> list[LossExit]:
    """Return a LossExit for every position with ``unrealized_plpc <= threshold``.

    ``unrealized_plpc`` is a decimal fraction (-0.005 == -0.5%), matching the
    Alpaca/engine convention. Positions with missing/NaN P/L are skipped (never
    force-closed on bad data).
    """
    out: list[LossExit] = []
    for p in positions or []:
        plpc = _as_float(getattr(p, "unrealized_plpc", None))
        sym = str(getattr(p, "symbol", "") or "").upper()
        if not sym or plpc is None:
            continue
        if plpc <= threshold:
            out.append(LossExit(symbol=sym, unrealized_plpc=plpc))
    return out


def _direction_sign(symbol: str) -> float:
    import universe
    try:
        return -1.0 if universe.classify_symbol(symbol).leverage < 0 else 1.0
    except Exception:  # noqa: BLE001
        return 1.0


def eligibility_from_features(features: Any, market: MarketContext,
                             config: MarketScanConfig) -> ReplacementEligibility:
    """Compute replacement eligibility from a candidate's CandidateFeatures.

    Required (all): confirmed seller exhaustion, positive momentum, DI+ > DI-,
    price > Bollinger midpoint, price > session VWAP, sufficient liquidity/spread,
    and regime/directional alignment. (Portfolio/position-sizing/execution risk
    gates are enforced by the engine's funnel downstream -> ``risk_ok=True`` here.)
    """
    row = getattr(features, "directional_row", {}) or {}
    snap = getattr(features, "snapshot", None)

    se = 0.0
    momentum_flow = 0.0
    if snap is not None and getattr(snap, "ok", False):
        se = max(_as_float(getattr(snap, "seller_exhaustion", 0.0)) or 0.0,
                 _as_float(getattr(snap, "seller_exhaustion_l1", 0.0)) or 0.0)
        momentum_flow = _as_float(getattr(snap, "buy_pressure", 0.0)) or 0.0

    short_mom = _as_float(row.get("short_mom")) or 0.0
    pdi = _as_float(row.get("pdi"))
    mdi = _as_float(row.get("mdi"))
    dir_sign = _direction_sign(getattr(features, "symbol", ""))

    e = ReplacementEligibility()
    e.require_seller_exhaustion = bool(
        getattr(config, "replacement_require_seller_exhaustion", True))
    e.seller_exhaustion = se >= SELLER_EXHAUSTION_MIN
    e.positive_momentum = (short_mom > 0.0) or (momentum_flow > 0.5)
    e.di_plus_gt_minus = (pdi is not None and mdi is not None and pdi > mdi)
    e.above_bb_mid = (_as_float(getattr(features, "bb_mid_dist", None)) or -1.0) > 0.0
    e.above_vwap = (_as_float(getattr(features, "vwap_dist", None)) or -1.0) > 0.0
    e.liquidity_ok = passes_liquidity(features, config)[0]
    e.regime_aligned = (market.regime_score * dir_sign) >= 0.0
    e.risk_ok = True
    e.reason = "ok" if e.eligible else "fails:" + ",".join(e.failing())
    return e


def select_replacement(
    ranked_universe: Iterable[str],
    *,
    evaluate: Callable[[str], Optional[ReplacementEligibility]],
    exclude: set[str],
    leverage_guard: Optional[Callable[[str], bool]] = None,
) -> tuple[Optional[str], dict]:
    """Pick the highest-ranked eligible replacement, or (None, ...) to hold cash.

    ``evaluate(symbol)`` returns a ReplacementEligibility (or None if the symbol
    has no usable features). ``exclude`` holds held/pending/no-reentry/just-closed
    symbols (prevents pyramiding + immediate re-entry). ``leverage_guard(symbol)``
    may veto a candidate that would create correlated leveraged exposure.
    """
    excl = {s.upper() for s in exclude}
    considered: list[str] = []
    for raw in ranked_universe or []:
        sym = str(raw).upper()
        if sym in excl:
            continue
        considered.append(sym)
        if leverage_guard is not None and leverage_guard(sym):
            continue
        elig = evaluate(sym)
        if elig is not None and elig.eligible:
            return sym, {"reason": "selected", "considered": considered}
    return None, {"reason": "no_eligible_candidate_hold_cash", "considered": considered}
