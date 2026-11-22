"""Membership-backed operational discovery — connects the master universe to the
scanner's discovery seam (directive §4/§25.1/§25.5).

The scanner already consumes the GPU feature_provider (screen_market +
technical_features) and the GPU rank_fn (rank_candidates), and publishes through
the sole UniversePublisher. The one remaining legacy dependency is its
``discovery_fn`` (default ``discover_us_listed``). This module supplies a drop-in
replacement that returns the **master universe** from the MembershipRegistry so
discovery operates over the full supported universe — including instruments
outside the former flat scanner subset — rather than a hard-coded list.

It matches the scanner's discovery contract: a callable accepting ``extra=`` and
returning an object with ``.symbols`` / ``.equities`` / ``.etfs`` /
``.source_counts`` / ``.error``. Fail-safe: any error returns an empty result with
``error`` set (the scanner keeps its previous discovered set), never raising into
the loop. Equities vs ETFs are split using the exposure registry (classified ETFs)
with the remainder treated as equities — never by ticker-name heuristics.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

from cengine.exposure_registry import ExposureRegistry
from cengine.membership import MembershipRegistry


def make_membership_discovery_fn(
    membership: MembershipRegistry,
    exposure: Optional[ExposureRegistry] = None,
    *,
    max_symbols: Optional[int] = None,
    tradable_only: bool = True,
) -> Callable[..., Any]:
    """Return a scanner-compatible discovery_fn over the master universe."""
    from market_scan.discovery import DiscoveryResult

    exposure = exposure or ExposureRegistry()

    def discover(*, extra: Any = (), **_kw: Any):
        try:
            base = (membership.tradable_master_symbols() if tradable_only
                    else membership.master_symbols())
            syms = set(base)
            for s in (extra or []):
                s = str(s).upper().strip()
                if s:
                    syms.add(s)
            ordered = sorted(syms)
            if max_symbols and len(ordered) > max_symbols:
                # Keep protected/extra first, then cap the rest deterministically.
                extra_set = {str(s).upper().strip() for s in (extra or [])}
                head = [s for s in ordered if s in extra_set]
                tail = [s for s in ordered if s not in extra_set]
                ordered = sorted(set(head + tail[: max(0, max_symbols - len(head))]))
            etfs = [s for s in ordered if exposure.get(s) is not None]
            equities = [s for s in ordered if exposure.get(s) is None]
            return DiscoveryResult(
                symbols=ordered, equities=equities, etfs=etfs,
                source_counts={"master": len(base), "etf_classified": len(etfs),
                               "equities": len(equities),
                               "membership_version": membership.snapshot_meta.membership_version},
                error=None)
        except Exception as exc:  # noqa: BLE001 — never raise into the scanner loop
            return DiscoveryResult(symbols=[], error=f"membership_discovery: {exc}")

    return discover
