"""Dynamic universe resubscription coordinator.

The market scanner republishes ``universe.txt`` as the canonical trading
universe changes. This coordinator keeps the engine's live market-data
subscription in step with it *without a restart*: on each pass it refreshes the
:class:`canonical_universe.CanonicalUniverseManager` from the file, computes the
required subscription set, and applies the delta via
:meth:`market_data.alpaca_stream.AlpacaMarketDataStream.update_symbols`.

Subscription set = canonical trading universe
                 ∪ open-position symbols
                 ∪ pending-order symbols
                 ∪ caller-supplied reference symbols.

This guarantees the spec invariant (§12): removing an instrument from the
canonical entry universe never unsubscribes a held position or a symbol with a
pending order — those stay in the required set and are therefore never dropped.
It runs as a periodic task on the engine loop and is fail-safe (a bad position
read or file read never stops the loop or blanks coverage).
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable, Iterable, Optional

logger = logging.getLogger("universe_sync")


class UniverseSubscriptionSync:
    def __init__(
        self,
        *,
        stream: Any,
        universe_manager: Any,
        positions: Any = None,
        open_orders_fn: Optional[Callable[[], Iterable[Any]]] = None,
        extra_symbols: Iterable[str] = (),
        interval_s: float = 10.0,
        now_fn: Optional[Callable[[], float]] = None,
        log_fn: Optional[Callable[[str], None]] = None,
    ) -> None:
        self._stream = stream
        self._manager = universe_manager
        self._positions = positions
        self._open_orders_fn = open_orders_fn
        self._extra = {s.strip().upper() for s in extra_symbols if s and s.strip()}
        self.interval_s = float(interval_s)
        self._now_fn = now_fn
        self._log = log_fn or (lambda m: logger.info(m))

    # ---------------------------------------------------------------- required
    def _position_symbols(self) -> set[str]:
        if self._positions is None:
            return set()
        try:
            return {
                str(getattr(p, "symbol", "") or "").upper()
                for p in self._positions.positions()
                if getattr(p, "quantity", 0)
            } - {""}
        except Exception as exc:  # noqa: BLE001 - fail-safe: never blank coverage
            logger.debug("universe-sync: position read failed: %s", exc)
            return set()

    def _pending_symbols(self) -> set[str]:
        if self._open_orders_fn is None:
            return set()
        try:
            out: set[str] = set()
            for order in (self._open_orders_fn() or []):
                intent = getattr(order, "intent", None)
                sym = getattr(intent, "symbol", None) if intent is not None else None
                if sym:
                    out.add(str(sym).upper())
            return out
        except Exception as exc:  # noqa: BLE001
            logger.debug("universe-sync: pending read failed: %s", exc)
            return set()

    # ------------------------------------------------------------- one pass
    async def sync_once(self) -> tuple[tuple[str, ...], tuple[str, ...]]:
        """Reconcile the live subscription to the required set once.

        Returns the ``(added, removed)`` symbols actually applied this pass.
        """
        try:
            snap = self._manager.refresh_from_file() or self._manager.current()
        except Exception as exc:  # noqa: BLE001 - keep last good; never blank
            logger.warning("universe-sync: manager refresh failed: %s", exc)
            return ((), ())

        desired = {s.upper() for s in snap.symbols}
        desired |= self._position_symbols()
        desired |= self._pending_symbols()
        desired |= self._extra

        current = {s.upper() for s in self._stream.symbols}
        add = sorted(desired - current)
        remove = sorted(current - desired)
        if not add and not remove:
            return ((), ())

        added, removed = await self._stream.update_symbols(add=add, remove=remove)
        if added or removed:
            self._log(
                f"[universe-sync] subscription delta: +{list(added)} -{list(removed)} "
                f"(now {len(self._stream.symbols)} symbols)"
            )
        return added, removed

    # ------------------------------------------------------------------- loop
    async def run(self, stop: asyncio.Event) -> None:
        self._log(
            f"[universe-sync] active: reconciling live subscriptions to the canonical "
            f"universe every {self.interval_s:.0f}s"
        )
        while not stop.is_set():
            try:
                await self.sync_once()
            except Exception as exc:  # noqa: BLE001 - the loop must never die
                logger.error("universe-sync loop iteration error: %s", exc)
            try:
                await asyncio.wait_for(stop.wait(), timeout=self.interval_s)
            except asyncio.TimeoutError:
                pass
