"""Engine-side independent protective-exit coordinator (mandatory −0.5% removal).

This is the single authoritative protective-exit path. It runs as a periodic
task on the engine's asyncio loop (a configurable backstop, default 5 s) and is
independent of the market scanner and of any entry/exit strategy: it reads the
authoritative in-engine position book and the live quote price (no broker REST
call), and when a position's unrealised return is at or below the mandatory
threshold it submits a close through the engine's *single* broker-writing path
(:meth:`order_manager.OrderManager.submit`) — the identical native risk gate +
venue + audit pathway every other order uses. It never opens a parallel broker
connection.

Safety properties:

  * duplicate-close prevention: it uses a dedicated ``strategy_id`` and the
    engine's single-pending guard (``StrategyStateBook.can_submit``) so it never
    stacks a second close while one is in flight;
  * oversell safety: the close quantity is sized from the *projected* position
    (held minus pending exits), and the native risk engine re-checks the
    projected position and may reduce/reject;
  * fail-safe: a per-symbol error (no quote yet, risk rejection, venue error)
    is isolated and never stops the other positions or the loop; the −0.5%
    threshold is an exit *trigger*, not a guaranteed execution price (gap/halt/
    spread/liquidity/slippage limits apply).

It deliberately keeps running whenever position state is available — regardless
of whether the scanner is healthy, the universe is fresh, or strategies are
enabled.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Callable, Optional

from order_manager import OrderIntent, OrderType, Side, TimeInForce

logger = logging.getLogger("protective_exit")

STRATEGY_ID = "protective_exit"


class ProtectiveExitCoordinator:
    def __init__(
        self,
        *,
        order_manager: Any,
        positions: Any,
        price_source: Any,
        reservations: Any,
        strategy_state: Any,
        threshold: float = -0.005,
        interval_s: float = 5.0,
        strategy_id: str = STRATEGY_ID,
        market_open_fn: Optional[Callable[[], bool]] = None,
        now_fn: Optional[Callable[[], int]] = None,
        log_fn: Optional[Callable[[str], None]] = None,
    ) -> None:
        self._orders = order_manager
        self._positions = positions
        self._prices = price_source
        self._reservations = reservations
        self._strategy_state = strategy_state
        # The threshold is an exit trigger; coerce to strictly-<= comparison.
        self.threshold = float(threshold)
        self.interval_s = float(interval_s)
        self.strategy_id = strategy_id
        # Optional "is the market open?" gate; off-hours a MARKET/DAY close would
        # be rejected/queued, so skip submission when closed.
        self.market_open_fn = market_open_fn
        self._now_fn = now_fn or time.time_ns
        self._log = log_fn or (lambda m: logger.info(m))

    # ------------------------------------------------------------------- math
    @staticmethod
    def unrealized_return(position: Any, price_ticks: float) -> float | None:
        """Fractional unrealised return for a position at the given price.

        Long: ``(price - avg) / avg``. Short (qty < 0): sign-inverted so a loss
        is negative. Returns ``None`` when it cannot be computed (zero quantity
        or zero cost basis).
        """
        qty = getattr(position, "quantity", 0)
        cost_ticks = getattr(position, "cost_ticks", 0)
        if not qty:
            return None
        avg = cost_ticks / qty
        if avg == 0:
            return None
        raw = (price_ticks - avg) / avg
        return raw if qty > 0 else -raw

    # --------------------------------------------------------------- one pass
    async def scan_and_exit(self) -> list[str]:
        """Evaluate every open position once; submit closes for breaches.

        Returns the list of symbols a close was submitted for this pass. Never
        raises: per-symbol failures are isolated and logged.
        """
        if self.market_open_fn is not None:
            try:
                if not self.market_open_fn():
                    return []
            except Exception as exc:  # noqa: BLE001 - treat a broken gate as closed
                logger.debug("market_open gate error (treating as closed): %s", exc)
                return []

        submitted: list[str] = []
        try:
            positions = list(self._positions.positions())
        except Exception as exc:  # noqa: BLE001 - never let a bad read kill the loop
            logger.error("protective-exit: position read failed: %s", exc)
            return submitted

        for pos in positions:
            try:
                sym = str(getattr(pos, "symbol", "") or "").upper()
                qty = getattr(pos, "quantity", 0)
                if not sym or not qty:
                    continue
                # Single-pending guard: never stack a second close.
                if not self._strategy_state.can_submit(self.strategy_id, sym):
                    continue
                try:
                    price = self._prices.price_ticks(sym)
                except Exception:  # noqa: BLE001 - no fresh quote yet -> skip, retry next pass
                    continue
                ret = self.unrealized_return(pos, price)
                if ret is None or ret > self.threshold:
                    continue
                # Size the close off the projected position (held minus pending
                # exits) so we never oversell; native risk re-checks this.
                try:
                    projected = self._reservations.projected_position(sym, qty)
                except Exception:  # noqa: BLE001
                    projected = qty
                close_qty = abs(int(projected))
                if close_qty <= 0:
                    continue
                side = Side.SELL if qty > 0 else Side.BUY
                ns = self._now_fn()
                intent = OrderIntent(
                    symbol=sym,
                    side=side,
                    quantity=close_qty,
                    order_type=OrderType.MARKET,
                    time_in_force=TimeInForce.DAY,
                    strategy_id=self.strategy_id,
                    correlation_id=f"protective:{sym}:{ns}",
                    created_ns=ns,
                )
                try:
                    await self._orders.submit(intent)
                except Exception as exc:  # noqa: BLE001 - isolate per symbol
                    self._log(
                        f"[protective-exit] submit FAILED {sym} qty={close_qty}: {exc}"
                    )
                    continue
                submitted.append(sym)
                self._log(
                    f"[protective-exit] CLOSE {sym} qty={close_qty} "
                    f"(unrealised {ret * 100:.3f}% <= {self.threshold * 100:.3f}%)"
                )
            except Exception as exc:  # noqa: BLE001 - per-position belt-and-braces
                logger.error("protective-exit: unexpected error on position: %s", exc)
                continue
        return submitted

    # ------------------------------------------------------------------- loop
    async def run(self, stop: asyncio.Event) -> None:
        """Periodic backstop loop; exits promptly when ``stop`` is set."""
        self._log(
            f"[protective-exit] coordinator active: threshold={self.threshold * 100:.2f}% "
            f"interval={self.interval_s:.0f}s (independent of scanner/strategies)"
        )
        while not stop.is_set():
            try:
                await self.scan_and_exit()
            except Exception as exc:  # noqa: BLE001 - the monitor must never die
                logger.error("protective-exit loop iteration error: %s", exc)
            try:
                await asyncio.wait_for(stop.wait(), timeout=self.interval_s)
            except asyncio.TimeoutError:
                pass
