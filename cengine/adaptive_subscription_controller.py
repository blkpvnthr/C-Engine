"""Adaptive subscription controller — makes the AdaptiveSubscriptionManager the
authoritative owner of the single SIP stock-data connection (directive item 2/§3/§4).

It translates the four-tier channel policy into per-channel subscribe/unsubscribe
frames on the ONE shared stream (``set_channel_subscriptions``), tracks server
acknowledgements and observation freshness, and reserves capacity for protected
instruments. It never opens a second connection.

Tier inputs (deduplicated by the manager):
  Tier 1  full master universe    -> bars (+ updated bars)
  Tier 2  ranked candidates       -> bars + trades
  Tier 3  actionable candidates   -> bars + trades + quotes
  Tier 4  protected instruments   -> all channels, never dropped for capacity

The manager enforces dedup, protected priority (degraded flag when a channel cap
cannot hold protected demand), dwell hysteresis, and the per-(symbol,channel)
state machine; this controller applies the resulting plan to the live socket.
"""

from __future__ import annotations

from typing import Any, Optional

from market_scan.adaptive_subscriptions import (
    AdaptiveSubscriptionManager,
    Channel,
    ChannelCaps,
)


class AdaptiveSubscriptionController:
    def __init__(self, stream: Any, *,
                 manager: Optional[AdaptiveSubscriptionManager] = None,
                 caps: Optional[ChannelCaps] = None) -> None:
        self._stream = stream
        self.manager = manager or AdaptiveSubscriptionManager(caps)
        self._last_plan: Any = None
        self._applies = 0

    async def apply(self, *, tier1_bars=None, tier2_trades=None, tier3_quotes=None,
                    protected=None, references=None, now: float = 0.0) -> dict:
        """Compute the per-channel plan and apply it to the shared stream. Returns a
        telemetry dict (per-channel target sizes, degraded flag, applied deltas)."""
        plan = self.manager.update_tiers(
            tier1_bars=tier1_bars, tier2_trades=tier2_trades,
            tier3_quotes=tier3_quotes, protected=protected, references=references,
            now=now)
        self._last_plan = plan
        applied: dict = {}
        try:
            applied = await self._stream.set_channel_subscriptions(
                bars=plan.target.get("bars", []),
                trades=plan.target.get("trades", []),
                quotes=plan.target.get("quotes", []))
        except Exception as exc:  # noqa: BLE001 — never let discovery break the feed
            self.manager.commit(plan, now)  # record intent even if send failed
            return {"applied": False, "error": f"{type(exc).__name__}: {exc}",
                    "degraded": plan.degraded}
        self.manager.commit(plan, now)
        # Treat the stream's returned adds as acknowledged requests advancing to
        # WARMING_UP (real provider acks arrive via the consume loop -> on_ack).
        for ch_name, (add, _remove) in applied.items():
            ch = Channel(ch_name)
            self.manager.on_ack(ch, add, now=now)
        self._applies += 1
        return {
            "applied": True, "degraded": plan.degraded,
            "targets": {c: len(plan.target.get(c, [])) for c in ("bars", "trades", "quotes")},
            "deltas": {c: {"added": len(a), "removed": len(r)}
                       for c, (a, r) in applied.items()},
            "mandatory_dropped": {c: plan.mandatory_dropped.get(c, []) for c in
                                  ("bars", "trades", "quotes") if plan.mandatory_dropped.get(c)},
        }

    def on_observation(self, channel: Channel, symbol: str, now: float = 0.0) -> None:
        self.manager.on_observation(channel, symbol, now)

    def mark_stale(self, now: float) -> int:
        return self.manager.mark_stale(now)

    def telemetry(self, now: float = 0.0) -> dict:
        t = self.manager.telemetry(now)
        t["applies"] = self._applies
        t["degraded"] = bool(self._last_plan and self._last_plan.degraded)
        return t
