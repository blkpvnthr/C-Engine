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
        self._confirmations = 0
        self._reconnects = 0

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
        # A successful send means REQUESTED only. Channels are NOT marked confirmed
        # here: genuine provider confirmation arrives via on_provider_confirmation()
        # (driven by real Alpaca `subscription` frames). This eliminates optimistic
        # acknowledgement-on-send.
        self._applies += 1
        return {
            "applied": True, "degraded": plan.degraded,
            "targets": {c: len(plan.target.get(c, [])) for c in ("bars", "trades", "quotes")},
            "deltas": {c: {"added": len(a), "removed": len(r)}
                       for c, (a, r) in applied.items()},
            "mandatory_dropped": {c: plan.mandatory_dropped.get(c, []) for c in
                                  ("bars", "trades", "quotes") if plan.mandatory_dropped.get(c)},
        }

    def on_provider_confirmation(self, confirmed_by_channel: dict,
                                 now: float = 0.0, *, generation: int | None = None) -> dict:
        """Process a genuine Alpaca ``subscription`` frame: reconcile the provider's
        authoritative per-channel confirmed membership into the manager so requested
        subscriptions advance REQUESTED -> PROVIDER_CONFIRMED (and unsubscriptions
        settle). ``confirmed_by_channel`` maps 'bars'/'trades'/'quotes' -> symbols."""
        summary = {}
        for ch_name in ("bars", "trades", "quotes"):
            if ch_name not in confirmed_by_channel:
                continue
            summary[ch_name] = self.manager.reconcile_confirmation(
                Channel(ch_name), confirmed_by_channel.get(ch_name) or [], now,
                generation=generation)
        self._confirmations += 1
        return summary

    def on_reconnect(self, now: float = 0.0) -> int:
        """Connection loss/reconnection: drop confirmed/fresh channels to RECOVERING
        so nothing is treated as live until the provider re-confirms on the new
        socket."""
        self._reconnects += 1
        return self.manager.mark_recovering(now)

    def fresh(self, channel: Channel) -> set:
        return self.manager.fresh(channel)

    def flowing(self, channel: Channel) -> set:
        return self.manager.flowing(channel)

    def fresh_trades(self) -> set:
        return self.manager.fresh(Channel.TRADES)

    def fresh_quotes(self) -> set:
        return self.manager.fresh(Channel.QUOTES)

    def on_observation(self, channel: Channel, symbol: str, now: float = 0.0) -> None:
        self.manager.on_observation(channel, symbol, now)

    def mark_stale(self, now: float) -> int:
        return self.manager.mark_stale(now)

    def telemetry(self, now: float = 0.0) -> dict:
        t = self.manager.telemetry(now)
        t["applies"] = self._applies
        t["provider_confirmations"] = self._confirmations
        t["reconnects"] = self._reconnects
        t["degraded"] = bool(self._last_plan and self._last_plan.degraded)
        return t
