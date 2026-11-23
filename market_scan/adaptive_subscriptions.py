"""Adaptive, channel-level SIP subscription management (directive Part II).

Under Alpaca SIP there is no documented per-symbol cap, so streaming capacity is
an *application* concern managed independently per channel (bars / trades /
quotes) across four overlapping tiers:

    Tier 1  full discovery universe   -> BARS (+ updated bars)
    Tier 2  expanded ranked screening -> BARS + TRADES
    Tier 3  actionable candidates     -> BARS + TRADES + QUOTES
    Tier 4  protected instruments     -> all required channels (never dropped)

Channel subscriptions are DEDUPLICATED (a Tier-3 symbol counts once per channel,
not once per tier) and PROTECTED instruments + references are mandatory: they are
never dropped for capacity. If protected demand exceeds a channel cap the manager
enters a degraded state (caller must suspend new elective exposure, §10).

This solves the discovery promotion deadlock (§7): Tier-1 symbols receive only
BARS, so preliminary structural discovery must NOT require the fresh developing
1-minute momentum that needs TRADES. A symbol is promoted Tier1->Tier2 (gains
TRADES) on preliminary rank, then Tier2->Tier3 (gains QUOTES) once actionable;
the strict final entry contract is unchanged and enforced downstream.

Pure/deterministic: holds only subscription state + a rotation cursor; the clock
is injected (`now`), so it is fully unit-testable with no network. One manager
owns all stock-data subscriptions for the single shared SIP connection (§11).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Channel(str, Enum):
    BARS = "bars"
    TRADES = "trades"
    QUOTES = "quotes"


class SubState(str, Enum):
    """Authoritative per-(symbol, channel) subscription lifecycle.

    The provider-driven progression is::

        DESIRED -> REQUESTED -> PROVIDER_CONFIRMED -> WARMING_UP -> FRESH -> STALE

    A channel is NEVER advanced past REQUESTED merely because a WebSocket send
    succeeded: ``PROVIDER_CONFIRMED`` requires a genuine Alpaca ``subscription``
    frame listing the symbol on that channel (see ``reconcile_confirmation``).
    ``WARMING_UP`` is the first observed tick after confirmation; ``FRESH`` is
    sustained flow; ``STALE`` is silence past the stale window. On reconnection a
    confirmed/fresh channel drops to ``RECOVERING`` until the provider re-confirms.
    """

    UNREQUESTED = "unrequested"
    DESIRED = "desired"                 # in target, subscribe frame not yet sent
    REQUESTED = "requested"             # subscribe frame sent; awaiting provider ack
    PROVIDER_CONFIRMED = "provider_confirmed"  # genuine provider subscription frame
    ACKNOWLEDGED = "acknowledged"       # legacy alias; retained for compatibility
    WARMING_UP = "warming_up"           # confirmed + first observation seen
    FRESH = "fresh"                     # confirmed + sustained observations
    STALE = "stale"                     # was fresh, silent past the stale window
    DEMOTION_PENDING = "demotion_pending"
    UNSUBSCRIBE_REQUESTED = "unsubscribe_requested"
    RECOVERING = "recovering"           # reconnect: awaiting re-confirmation
    FAILED = "failed"


@dataclass
class ChannelCaps:
    bars: int = 4096
    trades: int = 1000
    quotes: int = 250

    def get(self, ch: Channel) -> int:
        return {Channel.BARS: self.bars, Channel.TRADES: self.trades,
                Channel.QUOTES: self.quotes}[ch]


@dataclass
class _Sub:
    state: SubState = SubState.UNREQUESTED
    generation: int = 0
    requested_at: float = 0.0
    provider_confirmed_at: float = 0.0   # genuine provider ack (not send success)
    confirmed_at: float = 0.0
    first_obs_at: float = 0.0
    last_obs_at: float = 0.0
    promoted_at: float = 0.0
    demoted_at: float = 0.0


@dataclass
class ChannelPlan:
    # per-channel symbol lists
    to_subscribe: dict = field(default_factory=dict)
    to_unsubscribe: dict = field(default_factory=dict)
    target: dict = field(default_factory=dict)        # desired+capped confirmed set
    dropped: dict = field(default_factory=dict)        # elective symbols that did not fit
    mandatory_dropped: dict = field(default_factory=dict)  # protected that did not fit (!)
    degraded: bool = False                             # protected demand > a channel cap


class AdaptiveSubscriptionManager:
    def __init__(self, caps: ChannelCaps | None = None, *,
                 min_dwell_s: float = 30.0, demote_cooldown_s: float = 45.0,
                 stale_after_s: float = 20.0) -> None:
        self.caps = caps or ChannelCaps()
        self.min_dwell_s = float(min_dwell_s)
        self.demote_cooldown_s = float(demote_cooldown_s)
        self.stale_after_s = float(stale_after_s)
        # current[channel] = set(symbols) currently targeted; subs[(ch,sym)] = _Sub
        self._current: dict[Channel, set] = {c: set() for c in Channel}
        self._subs: dict[tuple, _Sub] = {}
        self._rotation: dict[Channel, int] = {c: 0 for c in Channel}

    # -- desired channel membership from tiers (deduplicated) ----------------
    @staticmethod
    def _desired_by_channel(tier1_bars, tier2_trades, tier3_quotes,
                            protected, references):
        def up(xs):
            return {str(s).upper().strip() for s in (xs or []) if str(s).strip()}
        t1, t2, t3 = up(tier1_bars), up(tier2_trades), up(tier3_quotes)
        prot, refs = up(protected), up(references)
        mandatory = prot | refs
        # Everything needs bars; trades for Tier2+; quotes for Tier3. Protected and
        # references get every channel. Dedup is automatic via sets.
        bars = t1 | t2 | t3 | mandatory
        trades = t2 | t3 | prot  # references need bars only unless also protected
        quotes = t3 | prot
        return ({Channel.BARS: bars, Channel.TRADES: trades, Channel.QUOTES: quotes},
                {Channel.BARS: mandatory, Channel.TRADES: prot, Channel.QUOTES: prot})

    def update_tiers(self, *, tier1_bars=None, tier2_trades=None, tier3_quotes=None,
                     protected=None, references=None, now: float = 0.0) -> ChannelPlan:
        desired, mandatory = self._desired_by_channel(
            tier1_bars, tier2_trades, tier3_quotes, protected, references)
        plan = ChannelPlan()
        degraded = False
        for ch in Channel:
            cap = self.caps.get(ch)
            want = desired[ch]
            must = mandatory[ch]
            current = self._current[ch]

            selected: list[str] = []
            seen: set = set()
            # 1) mandatory first (protected/references), never dropped for capacity
            mand_dropped: list[str] = []
            for s in sorted(must):
                if s in seen:
                    continue
                seen.add(s)
                if len(selected) < cap:
                    selected.append(s)
                else:
                    mand_dropped.append(s)  # degraded: cap cannot hold protected
            if mand_dropped:
                degraded = True
            # 2) elective fill with rotation (fair coverage), honoring dwell:
            #    a currently-subscribed elective within min_dwell is kept first.
            electives = [s for s in want if s not in seen]
            kept = [s for s in electives
                    if s in current and not self._demote_ready(ch, s, now)]
            fresh_cands = [s for s in electives if s not in set(kept)]
            for s in kept:
                if s in seen:
                    continue
                seen.add(s)
                if len(selected) < cap:
                    selected.append(s)
            free = max(0, cap - len(selected))
            dropped: list[str] = []
            if fresh_cands:
                fresh_cands.sort()
                n = len(fresh_cands)
                off = self._rotation[ch] % n if n else 0
                rotated = fresh_cands[off:] + fresh_cands[:off]
                take = [s for s in rotated if s not in seen][:free]
                for s in take:
                    seen.add(s)
                selected.extend(take)
                dropped = [s for s in rotated if s not in seen]
                self._rotation[ch] = (self._rotation[ch] + max(1, free)) % n if n else 0

            target = set(selected)
            plan.target[ch.value] = sorted(target)
            plan.to_subscribe[ch.value] = sorted(target - current)
            plan.to_unsubscribe[ch.value] = sorted(current - target)
            plan.dropped[ch.value] = sorted(dropped)
            plan.mandatory_dropped[ch.value] = sorted(mand_dropped)
        plan.degraded = degraded
        return plan

    def _demote_ready(self, ch: Channel, sym: str, now: float) -> bool:
        """An elective may be rotated out only after min_dwell since it became
        confirmed (prevents churn on minor rank fluctuations, §9)."""
        sub = self._subs.get((ch, sym))
        if sub is None or sub.confirmed_at == 0.0:
            return True  # never confirmed -> safe to replace
        return (now - sub.confirmed_at) >= self.min_dwell_s

    # -- lifecycle transitions ----------------------------------------------
    def commit(self, plan: ChannelPlan, now: float = 0.0) -> None:
        """Record the plan as applied: new subs -> REQUESTED, removed -> dropped."""
        for ch in Channel:
            for s in plan.to_subscribe.get(ch.value, []):
                sub = self._subs.setdefault((ch, s), _Sub())
                sub.state = SubState.REQUESTED
                sub.generation += 1
                sub.requested_at = now
            for s in plan.to_unsubscribe.get(ch.value, []):
                sub = self._subs.get((ch, s))
                if sub is not None:
                    # Unsubscribe frame sent; finalized to UNREQUESTED only when the
                    # provider confirms the reduced membership (reconcile_confirmation).
                    sub.state = SubState.UNSUBSCRIBE_REQUESTED
                    sub.demoted_at = now
            self._current[ch] = set(plan.target.get(ch.value, []))

    def on_ack(self, channel: Channel, symbols, now: float = 0.0) -> None:
        """Mark specific symbols provider-confirmed on a channel (REQUESTED or
        RECOVERING -> PROVIDER_CONFIRMED). This is driven by a GENUINE provider
        subscription frame, never by a successful send. Duplicate acks are
        idempotent. If an observation already arrived while REQUESTED (an
        observation racing ahead of the ack), the symbol advances straight to
        WARMING_UP on confirmation so no real tick is lost."""
        for s in symbols or []:
            sub = self._subs.get((channel, str(s).upper().strip()))
            if sub is None:
                continue
            if sub.state in (SubState.REQUESTED, SubState.RECOVERING,
                             SubState.DESIRED):
                sub.state = SubState.PROVIDER_CONFIRMED
                sub.provider_confirmed_at = now
                sub.confirmed_at = now
                if sub.last_obs_at > 0.0:  # observation arrived before the ack
                    sub.state = SubState.WARMING_UP

    def reconcile_confirmation(self, channel: Channel, confirmed, now: float = 0.0,
                               *, generation: int | None = None) -> dict:
        """Reconcile the AUTHORITATIVE provider-confirmed membership for a channel
        against desired membership from a real Alpaca ``subscription`` frame.

        Alpaca reports the FULL current per-channel subscription (not a delta), so:
          * symbols present + REQUESTED/RECOVERING/DESIRED -> PROVIDER_CONFIRMED
            (partial acks advance only what the provider actually listed);
          * symbols absent that we had asked to drop (UNSUBSCRIBE_REQUESTED) or no
            longer desire -> UNREQUESTED (unsubscription confirmed);
          * duplicate frames are idempotent.
        Returns a small reconciliation summary for telemetry/evidence."""
        conf = {str(s).upper().strip() for s in (confirmed or []) if str(s).strip()}
        newly = 0
        self.on_ack(channel, conf, now)
        for s in conf:
            sub = self._subs.get((channel, s))
            if sub is not None and sub.provider_confirmed_at == now:
                newly += 1
        unsubscribed = 0
        for (ch, s), sub in self._subs.items():
            if ch != channel or s in conf:
                continue
            if sub.state in (SubState.UNSUBSCRIBE_REQUESTED,) or s not in self._current[channel]:
                if sub.state not in (SubState.UNREQUESTED,):
                    sub.state = SubState.UNREQUESTED
                    sub.demoted_at = now
                    unsubscribed += 1
        return {"channel": channel.value, "confirmed": len(conf),
                "newly_confirmed": newly, "unsubscribed": unsubscribed,
                "generation": generation}

    def mark_recovering(self, now: float = 0.0) -> int:
        """On connection loss / reconnection, any confirmed-or-fresh channel must be
        re-confirmed by the provider on the new socket. Drop them to RECOVERING so
        a stale send-success can never be mistaken for a live subscription."""
        n = 0
        recover = (SubState.PROVIDER_CONFIRMED, SubState.WARMING_UP,
                   SubState.FRESH, SubState.STALE)
        for sub in self._subs.values():
            if sub.state in recover:
                sub.state = SubState.RECOVERING
                sub.provider_confirmed_at = 0.0
                sub.last_obs_at = 0.0
                n += 1
        return n

    def on_observation(self, channel: Channel, symbol: str, now: float = 0.0) -> None:
        """Advance a confirmed subscription on a real observed tick:
        PROVIDER_CONFIRMED -> WARMING_UP (first tick) -> FRESH (sustained);
        STALE -> FRESH on recovery. An observation that races ahead of the provider
        ack (state REQUESTED/RECOVERING) is recorded but does NOT confirm the
        subscription — confirmation still requires a genuine provider frame."""
        sub = self._subs.get((channel, str(symbol).upper().strip()))
        if sub is None:
            return
        if sub.first_obs_at == 0.0:
            sub.first_obs_at = now
        sub.last_obs_at = now
        if sub.state in (SubState.REQUESTED, SubState.RECOVERING, SubState.DESIRED):
            return  # early observation buffered; await provider confirmation
        if sub.state == SubState.PROVIDER_CONFIRMED:
            sub.state = SubState.WARMING_UP
        elif sub.state in (SubState.WARMING_UP, SubState.STALE, SubState.FRESH):
            sub.state = SubState.FRESH

    def mark_stale(self, now: float) -> int:
        n = 0
        for sub in self._subs.values():
            if (sub.state == SubState.FRESH and sub.last_obs_at > 0.0
                    and (now - sub.last_obs_at) > self.stale_after_s):
                sub.state = SubState.STALE
                n += 1
        return n

    # -- introspection -------------------------------------------------------
    def confirmed(self, channel: Channel) -> set:
        """Symbols the provider has genuinely confirmed on a channel (confirmed,
        warming up, fresh, or stale — i.e. past a real provider ack)."""
        return {s for (ch, s), sub in self._subs.items()
                if ch == channel and sub.state in (SubState.PROVIDER_CONFIRMED,
                                                    SubState.WARMING_UP,
                                                    SubState.FRESH, SubState.STALE)}

    def fresh(self, channel: Channel) -> set:
        """Symbols with sustained fresh observations on a channel (FRESH only)."""
        return {s for (ch, s), sub in self._subs.items()
                if ch == channel and sub.state == SubState.FRESH}

    def flowing(self, channel: Channel) -> set:
        """Symbols receiving observations on a channel (WARMING_UP or FRESH)."""
        return {s for (ch, s), sub in self._subs.items()
                if ch == channel and sub.state in (SubState.WARMING_UP, SubState.FRESH)}

    def state_of(self, channel: Channel, symbol: str) -> SubState:
        sub = self._subs.get((channel, str(symbol).upper().strip()))
        return sub.state if sub is not None else SubState.UNREQUESTED

    def telemetry(self, now: float = 0.0) -> dict:
        out = {"caps": {c.value: self.caps.get(c) for c in Channel},
               "channels": {}, "states": {}}
        state_counts: dict[str, int] = {}
        for ch in Channel:
            reqd = len(self._current[ch])
            conf = len(self.confirmed(ch))
            out["channels"][ch.value] = {
                "requested": reqd, "confirmed": conf,
                "cap": self.caps.get(ch),
                "utilization": round(reqd / self.caps.get(ch), 4) if self.caps.get(ch) else 0.0,
            }
        for sub in self._subs.values():
            state_counts[sub.state.value] = state_counts.get(sub.state.value, 0) + 1
        out["states"] = state_counts
        return out
