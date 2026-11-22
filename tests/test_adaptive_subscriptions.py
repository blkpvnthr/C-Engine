"""Adaptive channel-level SIP subscription manager (directive Part II §5-13)."""

from __future__ import annotations

from market_scan.adaptive_subscriptions import (
    AdaptiveSubscriptionManager,
    Channel,
    ChannelCaps,
    SubState,
)


def _mgr(**kw):
    caps = kw.pop("caps", ChannelCaps(bars=10, trades=5, quotes=3))
    return AdaptiveSubscriptionManager(caps, **kw)


def test_channel_dedup_across_tiers():
    m = _mgr()
    # AAA in all three tiers must subscribe once per channel, not three times.
    plan = m.update_tiers(tier1_bars=["AAA", "BBB"], tier2_trades=["AAA"],
                          tier3_quotes=["AAA"], now=0.0)
    assert plan.target["bars"] == ["AAA", "BBB"]
    assert plan.target["trades"] == ["AAA"]
    assert plan.target["quotes"] == ["AAA"]


def test_bars_superset_of_trades_superset_of_quotes():
    m = _mgr()
    plan = m.update_tiers(tier1_bars=["A", "B", "C"], tier2_trades=["A", "B"],
                          tier3_quotes=["A"], now=0.0)
    assert set(plan.target["trades"]).issubset(set(plan.target["bars"]))
    assert set(plan.target["quotes"]).issubset(set(plan.target["trades"]))


def test_discovery_promotion_progression_no_deadlock():
    """§7: a bar-only Tier-1 symbol must be promotable to TRADES then QUOTES
    without ever requiring the trade-dependent momentum up front."""
    m = _mgr()
    now = 0.0
    # 1) preliminary discovery: WIN is Tier 1 -> bars only (no trades yet).
    p1 = m.update_tiers(tier1_bars=["WIN", "OTH"], now=now)
    m.commit(p1, now)
    assert "WIN" in p1.target["bars"]
    assert "WIN" not in p1.target["trades"]  # cannot require 1m momentum yet
    # 2) promoted to Tier 2 on preliminary rank -> gains TRADES.
    now = 60.0
    p2 = m.update_tiers(tier1_bars=["WIN", "OTH"], tier2_trades=["WIN"], now=now)
    m.commit(p2, now)
    assert "WIN" in p2.to_subscribe["trades"]
    assert "WIN" in m.confirmed(Channel.BARS) or "WIN" in p2.target["bars"]
    # 3) actionable -> Tier 3 gains QUOTES.
    now = 120.0
    p3 = m.update_tiers(tier1_bars=["WIN", "OTH"], tier2_trades=["WIN"],
                        tier3_quotes=["WIN"], now=now)
    m.commit(p3, now)
    assert "WIN" in p3.to_subscribe["quotes"]
    assert "WIN" in p3.target["quotes"]


def test_protected_never_dropped_and_degraded_flag():
    # quotes cap = 3; 4 protected symbols demand quotes -> one cannot fit -> degraded.
    m = _mgr(caps=ChannelCaps(bars=10, trades=5, quotes=3))
    plan = m.update_tiers(protected=["P1", "P2", "P3", "P4"], now=0.0)
    assert plan.degraded is True
    assert len(plan.target["quotes"]) == 3
    assert len(plan.mandatory_dropped["quotes"]) == 1
    # protected fully fits bars (cap 10) -> not degraded on the bars channel
    assert plan.mandatory_dropped["bars"] == []


def test_protected_prioritized_over_electives():
    m = _mgr(caps=ChannelCaps(bars=10, trades=3, quotes=3))
    # 3 protected + 5 elective trades, cap 3 -> all 3 slots go to protected.
    plan = m.update_tiers(tier2_trades=["E1", "E2", "E3", "E4", "E5"],
                          protected=["P1", "P2", "P3"], now=0.0)
    assert set(plan.target["trades"]) == {"P1", "P2", "P3"}
    assert set(plan.dropped["trades"]) == {"E1", "E2", "E3", "E4", "E5"}


def test_dwell_prevents_churn():
    m = _mgr(caps=ChannelCaps(bars=10, trades=2, quotes=3), min_dwell_s=30.0)
    now = 0.0
    p1 = m.update_tiers(tier2_trades=["A", "B"], now=now)
    m.commit(p1, now)
    m.on_ack(Channel.TRADES, ["A", "B"], now)  # confirmed_at = 0
    # New higher-ranked candidates arrive within dwell; A,B must NOT be churned out.
    now = 10.0  # < min_dwell
    p2 = m.update_tiers(tier2_trades=["A", "B", "C", "D"], now=now)
    assert "A" in p2.target["trades"] and "B" in p2.target["trades"]
    assert p2.to_unsubscribe["trades"] == []  # no churn under dwell


def test_state_machine_transitions():
    # Provider-driven lifecycle: send -> REQUESTED; genuine provider ack ->
    # PROVIDER_CONFIRMED; first observation -> WARMING_UP; sustained -> FRESH;
    # silence -> STALE. A send that was never provider-confirmed never advances.
    m = _mgr()
    now = 0.0
    p = m.update_tiers(tier2_trades=["X"], now=now)
    m.commit(p, now)
    assert m._subs[(Channel.TRADES, "X")].state == SubState.REQUESTED
    m.on_ack(Channel.TRADES, ["X"], now=1.0)
    assert m._subs[(Channel.TRADES, "X")].state == SubState.PROVIDER_CONFIRMED
    m.on_observation(Channel.TRADES, "X", now=2.0)
    assert m._subs[(Channel.TRADES, "X")].state == SubState.WARMING_UP
    m.on_observation(Channel.TRADES, "X", now=3.0)
    assert m._subs[(Channel.TRADES, "X")].state == SubState.FRESH
    staled = m.mark_stale(now=100.0)  # > stale_after_s since last obs
    assert staled == 1
    assert m._subs[(Channel.TRADES, "X")].state == SubState.STALE


def test_no_confirmation_without_provider_ack():
    # A subscribe frame that is sent but NEVER provider-confirmed stays REQUESTED,
    # and observations racing ahead of the ack do not confirm it.
    m = _mgr()
    p = m.update_tiers(tier2_trades=["X"], now=0.0)
    m.commit(p, 0.0)
    m.on_observation(Channel.TRADES, "X", now=1.0)  # early obs, no ack yet
    assert m._subs[(Channel.TRADES, "X")].state == SubState.REQUESTED
    assert "X" not in m.confirmed(Channel.TRADES)
    # once the provider confirms, the buffered early observation advances it.
    m.on_ack(Channel.TRADES, ["X"], now=2.0)
    assert m._subs[(Channel.TRADES, "X")].state == SubState.WARMING_UP


def test_reconcile_confirmation_partial_and_unsubscribe():
    m = _mgr()
    p = m.update_tiers(tier2_trades=["A", "B"], now=0.0)
    m.commit(p, 0.0)
    # Provider confirms only A (partial ack): A -> confirmed, B stays REQUESTED.
    m.reconcile_confirmation(Channel.TRADES, ["A"], now=1.0)
    assert m.state_of(Channel.TRADES, "A") == SubState.PROVIDER_CONFIRMED
    assert m.state_of(Channel.TRADES, "B") == SubState.REQUESTED
    # Duplicate confirmation is idempotent.
    m.reconcile_confirmation(Channel.TRADES, ["A"], now=1.5)
    assert m.state_of(Channel.TRADES, "A") == SubState.PROVIDER_CONFIRMED
    # Drop B from tiers, then provider confirms the reduced set -> B unsubscribed.
    p2 = m.update_tiers(tier2_trades=["A"], now=100.0)
    m.commit(p2, 100.0)
    out = m.reconcile_confirmation(Channel.TRADES, ["A"], now=101.0)
    assert m.state_of(Channel.TRADES, "B") == SubState.UNREQUESTED
    assert out["unsubscribed"] == 1


def test_reconnect_marks_recovering():
    m = _mgr()
    p = m.update_tiers(tier2_trades=["X"], now=0.0)
    m.commit(p, 0.0)
    m.reconcile_confirmation(Channel.TRADES, ["X"], now=1.0)
    m.on_observation(Channel.TRADES, "X", now=2.0)
    m.on_observation(Channel.TRADES, "X", now=3.0)
    assert m.state_of(Channel.TRADES, "X") == SubState.FRESH
    n = m.mark_recovering(now=4.0)  # reconnect voids prior confirmation
    assert n == 1 and m.state_of(Channel.TRADES, "X") == SubState.RECOVERING
    assert "X" not in m.confirmed(Channel.TRADES)  # must be re-confirmed
    m.reconcile_confirmation(Channel.TRADES, ["X"], now=5.0)  # provider re-confirms
    assert m.state_of(Channel.TRADES, "X") == SubState.PROVIDER_CONFIRMED


def test_unsubscribe_on_tier_removal():
    m = _mgr()
    now = 0.0
    p1 = m.update_tiers(tier3_quotes=["Q"], now=now)
    m.commit(p1, now)
    assert "Q" in m._current[Channel.QUOTES]
    # Q drops out of all tiers after dwell -> unsubscribed from every channel.
    now = 100.0
    p2 = m.update_tiers(now=now)
    m.commit(p2, now)
    assert p2.to_unsubscribe["quotes"] == ["Q"]
    assert m._current[Channel.QUOTES] == set()


def test_telemetry_shape():
    m = _mgr(caps=ChannelCaps(bars=10, trades=5, quotes=3))
    p = m.update_tiers(tier1_bars=["A", "B"], tier2_trades=["A"], now=0.0)
    m.commit(p, 0.0)
    t = m.telemetry(now=0.0)
    assert t["caps"] == {"bars": 10, "trades": 5, "quotes": 3}
    assert t["channels"]["bars"]["requested"] == 2
    assert t["channels"]["trades"]["cap"] == 5
    assert 0.0 <= t["channels"]["bars"]["utilization"] <= 1.0


def test_deterministic_rotation_when_demand_exceeds_cap():
    """Directive item 6: when elective bar demand exceeds the cap, coverage ROTATES
    deterministically across successive ticks (no symbol is permanently excluded),
    and the per-tick target never exceeds the cap."""
    m = _mgr(caps=ChannelCaps(bars=2, trades=5, quotes=3), min_dwell_s=0.0)
    universe = ["A", "B", "C", "D"]
    covered: set = set()
    seen_targets = []
    for i in range(6):
        p = m.update_tiers(tier1_bars=universe, now=float(i))
        m.commit(p, float(i))
        assert len(p.target["bars"]) == 2          # never exceeds the cap
        assert len(p.dropped["bars"]) == 2         # overflow explicitly reported
        covered |= set(p.target["bars"])
        seen_targets.append(tuple(p.target["bars"]))
    # Every symbol is eventually covered by rotation (no permanent exclusion).
    assert covered == set(universe)
    # Rotation actually changes the covered subset over time.
    assert len(set(seen_targets)) > 1


def test_protected_never_rotated_out():
    """Protected instruments are exempt from rotation: they hold coverage every tick
    even when the cap is saturated by electives."""
    m = _mgr(caps=ChannelCaps(bars=2, trades=5, quotes=3), min_dwell_s=0.0)
    for i in range(5):
        p = m.update_tiers(tier1_bars=["A", "B", "C", "D"], protected=["HELD"],
                           now=float(i))
        m.commit(p, float(i))
        assert "HELD" in p.target["bars"]  # protected never rotated out
        assert not p.mandatory_dropped["bars"]
