"""Tests for market_scan.replacement — monitor-based loss removal + replacement.

Pure decision logic. The engine emits the actual close/buy through its existing
exit/rebalance funnels. Pins: -0.005 (decimal) loss detection is un-overridable
by bullish signals; replacement requires ALL eligibility gates; highest-ranked
eligible wins; excluded/just-closed symbols are skipped; no candidate -> cash.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from market_scan.config import MarketScanConfig
from market_scan.ranking import CandidateFeatures, MarketContext
from market_scan.replacement import (
    eligibility_from_features,
    find_loss_exits,
    select_replacement,
)

CFG = MarketScanConfig()
MKT = MarketContext(regime_label="EMERGING_BULLISH", regime_score=40.0,
                    spy_score=10.0, qqq_score=10.0)


def _pos(symbol, plpc):
    return SimpleNamespace(symbol=symbol, unrealized_plpc=plpc)


def _good_snap():
    return SimpleNamespace(ok=True, buy_pressure=0.7, seller_exhaustion=0.6,
                           seller_exhaustion_l1=0.6, cvd_trend=0.3,
                           last_trade_age_s=1.0, last_quote_age_s=1.0)


def _good_feat(symbol="CAND"):
    return CandidateFeatures(
        symbol=symbol, price=50.0, avg_daily_volume=5_000_000,
        avg_daily_dollar_volume=250_000_000, rel_volume=1.4,
        spread_pct=0.0004, quote_age_s=1.0,
        directional_row={"pdi": 30, "mdi": 10, "adx": 25, "vwap_dist": 0.01,
                         "short_mom": 0.02, "bb_dist": 0.01},
        snapshot=_good_snap(), vwap_dist=0.01, bb_mid_dist=0.01,
    )


# ---- loss detection (decimal fraction, un-overridable) ----
def test_find_loss_exits_decimal_threshold():
    positions = [_pos("A", -0.006), _pos("B", -0.004), _pos("C", -0.005), _pos("D", 0.02)]
    exits = {e.symbol for e in find_loss_exits(positions, threshold=-0.005)}
    assert exits == {"A", "C"}   # <= -0.005; B (-0.4%) and D (+2%) stay


def test_loss_exit_not_overridable_by_bullish_context():
    # the module only detects; bullish features cannot suppress a -0.5% loss
    positions = [_pos("A", -0.01)]
    exits = find_loss_exits(positions, threshold=-0.005)
    assert [e.symbol for e in exits] == ["A"]


def test_find_loss_exits_handles_bad_plpc_safely():
    positions = [_pos("A", None), _pos("B", float("nan")), _pos("C", -0.02)]
    exits = {e.symbol for e in find_loss_exits(positions, threshold=-0.005)}
    assert exits == {"C"}


# ---- eligibility ----
def test_eligibility_all_gates_pass():
    e = eligibility_from_features(_good_feat(), MKT, CFG)
    assert e.eligible is True


def test_eligibility_seller_exhaustion_relaxed_by_default():
    # Default policy (relaxed): seller exhaustion is NOT required, so a candidate
    # passing the trend/structure/regime gates is eligible without it.
    f = _good_feat()
    f.snapshot = SimpleNamespace(ok=True, buy_pressure=0.7, seller_exhaustion=0.0,
                                 seller_exhaustion_l1=0.0, cvd_trend=0.3,
                                 last_trade_age_s=1.0)
    assert eligibility_from_features(f, MKT, CFG).eligible is True


def test_eligibility_requires_seller_exhaustion_when_configured_strict():
    strict = MarketScanConfig(replacement_require_seller_exhaustion=True)
    f = _good_feat()
    f.snapshot = SimpleNamespace(ok=True, buy_pressure=0.7, seller_exhaustion=0.0,
                                 seller_exhaustion_l1=0.0, cvd_trend=0.3,
                                 last_trade_age_s=1.0)
    assert eligibility_from_features(f, MKT, strict).eligible is False
    # ... but the OTHER gates still bind even in relaxed mode:
    g = _good_feat()
    g.directional_row = dict(g.directional_row, pdi=5, mdi=40)   # DI fails
    assert eligibility_from_features(g, MKT, CFG).eligible is False


def test_eligibility_requires_di_plus_over_minus():
    f = _good_feat()
    f.directional_row = dict(f.directional_row, pdi=10, mdi=30)
    assert eligibility_from_features(f, MKT, CFG).eligible is False


def test_eligibility_requires_above_bb_mid_and_vwap():
    f = _good_feat(); f.bb_mid_dist = -0.01
    assert eligibility_from_features(f, MKT, CFG).eligible is False
    g = _good_feat(); g.vwap_dist = -0.01
    assert eligibility_from_features(g, MKT, CFG).eligible is False


def test_eligibility_requires_liquidity():
    f = _good_feat(); f.price = 2.0  # below floor
    assert eligibility_from_features(f, MKT, CFG).eligible is False


# ---- selection ----
def _eval_from(features):
    def _evaluate(sym):
        f = features.get(sym)
        return eligibility_from_features(f, MKT, CFG) if f is not None else None
    return _evaluate


def test_select_highest_ranked_eligible():
    ranked = ["TOP", "MID", "LOW"]
    feats = {s: _good_feat(s) for s in ranked}
    sym, detail = select_replacement(ranked, evaluate=_eval_from(feats), exclude=set())
    assert sym == "TOP"


def test_select_skips_excluded_and_just_closed():
    ranked = ["HELD", "JUSTCLOSED", "GOOD"]
    feats = {s: _good_feat(s) for s in ranked}
    sym, _ = select_replacement(ranked, evaluate=_eval_from(feats),
                                exclude={"HELD", "JUSTCLOSED"})
    assert sym == "GOOD"   # no immediate re-entry into a just-closed name


def test_select_returns_none_when_nothing_eligible_hold_cash():
    ranked = ["A", "B"]
    bad = _good_feat("A"); bad.directional_row = dict(bad.directional_row, pdi=5, mdi=40)
    feats = {"A": bad, "B": bad}
    sym, detail = select_replacement(ranked, evaluate=_eval_from(feats), exclude=set())
    assert sym is None   # remain in cash; never force a replacement


def test_select_skips_symbols_without_features():
    ranked = ["NOFEATS", "GOOD"]
    feats = {"GOOD": _good_feat("GOOD")}  # NOFEATS absent -> evaluate returns None
    sym, _ = select_replacement(ranked, evaluate=_eval_from(feats), exclude=set())
    assert sym == "GOOD"
