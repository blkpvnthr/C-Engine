"""Tests for market_scan.ranking — composite candidate ranking + eligibility.

Pins contracts (bounds, monotonicity, filtering, sort order, regime direction),
NOT predictive power. Missing order-flow never reads as bullish.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from market_scan.config import MarketScanConfig
from market_scan.ranking import (
    CandidateFeatures,
    MarketContext,
    passes_liquidity,
    rank,
    score_candidate,
)

CFG = MarketScanConfig()


def _snap(**kw):
    base = dict(ok=True, buy_pressure=0.5, seller_exhaustion=0.3,
               seller_exhaustion_l1=0.3, cvd_trend=0.2, pressure_score=0.1)
    base.update(kw)
    return SimpleNamespace(**base)


def _feat(symbol="AAA", **kw):
    base = dict(
        price=50.0, avg_daily_volume=2_000_000, avg_daily_dollar_volume=100_000_000,
        rel_volume=1.2, spread_pct=0.0005, quote_age_s=2.0,
        directional_row={"pdi": 30, "mdi": 10, "adx": 25, "vwap_dist": 0.01,
                         "short_mom": 0.01, "bb_dist": 0.01},
        snapshot=_snap(), vwap_dist=0.01, bb_mid_dist=0.01,
    )
    base.update(kw)
    return CandidateFeatures(symbol=symbol, **base)


def _mkt(regime_score=40.0, label="EMERGING_BULLISH"):
    return MarketContext(regime_label=label, regime_score=regime_score,
                         spy_score=10.0, qqq_score=10.0)


# ---- eligibility ----
def test_price_floor_blocks():
    ok, reason = passes_liquidity(_feat(price=3.0), CFG)
    assert not ok and reason == "price"


def test_volume_floor_blocks():
    assert not passes_liquidity(_feat(avg_daily_volume=100_000), CFG)[0]


def test_dollar_volume_floor_blocks():
    assert not passes_liquidity(_feat(avg_daily_dollar_volume=1_000_000), CFG)[0]


def test_wide_spread_blocks_when_known():
    assert not passes_liquidity(_feat(spread_pct=0.02), CFG)[0]


def test_stale_quote_blocks_when_known():
    assert not passes_liquidity(_feat(quote_age_s=30.0), CFG)[0]


def test_eligible_passes():
    assert passes_liquidity(_feat(), CFG)[0]


# ---- scoring ----
def test_score_bounded_0_1():
    s, _d = score_candidate(_feat(), _mkt())
    assert 0.0 <= s <= 1.0


def test_higher_buy_pressure_scores_higher():
    lo, _ = score_candidate(_feat(snapshot=_snap(buy_pressure=0.1)), _mkt())
    hi, _ = score_candidate(_feat(snapshot=_snap(buy_pressure=0.95)), _mkt())
    assert hi > lo


def test_higher_seller_exhaustion_scores_higher():
    lo, _ = score_candidate(_feat(snapshot=_snap(seller_exhaustion=0.0, seller_exhaustion_l1=0.0)), _mkt())
    hi, _ = score_candidate(_feat(snapshot=_snap(seller_exhaustion=0.9, seller_exhaustion_l1=0.9)), _mkt())
    assert hi > lo


def test_missing_snapshot_not_bullish():
    with_flow, _ = score_candidate(_feat(snapshot=_snap(buy_pressure=0.9)), _mkt())
    no_flow, _ = score_candidate(_feat(snapshot=None), _mkt())
    assert no_flow < with_flow


def test_relative_strength_beats_market():
    strong = _feat(directional_row={"pdi": 40, "mdi": 5, "adx": 35, "vwap_dist": 0.02,
                                    "short_mom": 0.03, "bb_dist": 0.02})
    weak = _feat(directional_row={"pdi": 8, "mdi": 30, "adx": 35, "vwap_dist": -0.02,
                                  "short_mom": -0.03, "bb_dist": -0.02})
    assert score_candidate(strong, _mkt())[0] > score_candidate(weak, _mkt())[0]


def test_inverse_etf_favored_in_bearish_regime():
    # SQQQ is a registered inverse ETF; it should rank relatively higher when the
    # regime is bearish than when bullish (direction-aware regime alignment).
    f = _feat(symbol="SQQQ")
    bear, _ = score_candidate(f, _mkt(regime_score=-60.0, label="CONFIRMED_BEARISH"))
    bull, _ = score_candidate(f, _mkt(regime_score=60.0, label="CONFIRMED_BULLISH"))
    assert bear > bull


# ---- rank ordering ----
def test_rank_sorts_descending_and_eligible_lead():
    feats = [
        _feat(symbol="LOWP", price=2.0),                   # ineligible
        _feat(symbol="WEAK", snapshot=_snap(buy_pressure=0.05, cvd_trend=-0.5)),
        _feat(symbol="STRONG", snapshot=_snap(buy_pressure=0.95, seller_exhaustion=0.8,
                                              seller_exhaustion_l1=0.8, cvd_trend=0.8)),
    ]
    ranked = rank(feats, config=CFG, market=_mkt())
    eligible = [r.symbol for r in ranked if r.eligible]
    assert eligible[0] == "STRONG"
    assert "LOWP" not in eligible
    # eligible candidates appear before ineligible ones in the full list
    syms = [r.symbol for r in ranked]
    assert syms.index("STRONG") < syms.index("LOWP")
