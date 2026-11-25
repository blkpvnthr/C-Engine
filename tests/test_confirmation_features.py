"""Genuine confirmation-feature computation from the native bar window.

These standard indicators (relative volume, momentum acceleration, higher-timeframe
direction, Donchian channel high, ATR expansion) are referenced by the strategy
family evaluators but were never supplied by the feature pipeline, so every
confirmation gate failed and the regime path produced zero candidates regardless of
market (verified via OOS SIP replay). Computing them from real bars is a defect fix,
not signal manufacturing or condition relaxation.
"""

from __future__ import annotations

from cengine.gpu_discovery import _confirmation_features


def test_relative_volume_current_vs_prior_mean():
    vols5 = [100.0] * 20 + [300.0]
    f = _confirmation_features([100.0, 101.0, 102.0], [100.0] * 21, [101.0] * 21,
                               [99.0] * 21, vols5, scale=10_000)
    assert abs(f["rel_volume"] - 3.0) < 1e-9


def test_momentum_accel_second_difference_of_1m_returns():
    # returns: (101-100)/100=0.01 then (103-101)/101≈0.0198 -> accel≈0.0098 > 0
    f = _confirmation_features([100.0, 101.0, 103.0], [100.0] * 5, [100.0] * 5,
                               [100.0] * 5, [1.0] * 5, scale=10_000)
    assert f["momentum_accel"] > 0

    # decelerating -> negative
    g = _confirmation_features([100.0, 103.0, 104.0], [100.0] * 5, [100.0] * 5,
                               [100.0] * 5, [1.0] * 5, scale=10_000)
    assert g["momentum_accel"] < 0


def test_htf_direction_sign_vs_mean():
    up = _confirmation_features([1.0, 1.0, 1.0], list(range(1, 22)),
                                [21.0] * 21, [1.0] * 21, [1.0] * 21, scale=10_000)
    assert up["htf_direction"] == 1.0  # rising series -> last above mean
    down = _confirmation_features([1.0, 1.0, 1.0], list(range(21, 0, -1)),
                                  [21.0] * 21, [1.0] * 21, [1.0] * 21, scale=10_000)
    assert down["htf_direction"] == -1.0


def test_donchian_high_prior_channel_in_price_units_excludes_current():
    highs5 = [1000.0] * 20 + [50_000.0]        # ticks; current bar is the spike
    f = _confirmation_features([1.0, 1.0, 1.0], [1.0] * 21, highs5,
                               [1.0] * 21, [1.0] * 21, scale=10_000)
    # prior-channel high excludes the current bar -> 1000 ticks / 10000 = 0.1 price
    assert abs(f["donchian_high"] - 0.1) < 1e-9


def test_insufficient_history_omits_features_not_fabricates():
    f = _confirmation_features([100.0], [100.0], [100.0], [99.0], [1.0], scale=10_000)
    assert "momentum_accel" not in f  # needs >=3 1m closes
    assert "htf_direction" not in f   # needs >=5 5m closes
