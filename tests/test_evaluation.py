"""Tests for the learning evaluation layer (economic metrics + offline policy evaluation).

All data here is EXPLICITLY SYNTHETIC: fixtures are hand-constructed so expected values can be
verified by hand. The tests assert the module's honesty gates hold — synthetic evidence never
claims profitability, and IPS/doubly-robust are gated off when behavior probabilities are
unknown (the real gold ``offline_rl`` contract where ``action_probability_if_known`` is NULL).
This module is pure numpy/scipy (CPU); there are no GPU code paths to guard.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from cengine.learning.evaluation import (
    DATA_LABEL_REAL,
    DATA_LABEL_SYNTHETIC,
    DirectMethodEvaluator,
    FittedQEvaluator,
    LinearRewardModel,
    LoggedDecisions,
    PerActionMeanRewardModel,
    action_support_audit,
    bootstrap_metric_ci,
    compare_policies,
    doubly_robust_value,
    economic_metrics,
    evaluate_returns,
    ips_value,
    sample_size_report,
    validate_data_label,
)

# A known series with hand-computable statistics.
_SERIES = [0.02, -0.01, 0.03, -0.02, 0.01]


# ------------------------------------------------------------------ economic metrics


def test_economic_metrics_match_hand_computed_values():
    m = economic_metrics(_SERIES, data_label=DATA_LABEL_SYNTHETIC, drawdown_mode="additive")
    assert m.n == 5
    assert m.net_return == pytest.approx(0.03)
    assert m.net_expectancy == pytest.approx(0.006)
    assert m.gross_profit == pytest.approx(0.06)
    assert m.gross_loss == pytest.approx(0.03)
    assert m.profit_factor == pytest.approx(2.0)
    assert m.win_rate == pytest.approx(0.6)
    assert m.n_wins == 3
    assert m.n_losses == 2
    assert m.avg_win == pytest.approx(0.02)
    assert m.avg_loss == pytest.approx(-0.015)
    # std(ddof=1) = sqrt(0.00172/4) = 0.0207364 ; sharpe = 0.006 / std
    assert m.sharpe == pytest.approx(0.006 / math.sqrt(0.00172 / 4.0), rel=1e-9)
    # downside deviation = sqrt((0.01^2 + 0.02^2)/5) = 0.01 ; sortino = 0.006/0.01 = 0.6
    assert m.sortino == pytest.approx(0.6, rel=1e-9)
    # additive equity cumsum peak-to-trough worst decline is 0.02
    assert m.max_drawdown == pytest.approx(0.02)
    # compounded total return
    assert m.net_return_compounded == pytest.approx(
        np.prod(1.0 + np.array(_SERIES)) - 1.0, rel=1e-12)


def test_compound_drawdown_matches_equity_curve():
    m = economic_metrics(_SERIES, data_label=DATA_LABEL_SYNTHETIC, drawdown_mode="compound")
    equity = np.cumprod(1.0 + np.array(_SERIES))
    peak = np.maximum.accumulate(equity)
    expected = float(-((equity - peak) / peak).min())
    assert m.max_drawdown == pytest.approx(expected)


def test_profit_factor_infinite_with_no_losses():
    m = economic_metrics([0.01, 0.02, 0.0, 0.03], data_label=DATA_LABEL_SYNTHETIC)
    assert math.isinf(m.profit_factor)
    assert m.gross_loss == 0.0


def test_turnover_from_weights():
    # weights step by 0.1, 0.0, 0.2 -> mean |diff| = 0.1
    m = economic_metrics(_SERIES, data_label=DATA_LABEL_SYNTHETIC,
                         weights=[0.0, 0.1, 0.1, 0.3, 0.3])
    assert m.turnover == pytest.approx((0.1 + 0.0 + 0.2 + 0.0) / 4.0)


def test_empty_series_is_research_only_not_a_crash():
    m = economic_metrics([], data_label=DATA_LABEL_SYNTHETIC)
    assert m.n == 0
    assert m.profitability_claim_allowed is False
    assert "insufficient_data" in m.note


# --------------------------------------------------------- provenance / honesty gate


def test_data_label_is_validated():
    with pytest.raises(ValueError):
        validate_data_label("backtest")
    with pytest.raises(ValueError):
        economic_metrics(_SERIES, data_label="not-a-label")


def test_synthetic_never_claims_profitability():
    big = (list(_SERIES) * 40)  # n=200, clearly positive expectancy
    m = economic_metrics(big, data_label=DATA_LABEL_SYNTHETIC)
    assert m.net_expectancy > 0
    assert m.profitability_claim_allowed is False
    assert "synthetic" in m.note.lower()


def test_real_small_sample_is_research_only():
    m = economic_metrics(_SERIES, data_label=DATA_LABEL_REAL)
    # only 5 decisions -> not enough for a profitability claim even though real
    assert m.profitability_claim_allowed is False


def test_real_adequate_sample_allows_claim_flag():
    m = economic_metrics(list(_SERIES) * 40, data_label=DATA_LABEL_REAL)
    assert m.n >= 30
    assert m.profitability_claim_allowed is True


# ----------------------------------------------------- dependence-aware uncertainty


def test_bootstrap_ci_brackets_the_mean():
    rng = np.random.default_rng(0)
    series = rng.normal(loc=0.001, scale=0.01, size=400)
    ci = bootstrap_metric_ci(series, data_label=DATA_LABEL_SYNTHETIC, n_boot=500, seed=1)
    assert ci.ci_low < ci.ci_high
    # the bootstrap percentile interval must bracket the sample mean (the point estimate)
    assert ci.brackets(float(np.mean(series)))
    assert ci.point_estimate == pytest.approx(float(np.mean(series)))
    assert 0 < ci.effective_sample_size <= ci.n


def test_moving_block_bootstrap_runs_and_brackets_mean():
    rng = np.random.default_rng(2)
    series = rng.normal(0.0, 0.02, size=300)
    ci = bootstrap_metric_ci(series, data_label=DATA_LABEL_SYNTHETIC, method="moving_block",
                             n_boot=400, seed=3)
    assert ci.method == "moving_block"
    assert ci.brackets(float(np.mean(series)))


def test_effective_sample_size_shrinks_under_autocorrelation():
    rng = np.random.default_rng(4)
    # strongly positively autocorrelated series via cumulative smoothing
    noise = rng.normal(0, 1, size=500)
    ar = np.zeros_like(noise)
    for i in range(1, len(noise)):
        ar[i] = 0.8 * ar[i - 1] + noise[i]
    report = sample_size_report(ar, data_label=DATA_LABEL_SYNTHETIC)
    assert report.integrated_autocorr_time > 1.0
    assert report.effective_sample_size < report.n
    assert report.autocorr_lag1 > 0.2


def test_evaluate_returns_bundle_is_synthetic_labeled():
    ev = evaluate_returns(list(_SERIES) * 40, data_label=DATA_LABEL_SYNTHETIC,
                          include_sharpe_ci=True, n_boot=300)
    assert ev.profitability_claim_allowed is False
    assert ev.verdict.startswith("SYNTHETIC")
    assert ev.sharpe_ci is not None
    assert ev.expectancy_ci.brackets(ev.metrics.net_expectancy)


# ----------------------------------------------------------- OPE: Direct Method


def _bandit_fixture(*, with_probs=False):
    actions = np.array([0, 0, 1, 1, 0, 1], dtype=np.int64)
    rewards = np.array([1.0, 0.0, 2.0, 4.0, 1.0, 0.0], dtype=np.float64)
    bap = np.full(6, 0.5) if with_probs else None
    return LoggedDecisions(actions=actions, rewards=rewards, action_space=["hold", "trade"],
                           data_label=DATA_LABEL_SYNTHETIC, behavior_action_probs=bap)


def test_direct_method_runs_and_matches_hand_value():
    logged = _bandit_fixture()
    ev = DirectMethodEvaluator(PerActionMeanRewardModel())
    # deterministic policy: always choose action 1 ("trade"); r_hat(1) = mean(2,4,0) = 2.0
    result = ev.estimate(logged, np.ones(6, dtype=np.int64))
    assert result.status == "ok"
    assert result.value == pytest.approx(2.0)
    # always action 0 ("hold"); r_hat(0) = mean(1,0,1) = 2/3
    result0 = ev.estimate(logged, np.zeros(6, dtype=np.int64))
    assert result0.value == pytest.approx(2.0 / 3.0)


def test_direct_method_stochastic_policy_matrix():
    logged = _bandit_fixture()
    ev = DirectMethodEvaluator()
    # 50/50 over the two actions -> 0.5*(2/3) + 0.5*2.0
    mat = np.tile([0.5, 0.5], (6, 1))
    result = ev.estimate(logged, mat)
    assert result.value == pytest.approx(0.5 * (2.0 / 3.0) + 0.5 * 2.0)


def test_direct_method_no_realized_rewards_is_insufficient():
    logged = LoggedDecisions(actions=np.array([0, 1, 0]),
                             rewards=np.array([np.nan, np.nan, np.nan]),
                             action_space=["a", "b"], data_label=DATA_LABEL_SYNTHETIC)
    result = DirectMethodEvaluator().estimate(logged, np.zeros(3, dtype=np.int64))
    assert result.status == "insufficient_data"
    assert result.value is None


def test_linear_reward_model_runs_with_context():
    rng = np.random.default_rng(7)
    n = 60
    contexts = rng.normal(size=(n, 2))
    actions = rng.integers(0, 2, size=n)
    rewards = contexts[:, 0] + (actions == 1) * 0.5 + rng.normal(0, 0.01, size=n)
    logged = LoggedDecisions(actions=actions, rewards=rewards, action_space=["a", "b"],
                             data_label=DATA_LABEL_SYNTHETIC, contexts=contexts)
    res = DirectMethodEvaluator(LinearRewardModel()).estimate(logged, np.ones(n, dtype=np.int64))
    assert res.status == "ok"
    assert np.isfinite(res.value)


# ----------------------------------------------------------- OPE: IPS / DR gating


def test_ips_unavailable_when_behavior_probs_unknown():
    logged = _bandit_fixture(with_probs=False)
    assert logged.behavior_probs_known is False
    res = ips_value(logged, np.ones(6, dtype=np.int64))
    assert res.status == "unavailable"
    assert res.value is None
    assert "behavior probabilities unknown" in res.reason
    assert res.diagnostics.get("gated_off") is True


def test_doubly_robust_unavailable_when_behavior_probs_unknown():
    logged = _bandit_fixture(with_probs=False)
    res = doubly_robust_value(logged, np.ones(6, dtype=np.int64))
    assert res.status == "unavailable"
    assert res.value is None
    assert "behavior probabilities unknown" in res.reason


def test_null_probability_column_is_treated_as_unknown():
    # an all-NaN propensity column (the real gold contract) must gate IPS off
    logged = LoggedDecisions(actions=np.array([0, 1]), rewards=np.array([1.0, 2.0]),
                             action_space=["a", "b"], data_label=DATA_LABEL_REAL,
                             behavior_action_probs=np.array([np.nan, np.nan]))
    assert logged.behavior_probs_known is False
    assert ips_value(logged, np.zeros(2, dtype=np.int64)).status == "unavailable"


def test_ips_runs_when_probs_are_known():
    logged = _bandit_fixture(with_probs=True)
    assert logged.behavior_probs_known is True
    res = ips_value(logged, np.ones(6, dtype=np.int64))
    assert res.status == "ok"
    assert np.isfinite(res.value)


def test_doubly_robust_runs_when_probs_are_known():
    logged = _bandit_fixture(with_probs=True)
    res = doubly_robust_value(logged, np.ones(6, dtype=np.int64))
    assert res.status == "ok"
    assert np.isfinite(res.value)


# ----------------------------------------------------------- action-support audit


def test_action_support_audit_flags_extrapolation():
    # 3 actions in the space, but only actions 0 and 1 ever appear in the logs
    logged = LoggedDecisions(actions=np.array([0, 0, 1, 1]),
                             rewards=np.array([1.0, 1.0, 2.0, 2.0]),
                             action_space=["a", "b", "c"], data_label=DATA_LABEL_SYNTHETIC)
    # target always chooses the never-logged action 2 ("c")
    audit = action_support_audit(logged, np.full(4, 2, dtype=np.int64))
    assert "c" in audit.unsupported_actions
    assert audit.extrapolation_fraction == pytest.approx(1.0)
    assert audit.warning is not None
    assert audit.reward_coverage == pytest.approx(1.0)


def test_action_support_audit_reports_reward_coverage():
    logged = LoggedDecisions(actions=np.array([0, 1, 0, 1]),
                             rewards=np.array([1.0, np.nan, np.nan, 2.0]),
                             action_space=["a", "b"], data_label=DATA_LABEL_SYNTHETIC)
    audit = action_support_audit(logged, np.zeros(4, dtype=np.int64))
    assert audit.reward_coverage == pytest.approx(0.5)


# ----------------------------------------------------------- Fitted-Q hook


def test_fitted_q_reduces_to_direct_method_without_next_states():
    logged = _bandit_fixture()
    dm = DirectMethodEvaluator().estimate(logged, np.ones(6, dtype=np.int64))
    fqe = FittedQEvaluator(gamma=0.9).estimate(logged, np.ones(6, dtype=np.int64))
    assert fqe.status == "ok"
    assert fqe.value == pytest.approx(dm.value)
    assert "reduces to the Direct Method" in fqe.diagnostics["reduction"]


# ----------------------------------------------------------- candidate vs baseline


def test_compare_policies_on_identical_decisions():
    logged = _bandit_fixture()
    cand = np.ones(6, dtype=np.int64)   # always "trade" -> DM value 2.0
    base = np.zeros(6, dtype=np.int64)  # always "hold"  -> DM value 2/3
    cmp = compare_policies(logged, cand, base, estimator="direct_method", n_boot=200)
    assert cmp.identical_decisions is True
    assert cmp.candidate.value == pytest.approx(2.0)
    assert cmp.baseline.value == pytest.approx(2.0 / 3.0)
    assert cmp.value_difference == pytest.approx(2.0 - 2.0 / 3.0)
    assert cmp.difference_ci is not None
    # synthetic -> no superiority/profitability may be inferred
    assert cmp.profitability_claim_allowed is False
    assert cmp.verdict.startswith("SYNTHETIC")


def test_compare_policies_ips_inherits_gate():
    logged = _bandit_fixture(with_probs=False)
    cmp = compare_policies(logged, np.ones(6, dtype=np.int64), np.zeros(6, dtype=np.int64),
                           estimator="ips")
    assert cmp.candidate.status == "unavailable"
    assert cmp.value_difference is None
    assert "unavailable" in cmp.verdict.lower()
