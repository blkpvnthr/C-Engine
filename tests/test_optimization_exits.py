"""Protective-exit research tests (Milestone 6B, §30.4).

Pure-logic tests (adaptive policy, state machine, native candidate emergency stop) run
offline; the counterfactual + EXIT-matrix tests run against genuine 1m bars in the default
lakehouse when present and are skipped otherwise (so CI stays deterministic without the data).
The production -0.5% exit and native financial authority invariants are asserted.
"""

from __future__ import annotations

import pytest

from cengine.optimization.exits.adaptive import (
    PRODUCTION_EXIT_PCT,
    AdaptiveExitConfig,
    AdaptiveExitPolicy,
    ExitAction,
    asset_class_stop_range,
    risk_normalized_notional,
    volatility_adjusted_stop,
)
from cengine.optimization.exits.state_machine import (
    ExitEvent,
    ExitState,
    ProtectiveExitStateMachine,
)

# -- adaptive policy ---------------------------------------------------------

def test_production_exit_preserved():
    assert PRODUCTION_EXIT_PCT == -0.005


def test_adaptive_hold_on_validated_recovery():
    p = AdaptiveExitPolicy(AdaptiveExitConfig())
    d = p.decide(total_pl_frac=-0.006, active_stop_pct=-0.01,
                 intelligence={"valid": True, "recovery_probability": 0.8,
                               "seller_exhaustion": True, "momentum_confirmation": True,
                               "downside_prediction": -0.004})
    assert d.action == ExitAction.HOLD.value


def test_adaptive_fallback_on_invalid_intelligence():
    p = AdaptiveExitPolicy(AdaptiveExitConfig())
    d = p.decide(total_pl_frac=-0.006, active_stop_pct=-0.01, intelligence={"valid": False})
    assert d.action == ExitAction.EXIT_POSITION.value and d.fallback is True


def test_emergency_and_stop_never_bypassed_even_with_recovery():
    p = AdaptiveExitPolicy(AdaptiveExitConfig())
    strong = {"valid": True, "recovery_probability": 0.99, "seller_exhaustion": True,
              "momentum_confirmation": True}
    assert p.decide(total_pl_frac=-0.031, active_stop_pct=-0.01,
                    intelligence=strong).emergency is True
    # at/under the active adaptive stop -> exit regardless of recovery evidence
    assert p.decide(total_pl_frac=-0.012, active_stop_pct=-0.01,
                    intelligence=strong).action == ExitAction.EXIT_POSITION.value


def test_volatility_adjusted_stop_and_risk_normalized_sizing():
    lo, hi = asset_class_stop_range("large_cap")
    d1 = volatility_adjusted_stop(atr=2.0, reference_price=200.0, asset_class="large_cap")
    assert lo <= d1 <= hi
    # a WIDER stop yields a SMALLER position (risk held constant)
    n_tight = risk_normalized_notional(100.0, 0.01)
    n_wide = risk_normalized_notional(100.0, 0.02)
    assert n_wide < n_tight


def test_percentage_unit_conventions():
    # signed P/L fractions are negative; stop distances are positive fractions
    assert PRODUCTION_EXIT_PCT < 0
    assert volatility_adjusted_stop(2.0, 200.0, asset_class="large_cap") > 0


# -- state machine -----------------------------------------------------------

def test_state_machine_emergency_precedence_and_duplicate_guard():
    sm = ProtectiveExitStateMachine()
    t = sm.step(ExitEvent("e1", total_pl_frac=-0.031, active_stop_pct=-0.01, now_ns=1))
    assert sm.state == ExitState.MANDATORY_EXIT and t.reason == "emergency_trigger"
    assert sm.step(ExitEvent("e1", total_pl_frac=-0.031, active_stop_pct=-0.01,
                             now_ns=1)).reason == "duplicate_event"


def test_state_machine_stale_data_fallback():
    sm = ProtectiveExitStateMachine()
    sm.step(ExitEvent("e1", total_pl_frac=-0.006, active_stop_pct=-0.01, now_ns=1, stale=True))
    assert sm.state == ExitState.MANDATORY_EXIT


def test_state_machine_hold_path():
    sm = ProtectiveExitStateMachine()
    sm.step(ExitEvent("e1", total_pl_frac=-0.006, active_stop_pct=-0.01, now_ns=1,
                      intelligence={"valid": True, "recovery_probability": 0.8,
                                    "seller_exhaustion": True, "momentum_confirmation": True,
                                    "downside_prediction": -0.004}))
    assert sm.state == ExitState.ACTIVE_POSITION


# -- native candidate emergency stop (inactive) ------------------------------

def test_native_emergency_candidate_inactive_and_safe():
    import _cengine_native as n
    assert n.evaluate_emergency_exit(-0.031).triggered is True
    assert n.evaluate_emergency_exit(-0.02).triggered is False
    assert n.evaluate_emergency_exit(-0.031).overrides_tighter_stop is False
    # the optimizer can never loosen the emergency threshold
    assert n.clamp_emergency_threshold(-0.05, -0.03) == -0.03     # looser rejected
    assert n.clamp_emergency_threshold(-0.02, -0.03) == -0.02     # tighter allowed
    # RiskEngine remains the independent authority (candidate not wired into evaluate)
    assert hasattr(n, "RiskEngine")


# -- genuine-data tests (skipped without 1m bars) ----------------------------

def _has_1m():
    import duckdb

    from cengine.lakehouse.config import LakehouseConfig
    cfg = LakehouseConfig.from_env()
    base = cfg.dataset_path("bronze", "market_bars")
    if not base.exists() or not any(base.rglob("*.parquet")):
        return False
    glob = str(base / "**" / "*.parquet").replace("'", "''")
    try:
        n = duckdb.connect().execute(
            f"SELECT count(*) FROM read_parquet('{glob}', union_by_name=true) "
            f"WHERE timeframe='1Min' AND symbol_id='AAPL'").fetchone()[0]
        return n > 1000
    except Exception:  # noqa: BLE001
        return False


@pytest.mark.skipif(not _has_1m(), reason="no genuine 1m bars in the lakehouse")
def test_counterfactual_on_genuine_bars():
    from cengine.optimization.exits.counterfactual import analyze
    cf = analyze(None, "AAPL", max_events=500)
    assert cf.exit_events > 0
    assert cf.recovery_above_exit_rate["h_60m"] is not None


@pytest.mark.skipif(not _has_1m(), reason="no genuine 1m bars in the lakehouse")
def test_exit_matrix_on_genuine_bars_has_baseline_and_research_only():
    from cengine.optimization.exits.experiment_matrix import run_exit_matrix
    res = run_exit_matrix(None, "AAPL")
    assert res.per_policy["EXIT-00"].get("baseline") is True
    assert any(v.get("status") == "RESEARCH_ONLY" for v in res.per_policy.values())
    completed = [v for v in res.per_policy.values() if v.get("status") == "COMPLETED"]
    assert len(completed) >= 5
