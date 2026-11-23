"""Five-family architecture (milestone §7-13): strict entry contract, family specs,
eligibility safety (never auto-enable). Deterministic; no device/network."""

from __future__ import annotations

from cengine.strategy_families import (
    FAMILY_SPECS,
    EligibilityState,
    StrategyFamily,
    StrategyFamilyRegistry,
    passes_entry_contract,
)


def _contract_row(**over):
    row = {"di_plus_5m": 30.0, "di_minus_5m": 10.0, "price": 101.0, "vwap": 100.0,
           "sma20_5m": 99.0, "bb_middle_5m": 100.5,
           "developing_1m_price": 101.2, "previous_completed_1m_close": 100.9}
    row.update(over)
    return row


def test_all_five_families_present_with_metadata():
    assert set(FAMILY_SPECS) == {
        StrategyFamily.MOMENTUM_CONTINUATION, StrategyFamily.VOLATILITY_BREAKOUT,
        StrategyFamily.TREND_PULLBACK, StrategyFamily.MEAN_REVERSION,
        StrategyFamily.MULTI_DAY_SWING}
    for spec in FAMILY_SPECS.values():
        assert spec.entry_conditions and spec.required_indicators
        assert spec.invalidation_conditions and spec.strategy_version
        assert spec.config_version and spec.est_transaction_cost_bps > 0


def test_entry_contract_all_clauses():
    ok, why = passes_entry_contract(_contract_row())
    assert ok and why == "entry_contract_satisfied"
    # DI+ must exceed DI-
    assert not passes_entry_contract(_contract_row(di_plus_5m=5.0))[0]
    # price must beat >=1 reference (set all refs above price)
    bad = _contract_row(price=98.0, vwap=100.0, sma20_5m=100.0, bb_middle_5m=100.0)
    assert passes_entry_contract(bad)[1] == "price_below_all_valid_references"
    # developing 1m must exceed previous 1m close
    below = _contract_row(developing_1m_price=100.0, previous_completed_1m_close=100.9)
    assert passes_entry_contract(below)[1] == "developing_1m_not_above_previous_1m_close"


def test_entry_contract_reference_alternatives_are_or():
    # Only ONE reference above price is required (explicit OR with parentheses).
    only_sma = _contract_row(vwap=200.0, bb_middle_5m=200.0, sma20_5m=99.0, price=101.0)
    assert passes_entry_contract(only_sma)[0]
    only_bb = _contract_row(vwap=200.0, sma20_5m=200.0, bb_middle_5m=100.0, price=101.0)
    assert passes_entry_contract(only_bb)[0]


def test_entry_contract_insufficient_observations():
    assert passes_entry_contract({"di_plus_5m": 30.0})[1] == "insufficient_observations"


def test_registry_defaults_to_non_execution():
    reg = StrategyFamilyRegistry()
    # No family is EXECUTION-eligible by default (never auto-enable unvalidated live).
    for fam in reg.families():
        assert reg.eligibility(fam) == EligibilityState.RESEARCH
        assert not reg.is_execution_eligible(fam)


def test_mark_execution_requires_tests_and_validation():
    reg = StrategyFamilyRegistry()
    fam = StrategyFamily.MOMENTUM_CONTINUATION
    # validated=False never promotes
    reg.mark_execution_eligible(fam, validated=False)
    assert not reg.is_execution_eligible(fam)
    # validated=True promotes only because the spec has regression tests
    reg.mark_execution_eligible(fam, validated=True)
    assert reg.is_execution_eligible(fam)


def test_registry_instance_isolation():
    a = StrategyFamilyRegistry()
    b = StrategyFamilyRegistry()
    a.mark_execution_eligible(StrategyFamily.MEAN_REVERSION, validated=True)
    # promoting one registry must not leak into another or into the module default
    assert not b.is_execution_eligible(StrategyFamily.MEAN_REVERSION)
    assert FAMILY_SPECS[StrategyFamily.MEAN_REVERSION].eligibility == EligibilityState.RESEARCH


def test_multi_day_swing_is_new_and_flags_protective_exit_viability():
    spec = FAMILY_SPECS[StrategyFamily.MULTI_DAY_SWING]
    assert spec.maps_to.startswith("NEW")
    assert any("-0.5%" in c for c in spec.invalidation_conditions)
