"""Unit tests for the probabilistic regime transition-frequency model (M6).

All fixtures are EXPLICITLY SYNTHETIC regime label sequences — no market data, no
fabricated rewards/labels. They verify the contract: Laplace-smoothed transition
matrices whose rows sum to 1, honest persistence estimates, the insufficient-data
research-only path, the derived strategy-facing probabilities, and that the model
only reads observed regimes (never writes them back).
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest

from cengine.learning.regime_models import (
    REGIME_DIMENSIONS,
    TREND_STATES,
    RegimeTransitionModel,
    TransitionMatrix,
    fit_regime_transition_model,
)

# -- synthetic fixtures ------------------------------------------------------


def _persistent_trend_episode(n: int = 60) -> list[str]:
    """A trend history that overwhelmingly stays 'persistent'."""
    return ["persistent"] * n


def _alternating(a: str, b: str, pairs: int = 30) -> list[str]:
    seq: list[str] = []
    for _ in range(pairs):
        seq.extend([a, b])
    return seq


def _full_sequences() -> dict[str, list[list[str]]]:
    return {
        "direction": [_alternating("bullish", "neutral", 30)],
        "trend": [_alternating("exhaustion", "recovery", 30)],
        "volatility": [_alternating("normal", "elevated", 30)],
        "liquidity": [_alternating("normal", "degraded", 30)],
    }


# -- core matrix properties --------------------------------------------------


def test_transition_matrix_rows_sum_to_one_all_dimensions():
    model = RegimeTransitionModel()
    model.fit_sequences(_full_sequences())
    for dim, matrix in model.matrices.items():
        row_sums = matrix.probabilities.sum(axis=1)
        assert np.allclose(row_sums, 1.0), f"{dim} rows do not sum to 1: {row_sums}"
        # smoothed probabilities are strictly positive (no zero-probability transitions)
        assert np.all(matrix.probabilities > 0.0)


def test_empty_rows_collapse_to_uniform_prior():
    model = RegimeTransitionModel(min_observations=1, min_transitions=1)
    model.fit_sequences({"trend": [_persistent_trend_episode(50)]})
    m = model.matrix("trend")
    k = len(TREND_STATES)
    # 'recovery' was never an origin -> its row is the uniform prior and sums to 1.
    rec = m.states.index("recovery")
    assert np.allclose(m.probabilities[rec], 1.0 / k)
    assert np.isclose(m.probabilities[rec].sum(), 1.0)


def test_persistence_estimate_high_for_sticky_regime():
    model = RegimeTransitionModel(min_observations=1, min_transitions=1)
    model.fit_sequences({"trend": [_persistent_trend_episode(60)]})
    persistence = model.persistence("trend")
    # 59 self-transitions, Laplace alpha=1, K=6 -> 60/65 ~= 0.923
    assert persistence["persistent"] > 0.85
    # a state never observed as an origin has uniform (low) persistence
    assert persistence["recovery"] == pytest.approx(1.0 / len(TREND_STATES))


def test_counts_are_raw_and_probabilities_are_smoothed():
    model = RegimeTransitionModel(min_observations=1, min_transitions=1)
    model.fit_sequences({"trend": [_persistent_trend_episode(10)]})
    m = model.matrix("trend")
    i = m.states.index("persistent")
    assert m.counts[i, i] == 9           # raw count untouched by smoothing
    assert m.n_observations == 10
    assert m.n_transitions == 9


# -- insufficient-data path --------------------------------------------------


def test_insufficient_data_flagged_and_still_valid():
    model = RegimeTransitionModel()  # default thresholds 50 obs / 30 transitions
    result = model.fit_sequences({"trend": [["persistent", "recovery", "expansion"]]})
    assert result.insufficient_data is True
    assert result.per_dimension_insufficient["trend"] is True
    assert result.promoted is False
    # even insufficient, the matrix is well-formed (rows sum to 1) and usable for research
    m = model.matrix("trend")
    assert np.allclose(m.probabilities.sum(axis=1), 1.0)
    assert "observations" in m.reason or "transitions" in m.reason


def test_no_observations_reports_no_data():
    model = RegimeTransitionModel()
    result = model.fit_sequences({d: [] for d in REGIME_DIMENSIONS})
    assert result.insufficient_data is True
    assert result.promoted is False
    for dim in REGIME_DIMENSIONS:
        m = model.matrix(dim)
        assert m.n_observations == 0
        assert m.reason == "no observations"
        assert np.allclose(m.probabilities.sum(axis=1), 1.0)


def test_unknown_labels_break_the_chain_and_are_counted():
    model = RegimeTransitionModel(min_observations=1, min_transitions=1)
    # the 'garbage' label is not a valid trend state -> no transition is counted across it
    model.fit_sequences({"trend": [["persistent", "garbage", "persistent"]]})
    m = model.matrix("trend")
    assert m.unknown_labels == 1
    i = m.states.index("persistent")
    assert m.counts[i, i] == 0           # chain was broken by the unknown label
    assert m.n_observations == 2         # two valid observations


# -- derived strategy-facing probabilities -----------------------------------


def test_recovery_expansion_exhaustion_probabilities_conditional():
    model = RegimeTransitionModel(min_observations=1, min_transitions=1)
    model.fit_sequences({"trend": [_alternating("exhaustion", "recovery", 30)]})
    # exhaustion is almost always followed by recovery in this fixture
    assert model.recovery_probability(from_state="exhaustion") > 0.7
    # expansion basically never follows exhaustion here
    assert model.expansion_probability(from_state="exhaustion") < 0.1
    # from_state=None returns an occupancy-weighted probability in [0, 1]
    assert 0.0 <= model.exhaustion_probability() <= 1.0


def test_volatility_escalation_probability():
    model = RegimeTransitionModel(min_observations=1, min_transitions=1)
    model.fit_sequences({"volatility": [_alternating("normal", "elevated", 30)]})
    # 'normal' escalates to a more severe state (elevated) almost every time
    assert model.volatility_escalation_probability(from_state="normal") > 0.8
    # 'normal' rarely de-escalates to 'low'
    vol = model.matrix("volatility")
    assert vol.severity_decrease_probability(from_state="normal") < 0.1


def test_liquidity_deterioration_probability():
    model = RegimeTransitionModel(min_observations=1, min_transitions=1)
    model.fit_sequences({"liquidity": [_alternating("normal", "degraded", 30)]})
    assert model.liquidity_deterioration_probability(from_state="normal") > 0.8


def test_severity_methods_reject_unordered_dimensions():
    model = RegimeTransitionModel(min_observations=1, min_transitions=1)
    model.fit_sequences({"direction": [_alternating("bullish", "bearish", 10)]})
    with pytest.raises(ValueError):
        model.matrix("direction").severity_increase_probability("bullish")


def test_next_distribution_and_most_likely_next():
    model = RegimeTransitionModel(min_observations=1, min_transitions=1)
    model.fit_sequences({"trend": [_alternating("exhaustion", "recovery", 30)]})
    dist = model.next_distribution("trend", "exhaustion")
    assert dist["recovery"] == pytest.approx(max(dist.values()))
    state, prob = model.most_likely_next("trend", "exhaustion")
    assert state == "recovery"
    assert prob == pytest.approx(dist["recovery"])


# -- uncertainty -------------------------------------------------------------


def test_uncertainty_interval_tightens_with_support():
    model = RegimeTransitionModel(min_observations=1, min_transitions=1)
    model.fit_sequences({"trend": [_persistent_trend_episode(200)]})
    m = model.matrix("trend")
    lo_hi_strong = m.transition_probability_interval("persistent", "persistent")
    lo_hi_weak = m.transition_probability_interval("recovery", "recovery")
    width_strong = lo_hi_strong[1] - lo_hi_strong[0]
    width_weak = lo_hi_weak[1] - lo_hi_weak[0]
    assert width_strong < width_weak
    for lo, hi in (lo_hi_strong, lo_hi_weak):
        assert 0.0 <= lo <= hi <= 1.0


# -- dataframe / grouping path -----------------------------------------------


def test_dataframe_fit_does_not_cross_symbol_boundaries():
    import pandas as pd

    df = pd.DataFrame(
        {
            "symbol_id": ["A", "A", "B", "B"],
            "event_timestamp_ns": [1, 2, 3, 4],
            "trend": ["persistent", "exhaustion", "recovery", "expansion"],
        }
    )
    model = RegimeTransitionModel(min_observations=1, min_transitions=1,
                                  dimensions=("trend",))
    model.fit(df)
    m = model.matrix("trend")
    exh = m.states.index("exhaustion")
    rec = m.states.index("recovery")
    per = m.states.index("persistent")
    # within-symbol transition exists ...
    assert m.counts[per, exh] == 1
    # ... but the cross-symbol boundary (A's last -> B's first) is NOT a transition
    assert m.counts[exh, rec] == 0


def test_dataframe_fit_orders_by_time():
    import pandas as pd

    # rows deliberately out of time order; fit must sort by event_timestamp_ns
    df = pd.DataFrame(
        {
            "symbol_id": ["A", "A", "A"],
            "event_timestamp_ns": [3, 1, 2],
            "volatility": ["extreme", "low", "elevated"],
        }
    )
    model = RegimeTransitionModel(min_observations=1, min_transitions=1,
                                  dimensions=("volatility",))
    model.fit(df)
    m = model.matrix("volatility")
    low = m.states.index("low")
    elev = m.states.index("elevated")
    ext = m.states.index("extreme")
    assert m.counts[low, elev] == 1      # low(t=1) -> elevated(t=2)
    assert m.counts[elev, ext] == 1      # elevated(t=2) -> extreme(t=3)


# -- warehouse integration ---------------------------------------------------


class _FakeWarehouse:
    """Minimal warehouse stub exposing query_df -> pyarrow.Table."""

    def __init__(self, table: pa.Table) -> None:
        self._table = table

    def query_df(self, sql: str) -> pa.Table:
        del sql  # signature parity with DuckDBWarehouse.query_df
        return self._table


class _RaisingWarehouse:
    def query_df(self, sql: str):
        del sql
        raise RuntimeError("view market_regimes not materialized")


def _synthetic_regime_table(n: int = 60) -> pa.Table:
    # one symbol, instrument level, sticky-persistent trend, escalating volatility
    trend = ["persistent"] * n
    vol = (["normal", "elevated"] * (n // 2))[:n]
    liq = ["normal"] * n
    direction = (["bullish", "neutral"] * (n // 2))[:n]
    return pa.table(
        {
            "symbol_id": ["SPY"] * n,
            "level": ["instrument"] * n,
            "direction": direction,
            "trend": trend,
            "volatility": vol,
            "liquidity": liq,
            "event_timestamp_ns": list(range(n)),
        }
    )


def test_fit_from_warehouse_success_path():
    wh = _FakeWarehouse(_synthetic_regime_table(60))
    model, result = fit_regime_transition_model(wh, min_observations=10,
                                                min_transitions=10)
    assert result.device == "cpu"
    assert result.insufficient_data is False
    assert result.promoted is False
    assert model.matrix("trend").persistence()["persistent"] > 0.85
    for m in model.matrices.values():
        assert np.allclose(m.probabilities.sum(axis=1), 1.0)


def test_fit_from_warehouse_missing_view_is_honest_not_fatal():
    model = RegimeTransitionModel()
    result = model.fit_from_warehouse(_RaisingWarehouse())
    assert result.insufficient_data is True
    assert result.promoted is False
    assert "market_regimes unavailable" in result.reason
    # matrices still well-formed (uniform priors) so downstream research code never crashes
    for m in model.matrices.values():
        assert np.allclose(m.probabilities.sum(axis=1), 1.0)


def test_fit_from_warehouse_symbol_filter():
    n = 40
    tbl = pa.table(
        {
            "symbol_id": ["AAA"] * n + ["BBB"] * n,
            "level": ["instrument"] * (2 * n),
            "direction": ["bullish"] * (2 * n),
            "trend": ["persistent"] * n + ["recovery"] * n,
            "volatility": ["normal"] * (2 * n),
            "liquidity": ["normal"] * (2 * n),
            "event_timestamp_ns": list(range(2 * n)),
        }
    )
    model = RegimeTransitionModel(min_observations=1, min_transitions=1)
    model.fit_from_warehouse(_FakeWarehouse(tbl), symbol_id="AAA")
    m = model.matrix("trend")
    # only AAA (persistent) was kept; recovery never appears as an origin
    assert m.counts[m.states.index("persistent"), m.states.index("persistent")] > 0
    assert m.row_support[m.states.index("recovery")] == 0


# -- device placement + recommendation-only invariants -----------------------


def test_fit_records_cpu_placement_decision():
    """Transition counting is a small CPU workload (no GPU kernel); it must say so.

    This holds whether or not an MLX/Metal GPU is present — the op is declared
    gpu_supported=False, so there is no GPU assertion to skip here.
    """
    model = RegimeTransitionModel(min_observations=1, min_transitions=1)
    result = model.fit_sequences({"trend": [_persistent_trend_episode(40)]})
    assert result.device == "cpu"
    assert result.placement_reason  # non-empty, observable reason
    summary = model.summary()
    assert summary["device"] == "cpu"
    assert summary["promoted"] is False
    assert summary["device_policy"]["placements"].get("cpu", 0) >= 1


def test_model_is_read_only_no_write_back_path():
    """Invariant: predicted probabilities never overwrite observed regimes.

    The model exposes no persistence/write API into the lakehouse.
    """
    model = RegimeTransitionModel()
    for forbidden in ("save", "write", "persist", "materialize", "upsert", "commit"):
        assert not hasattr(model, forbidden)


def test_accessors_require_fit():
    model = RegimeTransitionModel()
    with pytest.raises(RuntimeError):
        model.persistence("trend")
    with pytest.raises(RuntimeError):
        model.recovery_probability()


def test_unknown_dimension_rejected():
    with pytest.raises(ValueError):
        RegimeTransitionModel(dimensions=("not_a_dimension",))


def test_matrix_to_dict_is_json_friendly():
    model = RegimeTransitionModel(min_observations=1, min_transitions=1)
    model.fit_sequences({"trend": [_persistent_trend_episode(40)]})
    d: dict = model.matrix("trend").to_dict()
    assert isinstance(d["probabilities"], list)
    assert isinstance(d["counts"], list)
    assert d["insufficient_data"] is False
    assert isinstance(d["persistence"], dict)


def test_transition_matrix_type_exported():
    assert issubclass(TransitionMatrix, object)
