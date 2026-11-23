"""First-order Markov transition-frequency model over observed market regimes (M6).

This is the *probabilistic* extension of the deterministic ``MarketRegimeEngine``
(``cengine/regime_engine.py``). The deterministic engine stays the transparent
baseline and the fallback: it is the thing that actually *labels* each bar's regime.
This module only *reads* those already-observed labels out of the silver
``market_regimes`` dataset and estimates, per regime dimension, a Laplace-smoothed
first-order Markov transition matrix. From that single estimator it derives the
quantities the strategy layer cares about:

    * regime persistence           P(stay in the current state)
    * full transition probabilities P(state_t -> state_{t+1})
    * recovery probability          P(-> trend "recovery")
    * expansion probability         P(-> trend "expansion")
    * exhaustion probability        P(-> trend "exhaustion")
    * volatility-escalation prob.   P(-> a strictly more severe volatility state)
    * liquidity-deterioration prob. P(-> a strictly worse liquidity state)

Design discipline (CLAUDE.md "Simplicity Wins" / "Falsification First"):
  * This simple transition-frequency estimator is evaluated and shipped *before*
    any complex sequence classifier. A classifier must beat this honest baseline
    out-of-sample before it is worth its complexity. This module is that baseline.
  * Laplace (add-alpha) smoothing gives well-defined probabilities even for states
    that were never observed as an origin, and doubles as a Dirichlet prior so we
    can report honest credible intervals on every probability.
  * When there are too few observations we say so (``insufficient_data=True`` with a
    reason) and the result is research-only — it is NEVER promoted and NEVER used to
    overwrite an observed historical regime label.

INVARIANTS (enforced here in code + docstrings):
  * RECOMMENDATION-ONLY research infrastructure. This model issues no orders, touches
    no broker, and never sits on or bypasses the mandatory -0.5% protective-exit path.
    Mandatory exits are environment dynamics, not modelled agent actions.
  * Predicted transition probabilities NEVER overwrite observed historical regimes.
    The model has no write path back into the lakehouse; it only consumes observed
    ``direction/trend/volatility/liquidity`` labels produced by the deterministic
    engine. No regime label is ever fabricated.
  * Insufficient data still produces a working, unit-tested, explicitly-untrained
    result (uniform priors, flagged) rather than a crash or a false-confident model.

Compute placement (``cengine/learning/hardware.py`` ``DevicePolicy``): counting
first-order transitions is a tiny, sequential, memory-bound CPU workload — there is
no GPU kernel for it and host/device sync would dominate, so this op is declared
``gpu_supported=False`` and runs on the CPU. The decision is recorded on every fit so
telemetry can prove where it ran; the platform never silently claims GPU here.

Pure ``numpy`` + ``duckdb`` (via the lakehouse warehouse). No MLX required.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Optional

import numpy as np

from cengine.learning.hardware import DevicePolicy

REGIME_TRANSITION_MODEL_VERSION = "regime-transition-1.0.0"


# -- regime dimension specifications -----------------------------------------
# State vocabularies mirror the deterministic ``cengine.regime_engine`` enums but are
# kept as plain lowercase strings so this learning module stays decoupled (pure
# numpy/duckdb) and consumes exactly what silver ``market_regimes`` stores.
#
# ``ordered=True`` means the state tuple is a severity ladder (index increasing ==
# more severe): escalation = moving up the ladder, improvement = moving down it.
# Direction and trend are categorical (no single severity axis), so ``ordered=False``.


@dataclass(frozen=True)
class DimensionSpec:
    """A regime dimension: its name and its (optionally severity-ordered) states."""

    name: str
    states: tuple[str, ...]
    ordered: bool

    def index_of(self, state: str) -> int:
        return self.states.index(state)


DIRECTION_STATES: tuple[str, ...] = ("bearish", "neutral", "uncertain", "bullish")
TREND_STATES: tuple[str, ...] = (
    "compression", "range_bound", "persistent", "expansion", "exhaustion", "recovery",
)
# severity ladders (ascending badness)
VOLATILITY_STATES: tuple[str, ...] = ("low", "normal", "elevated", "extreme")
LIQUIDITY_STATES: tuple[str, ...] = ("normal", "degraded", "insufficient")

REGIME_DIMENSIONS: dict[str, DimensionSpec] = {
    "direction": DimensionSpec("direction", DIRECTION_STATES, ordered=False),
    "trend": DimensionSpec("trend", TREND_STATES, ordered=False),
    "volatility": DimensionSpec("volatility", VOLATILITY_STATES, ordered=True),
    "liquidity": DimensionSpec("liquidity", LIQUIDITY_STATES, ordered=True),
}

# derived-probability target states (semantic trend events)
RECOVERY_STATE = "recovery"
EXPANSION_STATE = "expansion"
EXHAUSTION_STATE = "exhaustion"


def _normalize_state(value: object) -> Optional[str]:
    """Normalize a raw cell to a lowercase state string, or None if missing.

    None, NaN (float or pandas NA), and empty strings become None (a chain break);
    everything else is ``str(value).strip().lower()``. No value is ever invented.
    """
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    try:  # pandas / numpy NA sentinels without importing pandas here
        if value != value:  # NaN-like
            return None
    except Exception:  # noqa: BLE001 - defensive only
        pass
    s = str(value).strip().lower()
    return s or None


# -- transition matrix -------------------------------------------------------


@dataclass
class TransitionMatrix:
    """A Laplace-smoothed first-order Markov transition matrix for one dimension.

    ``counts`` are the raw observed transition counts (never smoothed). ``probabilities``
    is the add-alpha smoothed row-stochastic matrix; every row sums to 1.0 by
    construction, including rows for states never seen as an origin (which collapse to
    the uniform prior). All probability queries read ``probabilities``; uncertainty
    queries use ``counts`` + ``alpha`` as a Dirichlet posterior.
    """

    dimension: str
    states: tuple[str, ...]
    counts: np.ndarray            # (K, K) int observed transition counts
    probabilities: np.ndarray     # (K, K) float, rows sum to 1
    alpha: float
    ordered: bool
    n_observations: int           # total labelled observations consumed
    n_transitions: int            # total consecutive transitions counted
    row_support: np.ndarray       # (K,) per-origin-state transition counts (row sums)
    insufficient_data: bool
    reason: str
    unknown_labels: int = 0       # labels seen that are not in ``states`` (chain breaks)

    # -- lookups -------------------------------------------------------------
    def _idx(self, state: str) -> int:
        s = _normalize_state(state)
        if s is None or s not in self.states:
            raise ValueError(
                f"unknown {self.dimension} state {state!r}; valid: {self.states}")
        return self.states.index(s)

    def transition_probability(self, from_state: str, to_state: str) -> float:
        """Smoothed P(from_state -> to_state)."""
        return float(self.probabilities[self._idx(from_state), self._idx(to_state)])

    def next_distribution(self, from_state: str) -> dict[str, float]:
        """Smoothed full next-state distribution given ``from_state``."""
        row = self.probabilities[self._idx(from_state)]
        return {s: float(p) for s, p in zip(self.states, row, strict=True)}

    def most_likely_next(self, from_state: str) -> tuple[str, float]:
        row = self.probabilities[self._idx(from_state)]
        j = int(np.argmax(row))
        return self.states[j], float(row[j])

    def persistence(self) -> dict[str, float]:
        """Per-state persistence P(i -> i) (the matrix diagonal)."""
        return {s: float(self.probabilities[i, i]) for i, s in enumerate(self.states)}

    def marginal_state_distribution(self) -> dict[str, float]:
        """Empirical occupancy of *origin* states (row-support normalized).

        Used to weight from_state-agnostic derived probabilities. Falls back to a
        uniform distribution when no transitions were observed.
        """
        total = float(self.row_support.sum())
        if total <= 0.0:
            k = len(self.states)
            return {s: 1.0 / k for s in self.states}
        return {s: float(c) / total
                for s, c in zip(self.states, self.row_support, strict=True)}

    # -- aggregate forward probabilities -------------------------------------
    def forward_probability(self, to_states: Iterable[str],
                            from_state: Optional[str] = None) -> float:
        """P(next in ``to_states``).

        If ``from_state`` is given, this is a single matrix row summed over the target
        columns. If ``from_state`` is None, it is the one-step-ahead marginal under the
        empirical origin-state occupancy: sum_i pi_i * sum_{j in targets} P(i->j).
        """
        cols = [self.states.index(s) for s in
                (_normalize_state(t) for t in to_states) if s in self.states]
        if not cols:
            return 0.0
        col_idx = np.asarray(cols, dtype=int)
        if from_state is not None:
            i = self._idx(from_state)
            return float(self.probabilities[i, col_idx].sum())
        pi = np.asarray(list(self.marginal_state_distribution().values()), dtype=float)
        per_origin = self.probabilities[:, col_idx].sum(axis=1)
        return float((pi * per_origin).sum())

    def severity_increase_probability(self, from_state: Optional[str] = None) -> float:
        """P(-> a strictly MORE severe state). Ordered dimensions only."""
        if not self.ordered:
            raise ValueError(f"{self.dimension} is not severity-ordered")
        return self._severity_prob(from_state, increase=True)

    def severity_decrease_probability(self, from_state: Optional[str] = None) -> float:
        """P(-> a strictly LESS severe state). Ordered dimensions only."""
        if not self.ordered:
            raise ValueError(f"{self.dimension} is not severity-ordered")
        return self._severity_prob(from_state, increase=False)

    def _severity_prob(self, from_state: Optional[str], *, increase: bool) -> float:
        k = len(self.states)
        if from_state is not None:
            i = self._idx(from_state)
            cols = range(i + 1, k) if increase else range(0, i)
            return float(sum(self.probabilities[i, j] for j in cols))
        pi = self.marginal_state_distribution()
        acc = 0.0
        for i, s in enumerate(self.states):
            cols = range(i + 1, k) if increase else range(0, i)
            acc += pi[s] * float(sum(self.probabilities[i, j] for j in cols))
        return acc

    # -- uncertainty ---------------------------------------------------------
    def transition_probability_interval(self, from_state: str, to_state: str, *,
                                        z: float = 1.96) -> tuple[float, float]:
        """Approximate credible interval for P(from->to) under a Dirichlet posterior.

        With add-alpha smoothing the posterior for a single cell is
        Beta(c + alpha, (row_total - c) + alpha*(K-1)); we report a normal-approximation
        interval around the posterior mean, clipped to [0, 1]. Honest uncertainty: wide
        when the originating state has little support, tight when it has a lot.
        """
        i = self._idx(from_state)
        j = self._idx(to_state)
        k = len(self.states)
        c = float(self.counts[i, j])
        n = float(self.row_support[i])
        denom = n + self.alpha * k
        if denom <= 0.0:
            return 0.0, 1.0
        mean = (c + self.alpha) / denom
        var = mean * (1.0 - mean) / (denom + 1.0)
        half = z * math.sqrt(max(var, 0.0))
        return max(0.0, mean - half), min(1.0, mean + half)

    def to_dict(self) -> dict:
        return {
            "dimension": self.dimension,
            "states": list(self.states),
            "ordered": self.ordered,
            "alpha": self.alpha,
            "n_observations": self.n_observations,
            "n_transitions": self.n_transitions,
            "row_support": self.row_support.astype(int).tolist(),
            "counts": self.counts.astype(int).tolist(),
            "probabilities": self.probabilities.astype(float).tolist(),
            "persistence": self.persistence(),
            "insufficient_data": self.insufficient_data,
            "reason": self.reason,
            "unknown_labels": self.unknown_labels,
        }


# -- fit result --------------------------------------------------------------


@dataclass
class TransitionFitResult:
    """Outcome of fitting the transition model. Always research-only (never promoted)."""

    device: str
    placement_reason: str
    dimensions_fitted: tuple[str, ...]
    n_observations: dict[str, int]
    n_transitions: dict[str, int]
    per_dimension_insufficient: dict[str, bool]
    insufficient_data: bool
    reason: str = ""
    promoted: bool = False  # INVARIANT: this estimator is never auto-promoted
    version: str = REGIME_TRANSITION_MODEL_VERSION

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "device": self.device,
            "placement_reason": self.placement_reason,
            "dimensions_fitted": list(self.dimensions_fitted),
            "n_observations": dict(self.n_observations),
            "n_transitions": dict(self.n_transitions),
            "per_dimension_insufficient": dict(self.per_dimension_insufficient),
            "insufficient_data": self.insufficient_data,
            "reason": self.reason,
            "promoted": self.promoted,
        }


# -- model -------------------------------------------------------------------


class RegimeTransitionModel:
    """Probabilistic first-order Markov model over observed regime dimensions.

    Fit from (a) explicit per-dimension state sequences, (b) a dataframe-like table of
    observed regimes, or (c) the lakehouse silver ``market_regimes`` dataset. Produces a
    ``TransitionMatrix`` per dimension plus the derived strategy-facing probabilities.

    This class deliberately does NOT subclass ``LearningModel`` (whose fit signature is
    supervised ``(X, y)`` arrays and whose regression/classification metrics do not apply
    to a Markov chain). It mirrors the same discipline: a recorded device on every fit,
    honest uncertainty, and an explicit untrained/insufficient-data path.
    """

    def __init__(self, *, alpha: float = 1.0, min_observations: int = 50,
                 min_transitions: int = 30,
                 dimensions: Optional[Sequence[str]] = None,
                 device_policy: Optional[DevicePolicy] = None) -> None:
        if alpha < 0:
            raise ValueError("alpha (Laplace smoothing) must be >= 0")
        self.alpha = float(alpha)
        self.min_observations = int(min_observations)
        self.min_transitions = int(min_transitions)
        names = tuple(dimensions) if dimensions is not None else tuple(REGIME_DIMENSIONS)
        for n in names:
            if n not in REGIME_DIMENSIONS:
                raise ValueError(f"unknown regime dimension {n!r}")
        self.dimensions: tuple[str, ...] = names
        self.policy = device_policy or DevicePolicy()
        self.matrices: dict[str, TransitionMatrix] = {}
        self.device: str = "cpu"
        self.placement_reason: str = ""
        self.fitted: bool = False
        self.version = REGIME_TRANSITION_MODEL_VERSION

    # -- core fit ------------------------------------------------------------
    def fit_sequences(
        self, sequences_by_dimension: Mapping[str, Sequence[Sequence[object]]]
    ) -> TransitionFitResult:
        """Fit from explicit per-dimension episode sequences.

        ``sequences_by_dimension`` maps a dimension name to a list of *episodes*; each
        episode is an ordered sequence of observed state labels. Transitions are counted
        only WITHIN an episode (never across episode boundaries) so unrelated symbols or
        time-gaps never create a spurious transition. Missing/unknown labels break the
        chain rather than being imputed.
        """
        total_transitions = 0
        for dim in self.dimensions:
            episodes = list(sequences_by_dimension.get(dim, []))
            total_transitions += sum(max(0, len(ep) - 1) for ep in episodes)

        # Record the placement decision. Transition counting has no GPU kernel and is a
        # small sequential workload: declared gpu_supported=False -> CPU, observably.
        decision = self.policy.choose(
            "regime_transition_fit",
            n_elements=max(1, total_transitions),
            gpu_supported=False,
        )
        self.device = decision.device
        self.placement_reason = (
            f"{decision.reason}; transition-frequency counting is a small sequential "
            f"CPU workload (no GPU kernel)"
        )

        self.matrices = {}
        n_obs: dict[str, int] = {}
        n_trans: dict[str, int] = {}
        insufficient: dict[str, bool] = {}
        for dim in self.dimensions:
            spec = REGIME_DIMENSIONS[dim]
            episodes = list(sequences_by_dimension.get(dim, []))
            m = self._fit_dimension(spec, episodes)
            self.matrices[dim] = m
            n_obs[dim] = m.n_observations
            n_trans[dim] = m.n_transitions
            insufficient[dim] = m.insufficient_data

        self.fitted = True
        any_insufficient = (not self.dimensions) or any(insufficient.values())
        reason = ("ok" if not any_insufficient else
                  "insufficient_data in: "
                  + ", ".join(d for d, v in insufficient.items() if v))
        return TransitionFitResult(
            device=self.device,
            placement_reason=self.placement_reason,
            dimensions_fitted=self.dimensions,
            n_observations=n_obs,
            n_transitions=n_trans,
            per_dimension_insufficient=insufficient,
            insufficient_data=any_insufficient,
            reason=reason,
        )

    def _fit_dimension(self, spec: DimensionSpec,
                       episodes: Sequence[Sequence[object]]) -> TransitionMatrix:
        k = len(spec.states)
        index = {s: i for i, s in enumerate(spec.states)}
        counts = np.zeros((k, k), dtype=np.int64)
        n_obs = 0
        unknown = 0
        for episode in episodes:
            prev: Optional[int] = None
            for raw in episode:
                s = _normalize_state(raw)
                if s is None:
                    prev = None
                    continue
                if s not in index:
                    unknown += 1
                    prev = None
                    continue
                cur = index[s]
                n_obs += 1
                if prev is not None:
                    counts[prev, cur] += 1
                prev = cur

        row_support = counts.sum(axis=1)
        n_transitions = int(counts.sum())
        probabilities = self._smooth(counts, row_support, k)

        insufficient = (
            n_obs == 0
            or n_obs < self.min_observations
            or n_transitions < self.min_transitions
        )
        if n_obs == 0:
            reason = "no observations"
        elif n_obs < self.min_observations:
            reason = f"only {n_obs} observations (< {self.min_observations})"
        elif n_transitions < self.min_transitions:
            reason = f"only {n_transitions} transitions (< {self.min_transitions})"
        else:
            reason = "ok"

        return TransitionMatrix(
            dimension=spec.name,
            states=spec.states,
            counts=counts,
            probabilities=probabilities,
            alpha=self.alpha,
            ordered=spec.ordered,
            n_observations=n_obs,
            n_transitions=n_transitions,
            row_support=row_support,
            insufficient_data=insufficient,
            reason=reason,
            unknown_labels=unknown,
        )

    def _smooth(self, counts: np.ndarray, row_support: np.ndarray, k: int) -> np.ndarray:
        """Add-alpha smoothing -> row-stochastic matrix (rows sum to exactly 1)."""
        denom = row_support.astype(np.float64) + self.alpha * k
        probs = (counts.astype(np.float64) + self.alpha)
        # Guard the alpha==0 + empty-row case: fall back to a uniform row.
        empty = denom <= 0.0
        denom_safe = np.where(empty, 1.0, denom)
        probs = probs / denom_safe[:, None]
        if np.any(empty):
            probs[empty] = 1.0 / k
        return probs

    # -- dataframe / warehouse fit -------------------------------------------
    def fit(self, data, *, time_col: str = "event_timestamp_ns",
            group_cols: Optional[Sequence[str]] = None) -> TransitionFitResult:
        """Fit from a dataframe-like table of observed regimes.

        ``data`` must expose the dimension columns being modelled (a subset of
        ``direction/trend/volatility/liquidity``). Rows are grouped into episodes by
        ``group_cols`` (default: whichever of ``symbol_id``/``level`` are present) and
        ordered by ``time_col`` when present, so transitions are counted only within a
        contiguous per-symbol/per-level history.
        """
        import pandas as pd

        df = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
        if group_cols is None:
            group_cols = [c for c in ("symbol_id", "level") if c in df.columns]
        else:
            group_cols = [c for c in group_cols if c in df.columns]

        sequences: dict[str, list[list[object]]] = {d: [] for d in self.dimensions}
        present = [d for d in self.dimensions if d in df.columns]

        def _emit(group: "pd.DataFrame") -> None:
            g = group.sort_values(time_col) if time_col in group.columns else group
            for dim in present:
                col = g[dim].tolist()
                sequences[dim].append([None if pd.isna(v) else v for v in col])

        if df.empty:
            pass
        elif group_cols:
            for _, grp in df.groupby(group_cols, dropna=False, sort=False):
                _emit(grp)
        else:
            _emit(df)

        return self.fit_sequences(sequences)

    def fit_from_warehouse(self, warehouse=None, *, level: Optional[str] = None,
                           symbol_id: Optional[object] = None,
                           config=None,
                           time_col: str = "event_timestamp_ns") -> TransitionFitResult:
        """Fit from the lakehouse silver ``market_regimes`` dataset.

        Reads observed regime labels only; never writes back. If the view is not yet
        materialized (or the read fails) this returns an honest insufficient-data,
        research-only result with uniform-prior matrices rather than raising.
        """
        from cengine.lakehouse.warehouse import DuckDBWarehouse

        wh = warehouse or DuckDBWarehouse(config)
        cols = ["symbol_id", "level", "direction", "trend", "volatility", "liquidity",
                time_col]
        sql = f"SELECT {', '.join(cols)} FROM market_regimes"
        try:
            table = wh.query_df(sql)
            df = table.to_pandas()
        except Exception as exc:  # noqa: BLE001 - missing view degrades, not crashes
            empty = self.fit_sequences({d: [] for d in self.dimensions})
            empty.reason = f"market_regimes unavailable: {type(exc).__name__}: {exc}"
            return empty

        if level is not None and "level" in df.columns:
            df = df[df["level"].astype(str).str.lower() == str(level).strip().lower()]
        if symbol_id is not None and "symbol_id" in df.columns:
            df = df[df["symbol_id"].astype(str) == str(symbol_id)]
        return self.fit(df, time_col=time_col)

    # -- guarded accessors ---------------------------------------------------
    def _matrix(self, dimension: str) -> TransitionMatrix:
        if not self.fitted:
            raise RuntimeError("model is not fitted; call fit*/fit_sequences first")
        if dimension not in self.matrices:
            raise ValueError(f"dimension {dimension!r} was not fitted")
        return self.matrices[dimension]

    def matrix(self, dimension: str) -> TransitionMatrix:
        """Return the fitted ``TransitionMatrix`` for a dimension."""
        return self._matrix(dimension)

    def persistence(self, dimension: Optional[str] = None):
        """Per-state persistence for one dimension, or all dimensions if None."""
        if dimension is not None:
            return self._matrix(dimension).persistence()
        return {d: self._matrix(d).persistence() for d in self.matrices}

    def transition_probability(self, dimension: str, from_state: str,
                               to_state: str) -> float:
        return self._matrix(dimension).transition_probability(from_state, to_state)

    def next_distribution(self, dimension: str, from_state: str) -> dict[str, float]:
        return self._matrix(dimension).next_distribution(from_state)

    def most_likely_next(self, dimension: str, from_state: str) -> tuple[str, float]:
        return self._matrix(dimension).most_likely_next(from_state)

    def transition_probability_interval(self, dimension: str, from_state: str,
                                        to_state: str, *, z: float = 1.96
                                        ) -> tuple[float, float]:
        return self._matrix(dimension).transition_probability_interval(
            from_state, to_state, z=z)

    # -- derived strategy-facing probabilities -------------------------------
    def recovery_probability(self, from_state: Optional[str] = None) -> float:
        """P(trend -> recovery), optionally conditioned on the current trend state."""
        return self._matrix("trend").forward_probability([RECOVERY_STATE], from_state)

    def expansion_probability(self, from_state: Optional[str] = None) -> float:
        """P(trend -> expansion)."""
        return self._matrix("trend").forward_probability([EXPANSION_STATE], from_state)

    def exhaustion_probability(self, from_state: Optional[str] = None) -> float:
        """P(trend -> exhaustion)."""
        return self._matrix("trend").forward_probability([EXHAUSTION_STATE], from_state)

    def volatility_escalation_probability(self, from_state: Optional[str] = None) -> float:
        """P(volatility -> a strictly more severe state)."""
        return self._matrix("volatility").severity_increase_probability(from_state)

    def liquidity_deterioration_probability(self, from_state: Optional[str] = None
                                            ) -> float:
        """P(liquidity -> a strictly worse state)."""
        return self._matrix("liquidity").severity_increase_probability(from_state)

    # -- reporting -----------------------------------------------------------
    def summary(self) -> dict:
        """Telemetry/reporting snapshot (also records where computation ran)."""
        return {
            "version": self.version,
            "fitted": self.fitted,
            "device": self.device,
            "placement_reason": self.placement_reason,
            "alpha": self.alpha,
            "min_observations": self.min_observations,
            "min_transitions": self.min_transitions,
            "dimensions": list(self.dimensions),
            "device_policy": self.policy.summary(),
            "matrices": {d: m.to_dict() for d, m in self.matrices.items()},
            "promoted": False,
        }


def fit_regime_transition_model(warehouse=None, *, level: Optional[str] = None,
                                symbol_id: Optional[object] = None,
                                config=None, alpha: float = 1.0,
                                min_observations: int = 50,
                                min_transitions: int = 30,
                                ) -> tuple[RegimeTransitionModel, TransitionFitResult]:
    """Convenience: build a model and fit it from the silver ``market_regimes`` view."""
    model = RegimeTransitionModel(alpha=alpha, min_observations=min_observations,
                                  min_transitions=min_transitions)
    result = model.fit_from_warehouse(warehouse, level=level, symbol_id=symbol_id,
                                      config=config)
    return model, result


# Re-export for callers that want the raw field type without importing dataclasses.
__all__ = [
    "REGIME_TRANSITION_MODEL_VERSION",
    "DimensionSpec",
    "REGIME_DIMENSIONS",
    "DIRECTION_STATES",
    "TREND_STATES",
    "VOLATILITY_STATES",
    "LIQUIDITY_STATES",
    "TransitionMatrix",
    "TransitionFitResult",
    "RegimeTransitionModel",
    "fit_regime_transition_model",
]
