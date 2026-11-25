"""Dataset sufficiency profiling + per-task eligibility gating (Milestone 6, §3).

Before ANY training, independently profiles every learning dataset directly from the
lakehouse (DuckDB over Parquet) and assigns an explicit per-task eligibility state. The
cardinal rule: a model with insufficient evidence receives NO fabricated examples and NO
automatic promotion — it stays UNTRAINED / research-only while evidence accumulates.

Crucially, raw bar counts are NOT treated as independent trading outcomes: 24k historical
bars are ~24k serially-dependent observations of a handful of instruments, not 24k labeled
lifecycles. Each task declares what it actually needs (labeled outcomes, instruments,
sessions, action support, microstructure availability) and is assessed against the honest
profile.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import duckdb

from cengine.lakehouse.config import LakehouseConfig

ELIGIBILITY_VERSION = "dataset-eligibility-1.0.0"


class EligibilityState(str, Enum):
    INSUFFICIENT_DATA = "insufficient_data"
    DATA_QUALITY_FAILED = "data_quality_failed"
    RESEARCH_ELIGIBLE = "research_eligible"       # explore/prototype only; not for a champion
    TRAINING_ELIGIBLE = "training_eligible"       # enough to fit a model
    VALIDATION_ELIGIBLE = "validation_eligible"   # enough for a held-out economic verdict


@dataclass
class DatasetProfile:
    layer: str
    dataset: str
    available: bool = False
    rows: int = 0
    unique_instruments: int = 0
    trading_sessions: int = 0
    date_start: Optional[str] = None
    date_end: Optional[str] = None
    labeled_rows: int = 0                 # realized (non-censored) labels
    censored_rows: int = 0
    reward_rows: int = 0                  # transitions with a non-null reward
    unique_actions: int = 0
    positive_label_rate: Optional[float] = None
    point_in_time_violations: int = 0
    missing_feature_rate: Optional[float] = None
    execution_environments: dict = field(default_factory=dict)  # env -> count
    completed_lifecycles: int = 0
    notes: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


def _glob(config: LakehouseConfig, layer: str, dataset: str) -> Optional[str]:
    base = config.dataset_path(layer, dataset)
    if not base.exists() or not any(base.rglob("*.parquet")):
        return None
    return str(base / "**" / "*.parquet").replace("'", "''")


class DatasetProfiler:
    """Profiles learning datasets straight from the lakehouse Parquet via DuckDB."""

    def __init__(self, config: Optional[LakehouseConfig] = None) -> None:
        self.config = config or LakehouseConfig.from_env()

    def _columns(self, con, src: str) -> set:
        try:
            return set(con.sql(f"SELECT * FROM {src} LIMIT 0").to_arrow_table().schema.names)
        except Exception:  # noqa: BLE001
            return set()

    def _scalar(self, con, sql: str, default=0):
        try:
            v = con.execute(sql).fetchone()[0]
            return v if v is not None else default
        except Exception:  # noqa: BLE001
            return default

    def profile(self, layer: str, dataset: str) -> DatasetProfile:
        prof = DatasetProfile(layer=layer, dataset=dataset)
        glob = _glob(self.config, layer, dataset)
        if glob is None:
            prof.notes.append("no_parquet_parts")
            return prof
        prof.available = True
        src = f"read_parquet('{glob}', union_by_name=true)"
        con = duckdb.connect()
        try:
            cols = self._columns(con, src)
            prof.rows = int(self._scalar(con, f"SELECT count(*) FROM {src}"))
            if "symbol_id" in cols:
                prof.unique_instruments = int(self._scalar(
                    con, f"SELECT count(DISTINCT symbol_id) FROM {src}"))
            # trading sessions + date coverage from any available event-time column
            ts_col = next((c for c in ("event_timestamp_ns", "decision_ts",
                                       "decision_timestamp", "entry_timestamp")
                           if c in cols), None)
            if ts_col:
                prof.trading_sessions = int(self._scalar(
                    con, f"SELECT count(DISTINCT CAST(to_timestamp({ts_col}/1e9) AS DATE)) "
                         f"FROM {src} WHERE {ts_col} IS NOT NULL"))
                rng = con.execute(
                    f"SELECT min(CAST(to_timestamp({ts_col}/1e9) AS DATE)), "
                    f"max(CAST(to_timestamp({ts_col}/1e9) AS DATE)) FROM {src} "
                    f"WHERE {ts_col} IS NOT NULL").fetchone()
                prof.date_start = str(rng[0]) if rng and rng[0] else None
                prof.date_end = str(rng[1]) if rng and rng[1] else None
            # point-in-time correctness
            if {"availability_timestamp_ns", "event_timestamp_ns"} <= cols:
                prof.point_in_time_violations = int(self._scalar(
                    con, f"SELECT count(*) FROM {src} WHERE availability_timestamp_ns "
                         f"< event_timestamp_ns"))
            # supervised labels
            if "label_status" in cols:
                prof.labeled_rows = int(self._scalar(
                    con, f"SELECT count(*) FROM {src} WHERE label_status = 'realized'"))
                prof.censored_rows = int(self._scalar(
                    con, f"SELECT count(*) FROM {src} WHERE label_status != 'realized'"))
            if "label_positive_return" in cols:
                pr = self._scalar(
                    con, f"SELECT avg(CAST(label_positive_return AS DOUBLE)) FROM {src} "
                         f"WHERE label_positive_return IS NOT NULL", default=None)
                prof.positive_label_rate = float(pr) if pr is not None else None
            # offline-RL transitions
            if "reward" in cols:
                prof.reward_rows = int(self._scalar(
                    con, f"SELECT count(*) FROM {src} WHERE reward IS NOT NULL"))
            if "selected_action" in cols:
                prof.unique_actions = int(self._scalar(
                    con, f"SELECT count(DISTINCT selected_action) FROM {src}"))
            # execution environment breakdown
            env_col = "execution_environment" if "execution_environment" in cols else None
            if env_col:
                for row in con.execute(
                        f"SELECT {env_col}, count(*) FROM {src} GROUP BY {env_col}").fetchall():
                    prof.execution_environments[str(row[0])] = int(row[1])
            # completed lifecycles (position_outcomes)
            if "holding_duration_ns" in cols and "net_return" in cols:
                prof.completed_lifecycles = int(self._scalar(
                    con, f"SELECT count(*) FROM {src} WHERE net_return IS NOT NULL"))
        finally:
            con.close()
        return prof


# -- learning-task eligibility specs -----------------------------------------

@dataclass(frozen=True)
class LearningTaskSpec:
    """Declares what a learning task actually needs from the data."""
    task: str
    source_layer: str
    source_dataset: str
    kind: str                              # 'supervised' | 'bandit' | 'offline_rl'
    min_rows_research: int = 200
    min_rows_training: int = 1_000
    min_rows_validation: int = 3_000
    min_instruments: int = 2
    min_sessions: int = 5
    requires_labeled_outcomes: bool = False
    requires_reward: bool = False
    requires_action_support: int = 0       # min distinct actions for bandit/RL
    note: str = ""


def default_task_specs() -> list:
    """The M6 learning tasks and their honest data requirements."""
    return [
        # Forward-return prediction on genuine market bars (the task real bar data supports).
        LearningTaskSpec(
            task="forward_return_prediction", source_layer="bronze",
            source_dataset="market_bars", kind="supervised",
            min_rows_research=500, min_rows_training=2_000, min_rows_validation=5_000,
            note="predicts theoretical forward bar return from technical features; "
                 "NOT realized strategy return under the protective exit"),
        # Strategy-outcome supervised learning (needs realized labeled lifecycles).
        LearningTaskSpec(
            task="strategy_net_return", source_layer="gold", source_dataset="supervised",
            kind="supervised", requires_labeled_outcomes=True,
            min_rows_training=1_000, min_rows_validation=3_000,
            note="realized strategy net return under the mandatory exit"),
        LearningTaskSpec(
            task="probability_positive_return", source_layer="gold",
            source_dataset="supervised", kind="supervised",
            requires_labeled_outcomes=True, min_rows_training=1_000,
            note="calibrated P(net_return>0)"),
        # Regime transition frequencies (needs regime observations).
        LearningTaskSpec(
            task="regime_transition", source_layer="silver",
            source_dataset="market_regimes", kind="supervised",
            min_rows_research=100, min_rows_training=1_000,
            note="transition-frequency model; deterministic engine stays the baseline"),
        # Execution-quality (needs genuine order/fill outcomes).
        LearningTaskSpec(
            task="execution_quality", source_layer="silver",
            source_dataset="transaction_costs", kind="supervised",
            requires_labeled_outcomes=False, min_rows_training=1_000,
            note="expected slippage / shortfall; mock/paper fills carry env limitation"),
        # Contextual bandit over authorized policies (needs reward + action support).
        LearningTaskSpec(
            task="policy_bandit", source_layer="gold", source_dataset="offline_rl",
            kind="bandit", requires_reward=True, requires_action_support=2,
            min_rows_training=1_000,
            note="selects among authorized policies; behavior probs unknown (no IPS)"),
        # Constrained offline RL (needs reward + action support + sequential transitions).
        LearningTaskSpec(
            task="offline_rl_execution", source_layer="gold", source_dataset="offline_rl",
            kind="offline_rl", requires_reward=True, requires_action_support=2,
            min_rows_training=5_000, min_rows_validation=20_000,
            note="sequential execution/holding decisions; conservative methods only"),
        # Universe-selection contextual bandit (§9): needs logged selection
        # decisions + per-symbol realized reward. Shadow-only; deterministic
        # rotation stays the champion.
        LearningTaskSpec(
            task="universe_selection", source_layer="gold",
            source_dataset="universe_selection", kind="bandit",
            requires_reward=True, requires_action_support=2,
            min_rows_research=300, min_rows_training=1_000, min_rows_validation=3_000,
            note="per-candidate include/exclude; reward is realized net return "
                 "attributed to selected symbols (NULL when unobserved, never faked)"),
    ]


def assess_task(spec: LearningTaskSpec, profile: DatasetProfile) -> tuple:
    """Return (EligibilityState, reasons: list[str]) for a task given its data profile."""
    reasons: list[str] = []
    if not profile.available or profile.rows == 0:
        return EligibilityState.INSUFFICIENT_DATA, ["dataset_not_materialized"]
    # data-quality gate first
    if profile.point_in_time_violations > 0:
        reasons.append(f"point_in_time_violations={profile.point_in_time_violations}")
        return EligibilityState.DATA_QUALITY_FAILED, reasons

    # effective sample size = labeled/reward rows when required, else raw rows
    if spec.requires_labeled_outcomes:
        effective = profile.labeled_rows
        reasons.append(f"labeled_rows={profile.labeled_rows} (censored={profile.censored_rows})")
    elif spec.requires_reward:
        effective = profile.reward_rows
        reasons.append(f"reward_rows={profile.reward_rows}")
    else:
        effective = profile.rows
        reasons.append(f"rows={profile.rows}")

    if spec.requires_action_support and profile.unique_actions < spec.requires_action_support:
        reasons.append(f"action_support={profile.unique_actions}"
                       f"<{spec.requires_action_support}")
        return EligibilityState.INSUFFICIENT_DATA, reasons
    if profile.unique_instruments < spec.min_instruments:
        reasons.append(f"instruments={profile.unique_instruments}<{spec.min_instruments}")
    if profile.trading_sessions < spec.min_sessions:
        reasons.append(f"sessions={profile.trading_sessions}<{spec.min_sessions}")

    enough_instruments = profile.unique_instruments >= spec.min_instruments
    enough_sessions = profile.trading_sessions >= spec.min_sessions

    if effective < spec.min_rows_research:
        return EligibilityState.INSUFFICIENT_DATA, reasons
    if effective < spec.min_rows_training or not (enough_instruments and enough_sessions):
        return EligibilityState.RESEARCH_ELIGIBLE, reasons
    if effective < spec.min_rows_validation:
        return EligibilityState.TRAINING_ELIGIBLE, reasons
    return EligibilityState.VALIDATION_ELIGIBLE, reasons


def assess_all(config: Optional[LakehouseConfig] = None, *,
               specs: Optional[list] = None) -> dict:
    """Profile + assess every learning task. Returns a machine-readable report."""
    profiler = DatasetProfiler(config)
    specs = specs or default_task_specs()
    out = {"report": "training_dataset_eligibility", "version": ELIGIBILITY_VERSION,
           "tasks": []}
    cache: dict = {}
    for spec in specs:
        key = (spec.source_layer, spec.source_dataset)
        if key not in cache:
            cache[key] = profiler.profile(*key)
        prof = cache[key]
        state, reasons = assess_task(spec, prof)
        out["tasks"].append({
            "task": spec.task, "kind": spec.kind,
            "source": f"{spec.source_layer}/{spec.source_dataset}",
            "eligibility": state.value, "reasons": reasons,
            "note": spec.note, "profile": prof.to_dict()})
    return out
