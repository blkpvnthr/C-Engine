"""Offline contextual-bandit dataset builder over gold ``offline_rl`` (Milestone 6).

Builds a ``(context, selected_action, reward)`` dataset for OFFLINE contextual-bandit
research over a FIXED, caller-supplied set of AUTHORIZED actions (e.g. execution policies
or strategy families). The data is read straight from the lakehouse gold ``offline_rl``
dataset (Parquet via DuckDB — the same mechanism ``DuckDBWarehouse`` uses) so profiling,
training and evaluation all see one authoritative analytical surface.

CRITICAL invariants (enforced in code, not by convention):
  * Actions are NEVER invented. The caller supplies the authorized action list; a logged
    transition whose ``selected_action`` is not in that list is DROPPED (and counted),
    never appended as a new action.
  * Behavior-policy probabilities are UNKNOWN. Gold ``action_probability_if_known`` is
    always NULL and is never fabricated, so this builder does not even read it. Any
    off-policy value estimator that REQUIRES those probabilities (IPS / weighted-IPS /
    doubly-robust / propensity correction) is therefore UNSUPPORTED by construction:
    ``behavior_policy_known`` is False and ``off_policy_estimation_supported`` is False.
    Only DIRECT reward-model scoring (and clearly-labeled offline Thompson sampling) is
    available downstream.
  * Reward is used ONLY where present. Gold ``reward`` is NULL for not-yet-closed
    positions; those rows are retained but masked out of model fitting. No reward is
    fabricated. Mandatory protective exits are ENVIRONMENT DYNAMICS, not agent actions.
  * Insufficient / absent data still yields a well-formed (possibly empty) research-only
    dataset — never a crash and never a promoted model.

Everything here is RECOMMENDATION-ONLY research infrastructure: it produces no orders,
touches no broker, and never sits on the mandatory -0.5% protective-exit path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import duckdb
import numpy as np

from cengine.lakehouse.config import LakehouseConfig

CONTEXTUAL_BANDIT_VERSION = "contextual-bandit-1.0.0"

# Off-policy correction (IPS / doubly-robust) requires logged behavior-policy action
# probabilities. Those are unknown (gold.action_probability_if_known is always NULL) and
# are never fabricated, so off-policy correction is gated OFF platform-wide.
OFF_POLICY_CORRECTION_SUPPORTED = False

# Pre-decision numeric context columns present in gold ``offline_rl``. ``current_weight``
# is genuine pre-decision state. ``target_weight`` is EXCLUDED by default because it is an
# OUTPUT of the chosen allocation policy (the action) and would leak the action into the
# context; callers may opt in explicitly via ``feature_columns`` for ablations.
DEFAULT_CONTEXT_COLUMNS: tuple[str, ...] = ("current_weight",)


def off_policy_correction_reason() -> str:
    """Human-readable reason off-policy (IPS/DR) estimation is unavailable."""
    return ("behavior-policy action probabilities are unknown "
            "(gold.action_probability_if_known is always NULL and never fabricated); "
            "IPS / weighted-IPS / doubly-robust estimators are gated off — use direct "
            "reward-model scoring only")


@dataclass
class BanditDataset:
    """An offline contextual-bandit dataset over a FIXED authorized action set.

    ``contexts`` is ``(n, d)``; ``actions`` holds the index into ``action_names`` for each
    row; ``rewards`` carries the observed reward where ``reward_mask`` is True (and 0.0,
    ignored, elsewhere). ``behavior_policy_known`` is always False — see module docstring.
    """

    contexts: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    reward_mask: np.ndarray
    action_names: list[str]
    feature_names: list[str]
    behavior_policy_known: bool = False
    provenance: dict = field(default_factory=dict)

    @property
    def n_rows(self) -> int:
        return int(self.contexts.shape[0])

    @property
    def n_actions(self) -> int:
        return len(self.action_names)

    @property
    def n_features(self) -> int:
        return len(self.feature_names)

    @property
    def n_labeled(self) -> int:
        """Rows with an observed reward — the only rows used to fit a reward model."""
        return int(np.count_nonzero(self.reward_mask))

    @property
    def off_policy_estimation_supported(self) -> bool:
        """Always False: behavior-policy probabilities are unknown and never fabricated.

        Any IPS / doubly-robust estimator needs logged action probabilities; those are
        NULL in the lakehouse by construction, so off-policy correction is unavailable.
        """
        return OFF_POLICY_CORRECTION_SUPPORTED and self.behavior_policy_known

    def off_policy_estimation_reason(self) -> str:
        return off_policy_correction_reason()

    def labeled(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(X, actions, rewards)`` restricted to rows with an observed reward."""
        m = self.reward_mask
        return self.contexts[m], self.actions[m], self.rewards[m]

    def is_fittable(self) -> bool:
        """True iff there is at least one labeled row and >=1 action/feature present."""
        return self.n_labeled > 0 and self.n_actions > 0 and self.n_features > 0


def _glob(config: LakehouseConfig, layer: str, dataset: str) -> Optional[str]:
    base = config.dataset_path(layer, dataset)
    if not base.exists() or not any(base.rglob("*.parquet")):
        return None
    return str(base / "**" / "*.parquet").replace("'", "''")


def _columns(con: duckdb.DuckDBPyConnection, src: str) -> set:
    try:
        return set(con.sql(f"SELECT * FROM {src} LIMIT 0").to_arrow_table().schema.names)
    except Exception:  # noqa: BLE001
        return set()


def _empty_dataset(action_names: list[str], feature_names: list[str],
                   note: str) -> BanditDataset:
    d = len(feature_names)
    return BanditDataset(
        contexts=np.zeros((0, d), dtype=np.float32),
        actions=np.zeros((0,), dtype=np.int64),
        rewards=np.zeros((0,), dtype=np.float32),
        reward_mask=np.zeros((0,), dtype=bool),
        action_names=list(action_names),
        feature_names=list(feature_names),
        behavior_policy_known=False,
        provenance={"version": CONTEXTUAL_BANDIT_VERSION, "rows_total": 0,
                    "rows_kept": 0, "n_labeled": 0, "dropped_unauthorized": 0,
                    "note": note, "off_policy_correction": off_policy_correction_reason()},
    )


def build_bandit_dataset(
    actions: Sequence[str],
    *,
    config: Optional[LakehouseConfig] = None,
    feature_columns: Optional[Sequence[str]] = None,
    source_layer: str = "gold",
    source_dataset: str = "offline_rl",
) -> BanditDataset:
    """Build an offline contextual-bandit dataset from the lakehouse ``offline_rl`` gold.

    Args:
        actions: the FIXED authorized action list. Logged ``selected_action`` values
            outside this list are dropped (counted in provenance), never invented.
        config: lakehouse config (defaults to :meth:`LakehouseConfig.from_env`).
        feature_columns: pre-decision numeric context columns to use. Defaults to
            :data:`DEFAULT_CONTEXT_COLUMNS`; missing columns are skipped and noted. If
            none resolve, a single constant ``bias_intercept`` feature is used so the
            pipeline still produces a (degenerate, research-only) non-contextual dataset.
        source_layer / source_dataset: override the lakehouse source (testing/ablation).

    Returns:
        A :class:`BanditDataset`. ``action_probability_if_known`` is never read; off-policy
        correction stays unavailable.
    """
    action_names = list(dict.fromkeys(str(a) for a in actions))  # de-dup, preserve order
    requested = list(feature_columns) if feature_columns is not None \
        else list(DEFAULT_CONTEXT_COLUMNS)

    config = config or LakehouseConfig.from_env()
    glob = _glob(config, source_layer, source_dataset)
    if glob is None:
        return _empty_dataset(action_names, requested or ["bias_intercept"],
                              "no_parquet_parts")

    src = f"read_parquet('{glob}', union_by_name=true)"
    con = duckdb.connect()
    try:
        cols = _columns(con, src)
        if "selected_action" not in cols:
            return _empty_dataset(action_names, requested or ["bias_intercept"],
                                  "missing_selected_action_column")
        has_reward = "reward" in cols
        feat_cols = [c for c in requested if c in cols]
        missing_feats = [c for c in requested if c not in cols]
        use_bias = len(feat_cols) == 0
        feature_names = list(feat_cols) if feat_cols else ["bias_intercept"]

        select = ["selected_action"]
        if has_reward:
            select.append("reward")
        select.extend(feat_cols)
        data = con.sql(f"SELECT {', '.join(select)} FROM {src}").to_arrow_table().to_pydict()
    finally:
        con.close()

    raw_actions = [None if a is None else str(a) for a in data["selected_action"]]
    n_all = len(raw_actions)
    if n_all == 0:
        return _empty_dataset(action_names, feature_names, "zero_rows")

    index_of = {name: i for i, name in enumerate(action_names)}
    keep = np.array([a in index_of for a in raw_actions], dtype=bool)
    action_idx_all = np.array([index_of.get(a, -1) for a in raw_actions], dtype=np.int64)

    if has_reward:
        reward_all = np.array(
            [np.nan if v is None else float(v) for v in data["reward"]], dtype=np.float64)
    else:
        reward_all = np.full(n_all, np.nan, dtype=np.float64)
    reward_mask_all = ~np.isnan(reward_all)

    d = len(feature_names)
    X_all = np.zeros((n_all, d), dtype=np.float32)
    missing_values = 0
    if use_bias:
        X_all[:, 0] = 1.0
    else:
        for j, c in enumerate(feat_cols):
            col = np.array([np.nan if v is None else float(v) for v in data[c]],
                           dtype=np.float64)
            miss = np.isnan(col)
            missing_values += int(np.count_nonzero(miss))
            col[miss] = 0.0
            X_all[:, j] = col.astype(np.float32)

    X = X_all[keep]
    a = action_idx_all[keep]
    rew = np.where(reward_mask_all[keep], reward_all[keep], 0.0).astype(np.float32)
    rmask = reward_mask_all[keep]

    provenance = {
        "version": CONTEXTUAL_BANDIT_VERSION,
        "source": f"{source_layer}/{source_dataset}",
        "rows_total": n_all,
        "rows_kept": int(keep.sum()),
        "dropped_unauthorized": int((~keep).sum()),
        "n_labeled": int(np.count_nonzero(rmask)),
        "authorized_actions": list(action_names),
        "feature_columns_used": list(feature_names),
        "feature_columns_missing": missing_feats,
        "missing_feature_values_filled_zero": missing_values,
        "reward_column_present": has_reward,
        "behavior_policy_known": False,
        "off_policy_correction": off_policy_correction_reason(),
        "note": ("recommendation-only research dataset; mandatory protective exits are "
                 "environment dynamics, not agent actions; rewards used only where present"),
    }
    return BanditDataset(contexts=X, actions=a, rewards=rew, reward_mask=rmask,
                         action_names=action_names, feature_names=feature_names,
                         behavior_policy_known=False, provenance=provenance)
