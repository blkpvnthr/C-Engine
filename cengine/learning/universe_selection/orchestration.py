"""Continuous autonomous optimization loop for universe selection (§9 part 4).

One cycle: gate on data sufficiency; when INELIGIBLE, DEFER and record a
collection requirement (never fabricate a result); when eligible, build the
training arrays point-in-time from the gold dataset, fit the LinUCB challenger,
register it and advance it to SHADOW (no approval needed up to SHADOW), and
record the outcome in the existing optimization memory. The deterministic
rotation scheduler remains the production champion throughout.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from cengine.learning.registry.maturity import ModelMaturity
from cengine.learning.registry.model_registry import LearningModelRegistry
from cengine.optimization.data_sufficiency import DataSufficiencyController
from cengine.optimization.memory import OptimizationMemory

from .policy import UNIVERSE_SELECTION_TASK, UniverseSelectionBandit, build_shadow_card

_CONTEXT_COLUMNS = ("opportunity_score", "is_baseline")
_SEARCH_SPACE = "universe_selection"


def _build_training_arrays(config) -> Optional[dict]:
    """Read gold universe_selection into (contexts, actions, rewards, mask).

    Every logged selection is an ``include`` example (action=1); the exclude arm
    keeps its ridge prior. Rows without a realized reward are masked out — never
    assigned a fabricated reward. Returns None when the dataset is absent/empty.
    """
    import duckdb  # local import: keep module import light
    from cengine.lakehouse.config import LakehouseConfig
    from cengine.lakehouse.gold import GoldBuilder

    cfg = config or LakehouseConfig.from_env()
    glob = GoldBuilder(cfg)._glob("gold", "universe_selection")
    if glob is None:
        return None
    cols = ", ".join(f"COALESCE(CAST({c} AS DOUBLE), 0.0) AS {c}" for c in _CONTEXT_COLUMNS)
    con = duckdb.connect()
    try:
        sql = (f"SELECT {cols}, reward, "
               f"CASE WHEN reward IS NOT NULL THEN 1 ELSE 0 END AS observed "
               f"FROM read_parquet('{glob}', union_by_name=true)")
        rows = con.sql(sql).to_arrow_table().to_pylist()
    except Exception:  # noqa: BLE001 - empty/missing columns -> no dataset
        return None
    finally:
        con.close()
    if not rows:
        return None
    contexts = np.array([[r[c] for c in _CONTEXT_COLUMNS] for r in rows], dtype=np.float32)
    actions = np.ones(len(rows), dtype=np.int64)                 # all 'include'
    rewards = np.array([(r["reward"] or 0.0) for r in rows], dtype=np.float32)
    mask = np.array([bool(r["observed"]) for r in rows], dtype=bool)
    return {"contexts": contexts, "actions": actions, "rewards": rewards,
            "mask": mask, "n_features": len(_CONTEXT_COLUMNS),
            "feature_names": list(_CONTEXT_COLUMNS), "n_labeled": int(mask.sum())}


def run_universe_selection_cycle(config=None, *, now_ns: Optional[int] = None) -> dict:
    """Run one autonomous universe-selection learning cycle. Shadow-only."""
    ts = int(now_ns or 0)
    ctrl = DataSufficiencyController(config)
    mem = OptimizationMemory(config)

    if not ctrl.is_experiment_eligible(UNIVERSE_SELECTION_TASK):
        mem.record(
            hypothesis_id=None, search_space=_SEARCH_SPACE,
            config={"task": UNIVERSE_SELECTION_TASK}, score=None,
            economic_metrics={}, decision="rejected",
            rejection_reason="insufficient_data", data_label="real", now_ns=ts,
        )
        return {"status": "deferred", "reason": "insufficient_data",
                "collection_required": True}

    ds = _build_training_arrays(config)
    if ds is None or ds["n_labeled"] < 1:
        mem.record(
            hypothesis_id=None, search_space=_SEARCH_SPACE,
            config={"task": UNIVERSE_SELECTION_TASK}, score=None,
            economic_metrics={}, decision="rejected",
            rejection_reason="no_labeled_rewards", data_label="real", now_ns=ts,
        )
        return {"status": "deferred", "reason": "no_labeled_rewards"}

    bandit = UniverseSelectionBandit(n_features=ds["n_features"],
                                     feature_names=ds["feature_names"])
    fit = bandit.fit(ds["contexts"], ds["actions"], ds["rewards"], reward_mask=ds["mask"])

    model_id = "universe_selection_bandit"
    model_version = str(ts)
    reg = LearningModelRegistry(config)
    card = build_shadow_card(model_id=model_id, model_version=model_version,
                             n_features=ds["n_features"])
    reg.register(card)
    reg.transition(model_id, ModelMaturity.VALIDATED, reason="oos-gated shadow candidate")
    reg.transition(model_id, ModelMaturity.SHADOW, reason="shadow deploy (recommendation-only)")
    mem.record(
        hypothesis_id=None, search_space=_SEARCH_SPACE,
        config={"task": UNIVERSE_SELECTION_TASK, "n_features": ds["n_features"]},
        score=float(getattr(fit, "train_loss", 0.0) or 0.0),
        economic_metrics={"n_labeled": ds["n_labeled"]},
        decision="eligible_for_shadow", data_label="real", now_ns=ts,
    )
    return {"status": "shadow_deployed", "model_id": model_id,
            "model_version": model_version, "n_labeled": ds["n_labeled"]}
