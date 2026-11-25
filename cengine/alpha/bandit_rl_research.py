"""Contextual-bandit (§11) and offline-RL (§12) research.

Both techniques need data C-Engine does not yet have from genuine operation:

* **Contextual bandit (§11)** — governed selection among eligible arms (strategy families, entry
  filters, ...) requires logged decisions with **behavior-policy propensities** and realized
  rewards. No live/paper decision log exists, so off-policy estimators (IPS/DR) are not valid and
  are correctly refused (never fabricate action probabilities). What IS possible, research-only, is
  a **direct-method** evaluation on historical replay where the context is the regime proxy and the
  reward is the family's realized net return: pick the best family per regime on TRAIN and measure
  the resulting policy on VALIDATION. This reuses the family×regime evidence; it is development
  research, not a certified off-policy estimate.

* **Offline RL (§12)** — BC/CQL/IQL/FQE need genuine transition tuples from executed experience.
  Historical replay provides single-step bandit-like transitions only (no genuine
  state→action→reward→next-state from real execution), so sequential-policy certification is
  **INSUFFICIENT_DATA** and deferred until genuine (paper/live) transition data accrues. The M6
  implementations remain available and unit-tested.

No live exploration, no fabricated propensities or rewards. Historical-replay, not live.
"""

from __future__ import annotations

import numpy as np

from cengine.alpha.family_research import FAMILIES
from cengine.alpha.features import build_entry_dataset, compute_features
from cengine.alpha.portfolio_replay import resolve_exit
from cengine.alpha.regime_research import _regime_label
from cengine.certification.contracts import DATA_LABEL_HISTORICAL_REPLAY, safe_ticker
from cengine.optimization.exits.experiment_matrix import _load_1m

BANDIT_RL_VERSION = "bandit-rl-research-1.0.0"
_MIN_CELL = 50


def run_contextual_bandit_research(symbols: list, *, config=None) -> dict:
    """Direct-method, research-only contextual bandit: best family per regime (TRAIN) -> VAL."""
    if config is None:
        from cengine.lakehouse.config import LakehouseConfig
        config = LakehouseConfig.from_env()
    # collect (regime, family, reward, split) rows across symbols
    atr_all = []
    rows = []
    for sym in symbols:
        s = safe_ticker(sym)
        bars = _load_1m(config, s)
        if bars is None:
            continue
        feats = compute_features(bars)
        atr_all.append(feats["atr_frac"][np.isfinite(feats["atr_frac"])])
        for fam, gen in FAMILIES.items():
            try:
                entries = gen(bars)
            except Exception:  # noqa: BLE001
                continue
            if not entries:
                continue
            ds = build_entry_dataset(bars, entries)
            n = len(ds["y"])
            if n < 20:
                continue
            cut = int(n * 0.70)
            tp = ds["X"]["trend_persist"]
            af = feats["atr_frac"][np.array(ds["entry_idx"], dtype=np.int64)]
            for j in range(n):
                rows.append((tp[j], af[j], fam, ds["y"][j], "train" if j < cut else "val"))
    if not rows:
        return {"version": BANDIT_RL_VERSION, "available": False, "note": "no data"}
    atr_cat = np.concatenate(atr_all) if atr_all else np.array([0.0])
    vol_lo, vol_hi = float(np.quantile(atr_cat, 0.33)), float(np.quantile(atr_cat, 0.66))

    train_cell: dict = {}
    val_cell: dict = {}
    for tp, af, fam, y, split in rows:
        reg = _regime_label(tp, af, vol_lo, vol_hi)
        (train_cell if split == "train" else val_cell).setdefault((reg, fam), []).append(y)

    # learn best arm per regime on TRAIN
    regimes = sorted({r for (r, _f) in train_cell})
    policy = {}
    for reg in regimes:
        arms = {fam: float(np.mean(v)) for (r, fam), v in train_cell.items()
                if r == reg and len(v) >= _MIN_CELL}
        if arms:
            best_fam = max(arms, key=arms.get)
            policy[reg] = {"arm": best_fam, "train_expectancy": round(arms[best_fam], 6)}

    # evaluate the learned policy on VAL (direct method), plus a uniform-random baseline value
    pol_rewards, uni_rewards = [], []
    for (reg, fam), v in val_cell.items():
        uni_rewards.extend(v)                       # uniform over observed arms = all rewards
        if reg in policy and policy[reg]["arm"] == fam:
            pol_rewards.extend(v)
    pol_val = float(np.mean(pol_rewards)) if pol_rewards else None
    uni_val = float(np.mean(uni_rewards)) if uni_rewards else None
    return {
        "version": BANDIT_RL_VERSION, "method": "direct_method (research-only)",
        "data_label": DATA_LABEL_HISTORICAL_REPLAY,
        "ips_or_dr_possible": False,
        "ips_refused_reason": "no logged behavior-policy propensities (never fabricated)",
        "regime_arm_policy": policy,
        "policy_val_expectancy": round(pol_val, 6) if pol_val is not None else None,
        "uniform_random_val_expectancy": round(uni_val, 6) if uni_val is not None else None,
        "policy_positive": bool(pol_val is not None and pol_val > 0),
        "certification_status": "INSUFFICIENT_DATA",
        "conclusion": (
            "Direct-method contextual-bandit research (best family per regime) "
            + (f"reaches {pol_val:+.5f} VAL vs uniform {uni_val:+.5f}" if pol_val is not None
               else "could not be evaluated")
            + "; off-policy (IPS/DR) certification is not possible without logged propensities. "
            "Research-only; certification INSUFFICIENT_DATA."),
    }


def run_offline_rl_research(symbols: list, *, config=None) -> dict:
    """Honest status of offline-RL applicability (no genuine transition data yet)."""
    if config is None:
        from cengine.lakehouse.config import LakehouseConfig
        config = LakehouseConfig.from_env()
    # count available single-step transitions (bandit-like) on replay — NOT genuine RL transitions
    transitions = 0
    for sym in symbols:
        s = safe_ticker(sym)
        bars = _load_1m(config, s)
        if bars is None:
            continue
        from cengine.optimization.exits.experiment_matrix import _entries_and_atr
        for e in _entries_and_atr(bars)[0]:
            if resolve_exit(bars, e, cost_frac=0.0004, protective_stop=0.005,
                            emergency_stop=0.03, max_hold=120) is not None:
                transitions += 1
    return {
        "version": BANDIT_RL_VERSION, "data_label": DATA_LABEL_HISTORICAL_REPLAY,
        "genuine_execution_transitions": 0,
        "historical_replay_single_step_transitions": transitions,
        "sequential_transition_data_available": False,
        "methods_available_unit_tested": ["behavior_cloning", "CQL", "IQL", "FQE"],
        "certification_status": "INSUFFICIENT_DATA",
        "conclusion": (
            "Offline RL (BC/CQL/IQL/FQE) requires genuine state→action→reward→next-state "
            "transitions from executed experience. Historical replay yields single-step "
            f"bandit-like transitions only ({transitions} available) and zero genuine EXECUTION "
            "transitions, so "
            "sequential-policy certification is INSUFFICIENT_DATA and is deferred until paper/live "
            "transition data accrues. No unrestricted live RL; no fabricated rewards."),
    }
