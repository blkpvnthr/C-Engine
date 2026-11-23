"""Exit-policy optimization runner for the MetaOptimizer (Milestone 6B, §38).

Evaluates a proposed protective-exit search config via multi-timeframe historical replay on
GENUINE 1m bars, OUT-OF-SAMPLE: candidate configs are selected/fit on the TRAIN window and
scored on the VALIDATION window, always against the EXIT-00 (-0.5%) incumbent + a simple
baseline on the SAME window. The untouched holdout is never accessed. This closes the
autonomous loop — the optimizer proposes exit configs, this runner returns an independent
economic verdict the controller accepts (shadow-only) or rejects.
"""

from __future__ import annotations

import time
from typing import Optional

from cengine.lakehouse.config import LakehouseConfig
from cengine.learning.evaluation import DATA_LABEL_HISTORICAL_REPLAY
from cengine.optimization.economic import evaluate_candidate
from cengine.optimization.exits.experiment_matrix import (
    _entries_and_atr,
    _load_1m,
    _simulate,
)
from cengine.optimization.validation import oos_windows

# search policies with no trained model/context yet -> runner declines (deferred, honest).
_RESEARCH_ONLY = {"recovery_prob", "rl", "hybrid"}
# map a search-space 'policy' to the simulator kind.
_POLICY_KIND = {"fixed": "fixed", "atr": "atr", "volatility": "volatility",
                "regime": "regime", "tightening_only": "tightening_only"}


def _config_to_sim(config: dict) -> Optional[tuple]:
    policy = config.get("policy", "fixed")
    if policy in _RESEARCH_ONLY:
        return None
    kind = _POLICY_KIND.get(policy, "fixed")
    stop = float(config.get("initial_stop_pct", 1.0)) / 100.0   # percent -> fraction
    if kind == "fixed":
        return "fixed", {"stop": stop}
    if kind == "atr":
        return "atr", {"mult": float(config.get("atr_multiplier", 2.0))}
    if kind == "volatility":
        return "volatility", {"mult": float(config.get("atr_multiplier", 2.5))}
    if kind == "regime":
        return "regime", {"calm": max(0.005, stop * 0.6), "volatile": stop}
    if kind == "tightening_only":
        return "tightening_only", {"initial": max(stop, 0.01), "floor": 0.005}
    return "fixed", {"stop": stop}


def make_exit_runner(symbol: str, *, config: Optional[LakehouseConfig] = None,
                     asset_class: str = "large_cap", cost_frac: float = 0.0004,
                     risk_normalized: bool = False):
    """Build a MetaOptimizer runner for the 'protective_exit' space on one symbol's genuine 1m
    bars with chronological train/validation windows (holdout reserved)."""
    config = config or LakehouseConfig.from_env()
    windows = oos_windows(config, symbol)
    cache: dict = {}

    def _bars(window):
        key = window
        if key not in cache:
            b = _load_1m(config, symbol, start_ns=window[0], end_ns=window[1])
            cache[key] = (b, (_entries_and_atr(b) if b else (None, None)))
        return cache[key]

    def _returns(window, kind, params):
        b, (entries, atr_frac) = _bars(window)
        if b is None or not entries:
            return None
        return _simulate(b, entries, atr_frac, kind, params, asset_class=asset_class,
                         cost_frac=cost_frac, risk_normalized=risk_normalized)

    def runner(space: str, cfg: dict, now_ns: int):
        if space != "protective_exit" or windows is None:
            return None
        sim = _config_to_sim(cfg)
        if sim is None:
            return None   # research-only policy -> deferred
        kind, params = sim
        t0 = time.perf_counter()
        # TRAIN sanity (must produce enough trades) then VALIDATION out-of-sample scoring
        train_ret = _returns(windows.train, kind, params)
        if not train_ret or len(train_ret) < 30:
            return None
        cand_ret = _returns(windows.validation, kind, params)
        inc_ret = _returns(windows.validation, "fixed", {"stop": 0.005})       # EXIT-00
        simple_ret = _returns(windows.validation, "fixed", {"stop": 0.010})    # baseline
        if not cand_ret or not inc_ret:
            return None
        cand = evaluate_candidate(cand_ret, label="candidate",
                                  data_label=DATA_LABEL_HISTORICAL_REPLAY)
        inc = evaluate_candidate(inc_ret, label="incumbent",
                                 data_label=DATA_LABEL_HISTORICAL_REPLAY)
        simple = evaluate_candidate(simple_ret or [], label="simple_statistical",
                                    data_label=DATA_LABEL_HISTORICAL_REPLAY)
        return {"evaluation": cand.to_dict(),
                "baselines": {"incumbent": inc.to_dict(),
                              "simple_statistical": simple.to_dict()},
                "data_label": DATA_LABEL_HISTORICAL_REPLAY,
                "runtime_s": round(time.perf_counter() - t0, 3), "device": "cpu",
                "evaluation_window": f"{windows.validation[0]}..{windows.validation[1]}",
                "training_window": f"{windows.train[0]}..{windows.train[1]}",
                "reward_definition": "net_return_after_costs"}

    return runner, windows
