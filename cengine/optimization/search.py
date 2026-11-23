"""Bounded search spaces + reproducible search methods (Milestone 6B, §9.1/§9.3).

Only parameters inside approved, versioned search spaces are tunable. Hard risk limits and
the mandatory emergency protection are NOT in any search space. Supports random, grid,
successive-halving, and a dependency-free TPE-style sampler that exploits prior history
(the Bayesian option). All sampling is seeded + reproducible; configs are validated and
clipped to bounds before use.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

SEARCH_VERSION = "search-spaces-1.0.0"


@dataclass(frozen=True)
class Choice:
    name: str
    options: tuple

    def sample(self, rng) -> object:
        return self.options[int(rng.integers(len(self.options)))]

    def clip(self, v):
        return v if v in self.options else self.options[0]


@dataclass(frozen=True)
class IntRange:
    name: str
    lo: int
    hi: int
    log: bool = False

    def sample(self, rng) -> int:
        if self.log:
            v = math.exp(rng.uniform(math.log(self.lo), math.log(self.hi)))
            return int(round(v))
        return int(rng.integers(self.lo, self.hi + 1))

    def clip(self, v):
        return int(min(self.hi, max(self.lo, v)))


@dataclass(frozen=True)
class FloatRange:
    name: str
    lo: float
    hi: float
    log: bool = False

    def sample(self, rng) -> float:
        if self.log:
            return float(math.exp(rng.uniform(math.log(self.lo), math.log(self.hi))))
        return float(rng.uniform(self.lo, self.hi))

    def clip(self, v):
        return float(min(self.hi, max(self.lo, v)))


@dataclass(frozen=True)
class SearchSpace:
    name: str
    params: tuple
    version: str = SEARCH_VERSION

    def sample(self, rng) -> dict:
        return {p.name: p.sample(rng) for p in self.params}

    def validate(self, config: dict) -> tuple:
        """Return (ok, reasons). A config must contain only approved params within bounds."""
        reasons = []
        names = {p.name for p in self.params}
        for k in config:
            if k not in names:
                reasons.append(f"unknown_param:{k}")
        for p in self.params:
            if p.name not in config:
                continue
            v = config[p.name]
            if isinstance(p, Choice) and v not in p.options:
                reasons.append(f"out_of_choice:{p.name}")
            elif isinstance(p, (IntRange, FloatRange)) and not (p.lo <= v <= p.hi):
                reasons.append(f"out_of_range:{p.name}")
        return (len(reasons) == 0, reasons)

    def clip(self, config: dict) -> dict:
        out = {}
        pmap = {p.name: p for p in self.params}
        for k, v in config.items():
            out[k] = pmap[k].clip(v) if k in pmap else v
        return out

    def grid_size(self) -> Optional[int]:
        total = 1
        for p in self.params:
            if isinstance(p, Choice):
                total *= len(p.options)
            elif isinstance(p, IntRange):
                total *= (p.hi - p.lo + 1)
            else:
                return None   # continuous -> no finite grid
        return total


# -- approved search spaces --------------------------------------------------

def default_search_spaces() -> dict:
    return {
        "supervised_model": SearchSpace("supervised_model", (
            Choice("model_family", ("mlx_quantile", "mlx_regressor", "sklearn_hgbr", "ridge")),
            IntRange("hidden_width", 16, 128, log=True),
            IntRange("hidden_depth", 1, 3),
            FloatRange("learning_rate", 1e-4, 5e-3, log=True),
            IntRange("batch_size", 512, 8192, log=True),
            IntRange("horizon_bars", 1, 12),
            FloatRange("weight_decay", 1e-5, 1e-2, log=True))),
        "rl_reward": SearchSpace("rl_reward", (
            FloatRange("turnover_penalty", 0.0, 1.0),
            FloatRange("drawdown_penalty", 0.0, 2.0),
            FloatRange("tail_risk_penalty", 0.0, 2.0),
            FloatRange("concentration_penalty", 0.0, 1.0))),
        "protective_exit": SearchSpace("protective_exit", (
            Choice("policy", ("fixed", "atr", "volatility", "regime", "recovery_prob",
                              "rl", "hybrid", "tightening_only")),
            FloatRange("initial_stop_pct", 0.5, 2.5),
            FloatRange("max_stop_pct", 1.0, 2.5),
            FloatRange("atr_multiplier", 0.5, 4.0))),
        "entry_policy": SearchSpace("entry_policy", (
            FloatRange("confidence_threshold", 0.5, 0.95),
            FloatRange("min_relative_volume", 1.0, 3.0))),
        "strategy_policy": SearchSpace("strategy_policy", (
            Choice("regime_conditioning", (True, False)),
            FloatRange("strategy_weight_decay", 0.0, 0.5))),
    }


# -- search methods ----------------------------------------------------------

def random_search(space: SearchSpace, n: int, *, seed: int = 7) -> list:
    rng = np.random.default_rng(seed)
    return [space.clip(space.sample(rng)) for _ in range(max(1, n))]


def grid_search(space: SearchSpace, *, max_points: int = 256) -> list:
    import itertools
    axes = []
    for p in space.params:
        if isinstance(p, Choice):
            axes.append([(p.name, o) for o in p.options])
        elif isinstance(p, IntRange):
            step = max(1, (p.hi - p.lo) // 4)
            axes.append([(p.name, v) for v in range(p.lo, p.hi + 1, step)])
        else:
            axes.append([(p.name, v) for v in np.linspace(p.lo, p.hi, 3)])
    out = []
    for combo in itertools.product(*axes):
        out.append(dict(combo))
        if len(out) >= max_points:
            break
    return out


def successive_halving(candidates: list, eval_fn: Callable[[dict, int], float], *,
                       budgets=(1, 3, 9), keep_fraction: float = 0.34) -> dict:
    """Multi-fidelity: evaluate all at low budget, keep the best fraction, escalate.

    eval_fn(config, budget) -> score (higher is better). Returns {best, survivors, history}.
    """
    survivors = list(candidates)
    history = []
    for budget in budgets:
        scored = [(c, float(eval_fn(c, budget))) for c in survivors]
        scored.sort(key=lambda x: x[1], reverse=True)
        history.append({"budget": budget,
                        "scores": [s for _, s in scored][:10], "n": len(scored)})
        keep = max(1, int(len(scored) * keep_fraction))
        survivors = [c for c, _ in scored[:keep]]
        if len(survivors) == 1:
            break
    best = survivors[0] if survivors else None
    return {"best": best, "survivors": survivors, "history": history}


def tpe_suggest(space: SearchSpace, history: list, n: int, *, seed: int = 7,
                good_quantile: float = 0.25) -> list:
    """Dependency-free TPE-style sampler: split prior (config, score) history into good/bad
    by a score quantile and sample preferentially near good configs. Falls back to random
    when history is thin. This is the 'Bayesian' option — honest about being a lightweight
    surrogate, not a full GP."""
    rng = np.random.default_rng(seed)
    scored = [(h["config"], h["score"]) for h in history
              if h.get("config") and h.get("score") is not None]
    if len(scored) < 8:
        return random_search(space, n, seed=seed)
    scores = np.array([s for _, s in scored])
    cut = np.quantile(scores, 1.0 - good_quantile)
    good = [c for c, s in scored if s >= cut] or [c for c, _ in scored]
    out = []
    for _ in range(max(1, n)):
        base = good[int(rng.integers(len(good)))]
        cand = {}
        for p in space.params:
            if isinstance(p, Choice):
                cand[p.name] = (base.get(p.name, p.options[0])
                                if rng.uniform() < 0.7 else p.sample(rng))
            else:
                anchor = base.get(p.name)
                if anchor is None or rng.uniform() > 0.7:
                    cand[p.name] = p.sample(rng)
                else:
                    span = (p.hi - p.lo) * 0.15
                    cand[p.name] = p.clip(anchor + rng.normal(0, span))
        out.append(space.clip(cand))
    return out
