"""Baseline hierarchy for strategy economic certification (§8).

A strategy edge is only real if it out-performs the trivial alternatives on the SAME window
under the SAME cost model. We build, from genuine 1-minute bars:

* **hold_to_horizon** — the exact same signal entries, but with NO protective stop (exit at the
  max-hold time stop). Isolates the economic *effect of the −0.5% protective exit* vs simply
  holding the entries.
* **buy_and_hold** — enter once at the window start, exit at the window end. The market baseline.
* **random_placebo** — the same NUMBER of entries at random times, run through the identical
  exit simulator and cost model. If random entries score as well as the signal entries, the
  entry signal carries no edge (a placebo / falsification control, per CLAUDE.md).
* **always_flat** — zero returns. Cash. The do-nothing floor.

Everything is net of realistic costs and labeled ``historical_replay`` — simulated outcomes on
genuine bars, never a live-profitability claim.
"""

from __future__ import annotations

import numpy as np

from cengine.certification.contracts import stable_id
from cengine.learning.evaluation import DATA_LABEL_HISTORICAL_REPLAY
from cengine.optimization.economic import evaluate_candidate
from cengine.optimization.exits.experiment_matrix import _simulate

BASELINES_VERSION = "certification-baselines-1.0.0"
_MAX_HOLD = 120


def hold_to_horizon_returns(bars: dict, entries: list, atr_frac, *, cost_frac: float,
                            max_hold: int = _MAX_HOLD) -> list:
    """Same entries, no protective stop: exit at the time horizon. Isolates the stop's effect.

    Implemented as a ``fixed`` policy with an effectively-unreachable stop so the exit simulator
    always reaches the time stop (keeping identical entry/cost/fill conventions)."""
    return _simulate(bars, entries, atr_frac, "fixed", {"stop": 0.999},
                     asset_class="large_cap", cost_frac=cost_frac, max_hold=max_hold)


def buy_and_hold_returns(bars: dict, *, cost_frac: float) -> list:
    """Single enter-at-start / exit-at-end holding of the symbol over the window."""
    c = bars["c"].astype(np.float64)
    o = bars["o"].astype(np.float64)
    if len(c) < 3:
        return []
    entry = o[1] * (1.0 + cost_frac)
    exit_px = c[-1]
    return [exit_px / entry - 1.0 - cost_frac]


def random_placebo_returns(bars: dict, atr_frac, n_entries: int, *, cost_frac: float,
                           symbol: str, max_hold: int = _MAX_HOLD) -> list:
    """``n_entries`` random entry times through the incumbent (−0.5%) exit simulator.

    Seeded deterministically off the symbol so the placebo is reproducible across runs."""
    n = len(bars["c"])
    if n < max_hold + 5 or n_entries <= 0:
        return []
    seed = int(stable_id("placebo", symbol), 16) % (2**32)
    rng = np.random.default_rng(seed)
    hi = n - max_hold - 2
    if hi <= 30:
        return []
    picks = sorted(set(rng.integers(30, hi, size=min(n_entries, hi - 30)).tolist()))
    return _simulate(bars, picks, atr_frac, "fixed", {"stop": 0.005},
                     asset_class="large_cap", cost_frac=cost_frac, max_hold=max_hold)


def evaluate_baselines(bars: dict, entries: list, atr_frac, *, cost_frac: float,
                       symbol: str) -> dict:
    """Return {name: CandidateEvaluation} for every baseline, scored on this window."""
    label = DATA_LABEL_HISTORICAL_REPLAY
    out: dict = {}
    hh = hold_to_horizon_returns(bars, entries, atr_frac, cost_frac=cost_frac)
    out["hold_to_horizon"] = evaluate_candidate(hh, label="hold_to_horizon", data_label=label)
    bh = buy_and_hold_returns(bars, cost_frac=cost_frac)
    if bh:
        out["buy_and_hold"] = evaluate_candidate(bh, label="buy_and_hold", data_label=label)
    pl = random_placebo_returns(bars, atr_frac, len(entries), cost_frac=cost_frac, symbol=symbol)
    if pl:
        out["random_placebo"] = evaluate_candidate(pl, label="random_placebo", data_label=label)
    out["always_flat"] = evaluate_candidate([0.0] * max(len(entries), 1),
                                            label="always_flat", data_label=label)
    return out
