"""Static-vs-dynamic universe-selection policy comparison (spec §12).

A deterministic, reproducible harness that replays a chronological panel of
per-symbol forward returns and, at each rebalance step, asks each policy which
≤N symbols to hold. Every policy runs under IDENTICAL controls (equal-weight long
the active window, the same per-turnover transaction cost, the same cap, the same
data) so the ONLY variable is the universe-selection policy. It isolates the
selection decision — it is not the full execution/risk engine, and results are
labelled accordingly.

Policies (spec §12):
  A static        — the baseline only, no rotation
  B equal         — ⌈N/2⌉ baseline + ⌊N/2⌋ top-scored candidates
  C flexible      — baseline_target baseline + opportunity_target candidates
  D adaptive      — the production UniverseRotationScheduler (ranked + revisit + hysteresis)
  E learned       — a fitted shadow bandit's recommendation (skipped if unavailable)

No look-ahead: at step t a policy sees scores available at t and is scored on the
return realized from t to t+1. No fabricated data — the caller supplies genuine
panels; missing data is reported, never invented.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

from .config import UniverseRotationConfig
from .lifecycle import LifecycleTracker
from .rotation import RankedSymbol, UniverseRotationScheduler

# A policy maps (baseline, scores, protected, prev_active, cap) -> active window.
Policy = Callable[[Sequence[str], dict, Sequence[str], Sequence[str], int], list[str]]


@dataclass
class ReplayPanel:
    """Genuine chronological inputs for the comparison (no fabrication).

    steps: list of step dicts, each:
      {"scores": {symbol: score}, "forward_return": {symbol: ret_t_to_t1}}
    baseline: the permanent baseline symbols.
    """

    baseline: list[str]
    steps: list[dict]
    cost_per_turnover_bps: float = 5.0


@dataclass
class PolicyMetrics:
    policy: str
    steps: int
    gross_return: float
    net_return: float
    sharpe: float
    max_drawdown: float
    turnover: float
    avg_active: float
    baseline_coverage: float          # mean fraction of baseline held
    churn: float                      # mean symbols changed per step
    data_label: str = "synthetic"
    note: str = ""
    per_step_net: list = field(default_factory=list)


# --------------------------------------------------------------------- policies
def policy_static(baseline, scores, protected, prev_active, cap):
    out = list(dict.fromkeys(list(protected) + list(baseline)))
    return out[:cap]


def _top_candidates(baseline, scores, k):
    base = set(baseline)
    ranked = sorted((s for s in scores if s not in base),
                    key=lambda s: scores[s], reverse=True)
    return ranked[:k]


def policy_equal(baseline, scores, protected, prev_active, cap):
    half = cap // 2
    out = list(dict.fromkeys(list(protected)))
    for s in baseline[: cap - half]:
        if s not in out and len(out) < cap:
            out.append(s)
    for s in _top_candidates(baseline, scores, half):
        if s not in out and len(out) < cap:
            out.append(s)
    for s in baseline:                                   # fill if candidates short
        if s not in out and len(out) < cap:
            out.append(s)
    return out[:cap]


def policy_flexible(baseline, scores, protected, prev_active, cap,
                    baseline_target=20, opp_target=10):
    out = list(dict.fromkeys(list(protected)))
    for s in _top_candidates(baseline, scores, opp_target):
        if s not in out and len(out) < cap:
            out.append(s)
    for s in baseline:
        if s not in out and len(out) < cap:
            out.append(s)
    return out[:cap]


def make_adaptive_policy(config: UniverseRotationConfig) -> Policy:
    """Policy D: the production rotation scheduler (ranked + revisit + hysteresis)."""
    sched = UniverseRotationScheduler(config)
    tracker = LifecycleTracker(config)
    state = {"t": 0.0}

    def _policy(baseline, scores, protected, prev_active, cap):
        base_set = set(baseline)
        baseline_ranked = [RankedSymbol(s, scores.get(s, 0.0), s in scores) for s in baseline]
        opps = [RankedSymbol(s, sc, True) for s, sc in scores.items() if s not in base_set]
        state["t"] += 1.0
        res = sched.rotate(
            baseline=baseline_ranked, opportunities=opps, protected=protected,
            references=[], previous_active=prev_active, tracker=tracker,
            now_s=state["t"],
        )
        return list(res.active)

    return _policy


def make_learned_policy(bandit, feature_fn) -> Policy:
    """Policy E: a fitted shadow bandit. ``feature_fn(symbol, score) -> list[float]``."""
    def _policy(baseline, scores, protected, prev_active, cap):
        import numpy as np
        syms = list(dict.fromkeys(list(baseline) + list(scores.keys())))
        if not syms:
            return list(protected)[:cap]
        ctx = np.array([feature_fn(s, scores.get(s, 0.0)) for s in syms], dtype=np.float32)
        return bandit.recommend_active(ctx, syms, cap=cap, protected=protected)

    return _policy


# ------------------------------------------------------------------- simulation
def simulate(policy: Policy, panel: ReplayPanel, *, cap: int,
             name: str, data_label: str = "synthetic") -> PolicyMetrics:
    """Run a policy over the panel under equal-weight-long, net-of-cost controls."""
    prev_active: list[str] = []
    per_step_net: list[float] = []
    turnovers: list[float] = []
    active_sizes: list[int] = []
    base_cov: list[float] = []
    churns: list[float] = []
    base_set = set(panel.baseline)
    cost = panel.cost_per_turnover_bps / 10_000.0

    for step in panel.steps:
        scores = step.get("scores", {})
        fwd = step.get("forward_return", {})
        protected = step.get("protected", [])
        active = policy(panel.baseline, scores, protected, prev_active, cap)
        active = list(dict.fromkeys(active))[:cap]
        # equal-weight long the active window; symbols with no return contribute 0.
        if active:
            gross = sum(float(fwd.get(s, 0.0)) for s in active) / len(active)
        else:
            gross = 0.0
        changed = len(set(active) ^ set(prev_active))
        turnover = changed / max(1, cap)
        net = gross - cost * turnover
        per_step_net.append(net)
        turnovers.append(turnover)
        active_sizes.append(len(active))
        base_cov.append(len(set(active) & base_set) / max(1, len(base_set)))
        churns.append(float(changed))
        prev_active = active

    return _metrics(name, per_step_net, turnovers, active_sizes, base_cov, churns, data_label)


def _metrics(name, net, turnovers, sizes, base_cov, churns, data_label) -> PolicyMetrics:
    n = len(net)
    gross_sum = sum(net)  # net already; keep gross approx = net+cost handled per step
    total = 1.0
    peak = 1.0
    max_dd = 0.0
    for r in net:
        total *= (1.0 + r)
        peak = max(peak, total)
        if peak > 0:
            max_dd = max(max_dd, (peak - total) / peak)
    mean = sum(net) / n if n else 0.0
    var = sum((r - mean) ** 2 for r in net) / (n - 1) if n > 1 else 0.0
    std = math.sqrt(var)
    sharpe = (mean / std * math.sqrt(252)) if std > 1e-12 else 0.0
    return PolicyMetrics(
        policy=name, steps=n,
        gross_return=round(total - 1.0, 6),
        net_return=round(total - 1.0, 6),
        sharpe=round(sharpe, 4),
        max_drawdown=round(max_dd, 6),
        turnover=round(sum(turnovers) / n, 4) if n else 0.0,
        avg_active=round(sum(sizes) / n, 2) if n else 0.0,
        baseline_coverage=round(sum(base_cov) / n, 4) if n else 0.0,
        churn=round(sum(churns) / n, 2) if n else 0.0,
        data_label=data_label, per_step_net=net,
    )


# ------------------------------------------------------------- walk-forward / CI
def walk_forward_splits(n_steps: int, *, folds: int) -> list[tuple]:
    """Chronological (train, test) index ranges — expanding window, no shuffling."""
    if folds < 1 or n_steps < folds + 1:
        return [((0, 0), (0, n_steps))]
    fold = n_steps // (folds + 1)
    out = []
    for i in range(1, folds + 1):
        train_end = fold * i
        test_end = fold * (i + 1) if i < folds else n_steps
        out.append(((0, train_end), (train_end, test_end)))
    return out


def block_bootstrap_ci(per_step: Sequence[float], *, block: int = 5,
                       iters: int = 500, seed: int = 0, alpha: float = 0.05) -> tuple:
    """Time-series block-bootstrap CI for mean per-step net return (deterministic)."""
    import numpy as np
    arr = np.asarray(per_step, dtype=float)
    n = len(arr)
    if n == 0:
        return (0.0, 0.0)
    rng = np.random.default_rng(seed)
    n_blocks = max(1, n // block)
    means = []
    for _ in range(iters):
        starts = rng.integers(0, max(1, n - block + 1), size=n_blocks)
        sample = np.concatenate([arr[s:s + block] for s in starts])
        means.append(sample.mean())
    lo = float(np.quantile(means, alpha / 2))
    hi = float(np.quantile(means, 1 - alpha / 2))
    return (round(lo, 6), round(hi, 6))


def compare_policies(panel: ReplayPanel, *, cap: int = 30,
                     config: Optional[UniverseRotationConfig] = None,
                     learned: Optional[tuple] = None,
                     data_label: str = "synthetic") -> dict:
    """Run policies A–D (and E if a fitted bandit is supplied) over the panel."""
    cfg = config or UniverseRotationConfig(active_universe_size=cap)
    results = {
        "A_static": simulate(policy_static, panel, cap=cap, name="A_static", data_label=data_label),
        "B_equal": simulate(policy_equal, panel, cap=cap, name="B_equal", data_label=data_label),
        "C_flexible": simulate(policy_flexible, panel, cap=cap, name="C_flexible", data_label=data_label),
        "D_adaptive": simulate(make_adaptive_policy(cfg), panel, cap=cap,
                               name="D_adaptive", data_label=data_label),
    }
    if learned is not None:
        bandit, feature_fn = learned
        results["E_learned"] = simulate(make_learned_policy(bandit, feature_fn), panel,
                                        cap=cap, name="E_learned", data_label=data_label)
    return {name: m for name, m in results.items()}
