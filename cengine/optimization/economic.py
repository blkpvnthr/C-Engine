"""Independent economic evaluation + gated scoring (Milestone 6B, §19).

Every optimization candidate is ultimately judged on transaction-cost-adjusted, risk-
adjusted TRADING performance — predictive metrics are supporting diagnostics only. Reuses
the Milestone 6 economic metrics + dependence-aware bootstrap CIs, and adds: a versioned
economic SCORE, MANDATORY risk gates that a high return can never buy its way past, and a
baseline hierarchy. The scoring method is fixed before candidates are observed; a high
return cannot compensate for a violated hard risk constraint.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from cengine.learning.evaluation import (
    DATA_LABEL_REAL,
    DATA_LABEL_SYNTHETIC,
    bootstrap_metric_ci,
    economic_metrics,
)

ECONOMIC_EVAL_VERSION = "optimization-economic-eval-1.0.0"


@dataclass
class EconomicScoreConfig:
    drawdown_penalty: float = 1.0
    tail_risk_penalty: float = 0.5
    turnover_penalty: float = 0.1
    instability_penalty: float = 0.5
    # mandatory gates (a candidate violating any is rejected regardless of return)
    max_drawdown_limit: float = 0.25          # 25% max drawdown ceiling
    min_sample: int = 30
    require_no_risk_expansion: bool = True


@dataclass
class CandidateEvaluation:
    label: str
    data_label: str
    n: int
    metrics: dict
    score: Optional[float]
    net_return_ci: Optional[tuple]
    gates_passed: bool
    gate_failures: list = field(default_factory=list)
    version: str = ECONOMIC_EVAL_VERSION

    def to_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items()}
        return d


def economic_score(metrics, cfg: EconomicScoreConfig) -> float:
    """Aggregate economic score (fixed formula; raw metrics preserved separately)."""
    net = float(getattr(metrics, "net_return", 0.0) or 0.0)
    dd = abs(float(getattr(metrics, "max_drawdown", 0.0) or 0.0))
    sortino = float(getattr(metrics, "sortino", 0.0) or 0.0)
    turnover = float(getattr(metrics, "turnover", 0.0) or 0.0)
    # instability proxy: negative sortino contribution
    instability = max(0.0, -sortino)
    return (net
            - cfg.drawdown_penalty * dd
            - cfg.tail_risk_penalty * _cvar_penalty(metrics)
            - cfg.turnover_penalty * turnover
            - cfg.instability_penalty * instability)


def _cvar_penalty(metrics) -> float:
    avg_loss = float(getattr(metrics, "avg_loss", 0.0) or 0.0)
    return abs(avg_loss)


def evaluate_candidate(net_returns, *, label: str, data_label: str = DATA_LABEL_SYNTHETIC,
                       cfg: Optional[EconomicScoreConfig] = None,
                       risk_expanded: bool = False) -> CandidateEvaluation:
    """Evaluate one candidate's per-decision net returns through metrics -> gates -> score."""
    cfg = cfg or EconomicScoreConfig()
    r = np.asarray(net_returns, dtype=np.float64)
    n = int(len(r))
    if n == 0:
        return CandidateEvaluation(label, data_label, 0, {}, None, None, False,
                                   ["no_decisions"])
    m = economic_metrics(r, data_label=data_label)
    gate_failures = []
    if n < cfg.min_sample:
        gate_failures.append(f"insufficient_sample:{n}<{cfg.min_sample}")
    if abs(float(m.max_drawdown or 0.0)) > cfg.max_drawdown_limit:
        gate_failures.append(f"drawdown_exceeds_limit:{m.max_drawdown}")
    if cfg.require_no_risk_expansion and risk_expanded:
        gate_failures.append("unauthorized_risk_expansion")
    ci = None
    try:
        b = bootstrap_metric_ci(r, data_label=data_label)
        ci = (b.low, b.high)
    except Exception:  # noqa: BLE001
        ci = None
    gates_passed = len(gate_failures) == 0
    return CandidateEvaluation(
        label=label, data_label=data_label, n=n,
        metrics={"net_return": m.net_return, "net_expectancy": m.net_expectancy,
                 "sharpe": m.sharpe, "sortino": m.sortino, "profit_factor": m.profit_factor,
                 "win_rate": m.win_rate, "max_drawdown": m.max_drawdown,
                 "avg_win": m.avg_win, "avg_loss": m.avg_loss},
        score=economic_score(m, cfg), net_return_ci=ci,
        gates_passed=gates_passed, gate_failures=gate_failures)


def compare_to_baselines(candidate: CandidateEvaluation,
                         baselines: dict) -> dict:
    """Compare a candidate against the baseline hierarchy (§19.3). Returns a verdict dict.

    ``baselines`` maps name -> CandidateEvaluation (e.g. 'incumbent', 'deterministic',
    'simple_statistical'). A candidate must pass its gates AND beat the incumbent + a simple
    baseline on score to be recommended for shadow — never promoted to live here.
    """
    if candidate.score is None or not candidate.gates_passed:
        return {"recommendation": "reject",
                "reason": candidate.gate_failures or "no_score",
                "beats": {}}
    beats = {name: (candidate.score > b.score)
             for name, b in baselines.items() if b.score is not None}
    incumbent_ok = beats.get("incumbent", True)
    simple_ok = beats.get("simple_statistical", True) or beats.get("deterministic", True)
    if incumbent_ok and simple_ok:
        rec = "eligible_for_shadow"
    else:
        rec = "reject"
    return {"recommendation": rec, "beats": beats,
            "candidate_score": candidate.score,
            "data_label": candidate.data_label,
            "note": ("NEVER promoted to live here; shadow-eligible only (M7 certifies). "
                     "Synthetic data is never genuine economic evidence."
                     if candidate.data_label == DATA_LABEL_SYNTHETIC else
                     "real-data comparison; shadow-eligible only, M7 certifies live")}


REAL = DATA_LABEL_REAL
SYNTHETIC = DATA_LABEL_SYNTHETIC
