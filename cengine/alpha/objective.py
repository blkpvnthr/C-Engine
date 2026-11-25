"""Economic optimization objective + frozen challenger-acceptance gates + challenger registry
(M8 §19 / §21).

The challenger ranking objective (§19) is a governed penalty of net return — it never rewards win
rate, AUC or gross return alone:

    J = net_expectancy
        - drawdown_penalty   * max_drawdown
        - downside_penalty   * downside_risk
        - uncertainty_penalty * estimation_uncertainty

Penalty weights are illustrative research defaults, calibrated on development data only. The hard
**acceptance gates** (§21) are independent of J and are frozen by fingerprint (anti-retrofitting),
mirroring the M7 FrozenEconomicGates mechanism. A challenger is accepted ONLY if it clears every
hard gate AND beats the mandatory baselines (cash, incumbent, simple momentum/trend). Gates are the
directive's proposed M7 research criteria (PF>=1.30, Sharpe>=1.25, Sortino>=1.50, maxDD<=10%,
adequate sample) — not relaxed to manufacture a pass.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from cengine.certification.contracts import config_fingerprint

OBJECTIVE_VERSION = "alpha-economic-objective-1.0.0"


@dataclass
class EconomicObjectiveWeights:
    drawdown_penalty: float = 1.0
    downside_penalty: float = 0.5
    uncertainty_penalty: float = 0.5

    def to_dict(self) -> dict:
        return asdict(self)


def objective_score(metrics: dict, weights: EconomicObjectiveWeights | None = None) -> float:
    """Governed objective J over a metrics dict (net-return-centric, penalty-adjusted)."""
    w = weights or EconomicObjectiveWeights()
    net = float(metrics.get("net_expectancy", metrics.get("net_expectancy_per_trade", 0.0)) or 0.0)
    dd = abs(float(metrics.get("max_drawdown", 0.0) or 0.0))
    downside = abs(float(metrics.get("avg_loss", metrics.get("downside_risk", 0.0)) or 0.0))
    unc = abs(float(metrics.get("uncertainty", 0.0) or 0.0))
    return (net - w.drawdown_penalty * dd - w.downside_penalty * downside
            - w.uncertainty_penalty * unc)


@dataclass(frozen=True)
class ChallengerGates:
    """Frozen hard acceptance gates (§21). Any violation -> reject, regardless of J."""

    min_profit_factor: float = 1.30
    min_annualized_sharpe: float = 1.25
    min_sortino: float = 1.50
    max_portfolio_drawdown: float = 0.10
    min_net_expectancy: float = 0.0        # strictly positive
    min_effective_sample: float = 50.0
    must_beat_cash: bool = True
    must_beat_incumbent: bool = True
    must_beat_simple_baselines: bool = True

    def fingerprint(self) -> str:
        return config_fingerprint(asdict(self))

    def to_dict(self) -> dict:
        d = asdict(self)
        d["fingerprint"] = self.fingerprint()
        d["version"] = OBJECTIVE_VERSION
        return d


# Frozen fingerprint of the default challenger gates (anti-retrofitting; a test enforces it).
FROZEN_CHALLENGER_GATES_FINGERPRINT = "19b52641f3e837e7"


def default_gates() -> ChallengerGates:
    return ChallengerGates()


def evaluate_challenger(metrics: dict, *, effective_sample: float | None,
                        beats_cash: bool | None, beats_incumbent: bool | None,
                        beats_simple_baselines: bool | None,
                        gates: ChallengerGates | None = None) -> dict:
    """Score a challenger's metrics against the frozen gates. Returns accept/reject/defer."""
    g = gates or default_gates()
    failures: list = []
    insufficient = False

    if effective_sample is not None and effective_sample < g.min_effective_sample:
        insufficient = True
        failures.append(f"insufficient_effective_sample:{effective_sample:.0f}<"
                        f"{g.min_effective_sample}")

    def _m(k: str, default=0.0) -> float:
        v = metrics.get(k)
        return float(v) if v is not None else default

    if _m("net_expectancy", _m("net_expectancy_per_trade")) <= g.min_net_expectancy:
        failures.append("net_expectancy_not_positive")
    if _m("profit_factor") < g.min_profit_factor:
        failures.append(f"profit_factor_below:{_m('profit_factor'):.3f}<{g.min_profit_factor}")
    if _m("sharpe_annualized", _m("sharpe")) < g.min_annualized_sharpe:
        failures.append(f"sharpe_below:{_m('sharpe_annualized', _m('sharpe')):.3f}<"
                        f"{g.min_annualized_sharpe}")
    if _m("sortino_annualized", _m("sortino")) < g.min_sortino:
        failures.append("sortino_below")
    if abs(_m("max_drawdown")) > g.max_portfolio_drawdown:
        failures.append(f"drawdown_exceeds:{abs(_m('max_drawdown')):.3f}>"
                        f"{g.max_portfolio_drawdown}")
    if g.must_beat_cash and beats_cash is False:
        failures.append("does_not_beat_cash")
    if g.must_beat_incumbent and beats_incumbent is False:
        failures.append("does_not_beat_incumbent")
    if g.must_beat_simple_baselines and beats_simple_baselines is False:
        failures.append("does_not_beat_simple_baselines")

    if insufficient:
        decision = "defer"
    elif failures:
        decision = "reject"
    else:
        decision = "accept"
    return {"decision": decision, "insufficient": insufficient, "failures": failures,
            "objective_J": round(objective_score(metrics), 6),
            "gate_fingerprint": g.fingerprint()}


@dataclass
class ChallengerRegistry:
    """Append-only registry of evaluated challengers with accept/reject/defer outcomes (§21/§25)."""

    challengers: list = field(default_factory=list)
    gate_fingerprint: str = field(default_factory=lambda: default_gates().fingerprint())
    version: str = OBJECTIVE_VERSION

    def record(self, name: str, metrics: dict, evaluation: dict, *, provenance: str) -> dict:
        row = {"name": name, "decision": evaluation["decision"],
               "objective_J": evaluation.get("objective_J"),
               "failures": evaluation.get("failures", []), "metrics": metrics,
               "provenance": provenance, "gate_fingerprint": evaluation.get("gate_fingerprint")}
        self.challengers.append(row)
        return row

    def summary(self) -> dict:
        counts: dict = {"accept": 0, "reject": 0, "defer": 0}
        for c in self.challengers:
            counts[c["decision"]] = counts.get(c["decision"], 0) + 1
        return {"total": len(self.challengers), "by_decision": counts,
                "accepted": [c["name"] for c in self.challengers if c["decision"] == "accept"]}

    def to_dict(self) -> dict:
        return {"version": self.version, "gate_fingerprint": self.gate_fingerprint,
                "summary": self.summary(), "challengers": self.challengers}
