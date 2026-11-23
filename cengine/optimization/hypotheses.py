"""Hypothesis generation (Milestone 6B, §9.2).

Translates performance diagnostics into testable hypotheses, each with the existing
behavior, suspected weakness, proposed intervention, expected economic improvement,
evaluation methodology, acceptance criteria, applicable risk constraints, and a named
bounded search space. A hypothesis is a research question — it never changes production.
"""

from __future__ import annotations

from typing import Optional

from cengine.optimization.contracts import Diagnostic, Hypothesis, stable_id

HYPOTHESIS_GENERATOR_VERSION = "hypothesis-generator-1.0.0"

# diagnostic experiment-family -> (intervention, expected, methodology, search_space)
_FAMILY = {
    "protective_exit_matrix": (
        "replace the unconditional -0.5% exit with a volatility/regime-adjusted adaptive "
        "reassessment (HOLD/REDUCE/EXIT/REPLACE) + independent emergency stop",
        "fewer premature liquidations at equal-or-lower drawdown under equivalent risk budget",
        "EXIT-00..EXIT-15 matrix via deterministic replay on genuine bars; "
        "exit-isolation (Eval A) + risk-normalized portfolio (Eval B)",
        "protective_exit"),
    "entry_optimization": (
        "tune entry confidence thresholds / signal weights within approved bounds",
        "higher net expectancy on accepted entries without raising risk",
        "purged walk-forward on genuine labeled entries (when eligible)",
        "entry_policy"),
    "strategy_optimization": (
        "adjust regime-conditioned strategy eligibility / weighting",
        "fewer regime-specific losses",
        "chronological OOS by regime (when eligible)",
        "strategy_policy"),
    "model_optimization": (
        "search model family / features / horizon / hyperparameters",
        "OOS economic improvement over the sklearn baseline",
        "certified chronological split + baseline-gated promotion",
        "supervised_model"),
    "reward_optimization": (
        "tune approved RL reward coefficients (turnover/drawdown/tail penalties)",
        "better net risk-adjusted policy value under identical hard constraints",
        "FQE + simulator comparison on eligible transitions",
        "rl_reward"),
    "data_collection": (
        "acquire the missing genuine evidence before experimenting",
        "move the task from INSUFFICIENT_DATA to training-eligible",
        "historical acquisition + eligibility reassessment",
        "none"),
}


class HypothesisGenerator:
    def __init__(self) -> None:
        self.version = HYPOTHESIS_GENERATOR_VERSION

    def from_diagnostic(self, diag: Diagnostic) -> Optional[Hypothesis]:
        fam = diag.recommended_experiment_family or "model_optimization"
        spec = _FAMILY.get(fam)
        if spec is None:
            return None
        intervention, expected, methodology, search_space = spec
        return Hypothesis(
            hypothesis_id=stable_id("hyp", diag.diagnostic_id, fam),
            diagnostic_id=diag.diagnostic_id, component=diag.component,
            existing_behavior=f"current {diag.component} behavior over {diag.period}",
            suspected_weakness=diag.suspected_cause,
            intervention=intervention, expected_improvement=expected,
            methodology=methodology,
            acceptance_criteria={
                "beats_incumbent_net": True,
                "beats_simple_baseline": True,
                "no_risk_expansion": True,
                "drawdown_not_worse": True,
                "statistically_supported": True,
                "sufficient_sample": True},
            risk_constraints=[
                "hard risk limits are not tunable",
                "no leverage/capital/stop-loss relaxation",
                "native C++ financial authority retained",
                "mandatory emergency protection preserved"],
            search_space=search_space,
            priority=diag.priority)

    def generate(self, diagnostics: list) -> list:
        out = []
        seen = set()
        for d in sorted(diagnostics, key=lambda x: getattr(x, "priority", 0.0), reverse=True):
            h = self.from_diagnostic(d)
            if h is None or h.hypothesis_id in seen:
                continue
            seen.add(h.hypothesis_id)
            out.append(h)
        return out

    def standing(self, *, eligible_tasks: set) -> list:
        """Standing research hypotheses that don't require a prior diagnostic.

        The protective-exit matrix always runs on genuine bars via the replay env (it does
        not need closed-position outcomes). The supervised-model search fires whenever the
        forward-return task has eligible data. These keep the optimizer researching even when
        attribution is data-blocked.
        """
        out = []
        out.append(Hypothesis(
            hypothesis_id="hyp-standing-protective-exit",
            diagnostic_id=None, component="protective_exit",
            existing_behavior="unconditional mandatory -0.5% protective exit",
            suspected_weakness="may cause premature liquidation of recoverable positions",
            intervention=_FAMILY["protective_exit_matrix"][0],
            expected_improvement=_FAMILY["protective_exit_matrix"][1],
            methodology=_FAMILY["protective_exit_matrix"][2],
            acceptance_criteria={"beats_incumbent_net": True, "no_risk_expansion": True,
                                 "drawdown_not_worse": True, "sufficient_sample": True},
            risk_constraints=["production -0.5% exit preserved", "emergency stop preserved",
                              "shadow-only; M7 certifies"],
            search_space="protective_exit", priority=1.0))
        if "forward_return_prediction" in eligible_tasks:
            out.append(Hypothesis(
                hypothesis_id="hyp-standing-supervised",
                diagnostic_id=None, component="model:forward_return",
                existing_behavior="incumbent forward-return model (marginal edge)",
                suspected_weakness="features/horizon/architecture may be improvable",
                intervention=_FAMILY["model_optimization"][0],
                expected_improvement=_FAMILY["model_optimization"][1],
                methodology=_FAMILY["model_optimization"][2],
                acceptance_criteria={"beats_sklearn_baseline": True, "oos_only": True},
                risk_constraints=["recommendation-only"],
                search_space="supervised_model", priority=0.5))
        return out
