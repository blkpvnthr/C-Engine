"""Offline policy evaluation (OPE) for authorized-policy selection (Milestone 6, §21).

Estimates the value a *candidate* decision policy would have earned from decisions that were
actually logged, so a challenger can be compared against the champion/baselines on the SAME
historical decisions — without ever touching live capital, submitting an order, or disturbing
the mandatory -0.5% protective exit.

Hard honesty gates, enforced in code (not convention):

* **No fabricated behavior probabilities.** The gold ``offline_rl`` feed records
  ``action_probability_if_known`` as ALWAYS NULL — the engine's logging (behavior) policy is
  deterministic/unknown and is *never* invented. Inverse-propensity scoring (IPS) and
  doubly-robust (DR) estimators mathematically require that probability; when it is unknown
  they are GATED OFF and reported as ``status='unavailable'`` with the reason
  "behavior probabilities unknown", rather than silently substituting a fabricated value.

* **Direct Method + Fitted-Q hooks that run on what we have.** The Direct Method fits a
  reward model from REALIZED rewards only (``reward`` is NULL until a position closes — such
  rows contribute no label) and averages its predictions under the target policy. A
  Fitted-Q-Evaluation hook generalizes this to multi-step transitions; with single-step
  (bandit) logs and no next-state it provably reduces to the Direct Method.

* **Action-support / extrapolation audit.** Off-policy estimates are only trustworthy where
  the target policy chooses actions the logging policy actually explored. The audit reports
  logged support per action, the fraction of target decisions that fall on unsupported
  (out-of-distribution) actions, and realized-reward coverage.

* **Synthetic evidence is never a profitability claim.** Every result carries a
  ``data_label`` of ``'synthetic'``/``'real'`` and a ``profitability_claim_allowed`` flag.

Pure numpy (CPU). The optional lakehouse loader imports DuckDB lazily so the estimators stay
dependency-light and importable everywhere.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from cengine.learning.evaluation.economic import (
    DATA_LABEL_REAL,
    DATA_LABEL_SYNTHETIC,
    bootstrap_metric_ci,
    validate_data_label,
)

OPE_VERSION = "offline-policy-evaluation-1.0.0"

_MIN_REWARD_ROWS_FOR_VERDICT = 30


# --------------------------------------------------------------------------- data


@dataclass
class LoggedDecisions:
    """A set of logged decisions for off-policy evaluation.

    ``actions`` are integer action ids indexing ``action_space``. ``rewards`` are realized
    outcomes with ``np.nan`` wherever the reward is not yet known (unclosed position) — those
    rows are excluded from reward-model fitting and from IPS/DR, never fabricated.
    ``behavior_action_probs`` is the logging-policy probability of each taken action and is
    ``None`` whenever it is unknown (the real-data case), which gates IPS/DR off.
    """

    actions: np.ndarray                              # (N,) int action ids
    rewards: np.ndarray                              # (N,) float; NaN where unknown
    action_space: list                              # distinct action labels
    data_label: str
    contexts: Optional[np.ndarray] = None            # (N, d) state features or None
    behavior_action_probs: Optional[np.ndarray] = None  # (N,) P_b(a|s) or None (unknown)
    episode_ids: Optional[np.ndarray] = None         # (N,) for dependence-aware CIs
    terminal: Optional[np.ndarray] = None            # (N,) bool

    def __post_init__(self) -> None:
        self.data_label = validate_data_label(self.data_label)
        self.actions = np.asarray(self.actions, dtype=np.int64).ravel()
        self.rewards = np.asarray(self.rewards, dtype=np.float64).ravel()
        if self.actions.shape[0] != self.rewards.shape[0]:
            raise ValueError("actions and rewards must have the same length")
        if self.contexts is not None:
            self.contexts = np.asarray(self.contexts, dtype=np.float64)
            if self.contexts.ndim == 1:
                self.contexts = self.contexts.reshape(-1, 1)
            if self.contexts.shape[0] != self.actions.shape[0]:
                raise ValueError("contexts must have one row per decision")
        if self.behavior_action_probs is not None:
            bap = np.asarray(self.behavior_action_probs, dtype=np.float64).ravel()
            # An all-NaN / all-None probability column means "unknown" -> gate IPS/DR off.
            if bap.size == 0 or not np.any(np.isfinite(bap)):
                self.behavior_action_probs = None
            else:
                self.behavior_action_probs = bap

    @property
    def n(self) -> int:
        return int(self.actions.shape[0])

    @property
    def n_actions(self) -> int:
        return len(self.action_space)

    @property
    def reward_mask(self) -> np.ndarray:
        """Boolean mask of decisions whose reward is realized (finite)."""
        return np.isfinite(self.rewards)

    @property
    def behavior_probs_known(self) -> bool:
        return self.behavior_action_probs is not None

    @classmethod
    def from_offline_rl(
        cls,
        config=None,
        *,
        data_label: str = DATA_LABEL_REAL,
        sql: Optional[str] = None,
    ) -> "LoggedDecisions":
        """Load logged decisions from the gold ``offline_rl`` lakehouse view.

        Behavior probabilities arrive as ``action_probability_if_known`` (ALWAYS NULL) and are
        therefore preserved as ``None`` — IPS/DR stay gated off on real data, by contract.
        ``reward`` is NULL until a position closes; those rows carry ``np.nan`` rewards and are
        excluded from any reward-dependent estimate. Imports DuckDB lazily.
        """
        from cengine.lakehouse.warehouse import DuckDBWarehouse  # lazy: keep core dep-light

        wh = DuckDBWarehouse(config)
        query = sql or (
            "SELECT transition_id, episode_id, selected_action, target_weight, "
            "current_weight, action_probability_if_known, reward, terminal "
            "FROM offline_rl_transitions"
        )
        tbl = wh.query_df(query)
        df = tbl.to_pandas()
        if len(df) == 0:
            return cls(actions=np.empty(0, dtype=np.int64), rewards=np.empty(0),
                       action_space=[], data_label=data_label)
        labels = sorted(df["selected_action"].astype(str).unique().tolist())
        index = {lab: i for i, lab in enumerate(labels)}
        actions = df["selected_action"].astype(str).map(index).to_numpy(dtype=np.int64)
        rewards = df["reward"].to_numpy(dtype=np.float64)  # NaN where NULL
        contexts = None
        for cols in (["target_weight", "current_weight"],):
            if all(c in df.columns for c in cols):
                contexts = df[cols].to_numpy(dtype=np.float64)
                break
        bap_col = df.get("action_probability_if_known")
        bap = None if bap_col is None else bap_col.to_numpy(dtype=np.float64)
        episodes = (df["episode_id"].astype(str).to_numpy()
                    if "episode_id" in df.columns else None)
        terminal = (df["terminal"].to_numpy() if "terminal" in df.columns else None)
        return cls(
            actions=actions, rewards=rewards, action_space=labels, data_label=data_label,
            contexts=contexts, behavior_action_probs=bap, episode_ids=episodes,
            terminal=terminal,
        )


# ----------------------------------------------------------- target policy helper


def policy_action_matrix(policy, *, n: int, n_actions: int) -> np.ndarray:
    """Normalize a target policy into an (N, n_actions) action-probability matrix.

    Accepts a deterministic action array (shape (N,), integer action ids) or an already-formed
    (N, n_actions) probability matrix (rows renormalized to sum to 1).
    """
    p = np.asarray(policy)
    if p.ndim == 1:
        if p.shape[0] != n:
            raise ValueError("deterministic policy must have one action per decision")
        mat = np.zeros((n, n_actions), dtype=np.float64)
        mat[np.arange(n), p.astype(np.int64)] = 1.0
        return mat
    if p.ndim == 2:
        if p.shape != (n, n_actions):
            raise ValueError(f"policy matrix must be shape {(n, n_actions)}")
        mat = np.asarray(p, dtype=np.float64)
        row = mat.sum(axis=1, keepdims=True)
        row[row <= 0.0] = 1.0
        return mat / row
    raise ValueError("policy must be a 1-D action array or a 2-D probability matrix")


# --------------------------------------------------------------------- reward models


class RewardModel(ABC):
    """Reward model r_hat(context, action) for the Direct Method. Fit on REALIZED rewards."""

    @abstractmethod
    def fit(self, logged: "LoggedDecisions") -> "RewardModel": ...

    @abstractmethod
    def predict_all_actions(self, logged: "LoggedDecisions") -> np.ndarray:
        """Return an (N, n_actions) matrix of predicted rewards for every action per row."""


class PerActionMeanRewardModel(RewardModel):
    """Context-free reward model: r_hat(a) = mean realized reward of logged action ``a``.

    Robust default given the data reality (behavior unknown, few instruments). Actions with no
    realized reward fall back to the global realized mean, and that fallback is surfaced in the
    support audit as extrapolation risk.
    """

    def __init__(self) -> None:
        self._per_action: dict = {}
        self._global_mean: float = 0.0
        self._supported: set = set()

    def fit(self, logged: "LoggedDecisions") -> "PerActionMeanRewardModel":
        mask = logged.reward_mask
        if not np.any(mask):
            self._global_mean = 0.0
            return self
        self._global_mean = float(logged.rewards[mask].mean())
        for a in range(logged.n_actions):
            sel = mask & (logged.actions == a)
            if np.any(sel):
                self._per_action[a] = float(logged.rewards[sel].mean())
                self._supported.add(a)
        return self

    def predict_all_actions(self, logged: "LoggedDecisions") -> np.ndarray:
        row = np.array([self._per_action.get(a, self._global_mean)
                        for a in range(logged.n_actions)], dtype=np.float64)
        return np.tile(row, (logged.n, 1))


class LinearRewardModel(RewardModel):
    """Context-aware linear reward model over [context, one-hot(action)] via least squares.

    Fit on realized rewards only; if there is no context it degrades to a per-action intercept
    model. Pure numpy (``lstsq``) — no external ML dependency.
    """

    def __init__(self, ridge: float = 1e-6) -> None:
        self.ridge = float(ridge)
        self._coef: Optional[np.ndarray] = None
        self._n_ctx = 0
        self._global_mean = 0.0

    def _design(self, contexts: Optional[np.ndarray], actions: np.ndarray,
                n_actions: int, n_ctx: int) -> np.ndarray:
        n = actions.shape[0]
        onehot = np.zeros((n, n_actions), dtype=np.float64)
        onehot[np.arange(n), actions] = 1.0
        if n_ctx and contexts is not None:
            return np.hstack([contexts, onehot])
        return onehot

    def fit(self, logged: "LoggedDecisions") -> "LinearRewardModel":
        mask = logged.reward_mask
        self._n_ctx = logged.contexts.shape[1] if logged.contexts is not None else 0
        if not np.any(mask):
            self._coef = None
            return self
        self._global_mean = float(logged.rewards[mask].mean())
        ctx = logged.contexts[mask] if logged.contexts is not None else None
        X = self._design(ctx, logged.actions[mask], logged.n_actions, self._n_ctx)
        y = logged.rewards[mask]
        # ridge-regularized normal equations for numerical stability on collinear one-hots
        gram = X.T @ X + self.ridge * np.eye(X.shape[1])
        self._coef = np.linalg.solve(gram, X.T @ y)
        return self

    def predict_all_actions(self, logged: "LoggedDecisions") -> np.ndarray:
        if self._coef is None:
            return np.full((logged.n, logged.n_actions), self._global_mean, dtype=np.float64)
        out = np.empty((logged.n, logged.n_actions), dtype=np.float64)
        for a in range(logged.n_actions):
            acts = np.full(logged.n, a, dtype=np.int64)
            X = self._design(logged.contexts, acts, logged.n_actions, self._n_ctx)
            out[:, a] = X @ self._coef
        return out


# ------------------------------------------------------------------------ results


@dataclass
class OPEResult:
    """Outcome of one off-policy estimator for one target policy."""

    estimator: str
    value: Optional[float]
    status: str                       # 'ok' | 'unavailable' | 'insufficient_data'
    reason: Optional[str]
    n: int
    n_reward_available: int
    data_label: str
    profitability_claim_allowed: bool
    diagnostics: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


@dataclass
class ActionSupportAudit:
    """Overlap / extrapolation audit between a target policy and the logged actions."""

    n_decisions: int
    logged_action_counts: dict
    target_action_mass: dict
    unsupported_actions: list
    extrapolation_fraction: float
    reward_coverage: float
    min_logged_support: int
    data_label: str
    warning: Optional[str] = None

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


@dataclass
class PolicyComparison:
    """Candidate vs baseline evaluated on IDENTICAL logged decisions with one estimator."""

    estimator: str
    candidate: OPEResult
    baseline: OPEResult
    value_difference: Optional[float]
    difference_ci: Optional[tuple]
    identical_decisions: bool
    n: int
    data_label: str
    profitability_claim_allowed: bool
    verdict: str

    def to_dict(self) -> dict:
        return {
            "estimator": self.estimator,
            "candidate": self.candidate.to_dict(),
            "baseline": self.baseline.to_dict(),
            "value_difference": self.value_difference,
            "difference_ci": self.difference_ci,
            "identical_decisions": self.identical_decisions,
            "n": self.n,
            "data_label": self.data_label,
            "profitability_claim_allowed": self.profitability_claim_allowed,
            "verdict": self.verdict,
        }


def _claim_allowed(logged: LoggedDecisions) -> bool:
    return (logged.data_label == DATA_LABEL_REAL
            and int(np.sum(logged.reward_mask)) >= _MIN_REWARD_ROWS_FOR_VERDICT)


# ----------------------------------------------------------------- audit + estimators


def action_support_audit(logged: LoggedDecisions, policy) -> ActionSupportAudit:
    """Audit target-policy action support against the logged action distribution."""
    label = logged.data_label
    n, n_actions = logged.n, logged.n_actions
    mask = logged.reward_mask
    logged_counts = {logged.action_space[a]: int(np.sum(logged.actions == a))
                     for a in range(n_actions)}
    # actions explored with a REALIZED reward are the ones the reward model can actually learn
    supported = {a for a in range(n_actions) if np.any(mask & (logged.actions == a))}
    if n == 0:
        return ActionSupportAudit(
            n_decisions=0, logged_action_counts=logged_counts, target_action_mass={},
            unsupported_actions=[], extrapolation_fraction=0.0, reward_coverage=0.0,
            min_logged_support=0, data_label=label, warning="insufficient_data: no decisions",
        )
    mat = policy_action_matrix(policy, n=n, n_actions=n_actions)
    target_mass = mat.sum(axis=0) / n
    target_mass_named = {logged.action_space[a]: float(target_mass[a])
                         for a in range(n_actions)}
    unsupported = [logged.action_space[a] for a in range(n_actions)
                   if a not in supported and target_mass[a] > 0.0]
    unsupported_idx = [a for a in range(n_actions) if a not in supported]
    extrapolation = float(mat[:, unsupported_idx].sum() / n) if unsupported_idx else 0.0
    reward_coverage = float(np.sum(mask) / n)
    min_support = min((int(np.sum(mask & (logged.actions == a)))
                       for a in range(n_actions) if target_mass[a] > 0.0), default=0)
    warning = None
    if extrapolation > 0.0:
        warning = (f"extrapolation: {extrapolation:.1%} of target action mass falls on "
                   "actions with no realized logged reward (OOD) — estimates there are "
                   "unvalidated")
    elif reward_coverage < 0.5:
        warning = (f"low reward coverage ({reward_coverage:.1%}); most logged decisions have "
                   "no realized reward yet")
    return ActionSupportAudit(
        n_decisions=n, logged_action_counts=logged_counts,
        target_action_mass=target_mass_named, unsupported_actions=unsupported,
        extrapolation_fraction=extrapolation, reward_coverage=reward_coverage,
        min_logged_support=min_support, data_label=label, warning=warning,
    )


class DirectMethodEvaluator:
    """Direct Method OPE: fit a reward model, then average it under the target policy.

    ``V_DM(pi) = (1/N) sum_i sum_a pi(a | s_i) * r_hat(s_i, a)``. The reward model is fit on
    realized rewards ONLY. This estimator needs no behavior probabilities, so it is the
    workhorse when the logging policy is unknown — but it is model-based, so it is paired with
    the action-support audit to flag extrapolation.
    """

    estimator_name = "direct_method"

    def __init__(self, reward_model: Optional[RewardModel] = None) -> None:
        self.reward_model = reward_model or PerActionMeanRewardModel()

    def estimate(self, logged: LoggedDecisions, policy) -> OPEResult:
        n = logged.n
        n_reward = int(np.sum(logged.reward_mask))
        claim = _claim_allowed(logged)
        if n == 0:
            return OPEResult(self.estimator_name, None, "insufficient_data",
                             "no logged decisions", 0, 0, logged.data_label, False)
        if n_reward == 0:
            return OPEResult(self.estimator_name, None, "insufficient_data",
                             "no realized rewards: reward is NULL until positions close; "
                             "nothing to fit a reward model on", n, 0, logged.data_label,
                             False)
        self.reward_model.fit(logged)
        q = self.reward_model.predict_all_actions(logged)          # (N, A)
        mat = policy_action_matrix(policy, n=n, n_actions=logged.n_actions)
        per_decision = np.sum(mat * q, axis=1)                     # (N,)
        value = float(per_decision.mean())
        audit = action_support_audit(logged, policy)
        return OPEResult(
            self.estimator_name, value, "ok", None, n, n_reward, logged.data_label, claim,
            diagnostics={
                "reward_model": type(self.reward_model).__name__,
                "per_decision_value": per_decision,
                "extrapolation_fraction": audit.extrapolation_fraction,
                "reward_coverage": audit.reward_coverage,
                "synthetic": logged.data_label == DATA_LABEL_SYNTHETIC,
            },
        )


class FittedQEvaluator:
    """Fitted-Q Evaluation hook (§21) for multi-step transitions.

    FQE iterates ``Q(s,a) <- r + gamma * Q(s', pi(s'))`` using a reward/Q model. With
    single-step (bandit) logs and no next-state information it PROVABLY reduces to the Direct
    Method (``Q = r_hat``), which is what the gold ``offline_rl`` feed currently supports. The
    ``next_action_probs`` hook is where real sequential transitions plug in; it is accepted but
    conservatively ignored when next-states are absent, and that reduction is reported.
    """

    estimator_name = "fitted_q_evaluation"

    def __init__(self, *, gamma: float = 1.0, n_iterations: int = 1,
                 reward_model: Optional[RewardModel] = None) -> None:
        self.gamma = float(gamma)
        self.n_iterations = int(max(1, n_iterations))
        self.reward_model = reward_model or PerActionMeanRewardModel()

    def estimate(self, logged: LoggedDecisions, policy, *,
                 next_action_probs=None) -> OPEResult:
        base = DirectMethodEvaluator(self.reward_model).estimate(logged, policy)
        if base.status != "ok":
            return OPEResult(self.estimator_name, base.value, base.status, base.reason,
                             base.n, base.n_reward_available, logged.data_label,
                             base.profitability_claim_allowed, dict(base.diagnostics))
        diag = dict(base.diagnostics)
        if next_action_probs is None:
            diag["reduction"] = ("no next-state transitions provided; FQE reduces to the "
                                 "Direct Method (Q = r_hat), gamma unused")
            diag["gamma"] = self.gamma
            diag["iterations"] = self.n_iterations
            return OPEResult(self.estimator_name, base.value, "ok", None, base.n,
                             base.n_reward_available, logged.data_label,
                             base.profitability_claim_allowed, diag)
        # Multi-step backup hook: callers supplying genuine next-state action distributions can
        # extend this. The reduction above is the supported path for current logged data.
        raise NotImplementedError(
            "Multi-step FQE backups require genuine logged next-state transitions, which the "
            "current gold offline_rl feed does not provide. Supply them explicitly to extend "
            "this hook; synthetic next-states are never fabricated here."
        )


def ips_value(logged: LoggedDecisions, policy) -> OPEResult:
    """Inverse-propensity-scoring OPE — GATED OFF when behavior probabilities are unknown.

    When ``action_probability_if_known`` is NULL (the real-data contract) there is no known
    logging policy, so IPS is mathematically undefined. We return ``status='unavailable'``
    rather than invent a propensity, because a fabricated propensity would produce a fabricated
    value estimate.
    """
    n = logged.n
    n_reward = int(np.sum(logged.reward_mask))
    if not logged.behavior_probs_known:
        return OPEResult(
            "ips", None, "unavailable",
            "behavior probabilities unknown (action_probability_if_known is NULL); IPS "
            "requires a known logging policy and is not computed to avoid a fabricated "
            "estimate", n, n_reward, logged.data_label, False,
            diagnostics={"gated_off": True},
        )
    if n_reward == 0:
        return OPEResult("ips", None, "insufficient_data", "no realized rewards",
                         n, 0, logged.data_label, False)
    mat = policy_action_matrix(policy, n=n, n_actions=logged.n_actions)
    pi_taken = mat[np.arange(n), logged.actions]
    pb = np.asarray(logged.behavior_action_probs, dtype=np.float64)
    mask = logged.reward_mask & np.isfinite(pb) & (pb > 0.0)
    if not np.any(mask):
        return OPEResult("ips", None, "insufficient_data",
                         "no rows with positive known propensity and realized reward",
                         n, n_reward, logged.data_label, False)
    weights = pi_taken[mask] / pb[mask]
    value = float(np.mean(weights * logged.rewards[mask]))
    return OPEResult("ips", value, "ok", None, n, n_reward, logged.data_label,
                     _claim_allowed(logged),
                     diagnostics={"mean_importance_weight": float(np.mean(weights))})


def doubly_robust_value(logged: LoggedDecisions, policy,
                        reward_model: Optional[RewardModel] = None) -> OPEResult:
    """Doubly-robust OPE — GATED OFF when behavior probabilities are unknown.

    DR combines the Direct Method with an IPS correction term and therefore also needs the
    logging-policy propensities. With them NULL it is unavailable, exactly like IPS.
    """
    n = logged.n
    n_reward = int(np.sum(logged.reward_mask))
    if not logged.behavior_probs_known:
        return OPEResult(
            "doubly_robust", None, "unavailable",
            "behavior probabilities unknown (action_probability_if_known is NULL); the "
            "doubly-robust correction requires a known logging policy and is not computed",
            n, n_reward, logged.data_label, False, diagnostics={"gated_off": True},
        )
    dm = DirectMethodEvaluator(reward_model or PerActionMeanRewardModel())
    dm_res = dm.estimate(logged, policy)
    if dm_res.status != "ok":
        return OPEResult("doubly_robust", None, dm_res.status, dm_res.reason, n,
                         n_reward, logged.data_label, False)
    q = dm.reward_model.predict_all_actions(logged)
    mat = policy_action_matrix(policy, n=n, n_actions=logged.n_actions)
    dm_term = np.sum(mat * q, axis=1)
    q_taken = q[np.arange(n), logged.actions]
    pi_taken = mat[np.arange(n), logged.actions]
    pb = np.asarray(logged.behavior_action_probs, dtype=np.float64)
    mask = logged.reward_mask & np.isfinite(pb) & (pb > 0.0)
    correction = np.zeros(n, dtype=np.float64)
    correction[mask] = (pi_taken[mask] / pb[mask]) * (logged.rewards[mask] - q_taken[mask])
    value = float(np.mean(dm_term + correction))
    return OPEResult("doubly_robust", value, "ok", None, n, n_reward, logged.data_label,
                     _claim_allowed(logged),
                     diagnostics={"dm_value": float(dm_term.mean())})


# ---------------------------------------------------------------- candidate vs baseline


def compare_policies(
    logged: LoggedDecisions,
    candidate_policy,
    baseline_policy,
    *,
    estimator: str = "direct_method",
    reward_model: Optional[RewardModel] = None,
    n_boot: int = 2000,
    confidence: float = 0.95,
    seed: int = 7,
) -> PolicyComparison:
    """Compare a candidate vs a baseline policy on the SAME logged decisions.

    Both policies are scored with the identical estimator on the identical ``logged`` set, so
    the difference is apples-to-apples. For the Direct Method a dependence-aware bootstrap CI
    on the per-decision value difference is returned (conditional on the single reward-model
    fit — a first-order interval, noted as such). IPS/DR comparisons inherit the behavior-
    probability gate and report ``unavailable`` when propensities are unknown.
    """
    label = logged.data_label
    if estimator == "direct_method":
        ev = DirectMethodEvaluator(reward_model or PerActionMeanRewardModel())
        cand = ev.estimate(logged, candidate_policy)
        base = ev.estimate(logged, baseline_policy)
    elif estimator == "ips":
        cand = ips_value(logged, candidate_policy)
        base = ips_value(logged, baseline_policy)
    elif estimator == "doubly_robust":
        cand = doubly_robust_value(logged, candidate_policy, reward_model)
        base = doubly_robust_value(logged, baseline_policy, reward_model)
    else:
        raise ValueError("estimator must be 'direct_method', 'ips' or 'doubly_robust'")

    diff = None
    diff_ci = None
    if cand.status == "ok" and base.status == "ok" and cand.value is not None \
            and base.value is not None:
        diff = float(cand.value - base.value)
        cpd = cand.diagnostics.get("per_decision_value")
        bpd = base.diagnostics.get("per_decision_value")
        if cpd is not None and bpd is not None:
            per_decision_diff = np.asarray(cpd) - np.asarray(bpd)
            ci = bootstrap_metric_ci(
                per_decision_diff, data_label=label, statistic_name="value_difference",
                method="stationary", n_boot=n_boot, confidence=confidence, seed=seed,
            )
            diff_ci = (ci.ci_low, ci.ci_high)

    claim = (label == DATA_LABEL_REAL and cand.profitability_claim_allowed
             and base.profitability_claim_allowed)
    if cand.status != "ok" or base.status != "ok":
        verdict = (f"{estimator.upper()} unavailable/insufficient — "
                   f"candidate={cand.status}, baseline={base.status}: {cand.reason or base.reason}")
    elif label == DATA_LABEL_SYNTHETIC:
        verdict = ("SYNTHETIC — ordering shown for plumbing only; no profitability or "
                   "superiority may be inferred from synthetic data")
    elif diff_ci is not None and diff_ci[0] > 0.0:
        verdict = ("CANDIDATE SUPERIOR (real, DM) — value-difference CI excludes zero; "
                   "confirm with falsification before any promotion")
    elif diff_ci is not None and diff_ci[1] < 0.0:
        verdict = "BASELINE SUPERIOR (real, DM) — value-difference CI is below zero"
    else:
        verdict = "INCONCLUSIVE (real) — value-difference CI includes zero"

    return PolicyComparison(
        estimator=estimator, candidate=cand, baseline=base, value_difference=diff,
        difference_ci=diff_ci, identical_decisions=True, n=logged.n, data_label=label,
        profitability_claim_allowed=claim, verdict=verdict,
    )
