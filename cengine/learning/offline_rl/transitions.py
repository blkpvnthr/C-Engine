"""Versioned offline-RL transition dataset + training guard (Milestone 6, §3/§6/§7).

Loads logged transitions from the gold ``offline_rl_transitions`` view into an immutable,
versioned in-memory dataset and gates whether any learner is even allowed to train on it.

Honest-data invariants enforced here (not just documented):

* ``action_probability_if_known`` is ALWAYS ``None``. The behavior/logging policy that
  produced these decisions is unknown, so it is never invented. Any estimator that needs
  behavior probabilities (IPS / weighted doubly-robust) is therefore GATED OFF upstream —
  the methods we ship (BC / FQE / CQL / IQL) deliberately do not require them.
* The mandatory -0.5% protective exit is ENVIRONMENT DYNAMICS, not an agent action. Rows
  whose logged action is a reserved mandatory-exit label are excluded from the learnable
  action space (counted in the data-quality report) so no learner can ever model, imitate,
  or recommend the protective exit as a policy choice.
* Rewards are realized only for closed positions; unknown rewards stay ``NaN`` with a
  ``reward_known`` mask — no fabricated rewards.
* Insufficient / synthetic-as-real data cannot yield a promoted model. The guard refuses
  training on insufficient data and marks any result on explicitly-synthetic data as
  research-only / non-promotable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from cengine.learning.datasets.eligibility import EligibilityState, default_task_specs

TRANSITIONS_VERSION = "offline-rl-transitions-1.0.0"

# Reserved logged-action labels that represent ENVIRONMENT DYNAMICS (the mandatory
# protective exit and kin), never agent policy actions. Excluded from the action space.
MANDATORY_EXIT_ACTIONS = frozenset({
    "mandatory_exit", "mandatory_protective_exit", "protective_exit",
    "protective_stop", "stop_loss_exit", "forced_exit", "risk_mandatory_exit",
})

# Columns that are metadata / targets, never state features.
_RESERVED_COLUMNS = frozenset({
    "transition_id", "episode_id", "decision_timestamp", "symbol_id", "selected_action",
    "eligibility", "action_probability_if_known", "behavior_policy_version", "reward",
    "terminal", "truncated", "execution_environment", "constraint_status",
    "schema_version", "event_timestamp_ns", "source_feed", "available_actions",
    "next_state_id",
})

# Feature columns the current gold build actually carries (richer state comes from joining
# silver features, which a separate task owns — the loader accepts any extra numeric cols).
_DEFAULT_REAL_FEATURES = ("current_weight", "target_weight")


class InsufficientDataError(RuntimeError):
    """Raised when a learner is asked to train on data that fails the eligibility guard."""


class SyntheticDataError(RuntimeError):
    """Raised when explicitly-synthetic data is used where a promotable model is required."""


@dataclass
class DataQualityReport:
    """Observable provenance + quality signals for a loaded transition set."""

    source: str = ""
    synthetic: bool = False
    rows_raw: int = 0
    rows_kept: int = 0
    excluded_mandatory_exits: int = 0
    n_actions: int = 0
    n_episodes: int = 0
    n_reward_known: int = 0
    n_terminal: int = 0
    n_truncated: int = 0
    point_in_time_violations: int = 0
    has_behavior_probabilities: bool = False  # ALWAYS False — never fabricated
    feature_names: tuple = ()
    notes: list = field(default_factory=list)

    def to_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items()}
        d["feature_names"] = list(self.feature_names)
        return d


@dataclass(frozen=True)
class ActionSpace:
    """Discrete authorized-action space (allocation policies), mandatory exits excluded."""

    actions: tuple

    @property
    def n(self) -> int:
        return len(self.actions)

    def index(self, name: str) -> int:
        return self.actions.index(name)

    def name(self, idx: int) -> str:
        return self.actions[int(idx)]

    def to_dict(self) -> dict:
        return {"actions": list(self.actions), "n": self.n}


@dataclass
class TransitionDataset:
    """Immutable, versioned logged-transition set for conservative offline RL.

    Behavior probabilities are intentionally absent (``action_probability_if_known`` is
    ``None``). ``reward`` is ``NaN`` where unknown, masked by ``reward_known``.
    """

    states: np.ndarray                 # (N, d) float32
    actions: np.ndarray                # (N,) int32 — index into action_space
    rewards: np.ndarray                # (N,) float32 — NaN where unknown
    reward_known: np.ndarray           # (N,) bool
    next_states: np.ndarray            # (N, d) float32 (copy of state where terminal)
    next_actions: np.ndarray           # (N,) int32 — logged next action, -1 if none
    next_action_valid: np.ndarray      # (N,) bool
    terminal: np.ndarray               # (N,) bool
    truncated: np.ndarray              # (N,) bool
    action_mask: np.ndarray            # (N, n_actions) float32 {0,1}
    episode_ids: np.ndarray            # (N,) object
    is_initial: np.ndarray             # (N,) bool — first transition of an episode
    action_space: ActionSpace
    feature_names: tuple
    data_quality: DataQualityReport
    synthetic: bool = False
    source: str = ""
    version: str = TRANSITIONS_VERSION
    # ALWAYS None: the logging policy is unknown and is never fabricated.
    action_probability_if_known: Optional[np.ndarray] = None

    # -- derived -------------------------------------------------------------
    @property
    def n_transitions(self) -> int:
        return int(self.states.shape[0])

    @property
    def n_features(self) -> int:
        return int(self.states.shape[1]) if self.states.ndim == 2 else 0

    @property
    def n_actions(self) -> int:
        return self.action_space.n

    @property
    def n_reward_known(self) -> int:
        return int(np.count_nonzero(self.reward_known))

    @property
    def n_episodes(self) -> int:
        return int(len(np.unique(self.episode_ids)))

    def behavior_action_counts(self) -> dict:
        counts: dict = {}
        for a in self.actions:
            nm = self.action_space.name(int(a))
            counts[nm] = counts.get(nm, 0) + 1
        return counts

    def subset(self, idx) -> "TransitionDataset":
        """Return a view-dataset over a boolean mask or integer index (for train/val)."""
        idx = np.asarray(idx)
        return TransitionDataset(
            states=self.states[idx], actions=self.actions[idx], rewards=self.rewards[idx],
            reward_known=self.reward_known[idx], next_states=self.next_states[idx],
            next_actions=self.next_actions[idx], next_action_valid=self.next_action_valid[idx],
            terminal=self.terminal[idx], truncated=self.truncated[idx],
            action_mask=self.action_mask[idx], episode_ids=self.episode_ids[idx],
            is_initial=self.is_initial[idx], action_space=self.action_space,
            feature_names=self.feature_names, data_quality=self.data_quality,
            synthetic=self.synthetic, source=self.source, version=self.version)

    def summary(self) -> dict:
        return {"version": self.version, "source": self.source, "synthetic": self.synthetic,
                "n_transitions": self.n_transitions, "n_features": self.n_features,
                "n_actions": self.n_actions, "n_reward_known": self.n_reward_known,
                "n_episodes": self.n_episodes, "actions": list(self.action_space.actions),
                "has_behavior_probabilities": False,
                "data_quality": self.data_quality.to_dict()}


# -- loader ------------------------------------------------------------------

def _detect_feature_columns(records: list, explicit: Optional[tuple]) -> tuple:
    if explicit:
        return tuple(explicit)
    keys: list = []
    seen: set = set()
    for rec in records:
        for k, v in rec.items():
            if k in _RESERVED_COLUMNS or k in seen:
                continue
            if isinstance(v, bool):
                continue
            if isinstance(v, (int, float)) or v is None:
                keys.append(k)
                seen.add(k)
    if not keys:
        # fall back to the columns the gold build is known to carry
        present = [c for c in _DEFAULT_REAL_FEATURES
                   if any(c in rec for rec in records)]
        keys = present
    return tuple(sorted(keys))


def build_dataset_from_records(
    records: list, *, synthetic: bool, source: str,
    feature_columns: Optional[tuple] = None,
) -> TransitionDataset:
    """Build a :class:`TransitionDataset` from raw logged-transition dict rows.

    ``synthetic`` MUST be labeled truthfully by the caller: ``True`` for fixtures/placeholder
    data (never promotable), ``False`` only for genuine lakehouse rows. Next-state / next-action
    are reconstructed by ordering within ``episode_id`` by ``decision_timestamp``.
    """
    dq = DataQualityReport(source=source, synthetic=bool(synthetic), rows_raw=len(records))
    if not records:
        dq.notes.append("no_rows")
        empty_space = ActionSpace(actions=())
        return _empty_dataset(empty_space, feature_columns or (), dq, synthetic, source)

    feats = _detect_feature_columns(records, feature_columns)

    # 1) action space (exclude mandatory-exit environment dynamics)
    raw_actions = [str(r.get("selected_action")) for r in records]
    kept_action_names = sorted({a for a in raw_actions
                                if a not in MANDATORY_EXIT_ACTIONS and a != "None"})
    space = ActionSpace(actions=tuple(kept_action_names))
    action_to_idx = {a: i for i, a in enumerate(space.actions)}

    # 2) keep only rows whose logged action is a learnable (non-mandatory-exit) action
    kept_rows: list = []
    excluded = 0
    for r, a in zip(records, raw_actions, strict=False):
        if a in MANDATORY_EXIT_ACTIONS:
            excluded += 1
            continue
        if a not in action_to_idx:
            excluded += 1
            continue
        kept_rows.append(r)
    dq.excluded_mandatory_exits = excluded

    n = len(kept_rows)
    d = len(feats)
    states = np.zeros((n, d), dtype=np.float32)
    actions = np.zeros(n, dtype=np.int32)
    rewards = np.full(n, np.nan, dtype=np.float32)
    reward_known = np.zeros(n, dtype=bool)
    terminal = np.zeros(n, dtype=bool)
    truncated = np.zeros(n, dtype=bool)
    episode_ids = np.empty(n, dtype=object)
    order_keys: list = []
    action_mask = np.ones((n, max(1, space.n)), dtype=np.float32)
    pit_violations = 0

    for i, r in enumerate(kept_rows):
        for j, c in enumerate(feats):
            v = r.get(c)
            states[i, j] = float(v) if v is not None else 0.0
        actions[i] = action_to_idx[str(r.get("selected_action"))]
        rv = r.get("reward")
        if rv is not None and not (isinstance(rv, float) and np.isnan(rv)):
            rewards[i] = float(rv)
            reward_known[i] = True
        terminal[i] = bool(r.get("terminal", False))
        truncated[i] = bool(r.get("truncated", False))
        ep = r.get("episode_id", r.get("transition_id", i))
        episode_ids[i] = ep
        ts = r.get("decision_timestamp", r.get("event_timestamp_ns", i))
        order_keys.append((str(ep), ts if ts is not None else i, i))
        # honest behavior-prob check: must be absent/None (never fabricated)
        if r.get("action_probability_if_known") is not None:
            pit_violations += 1  # a fabricated behavior prob is a data-integrity violation
        # optional explicit availability mask (never fabricated when absent)
        avail = r.get("available_actions")
        if avail is not None and space.n:
            m = np.zeros(space.n, dtype=np.float32)
            for nm in avail:
                if str(nm) in action_to_idx:
                    m[action_to_idx[str(nm)]] = 1.0
            m[actions[i]] = 1.0  # the logged action was, by definition, available
            action_mask[i] = m

    # 3) reconstruct next_state / next_action within each episode (time order)
    next_states = states.copy()
    next_actions = np.full(n, -1, dtype=np.int32)
    next_action_valid = np.zeros(n, dtype=bool)
    is_initial = np.zeros(n, dtype=bool)
    order = sorted(range(n), key=lambda i: order_keys[i])
    by_ep: dict = {}
    for i in order:
        by_ep.setdefault(str(episode_ids[i]), []).append(i)
    for _ep, seq in by_ep.items():
        is_initial[seq[0]] = True
        for k in range(len(seq)):
            i = seq[k]
            if k + 1 < len(seq) and not terminal[i]:
                nxt = seq[k + 1]
                next_states[i] = states[nxt]
                next_actions[i] = actions[nxt]
                next_action_valid[i] = True
            else:
                terminal[i] = True  # last logged step of an episode is terminal for bootstrap

    if pit_violations:
        dq.notes.append("fabricated_behavior_probabilities_detected")
    dq.point_in_time_violations = pit_violations
    dq.rows_kept = n
    dq.n_actions = space.n
    dq.n_episodes = int(len(by_ep))
    dq.n_reward_known = int(np.count_nonzero(reward_known))
    dq.n_terminal = int(np.count_nonzero(terminal))
    dq.n_truncated = int(np.count_nonzero(truncated))
    dq.feature_names = feats

    return TransitionDataset(
        states=states, actions=actions, rewards=rewards, reward_known=reward_known,
        next_states=next_states, next_actions=next_actions,
        next_action_valid=next_action_valid, terminal=terminal, truncated=truncated,
        action_mask=action_mask, episode_ids=episode_ids, is_initial=is_initial,
        action_space=space, feature_names=feats, data_quality=dq,
        synthetic=bool(synthetic), source=source, action_probability_if_known=None)


def _empty_dataset(space, feats, dq, synthetic, source) -> TransitionDataset:
    z0 = np.zeros((0, max(1, len(feats))), dtype=np.float32)
    return TransitionDataset(
        states=z0, actions=np.zeros(0, dtype=np.int32),
        rewards=np.zeros(0, dtype=np.float32), reward_known=np.zeros(0, dtype=bool),
        next_states=z0.copy(), next_actions=np.zeros(0, dtype=np.int32),
        next_action_valid=np.zeros(0, dtype=bool), terminal=np.zeros(0, dtype=bool),
        truncated=np.zeros(0, dtype=bool),
        action_mask=np.zeros((0, max(1, space.n)), dtype=np.float32),
        episode_ids=np.empty(0, dtype=object), is_initial=np.zeros(0, dtype=bool),
        action_space=space, feature_names=tuple(feats), data_quality=dq,
        synthetic=synthetic, source=source, action_probability_if_known=None)


class TransitionLoader:
    """Loads gold ``offline_rl_transitions`` into a versioned :class:`TransitionDataset`.

    The lakehouse warehouse is imported lazily so this module imports on hosts without a
    materialized lakehouse; a non-materialized dataset yields an empty (research-only)
    dataset rather than raising.
    """

    def __init__(self, warehouse=None, *, feature_columns: Optional[tuple] = None) -> None:
        self._warehouse = warehouse
        self.feature_columns = feature_columns

    def _get_warehouse(self):
        if self._warehouse is not None:
            return self._warehouse
        from cengine.lakehouse.warehouse import DuckDBWarehouse
        self._warehouse = DuckDBWarehouse()
        return self._warehouse

    def load(self, *, view: str = "offline_rl_transitions",
             limit: Optional[int] = None) -> TransitionDataset:
        """Load genuine logged transitions (``synthetic=False``) from the lakehouse."""
        wh = self._get_warehouse()
        sql = f"SELECT * FROM {view}"
        if limit:
            sql += f" LIMIT {int(limit)}"
        try:
            table = wh.query_df(sql)
            records = table.to_pandas().to_dict("records")
        except Exception as exc:  # noqa: BLE001 - absence must degrade, not crash
            dq = DataQualityReport(source=view, synthetic=False)
            dq.notes.append(f"load_failed: {type(exc).__name__}: {exc}")
            return _empty_dataset(ActionSpace(()), self.feature_columns or (), dq, False, view)
        return build_dataset_from_records(
            records, synthetic=False, source=view,
            feature_columns=self.feature_columns or _DEFAULT_REAL_FEATURES)


# -- training guard ----------------------------------------------------------

@dataclass
class GuardDecision:
    """Explicit, machine-readable verdict on whether / how a learner may train."""

    method: str
    state: EligibilityState
    allowed_to_train: bool
    promotable: bool
    reasons: list = field(default_factory=list)

    @property
    def mode(self) -> str:
        return self.state.value

    def to_dict(self) -> dict:
        return {"method": self.method, "state": self.state.value,
                "allowed_to_train": self.allowed_to_train, "promotable": self.promotable,
                "reasons": list(self.reasons)}


def _offline_rl_thresholds() -> tuple:
    """Reuse the committed offline-RL eligibility spec so thresholds stay single-sourced."""
    spec = next((s for s in default_task_specs() if s.task == "offline_rl_execution"), None)
    if spec is None:  # pragma: no cover - spec is committed
        return 200, 5_000, 20_000, 2
    return (spec.min_rows_research, spec.min_rows_training,
            spec.min_rows_validation, spec.requires_action_support or 2)


# Methods here are reward-free (BC imitates logged actions); the rest need realized reward.
_REWARD_FREE_METHODS = frozenset({"bc"})
# No shipped method uses importance sampling, so none require behavior probabilities.
_BEHAVIOR_PROB_METHODS: frozenset = frozenset()


class TrainingGuard:
    """Refuses to train on insufficient / synthetic-as-real data (reuses §3 eligibility).

    ``check`` is non-raising (fit() uses it to return an untrained research-only result so
    the pipeline keeps working on thin data). ``require`` raises and is the hard gate used
    before any promotion-track training.
    """

    def __init__(self, *, min_research: Optional[int] = None,
                 min_training: Optional[int] = None,
                 min_validation: Optional[int] = None,
                 min_actions: Optional[int] = None) -> None:
        r, t, v, a = _offline_rl_thresholds()
        self.min_research = int(min_research if min_research is not None else r)
        self.min_training = int(min_training if min_training is not None else t)
        self.min_validation = int(min_validation if min_validation is not None else v)
        self.min_actions = int(min_actions if min_actions is not None else a)

    def check(self, dataset: TransitionDataset, *, method: str) -> GuardDecision:
        method = method.lower()
        reasons: list[str] = []

        if method in _BEHAVIOR_PROB_METHODS:  # pragma: no cover - none shipped
            reasons.append("requires_behavior_probabilities_which_are_unknown (IPS gated off)")
            return GuardDecision(method, EligibilityState.INSUFFICIENT_DATA, False, False,
                                 reasons)

        if dataset.data_quality.point_in_time_violations > 0:
            reasons.append("fabricated_behavior_probabilities_or_pit_violation")
            return GuardDecision(method, EligibilityState.DATA_QUALITY_FAILED, False, False,
                                 reasons)

        if dataset.n_actions < self.min_actions:
            reasons.append(f"action_support={dataset.n_actions}<{self.min_actions}")
            return GuardDecision(method, EligibilityState.INSUFFICIENT_DATA, False, False,
                                 reasons)

        needs_reward = method not in _REWARD_FREE_METHODS
        if needs_reward and dataset.n_reward_known == 0:
            reasons.append("no_realized_reward (closed-position reward required; none present)")
            return GuardDecision(method, EligibilityState.INSUFFICIENT_DATA, False, False,
                                 reasons)

        effective = dataset.n_reward_known if needs_reward else dataset.n_transitions
        reasons.append(f"effective_rows={effective} "
                       f"({'reward_known' if needs_reward else 'transitions'})")
        reasons.append(f"actions={dataset.n_actions}, episodes={dataset.n_episodes}")

        if effective < self.min_research:
            state = EligibilityState.INSUFFICIENT_DATA
        elif effective < self.min_training:
            state = EligibilityState.RESEARCH_ELIGIBLE
        elif effective < self.min_validation:
            state = EligibilityState.TRAINING_ELIGIBLE
        else:
            state = EligibilityState.VALIDATION_ELIGIBLE

        allowed = state in (EligibilityState.RESEARCH_ELIGIBLE,
                            EligibilityState.TRAINING_ELIGIBLE,
                            EligibilityState.VALIDATION_ELIGIBLE)
        promotable = (state == EligibilityState.VALIDATION_ELIGIBLE) and not dataset.synthetic
        if dataset.synthetic:
            reasons.append("synthetic_data: research-only, never promotable")
        return GuardDecision(method, state, allowed, promotable, reasons)

    def require(self, dataset: TransitionDataset, *, method: str,
                promote: bool = False) -> GuardDecision:
        """Hard gate: raise unless training is allowed (and promotable when ``promote``)."""
        decision = self.check(dataset, method=method)
        if not decision.allowed_to_train:
            raise InsufficientDataError(
                f"offline-RL training refused for method={method}: {decision.reasons}")
        if promote:
            if dataset.synthetic:
                raise SyntheticDataError(
                    "synthetic data can never produce a promotable offline-RL model")
            if not decision.promotable:
                raise InsufficientDataError(
                    f"not promotable for method={method}: {decision.reasons}")
        return decision


# -- explicitly-labeled synthetic fixtures (research/testing only) -----------

def make_synthetic_dataset(
    *, n_episodes: int = 60, steps: int = 5, n_features: int = 5, n_actions: int = 3,
    reward_fraction: float = 1.0, seed: int = 7, source: str = "synthetic_fixture",
    include_mandatory_exit: bool = False,
) -> TransitionDataset:
    """Build a LEARNABLE, explicitly-synthetic transition set (``synthetic=True``).

    The logged action depends on the state so BC can beat chance; closed-transition rewards
    depend on state+action so value learning has signal. Always labeled synthetic => the
    guard will mark any result research-only / non-promotable. Optionally injects a reserved
    mandatory-exit label to prove it is excluded from the learnable action space.
    """
    rng = np.random.default_rng(seed)
    pref = rng.normal(0.0, 1.0, size=(n_actions, n_features))
    rw = rng.normal(0.0, 0.5, size=n_features)
    action_effect = np.linspace(-0.01, 0.01, n_actions)
    action_names = [f"policy_{k}" for k in range(n_actions)]
    records: list = []
    for ep in range(n_episodes):
        s = rng.normal(0.0, 1.0, size=n_features)
        for t in range(steps):
            scores = pref @ s
            a = int(np.argmax(scores + rng.normal(0.0, 0.3, size=n_actions)))
            is_terminal = t == steps - 1
            rec = {"transition_id": f"ep{ep}-t{t}", "episode_id": f"ep{ep}",
                   "decision_timestamp": ep * 100 + t, "symbol_id": f"SYM{ep % 4}",
                   "selected_action": action_names[a],
                   "action_probability_if_known": None,  # NEVER fabricated
                   "terminal": is_terminal, "truncated": False,
                   "execution_environment": "synthetic", "eligibility": "eligible"}
            for j in range(n_features):
                rec[f"f{j}"] = float(s[j])
            if rng.random() <= reward_fraction:
                rec["reward"] = float(s @ rw + action_effect[a] + rng.normal(0.0, 0.01))
            else:
                rec["reward"] = None
            records.append(rec)
            s = 0.9 * s + 0.1 * rng.normal(0.0, 1.0, size=n_features)
    if include_mandatory_exit:
        base = dict(records[0])
        base["transition_id"] = "mandatory-exit-row"
        base["selected_action"] = "mandatory_protective_exit"
        base["reward"] = -0.005
        records.append(base)
    feats = tuple(f"f{j}" for j in range(n_features))
    return build_dataset_from_records(records, synthetic=True, source=source,
                                      feature_columns=feats)
