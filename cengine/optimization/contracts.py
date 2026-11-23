"""Shared contracts for the autonomous optimization platform (Milestone 6B).

Versioned, serializable dataclasses threaded through the closed loop:
attribution -> diagnostic -> hypothesis -> experiment -> evaluation -> memory. Kept in one
module so the engines share identity without circular imports. Nothing here executes orders
or grants financial authority — these are research records only.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Optional

OPTIMIZATION_CONTRACTS_VERSION = "optimization-contracts-1.0.0"


class Severity(str, Enum):
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ExperimentStatus(str, Enum):
    PROPOSED = "proposed"
    DEFERRED = "deferred"               # insufficient eligible data
    RUNNING = "running"
    COMPLETED = "completed"
    REJECTED = "rejected"
    RESEARCH_ONLY = "research_only"
    ELIGIBLE_FOR_SHADOW = "eligible_for_shadow"
    BLOCKED = "blocked"
    ERROR = "error"


@dataclass
class Diagnostic:
    """A detected, economically-significant weakness (§8.3)."""
    diagnostic_id: str
    detected_ns: int
    component: str                      # e.g. 'protective_exit' | 'entry' | 'model:forward_return'
    period: str
    evidence_refs: list = field(default_factory=list)
    sample_size: int = 0
    est_economic_impact: Optional[float] = None
    uncertainty: Optional[float] = None
    severity: str = Severity.INFO.value
    suspected_cause: str = ""
    alternatives: list = field(default_factory=list)
    recommended_experiment_family: str = ""
    required_eligibility: str = ""
    priority: float = 0.0
    status: str = "open"
    causal_note: str = ("descriptive attribution only; causation requires a controlled "
                        "experiment")

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Hypothesis:
    """A testable intervention derived from a diagnostic (§9.2)."""
    hypothesis_id: str
    diagnostic_id: str
    component: str
    existing_behavior: str
    suspected_weakness: str
    intervention: str
    expected_improvement: str
    methodology: str
    acceptance_criteria: dict = field(default_factory=dict)
    risk_constraints: list = field(default_factory=list)
    search_space: str = ""              # named search space to explore
    priority: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ExperimentManifest:
    """Reproducible experiment record (§23.1)."""
    experiment_id: str
    parent_experiment: Optional[str]
    diagnostic_id: Optional[str]
    hypothesis_id: Optional[str]
    search_space_version: str
    model_family: str
    model_architecture: str
    hyperparameters: dict
    feature_definition: str
    dataset_version: str
    training_window: str
    evaluation_window: str
    reward_definition: str
    policy_configuration: dict
    random_seed: int
    execution_device: str
    runtime_s: float
    resource: dict
    git_commit: str
    artifact_hashes: dict
    economic_metrics: dict
    risk_metrics: dict
    statistical_uncertainty: dict
    evaluation_result: dict
    decision: str
    rejection_reason: str
    shadow_observations: dict = field(default_factory=dict)
    data_label: str = "synthetic"       # 'real' | 'synthetic'
    created_ns: int = 0
    version: str = OPTIMIZATION_CONTRACTS_VERSION

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class CollectionTask:
    """A prioritized data-acquisition request (§7.3)."""
    task_id: str
    learning_task: str
    missing_evidence: str
    recoverable_historically: bool
    priority: float
    estimated_economic_importance: Optional[float] = None
    suggested_source: str = ""
    status: str = "open"

    def to_dict(self) -> dict:
        return asdict(self)


_TICKER_RE = re.compile(r"^[A-Z0-9.\-]{1,12}$")


def safe_ticker(symbol: str) -> str:
    """Validate a ticker before it is interpolated into SQL (defense against injection).

    Tickers are uppercase alphanumerics plus '.'/'-'; anything else is rejected. Returns the
    normalized symbol. Use this on ANY symbol that reaches a DuckDB query string.
    """
    s = str(symbol).strip().upper()
    if not _TICKER_RE.match(s):
        raise ValueError(f"invalid ticker symbol: {symbol!r} (expected ^[A-Z0-9.\\-]{{1,12}}$)")
    return s


def stable_id(prefix: str, *parts) -> str:
    basis = "|".join(str(p) for p in parts)
    return f"{prefix}-{hashlib.sha256(basis.encode()).hexdigest()[:16]}"


def config_fingerprint(config: dict) -> str:
    """Deterministic fingerprint of an experiment configuration (for dedup)."""
    return hashlib.sha256(
        json.dumps(config, sort_keys=True, default=str).encode()).hexdigest()[:24]
