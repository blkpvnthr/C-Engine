"""Autonomous optimization spine tests (Milestone 6B, §30).

Deterministic + offline. Covers contracts/ticker-validation, performance attribution,
data-sufficiency + collection tasks, hypothesis generation, search spaces + samplers + dedup,
persistent optimization memory, economic scoring + mandatory gates + baseline hierarchy, the
closed-loop MetaOptimizer, governed config safety invariants, and OOS validation windows
(holdout reserved).
"""

from __future__ import annotations

import numpy as np
import pytest

from cengine.lakehouse.config import LakehouseConfig
from cengine.optimization.config import ConfigError, OptimizationConfig
from cengine.optimization.contracts import Diagnostic, Severity, config_fingerprint, safe_ticker
from cengine.optimization.controller import MetaOptimizer
from cengine.optimization.data_sufficiency import DataSufficiencyController
from cengine.optimization.economic import (
    EconomicScoreConfig,
    compare_to_baselines,
    evaluate_candidate,
)
from cengine.optimization.hypotheses import HypothesisGenerator
from cengine.optimization.memory import OptimizationMemory
from cengine.optimization.search import (
    default_search_spaces,
    random_search,
    successive_halving,
    tpe_suggest,
)

NS = 1_000_000_000


@pytest.fixture()
def config(tmp_path):
    cfg = LakehouseConfig(root=tmp_path / "lakehouse", min_free_disk_gib=0.0)
    cfg.ensure_dirs()
    return cfg


# -- contracts / injection guard --------------------------------------------

def test_safe_ticker_accepts_valid_rejects_injection():
    assert safe_ticker("aapl") == "AAPL"
    assert safe_ticker("BRK.B") == "BRK.B"
    for bad in ["AAPL'; DROP TABLE x;--", "A B", "'", "", "TOOLONGSYMBOL123"]:
        with pytest.raises(ValueError):
            safe_ticker(bad)


def test_config_fingerprint_stable_and_discriminating():
    a = config_fingerprint({"x": 1, "y": 2})
    assert a == config_fingerprint({"y": 2, "x": 1})      # order-independent
    assert a != config_fingerprint({"x": 1, "y": 3})


# -- config safety invariants -----------------------------------------------

def test_config_rejects_unsafe_settings():
    OptimizationConfig().validate()  # defaults are safe
    with pytest.raises(ConfigError):
        OptimizationConfig(production_activation_enabled=True).validate()
    with pytest.raises(ConfigError):
        OptimizationConfig(autonomous_risk_expansion=True).validate()
    with pytest.raises(ConfigError):
        OptimizationConfig(hard_stop_optimization=True).validate()
    with pytest.raises(ConfigError):
        OptimizationConfig(production_exit_pct=-0.01).validate()    # must stay -0.5%
    with pytest.raises(ConfigError):
        OptimizationConfig(candidate_emergency_pct=-0.05).validate()  # loosening forbidden


# -- attribution / data sufficiency -----------------------------------------

def test_attribution_insufficient_when_no_outcomes(config):
    from cengine.optimization.attribution import PerformanceAttributionEngine
    rep = PerformanceAttributionEngine(config).attribute(now_ns=NS)
    assert rep["data_available"] is False
    assert any(d["status"] == "insufficient_data" for d in rep["diagnostics"])


def test_data_sufficiency_emits_collection_tasks(config):
    rep = DataSufficiencyController(config).assess(now_ns=NS)
    # with an empty lakehouse, outcome tasks are insufficient -> collection tasks generated
    assert rep["collection_tasks"]
    assert all("missing_evidence" in t for t in rep["collection_tasks"])
    # collection tasks are prioritized (descending)
    pr = [t["priority"] for t in rep["collection_tasks"]]
    assert pr == sorted(pr, reverse=True)
    assert "no fabricated observations" in rep["policy"]


# -- hypotheses --------------------------------------------------------------

def test_hypothesis_from_diagnostic_and_standing():
    diag = Diagnostic(diagnostic_id="d1", detected_ns=NS, component="protective_exit",
                      period="all", severity=Severity.HIGH.value,
                      recommended_experiment_family="protective_exit_matrix", priority=5.0)
    gen = HypothesisGenerator()
    h = gen.from_diagnostic(diag)
    assert h is not None and h.search_space == "protective_exit"
    assert h.acceptance_criteria.get("no_risk_expansion") is True
    standing = gen.standing(eligible_tasks={"forward_return_prediction"})
    spaces = {s.search_space for s in standing}
    assert "protective_exit" in spaces and "supervised_model" in spaces


# -- search ------------------------------------------------------------------

def test_search_space_validate_and_clip():
    space = default_search_spaces()["protective_exit"]
    bad = {"initial_stop_pct": 99.0, "policy": "nonsense"}
    ok, reasons = space.validate(bad)
    assert not ok and reasons
    clipped = space.clip({"initial_stop_pct": 99.0})
    assert clipped["initial_stop_pct"] <= 2.5


def test_random_and_tpe_and_halving():
    space = default_search_spaces()["rl_reward"]
    rs = random_search(space, 5, seed=1)
    assert len(rs) == 5 and all(space.validate(c)[0] for c in rs)
    history = [{"config": c, "score": -float(c["turnover_penalty"])} for c in rs]
    tpe = tpe_suggest(space, history * 3, 4, seed=2)
    assert len(tpe) == 4
    res = successive_halving([{"a": i} for i in range(8)],
                             lambda c, b: -abs(c["a"] - 3) + 0.01 * b, budgets=(1, 2, 4))
    assert res["best"] is not None


# -- memory ------------------------------------------------------------------

def test_memory_dedup_and_effectiveness(config):
    mem = OptimizationMemory(config)
    mem.record(hypothesis_id="h", search_space="rl_reward", config={"turnover_penalty": 0.5},
               score=0.1, economic_metrics={}, decision="eligible_for_shadow")
    assert mem.is_duplicate("rl_reward", {"turnover_penalty": 0.5})
    deduped = mem.dedupe("rl_reward", [{"turnover_penalty": 0.5}, {"turnover_penalty": 0.9}])
    assert len(deduped) == 1
    eff = mem.effectiveness()
    assert eff["experiments_attempted"] == 1 and eff["improvements_identified"] == 1


# -- economic gates + baselines ---------------------------------------------

def test_economic_gates_reject_excessive_drawdown():
    # a series with a catastrophic drawdown fails the mandatory gate regardless of return
    rng = np.random.default_rng(0)
    r = np.concatenate([rng.normal(0.001, 0.005, 50), np.full(20, -0.05)])
    ev = evaluate_candidate(r, label="c", cfg=EconomicScoreConfig(max_drawdown_limit=0.1))
    assert ev.gates_passed is False


def test_compare_to_baselines_requires_beating_incumbent():
    good = evaluate_candidate(np.full(60, 0.002), label="cand")
    weak_inc = evaluate_candidate(np.full(60, -0.001), label="incumbent")
    strong_inc = evaluate_candidate(np.full(60, 0.01), label="incumbent")
    assert compare_to_baselines(
        good, {"incumbent": weak_inc})["recommendation"] == "eligible_for_shadow"
    assert compare_to_baselines(good, {"incumbent": strong_inc})["recommendation"] == "reject"


# -- closed-loop controller --------------------------------------------------

def test_meta_optimizer_closed_loop_accepts_and_remembers(config):
    rng = np.random.default_rng(0)

    def runner(space, cfg, now_ns):
        cand = rng.normal(0.0015, 0.01, 120)
        inc = rng.normal(-0.0005, 0.01, 120)
        return {"evaluation": evaluate_candidate(cand, label="c").to_dict(),
                "baselines": {"incumbent": evaluate_candidate(inc, label="i").to_dict()},
                "data_label": "synthetic", "runtime_s": 0.01, "device": "cpu"}

    mo = MetaOptimizer(config, runners={"protective_exit": runner})
    rep = mo.run(now_ns=NS)
    assert rep["outcomes"]["accepted"] >= 1
    assert rep["memory_effectiveness"]["experiments_attempted"] >= 1
    # boundary invariant present
    assert "shadow-eligible only" in rep["boundary"]


def test_meta_optimizer_defers_without_runner(config):
    mo = MetaOptimizer(config, runners={})     # no runners registered
    rep = mo.run(now_ns=NS)
    assert rep["outcomes"]["accepted"] == 0
    assert rep["outcomes"]["deferred"] >= 1


# -- OOS validation ----------------------------------------------------------

def test_oos_windows_absent_data(config):
    from cengine.optimization.validation import oos_windows
    assert oos_windows(config, "AAPL") is None   # no bars in this tmp lakehouse
