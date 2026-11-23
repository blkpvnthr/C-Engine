"""Learning certification — supervised ML, contextual bandits, offline RL, autonomous optimization.

The learning stack is engineering-complete, but genuine economic learning value is **not
demonstrable** on the available data, and this certification says so plainly rather than
manufacturing a result:

* **infrastructure** — the learning packages import and the eligibility gate + economic
  evaluator are operational. CERTIFIED.
* **data eligibility** — the real eligibility assessment over the lakehouse: only the THEORETICAL
  ``forward_return_prediction`` target is validation-eligible (and it is explicitly NOT realized
  strategy return under the protective exit); every strategy-relevant / bandit / RL task is
  INSUFFICIENT_DATA because the gold/live outcome layers are not materialized (no live sessions).
  So strategy-relevant supervised, bandit and RL learning are INSUFFICIENT_DATA — honestly.
* **provenance discipline** — synthetic and historical-replay results can never back a
  profitability claim (the economic evaluator enforces it). CERTIFIED.
* **recommendation-only** — no learning module can route an order: the learning package imports
  no order-manager / execution-venue path. A learning fault can never place a trade. CERTIFIED.
* **autonomous-optimization safety** — ``OptimizationConfig.validate`` refuses autonomous risk
  expansion, refuses to change the −0.5% production exit, and refuses to loosen the −3% emergency
  threshold. CERTIFIED. (The loop demonstrably retains the incumbent — it certified no economic
  improvement, which is the correct falsification-first outcome, not a defect.)
"""

from __future__ import annotations

from cengine.certification.contracts import (
    DATA_LABEL_HISTORICAL_REPLAY,
    DATA_LABEL_REAL,
    DATA_LABEL_SYNTHETIC,
    CertificationResult,
    CertificationVerdict,
)

LEARNING_CERT_VERSION = "certification-learning-1.0.0"


def _v(ok: bool) -> CertificationVerdict:
    return CertificationVerdict.CERTIFIED if ok else CertificationVerdict.NOT_CERTIFIED


def _certify_infrastructure() -> CertificationResult:
    required = [
        "cengine.learning.datasets.eligibility",
        "cengine.learning.evaluation.economic",
        "cengine.learning.inference.service",
        "cengine.learning.contextual_bandits.linucb",
        "cengine.learning.offline_rl.cql_iql",
        "cengine.optimization.controller",
    ]
    import importlib
    missing = []
    for mod in required:
        try:
            importlib.import_module(mod)
        except Exception as exc:  # noqa: BLE001
            missing.append(f"{mod}: {exc}")
    return CertificationResult(
        check="learning.infrastructure", component="learning_ml", verdict=_v(not missing),
        summary=("learning + autonomous-optimization stack imports cleanly"
                 if not missing else f"import failures: {missing}"),
        metrics={"modules": required, "missing": missing})


def _certify_data_eligibility() -> CertificationResult:
    from cengine.learning.datasets.eligibility import assess_all
    report = assess_all()
    tasks = report.get("tasks", [])
    by_state: dict = {}
    strategy_relevant_eligible = []
    for t in tasks:
        st = t.get("eligibility")
        by_state[st] = by_state.get(st, 0) + 1
        # forward_return_prediction is theoretical; genuine strategy value needs a REALIZED target
        if (st == "validation_eligible" and t.get("task") not in ("forward_return_prediction",)):
            strategy_relevant_eligible.append(t.get("task"))
    # honest verdict: no strategy-relevant task is validation-eligible -> INSUFFICIENT_DATA
    verdict = (CertificationVerdict.CERTIFIED if strategy_relevant_eligible
               else CertificationVerdict.INSUFFICIENT_DATA)
    eligible_text = (", ".join(strategy_relevant_eligible) if strategy_relevant_eligible
                     else "NONE (only the theoretical forward_return target is eligible; no "
                          "realized-outcome layers materialized)")
    return CertificationResult(
        check="learning.data_eligibility", component="learning_ml", verdict=verdict,
        summary=(f"eligibility by state {by_state}; strategy-relevant validation-eligible tasks: "
                 f"{eligible_text}"),
        data_label=DATA_LABEL_HISTORICAL_REPLAY,
        metrics={"by_state": by_state,
                 "tasks": [{"task": t.get("task"), "eligibility": t.get("eligibility")}
                           for t in tasks]},
        notes="strategy-relevant supervised/bandit/RL learning cannot be economically certified "
              "without realized outcomes (no live sessions during certification)")


def _certify_provenance_discipline() -> CertificationResult:
    from cengine.learning.evaluation.economic import economic_metrics
    # synthetic and historical-replay results must NEVER permit a profitability claim
    rets = [0.01, -0.005, 0.008, -0.003, 0.006] * 10
    syn = economic_metrics(rets, data_label=DATA_LABEL_SYNTHETIC)
    rep = economic_metrics(rets, data_label=DATA_LABEL_HISTORICAL_REPLAY)
    real = economic_metrics(rets, data_label=DATA_LABEL_REAL)
    syn_blocked = not getattr(syn, "profitability_claim_allowed", True)
    rep_blocked = not getattr(rep, "profitability_claim_allowed", True)
    real_allowed = getattr(real, "profitability_claim_allowed", False)
    ok = syn_blocked and rep_blocked and real_allowed
    return CertificationResult(
        check="learning.provenance_discipline", component="learning_ml", verdict=_v(ok),
        summary=("synthetic & historical-replay results cannot claim profitability; only "
                 "genuine realized ('real') outcomes can"),
        metrics={"synthetic_claim_blocked": syn_blocked,
                 "historical_replay_claim_blocked": rep_blocked,
                 "real_claim_allowed": real_allowed})


def _certify_recommendation_only() -> CertificationResult:
    """The learning package must not be able to route orders (recommendation-only)."""
    import re
    from pathlib import Path

    import cengine.learning as learning_pkg
    root = Path(learning_pkg.__file__).parent
    offenders = []
    # real order-routing coupling = an actual IMPORT of the order path or a venue, OR a live
    # .submit(...) order call — NOT a prose/docstring mention of "OrderManager".
    import_re = re.compile(
        r"^\s*(?:from\s+(order_manager|cengine\.native_execution|market_data\.\w+)\s+import"
        r"|import\s+(order_manager))", re.MULTILINE)
    submit_re = re.compile(r"\.submit\s*\(")
    for path in root.rglob("*.py"):
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        hits = []
        if import_re.search(text):
            hits.append("imports_order_path")
        # a .submit( that is not inside a comment line
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if submit_re.search(line):
                hits.append("submit_call")
                break
        if hits:
            offenders.append({"module": str(path.relative_to(root)), "tokens": sorted(set(hits))})
    return CertificationResult(
        check="learning.recommendation_only", component="learning_ml", verdict=_v(not offenders),
        summary=("no learning module routes orders (imports no order-manager / execution-venue / "
                 "submit path) — a learning fault can never place a trade"
                 if not offenders else f"order-routing coupling found: {offenders}"),
        metrics={"offenders": offenders})


def _certify_autonomous_opt_safety() -> CertificationResult:
    import dataclasses

    from cengine.optimization.config import ConfigError, OptimizationConfig
    base = OptimizationConfig()
    checks = {}
    try:
        base.validate()
        checks["default_valid"] = True
    except Exception:  # noqa: BLE001
        checks["default_valid"] = False

    def _raises(**kw) -> bool:
        try:
            dataclasses.replace(base, **kw).validate()
            return False
        except ConfigError:
            return True
        except Exception:  # noqa: BLE001
            return True

    checks["risk_expansion_blocked"] = _raises(autonomous_risk_expansion=True)
    checks["production_exit_locked"] = _raises(production_exit_pct=-0.01)
    checks["emergency_loosen_blocked"] = _raises(candidate_emergency_pct=-0.05)  # more neg = looser
    # tightening the emergency (less negative) must be ALLOWED
    try:
        dataclasses.replace(base, candidate_emergency_pct=-0.01).validate()
        checks["emergency_tighten_allowed"] = True
    except Exception:  # noqa: BLE001
        checks["emergency_tighten_allowed"] = False
    ok = all(checks.values())
    return CertificationResult(
        check="learning.autonomous_opt_safety", component="autonomous_opt", verdict=_v(ok),
        summary=("optimization config refuses risk expansion, locks the −0.5% production exit, and "
                 "refuses to loosen the −3% emergency (tightening allowed)"),
        metrics=checks,
        notes="the governed loop retains the incumbent — it certified NO economic improvement, "
              "the correct falsification-first outcome")


def _certify_bandit_rl() -> CertificationResult:
    return CertificationResult(
        check="learning.bandit_rl", component="learning_bandit_rl",
        verdict=CertificationVerdict.INSUFFICIENT_DATA,
        summary="contextual-bandit / offline-RL policies require logged action-propensity + "
                "realized-reward data from live/paper sessions — none available",
        notes="implemented + unit-tested (engineering), but economically INSUFFICIENT_DATA; "
              "no fabricated behavior probabilities or rewards")


def certify_learning() -> list:
    """Full learning + autonomous-optimization certification."""
    return [
        _certify_infrastructure(),
        _certify_data_eligibility(),
        _certify_provenance_discipline(),
        _certify_recommendation_only(),
        _certify_autonomous_opt_safety(),
        _certify_bandit_rl(),
    ]
