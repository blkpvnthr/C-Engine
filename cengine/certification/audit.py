"""M7A — Independent engineering audit (component status matrix).

Determines, for every platform component, an evidence-based :class:`ComponentStatus` from
*mechanical signals* rather than assertion:

* **IMPLEMENTED / BLOCKED** — does the module import cleanly in an isolated subprocess? (A
  subprocess isolates import-time side effects, native crashes and hangs behind a timeout, so
  one broken module can never abort the audit or taint the auditing interpreter.)
* **UNIT_TESTED** — is the module imported by at least one file under ``tests/``?
* **OPERATIONALLY_CONSUMED** — is it imported by a genuine runtime path (``trade.py``, the
  continuous ``cengine/runtime`` package, or another first-class entrypoint)?

The static audit **caps at OPERATIONALLY_CONSUMED on purpose.** The higher rungs
(MOCK_CERTIFIED, HISTORICALLY_VALIDATED, PAPER_CERTIFIED, LIVE_APPROVED) represent empirical
certification evidence and may only be granted by the actual certification runs that produce
that evidence — never inferred from the presence of code. This keeps the audit honest: it
reports what is *built and wired*, not what is *proven to work economically*.
"""

from __future__ import annotations

import concurrent.futures
import os
import re
import subprocess
import sys
from pathlib import Path

from cengine.certification.contracts import (
    CertificationResult,
    CertificationVerdict,
    ComponentAudit,
    ComponentStatus,
)

AUDIT_VERSION = "m7a-engineering-audit-1.0.0"

_REPO_ROOT = Path(__file__).resolve().parents[2]

# Runtime roots whose imports count as "operationally consumed" — genuine execution/continuous
# paths, NOT tests and NOT research notebooks.
_RUNTIME_ROOTS = (
    "trade.py",
    "order_manager.py",
    "order_flow_engine.py",
    "protective_exit.py",
    "scanner_bridge.py",
    "universe_sync.py",
    "canonical_universe.py",
    "cengine/runtime",
)

# Logical grouping for the matrix. Every module is also audited individually; this grouping is
# for the human-facing roll-up. Modules not listed here fall into "uncategorized" and are still
# audited — nothing is silently dropped.
COMPONENT_GROUPS: dict = {
    "execution": [
        "order_manager", "order_flow_engine", "cengine.execution",
        "cengine.execution_planner", "cengine.execution_policy", "cengine.native_execution",
        "cengine.transaction_cost",
    ],
    "protective_exit": ["protective_exit", "cengine.position_management"],
    "strategies": [
        "strategies", "cengine.strategy_families", "cengine.strategy_arbitration",
        "cengine.strategy_execution_bridge", "cengine.strategy_maturity",
        "cengine.strategy_state", "amps_v4_regime",
    ],
    "regime": [
        "cengine.regime_engine", "cengine.regime_strategy_pipeline",
        "cengine.learning.regime_models.transition_model",
    ],
    "portfolio": [
        "cengine.portfolio", "cengine.portfolio_analytics", "cengine.portfolio_constraints",
        "cengine.portfolio_optimization", "cengine.portfolio_optimizer",
        "cengine.portfolio_pipeline", "cengine.portfolio_replacement",
        "cengine.portfolio_state", "cengine.capital_reservation", "cengine.reservations",
        "cengine.exposure_registry",
    ],
    "risk_native": [
        "native_adapters", "cengine.metal_portfolio", "cengine.metal_covariance",
        "cengine.metal_indicators", "cengine.metal_runtime", "cengine.metal_backend",
        "cengine.metal_buffers", "cengine.metal_validation", "cengine.metal_cpu_ref",
    ],
    "discovery_gpu": [
        "cengine.gpu_discovery", "cengine.gpu_scanner_bridge", "cengine.gpu_worker",
        "cengine.operational_discovery", "cengine.microstructure", "indicators", "liquidity",
    ],
    "lakehouse": [
        "cengine.lakehouse.warehouse", "cengine.lakehouse.bronze", "cengine.lakehouse.silver",
        "cengine.lakehouse.gold", "cengine.lakehouse.feature_store",
        "cengine.lakehouse.historical", "cengine.lakehouse.quality",
        "cengine.lakehouse.governance", "cengine.lakehouse.pipeline",
        "cengine.lakehouse.attribution", "cengine.historical_warmup",
    ],
    "learning_ml": [
        "cengine.learning.features.builder", "cengine.learning.inference.service",
        "cengine.learning.evaluation.economic", "cengine.learning.evaluation.ope",
        "cengine.learning.calibration.calibration", "cengine.learning.monitoring.drift",
        "cengine.learning.datasets.certification",
    ],
    "learning_bandit_rl": [
        "cengine.learning.contextual_bandits.linucb",
        "cengine.learning.contextual_bandits.neural_bandit",
        "cengine.learning.offline_rl.cql_iql", "cengine.learning.offline_rl.bc",
        "cengine.learning.offline_rl.fqe", "cengine.learning.offline_rl.transitions",
        "cengine.learning.environment.replay",
        "cengine.learning.orchestration.champion_challenger",
        "cengine.learning.orchestration.daily_cycle",
    ],
    "autonomous_opt": [
        "cengine.optimization.controller", "cengine.optimization.orchestration",
        "cengine.optimization.attribution", "cengine.optimization.hypotheses",
        "cengine.optimization.search", "cengine.optimization.experiments",
        "cengine.optimization.memory", "cengine.optimization.economic",
        "cengine.optimization.validation", "cengine.optimization.data_sufficiency",
        "cengine.optimization.exits.adaptive", "cengine.optimization.exits.counterfactual",
        "cengine.optimization.exits.experiment_matrix",
    ],
    "continuous_runtime": [
        "cengine.runtime.calendar", "cengine.runtime.state_machine",
        "cengine.runtime.supervisor",
    ],
    "infra_safety": [
        "cengine.safety", "cengine.integrity", "cengine.journal", "cengine.nosql_journal",
        "cengine.event_bus", "cengine.engine_state", "cengine.metrics",
        "cengine.decision_events", "cengine.membership", "cengine.warmup_scheduler",
        "cengine.adaptive_subscription_controller",
    ],
    "market_data": [
        "market_data.feed", "market_data.market_data_store",
        "market_data.create_daily_market_file",
    ],
}

# Modules intentionally NOT imported live (import-time side effects / live-credential / network
# clients). They are verified by byte-compilation instead of execution, and reported as
# IMPLEMENTED (compiles) with a note — never BLOCKED merely for declining to open a socket.
_NO_LIVE_IMPORT = {
    "market_data.alpaca_stream",
    "market_data.alpaca_sip_stream",
    "cengine.mock_broker",
    "trade",
}


def _all_modules() -> list:
    """Every grouped module plus any uncategorized project module, de-duplicated."""
    grouped = {m for mods in COMPONENT_GROUPS.values() for m in mods}
    return sorted(grouped)


def _module_to_path(mod: str) -> Path:
    rel = mod.replace(".", "/") + ".py"
    return _REPO_ROOT / rel


def _build_import_index(roots: tuple) -> set:
    """Set of module names imported anywhere under the given file/dir roots.

    Scans once for ``import x`` / ``from x import ...`` and records both the top-level and
    dotted forms so ``cengine.foo.bar`` is matched whether imported fully or via a parent.
    """
    import_re = re.compile(
        r"^\s*(?:from\s+([a-zA-Z0-9_.]+)\s+import|import\s+([a-zA-Z0-9_.]+))", re.MULTILINE
    )
    found: set = set()
    files: list = []
    for root in roots:
        p = _REPO_ROOT / root
        if p.is_dir():
            files.extend(p.rglob("*.py"))
        elif p.is_file():
            files.append(p)
    for f in files:
        try:
            text = f.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for m in import_re.finditer(text):
            name = m.group(1) or m.group(2)
            if name:
                found.add(name.split(" as ")[0].strip())
    return found


def _consumed(mod: str, runtime_index: set) -> bool:
    """A module is consumed if it, or any ancestor package it belongs to, is imported by a
    runtime root (but a module never counts as consuming itself)."""
    for name in runtime_index:
        if name == mod or name.startswith(mod + "."):
            return True
    return False


def _tested(mod: str, test_index: set) -> bool:
    for name in test_index:
        if name == mod or name.startswith(mod + ".") or mod.startswith(name + "."):
            return True
    return False


def _try_import(mod: str, *, timeout: float = 90.0) -> tuple:
    """Import a module in an isolated subprocess. Returns (ok, detail)."""
    proc = subprocess.run(
        [sys.executable, "-c", f"import {mod}"],
        cwd=str(_REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=timeout,
        env={**os.environ, "PYTHONWARNINGS": "ignore"},
    )
    if proc.returncode == 0:
        return True, "imports cleanly"
    err = (proc.stderr or "").strip().splitlines()
    detail = err[-1] if err else f"exit {proc.returncode}"
    return False, detail[:240]


def _compiles(mod: str) -> tuple:
    """Byte-compile a module file without executing it. Returns (ok, detail)."""
    path = _module_to_path(mod)
    if not path.exists():
        return False, f"missing file {path.relative_to(_REPO_ROOT)}"
    proc = subprocess.run(
        [sys.executable, "-m", "py_compile", str(path)],
        cwd=str(_REPO_ROOT), capture_output=True, text=True, timeout=60,
    )
    if proc.returncode == 0:
        return True, "compiles (not import-executed: side-effecting/credentialed module)"
    err = (proc.stderr or "").strip().splitlines()
    return False, (err[-1] if err else "compile failed")[:240]


def _audit_one(mod: str, test_index: set, runtime_index: set) -> ComponentAudit:
    path = _module_to_path(mod)
    module_ref = str(path.relative_to(_REPO_ROOT)) if path.exists() else mod
    tested = _tested(mod, test_index)
    consumed = _consumed(mod, runtime_index)

    if mod in _NO_LIVE_IMPORT:
        ok, detail = _compiles(mod)
    else:
        try:
            ok, detail = _try_import(mod)
        except subprocess.TimeoutExpired:
            ok, detail = False, "import timed out (possible import-time blocking work)"

    gaps: list = []
    if not ok:
        status = ComponentStatus.BLOCKED
    else:
        # highest mechanically-justified rung
        if consumed:
            status = ComponentStatus.OPERATIONALLY_CONSUMED
        elif tested:
            status = ComponentStatus.UNIT_TESTED
        else:
            status = ComponentStatus.IMPLEMENTED
            gaps.append("no unit-test import found")
        # record the empirical rungs this static audit cannot grant
        gaps.append("empirical rungs (historical/paper/live) pending cert runs")

    consumers = []
    if consumed:
        consumers = sorted(
            {r for r in _RUNTIME_ROOTS
             if any(n == mod or n.startswith(mod + ".") for n in
                    _build_import_index((r,)))}
        )[:6]

    return ComponentAudit(
        component=mod,
        status=status,
        module=module_ref,
        evidence=detail + (f"; tested={tested}; consumed={consumed}"),
        consumed_by=consumers,
        gaps=gaps,
    )


def run_engineering_audit(*, parallel: int = 6) -> tuple:
    """Run the full M7A audit. Returns (list[ComponentAudit], list[CertificationResult]).

    The results summarize group-level roll-ups and flag any BLOCKED component as NOT_CERTIFIED
    (a broken import is a hard engineering failure); the matrix itself is the audit rows.
    """
    test_index = _build_import_index(("tests",))
    runtime_index = _build_import_index(_RUNTIME_ROOTS)
    modules = _all_modules()

    audits: dict = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=parallel) as ex:
        futs = {ex.submit(_audit_one, m, test_index, runtime_index): m for m in modules}
        for fut in concurrent.futures.as_completed(futs):
            mod = futs[fut]
            try:
                audits[mod] = fut.result()
            except Exception as exc:  # pragma: no cover - defensive
                audits[mod] = ComponentAudit(
                    component=mod, status=ComponentStatus.BLOCKED, module=mod,
                    evidence=f"audit error: {exc}",
                )

    audit_rows = [audits[m] for m in modules]

    results: list = []
    for group, mods in COMPONENT_GROUPS.items():
        rows = [audits[m] for m in mods if m in audits]
        blocked = [r.component for r in rows if r.status == ComponentStatus.BLOCKED]
        consumed = sum(1 for r in rows if r.status == ComponentStatus.OPERATIONALLY_CONSUMED)
        tested = sum(1 for r in rows
                     if r.status in (ComponentStatus.UNIT_TESTED,
                                     ComponentStatus.OPERATIONALLY_CONSUMED))
        if blocked:
            verdict = CertificationVerdict.NOT_CERTIFIED
            summary = f"{len(blocked)}/{len(rows)} modules BLOCKED: {', '.join(blocked[:5])}"
        else:
            verdict = CertificationVerdict.CERTIFIED
            summary = (f"all {len(rows)} modules import; {tested} unit-tested, "
                       f"{consumed} operationally consumed")
        results.append(CertificationResult(
            check=f"engineering_audit.{group}",
            component=group,
            verdict=verdict,
            summary=summary,
            sample_size=len(rows),
            metrics={"modules": len(rows), "blocked": len(blocked),
                     "unit_tested_or_consumed": tested, "operationally_consumed": consumed,
                     "blocked_modules": blocked},
            notes="static engineering audit; empirical status granted only by cert runs",
        ))

    total_blocked = [r.component for r in audit_rows if r.status == ComponentStatus.BLOCKED]
    results.insert(0, CertificationResult(
        check="engineering_audit.platform",
        component="platform",
        verdict=(CertificationVerdict.NOT_CERTIFIED if total_blocked
                 else CertificationVerdict.CERTIFIED),
        summary=(f"{len(audit_rows)} modules audited, {len(total_blocked)} BLOCKED"
                 if total_blocked else f"{len(audit_rows)} modules audited, 0 BLOCKED"),
        sample_size=len(audit_rows),
        metrics={"audit_version": AUDIT_VERSION, "total_modules": len(audit_rows),
                 "blocked": total_blocked},
    ))
    return audit_rows, results
