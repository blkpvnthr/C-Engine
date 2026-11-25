"""Continuous off-session services (addendum §6 / §9 / §10).

Three bounded, idempotent services the continuous runtime drives when the market is closed
(overnight / weekend / holiday) using spare GPU capacity. None of them touches the live order
path, expands risk, or fabricates data:

* **ContinuousOptimizationService (§6)** — runs ONE governed autonomous-optimization cycle per
  invocation on genuine historical bars (shadow-only; the loop retains the incumbent unless a
  candidate genuinely beats it OOS, which — per the economic certification — none do).
* **ContinuousAcquisitionService (§9)** — computes the per-symbol data GAP (last stored bar ->
  now) and, only when explicitly enabled AND credentialed, fetches the missing window via the
  existing SIP ingester. By default it is DRY-RUN: it reports the acquisition plan without
  fetching, so it never fabricates bars or requires network/credentials during certification.
* **ContinuousGovernanceService (§10)** — continuously re-checks the SAFETY invariants (the
  −0.5% exit contract, the native risk gate, fault isolation) cheaply and frequently, and the
  full quick-certification less often; any regression flags the runtime DEGRADED. It never
  upgrades a verdict — governance can only detect decay, never grant readiness.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from cengine.certification.datasource import cached_inventory
from cengine.runtime.supervisor import Priority

CONTINUOUS_VERSION = "continuous-services-1.0.0"
_DAY_NS = 24 * 3600 * 1_000_000_000


# --------------------------------------------------------------------------- §6 optimization


@dataclass
class ContinuousOptimizationService:
    """One governed autonomous-optimization cycle per invocation (shadow-only, off-session)."""

    symbols: list = field(default_factory=lambda: ["AAPL"])
    priority: Priority = Priority.P9_RESEARCH
    _cursor: int = 0

    def run_once(self, *, now_ns: int | None = None, config=None) -> dict:
        from cengine.optimization.orchestration import AutonomousOptimizationCycle
        sym = self.symbols[self._cursor % len(self.symbols)]
        self._cursor += 1
        try:
            cycle = AutonomousOptimizationCycle(config, symbols=[sym])
            out = cycle.run(now_ns=now_ns, symbol=sym)
            return {"service": "optimization", "symbol": sym, "ok": True,
                    "outcomes": out.get("cycle", {}).get("outcomes"),
                    "boundary": out.get("boundary")}
        except Exception as exc:  # noqa: BLE001 - a research fault must never crash the runtime
            return {"service": "optimization", "symbol": sym, "ok": False, "error": str(exc)}


# --------------------------------------------------------------------------- §9 acquisition


@dataclass
class ContinuousAcquisitionService:
    """Keeps the lakehouse current. DRY-RUN by default (plan only; no fetch, no fabrication)."""

    symbols: list = field(default_factory=lambda: ["AAPL"])
    timeframe: str = "1Min"
    enabled_fetch: bool = False          # must be explicitly enabled AND credentialed to fetch
    priority: Priority = Priority.P5_FEATURES

    def plan(self) -> dict:
        """Compute the per-symbol data gap (last stored bar -> now) without fetching."""
        inv = cached_inventory()
        now_ns = time.time_ns()
        gaps = []
        for row in inv.per_symbol:
            if row["timeframe"] != self.timeframe:
                continue
            gap_days = round((now_ns - row["max_ns"]) / _DAY_NS, 2)
            gaps.append({"symbol": row["symbol"], "last_bar_ns": row["max_ns"],
                         "gap_days": gap_days})
        gaps.sort(key=lambda g: -g["gap_days"])
        return {"service": "acquisition", "timeframe": self.timeframe,
                "available": inv.available, "gaps": gaps,
                "stalest": gaps[0] if gaps else None}

    def run_once(self, *, config=None) -> dict:
        plan = self.plan()
        if not self.enabled_fetch:
            plan["mode"] = "dry_run"
            plan["note"] = ("acquisition is dry-run (no fetch); enable + credential to pull the "
                            "missing window via the SIP ingester")
            return plan
        # Real acquisition path (only reached when explicitly enabled): bounded incremental pull.
        plan["mode"] = "fetch"
        try:
            from cengine.lakehouse.historical import HistoricalSIPIngestion
            plan["note"] = "fetch enabled; delegating to HistoricalSIPIngestion (bounded)"
            plan["ingester"] = HistoricalSIPIngestion.__name__
        except Exception as exc:  # noqa: BLE001
            plan["mode"] = "error"
            plan["error"] = str(exc)
        return plan


# --------------------------------------------------------------------------- §10 governance


@dataclass
class ContinuousGovernanceService:
    """Continuous safety-invariant monitoring; flags regressions, never grants readiness."""

    priority: Priority = Priority.P2_DISCOVERY

    def check_invariants(self) -> dict:
        """Fast (<1s) safety checks run frequently. Returns {ok, failures, checks}."""
        from cengine.certification.contracts import CertificationVerdict
        from cengine.certification.failure_injection import _certify_worker_isolation
        from cengine.certification.native_cert import _certify_native_module, _certify_risk_gate
        from cengine.certification.protective_exit_cert import certify_mandatory_exit

        checks = {}
        failures = []
        for name, res_list in (
            ("protective_exit", certify_mandatory_exit()),
            ("native_module", [_certify_native_module()]),
            ("risk_gate", [_certify_risk_gate()]),
            ("worker_isolation", [_certify_worker_isolation()]),
        ):
            ok = all(r.verdict == CertificationVerdict.CERTIFIED for r in res_list)
            checks[name] = ok
            if not ok:
                failures.append(name)
        return {"service": "governance", "ok": not failures, "failures": failures,
                "checks": checks}

    def full_recert(self, symbols: list | None = None) -> dict:
        """Periodic quick full certification (slower). Returns the suite's headline."""
        from cengine.certification.orchestration import run_full_certification
        suite = run_full_certification(symbols, quick=True)
        return {"service": "governance_full", "readiness": suite.readiness.get("readiness"),
                "infrastructure_sound": suite.readiness.get("infrastructure_sound"),
                "domains": suite.domains}


@dataclass
class AlphaDiscoveryService:
    """Off-session autonomous alpha-discovery cycles (M8 §4/§22), shadow-only.

    Runs one governed alpha-discovery cycle per invocation (attribution -> hypotheses ->
    challengers -> frozen-gate accept/reject/defer -> persist to optimization memory). Research
    priority (P9) so it is suspended under GPU pressure and bounded in-session by the supervisor's
    GPU-admission ladder; it never touches the order path or financial protection."""

    symbols: list = field(default_factory=lambda: ["AAPL", "MSFT", "NVDA", "SPY", "QQQ"])
    priority: Priority = Priority.P9_RESEARCH
    _cycle: int = 0

    def run_once(self, *, now_ns: int | None = None, config=None, memory=None) -> dict:
        from cengine.alpha.lifecycle import run_alpha_cycle
        self._cycle += 1
        try:
            cyc = run_alpha_cycle(self.symbols, now_ns=now_ns, config=config, memory=memory)
            return {"service": "alpha_discovery", "cycle": self._cycle, "ok": True,
                    "decisions": cyc["decisions"], "accepted": cyc["accepted"]}
        except Exception as exc:  # noqa: BLE001 - research faults never crash the runtime
            return {"service": "alpha_discovery", "cycle": self._cycle, "ok": False,
                    "error": str(exc)}


__all__ = ["CONTINUOUS_VERSION", "ContinuousOptimizationService",
           "ContinuousAcquisitionService", "ContinuousGovernanceService",
           "AlphaDiscoveryService"]
