"""Continuous alpha-discovery dashboard (M8 §24).

Renders the alpha-discovery status fields for the terminal dashboard, STRICTLY separating
development results from independent certification results and never fabricating a metric. Values
are read from a cycle record + the validation protocol; absent values render as ``—``.
"""

from __future__ import annotations

DASHBOARD_VERSION = "alpha-discovery-dashboard-1.0.0"

_FIELDS = (
    "ALPHA_DISCOVERY_STATUS", "CURRENT_RESEARCH_HYPOTHESIS", "ACTIVE_EXPERIMENT", "RESEARCH_CYCLE",
    "EXPERIMENTS_COMPLETED", "EXPERIMENTS_REJECTED", "EXPERIMENTS_DEFERRED", "ELIGIBLE_CHALLENGERS",
    "BEST_DEVELOPMENT_NET_EXPECTANCY", "INCUMBENT_NET_EXPECTANCY", "BEST_DEVELOPMENT_PROFIT_FACTOR",
    "BEST_DEVELOPMENT_DRAWDOWN", "REGIME_APPLICABILITY", "GPU_TRAINING_STATUS",
    "OPTIMIZATION_MEMORY_SIZE", "DATA_SUFFICIENCY", "INDEPENDENT_CERTIFICATION_STATUS",
)


def build_status(cycle: dict | None = None, *, validation: dict | None = None,
                 memory_size: int = 0, gpu_training: str = "idle",
                 regime_applicability: str = "—", research_cycle: int = 0,
                 active_experiment: str = "—") -> dict:
    cycle = cycle or {}
    reg = cycle.get("challenger_registry", {})
    challengers = reg.get("challengers", [])
    decisions = cycle.get("decisions", {})
    best_dev = _best_development(challengers)
    hyps = cycle.get("hypotheses", [])
    status = {
        "ALPHA_DISCOVERY_STATUS": "active" if cycle else "idle",
        "CURRENT_RESEARCH_HYPOTHESIS": hyps[0]["hypothesis"][:80] if hyps else "—",
        "ACTIVE_EXPERIMENT": active_experiment,
        "RESEARCH_CYCLE": research_cycle,
        "EXPERIMENTS_COMPLETED": len([c for c in challengers if c["decision"] != "defer"]),
        "EXPERIMENTS_REJECTED": decisions.get("reject", 0),
        "EXPERIMENTS_DEFERRED": decisions.get("defer", 0),
        "ELIGIBLE_CHALLENGERS": decisions.get("accept", 0),
        "BEST_DEVELOPMENT_NET_EXPECTANCY": best_dev.get("net_expectancy", "—"),
        "INCUMBENT_NET_EXPECTANCY": cycle.get("incumbent_net_expectancy", "—"),
        "BEST_DEVELOPMENT_PROFIT_FACTOR": best_dev.get("profit_factor", "—"),
        "BEST_DEVELOPMENT_DRAWDOWN": best_dev.get("max_drawdown", "—"),
        "REGIME_APPLICABILITY": regime_applicability,
        "GPU_TRAINING_STATUS": gpu_training,
        "OPTIMIZATION_MEMORY_SIZE": memory_size,
        "DATA_SUFFICIENCY": "sufficient" if cycle.get("data_sufficient") else "limited",
        "INDEPENDENT_CERTIFICATION_STATUS":
            (validation or {}).get("independent_certification_status", "deferred"),
    }
    return status


def _best_development(challengers: list) -> dict:
    scored = [c for c in challengers if c["decision"] != "defer"
              and c.get("metrics", {}).get("net_expectancy") is not None]
    if not scored:
        return {}
    best = max(scored, key=lambda c: c["metrics"]["net_expectancy"])
    m = best["metrics"]
    return {"net_expectancy": round(m.get("net_expectancy", 0.0), 6),
            "profit_factor": round(m.get("profit_factor", 0.0), 3),
            "max_drawdown": round(m.get("max_drawdown", 0.0), 4), "name": best["name"]}


def render(status: dict) -> str:
    """Compact terminal panel. Development vs independent-certification are visually separated."""
    lines = ["ALPHA DISCOVERY (development) ───────────────"]
    dev_fields = ("ALPHA_DISCOVERY_STATUS", "RESEARCH_CYCLE", "EXPERIMENTS_COMPLETED",
                  "EXPERIMENTS_REJECTED", "EXPERIMENTS_DEFERRED", "ELIGIBLE_CHALLENGERS",
                  "BEST_DEVELOPMENT_NET_EXPECTANCY", "INCUMBENT_NET_EXPECTANCY",
                  "BEST_DEVELOPMENT_PROFIT_FACTOR", "BEST_DEVELOPMENT_DRAWDOWN",
                  "GPU_TRAINING_STATUS", "OPTIMIZATION_MEMORY_SIZE", "DATA_SUFFICIENCY")
    for k in dev_fields:
        lines.append(f"  {k:32} {status.get(k, '—')}")
    lines.append("INDEPENDENT CERTIFICATION ──────────────────")
    cert = status.get("INDEPENDENT_CERTIFICATION_STATUS")
    lines.append(f"  INDEPENDENT_CERTIFICATION_STATUS {cert}")
    lines.append("  (development results are NOT certification results)")
    return "\n".join(lines)
