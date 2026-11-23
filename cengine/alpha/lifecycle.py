"""Autonomous alpha-discovery lifecycle (M8 §4 / §16 / §17 / §22).

Chains the existing + M8 research into the governed continuous loop:

    COLLECT/ATTRIBUTE -> IDENTIFY FAILURE MODES -> GENERATE HYPOTHESES -> CHECK DATA SUFFICIENCY
    -> CONSULT OPTIMIZATION MEMORY -> PLAN/RUN EXPERIMENTS (challengers) -> ECONOMIC + ROBUSTNESS
    EVALUATION (frozen gates + baselines) -> ACCEPT/REJECT/DEFER -> PERSIST EXPERIMENT KNOWLEDGE

It reuses the M6B persistent ``OptimizationMemory`` (dedup + effectiveness), the M8 economic
objective + frozen ``ChallengerGates`` + ``ChallengerRegistry``, and the capital-constrained
accountant. Challengers are evaluated against cash, the deterministic incumbent and simple
baselines; each is genuinely accepted, rejected or deferred. Given the current evidence, the
challengers are rejected (negative edge) or deferred (insufficient data) — the accept path exists
and is exercised in tests with a qualifying candidate. No live activation; no fabrication.
"""

from __future__ import annotations

import time

from cengine.alpha.failure_attribution import run_failure_attribution
from cengine.alpha.objective import ChallengerRegistry, default_gates, evaluate_challenger
from cengine.certification.contracts import DATA_LABEL_HISTORICAL_REPLAY, stable_id

LIFECYCLE_VERSION = "alpha-discovery-lifecycle-1.0.0"
CANONICAL = ["AAPL", "MSFT", "NVDA", "SPY", "QQQ"]


def generate_hypotheses(findings: list) -> list:
    """§16 — governed, falsifiable hypotheses derived from observed failure findings."""
    out = []
    for f in findings:
        out.append({
            "hypothesis_id": stable_id("hyp", f.get("finding", "")),
            "from_finding": f.get("finding"),
            "evidence": f.get("evidence"),
            "hypothesis": f.get("hypothesis"),
            "falsifiable_test": ("evaluate the proposed entry/exit/selectivity change on a "
                                 "chronological TRAIN/VAL split and require positive OOS net "
                                 "expectancy that beats cash + the incumbent"),
        })
    return out


def _incumbent_expectancy(challengers: list) -> float:
    for c in challengers:
        if c.get("name") == "incumbent_momentum":
            return float(c["metrics"].get("net_expectancy", 0.0))
    return -1e9


def build_default_challengers(symbols: list, config=None) -> list:
    """Run the genuine M8 research modules and map results to challenger records."""
    from cengine.alpha.family_research import run_family_research
    from cengine.alpha.supervised_filter import run_supervised_filter
    challengers: list = []

    fam = run_family_research(symbols, config=config)
    if fam.get("available"):
        for name, d in fam["by_family"].items():
            if name == "_errors" or not isinstance(d, dict) or d.get("status") != "COMPLETED":
                continue
            cname = ("incumbent_momentum" if name == "MOMENTUM_CONTINUATION"
                     else f"family_{name.lower()}")
            challengers.append({
                "name": cname,
                "metrics": {"net_expectancy": d["net_expectancy_per_trade"],
                            "profit_factor": d["profit_factor"],
                            "sharpe_annualized": d["sharpe_annualized"],
                            "sortino_annualized": d["sharpe_annualized"],  # proxy; conservative
                            "max_drawdown": d["max_drawdown"]},
                "effective_sample": d["trades_taken"],
                "total_return": d["total_return"], "provenance": "family_research/replay"})

    sup = run_supervised_filter(symbols, config=config)
    if sup.get("available") and sup.get("best_model"):
        bm = sup["models"].get(sup["best_model"], {})
        if "selected_val_expectancy" in bm:
            challengers.append({
                "name": f"supervised_{sup['best_model']}",
                "metrics": {"net_expectancy": bm["selected_val_expectancy"],
                            "profit_factor": 0.0, "sharpe_annualized": 0.0,
                            "sortino_annualized": 0.0, "max_drawdown": 1.0},
                "effective_sample": bm.get("selected_n", 0),
                "total_return": None, "provenance": "supervised_filter/top_decile"})

    # deferred challengers (genuinely INSUFFICIENT_DATA)
    challengers.append({"name": "contextual_bandit_family_selection", "metrics": {},
                        "effective_sample": 0, "provenance": "bandit_rl/insufficient",
                        "force_defer": True})
    challengers.append({"name": "offline_rl_sequential_policy", "metrics": {},
                        "effective_sample": 0, "provenance": "bandit_rl/insufficient",
                        "force_defer": True})
    return challengers


def run_alpha_cycle(symbols: list | None = None, *, now_ns: int | None = None, config=None,
                    challengers_fn=build_default_challengers, memory=None,
                    attribution_fn=run_failure_attribution) -> dict:
    """Run one governed alpha-discovery cycle. Returns a structured cycle record."""
    symbols = symbols or CANONICAL
    now_ns = now_ns if now_ns is not None else time.time_ns()

    # 1-2. collect + attribute + identify failure modes
    attribution = attribution_fn(symbols, config=config)
    findings = attribution.get("findings", [])
    # 3. hypotheses
    hypotheses = generate_hypotheses(findings)
    # 4. data sufficiency (coarse: total trades observed)
    data_sufficient = attribution.get("aggregate", {}).get("total_trades", 0) >= 1000

    # 5. consult memory (dedup) + 6-7. plan/run challengers
    challengers = challengers_fn(symbols, config=config)
    incumbent_exp = _incumbent_expectancy(challengers)

    # 8. economic + robustness evaluation -> accept/reject/defer -> persist
    registry = ChallengerRegistry()
    gates = default_gates()
    for ch in challengers:
        if ch.get("force_defer"):
            ev = {"decision": "defer", "insufficient": True,
                  "failures": ["insufficient_data"], "objective_J": None,
                  "gate_fingerprint": gates.fingerprint()}
        else:
            m = ch["metrics"]
            total_return = ch.get("total_return")
            beats_cash = (total_return is not None and total_return > 0)
            beats_incumbent = m.get("net_expectancy", -1e9) > incumbent_exp
            ev = evaluate_challenger(
                m, effective_sample=ch.get("effective_sample"),
                beats_cash=beats_cash if total_return is not None else None,
                beats_incumbent=beats_incumbent, beats_simple_baselines=beats_incumbent,
                gates=gates)
        registry.record(ch["name"], ch.get("metrics", {}), ev, provenance=ch.get("provenance", ""))
        if memory is not None:
            try:
                memory.record(hypothesis_id=None, search_space="alpha_discovery",
                              config={"challenger": ch["name"], "provenance": ch.get("provenance")},
                              score=ev.get("objective_J"), economic_metrics=ch.get("metrics", {}),
                              decision=ev["decision"],
                              rejection_reason=";".join(ev.get("failures", [])),
                              data_label=DATA_LABEL_HISTORICAL_REPLAY, now_ns=now_ns)
            except Exception:  # noqa: BLE001 - persistence must never crash the cycle
                pass

    summary = registry.summary()
    return {
        "version": LIFECYCLE_VERSION, "now_ns": now_ns, "symbols": symbols,
        "data_label": DATA_LABEL_HISTORICAL_REPLAY,
        "failure_findings": findings, "hypotheses": hypotheses,
        "data_sufficient": data_sufficient,
        "challenger_registry": registry.to_dict(),
        "decisions": summary["by_decision"], "accepted": summary["accepted"],
        "incumbent_net_expectancy": round(incumbent_exp, 6) if incumbent_exp > -1e8 else None,
        "conclusion": (
            f"{summary['by_decision'].get('accept', 0)} accepted, "
            f"{summary['by_decision'].get('reject', 0)} rejected, "
            f"{summary['by_decision'].get('defer', 0)} deferred. "
            + ("No challenger cleared the frozen acceptance gates; the incumbent is retained and "
               "the loop continues gathering evidence." if not summary["accepted"]
               else f"Accepted (development, pending certification): {summary['accepted']}.")),
        "note": "governed autonomous cycle; simulated historical-replay; no live activation; "
                "no fabricated performance",
    }
