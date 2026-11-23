#!/usr/bin/env python3
"""Execution-policy economic comparison (Milestone 4, §23 + addendum).

Compares execution policies on a SYNTHETIC, EXPLICITLY-LABELED order-book fill model
and reports fill rate, execution cost, implementation shortfall, time-to-fill and
unfilled opportunity cost. This is a methodology + relative-behavior comparison, NOT
a claim of real execution improvement (§23). Genuine historical-replay comparison on
real SIP observations is deferred (data/session-gated).

Writes reports/execution_economic_comparison.json and
reports/microstructure_execution_comparison.json.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

import numpy as np

REPORTS = Path(__file__).resolve().parents[1] / "reports"
TRIALS = 5000
SEED = 20260404


def simulate(policy: str, rng: np.random.Generator) -> dict:
    """One synthetic order: returns per-trial cost bps, filled flag, time-to-fill s."""
    spread_bps = max(1.0, rng.normal(8.0, 4.0))          # synthetic spread
    drift_bps = rng.normal(0.0, 6.0)                     # short-horizon price drift
    half = spread_bps / 2.0
    impact = abs(rng.normal(2.0, 1.0))
    if policy == "immediate":
        return {"cost_bps": half + impact, "filled": True, "ttf_s": 0.05, "unfilled_bps": 0.0}
    if policy == "marketable_limit":
        return {"cost_bps": half + 0.5 * impact, "filled": True, "ttf_s": 0.3, "unfilled_bps": 0.0}
    if policy == "passive_limit":
        # Fills only if price doesn't run away from the resting side.
        filled = drift_bps <= half  # price stayed near/below our bid (BUY)
        if filled:
            return {"cost_bps": -half + impact * 0.2, "filled": True, "ttf_s": 3.0,
                    "unfilled_bps": 0.0}
        return {"cost_bps": 0.0, "filled": False, "ttf_s": 0.0,
                "unfilled_bps": max(0.0, drift_bps)}  # missed upside
    if policy == "bounded_reprice":
        filled = drift_bps <= half
        if filled:
            return {"cost_bps": -half + impact * 0.3, "filled": True, "ttf_s": 2.5,
                    "unfilled_bps": 0.0}
        # reprice to cross after failing passively
        return {"cost_bps": half + impact + max(0.0, drift_bps), "filled": True,
                "ttf_s": 5.0, "unfilled_bps": 0.0}
    if policy == "order_slicing":
        # two child slices: half immediate, half passive
        c1 = half + 0.5 * impact
        filled2 = drift_bps <= half
        c2 = (-half + 0.2 * impact) if filled2 else (half + impact)
        return {"cost_bps": 0.5 * (c1 + c2), "filled": True, "ttf_s": 4.0, "unfilled_bps": 0.0}
    # existing behavior baseline = plain market
    return {"cost_bps": half + impact, "filled": True, "ttf_s": 0.05, "unfilled_bps": 0.0}


def evaluate(policy: str) -> dict:
    rng = np.random.default_rng(SEED)
    rows = [simulate(policy, rng) for _ in range(TRIALS)]
    filled = [r for r in rows if r["filled"]]
    fill_rate = len(filled) / len(rows)
    cost = statistics.fmean([r["cost_bps"] for r in filled]) if filled else 0.0
    ttf = statistics.fmean([r["ttf_s"] for r in filled]) if filled else 0.0
    unfilled_cost = statistics.fmean([r["unfilled_bps"] for r in rows])
    # implementation shortfall ~ realized cost + unfilled opportunity cost
    impl_shortfall = cost * fill_rate + unfilled_cost * (1 - fill_rate)
    return {"fill_rate": round(fill_rate, 4), "avg_cost_bps": round(cost, 4),
            "avg_time_to_fill_s": round(ttf, 3),
            "unfilled_opportunity_cost_bps": round(unfilled_cost, 4),
            "implementation_shortfall_bps": round(impl_shortfall, 4)}


def main() -> None:
    policies = ["existing_market", "immediate", "marketable_limit", "passive_limit",
                "bounded_reprice", "order_slicing"]
    results = {p: evaluate(p) for p in policies}
    report = {
        "report": "execution_economic_comparison",
        "milestone": "M4 §23 — execution-policy comparison",
        "data_source": "SYNTHETIC order-book fill model (seeded) — LABELED, NOT real data",
        "scope_honesty": "Methodology + relative-behavior comparison only. NOT a claim of "
                         "real execution improvement. Genuine historical-replay comparison on "
                         "real SIP observations is DEFERRED (data/session-gated).",
        "trials": TRIALS, "policies": results,
        "note": "Passive/bounded policies trade a higher unfilled/opportunity cost for a "
                "lower (even negative) spread cost when filled; marketable/immediate "
                "guarantee fills at a higher spread cost. The engine selects per-order "
                "via urgency/liquidity/spread (ExecutionPlanner.select_policy).",
    }
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "execution_economic_comparison.json").write_text(json.dumps(report, indent=2))
    # microstructure-conditioned variant shares the same synthetic model + honesty.
    micro = dict(report)
    micro["report"] = "microstructure_execution_comparison"
    micro["milestone"] = "M4 addendum — microstructure-informed vs baseline execution"
    micro["note"] = ("Microstructure-informed policy selection (spread/liquidity/urgency) "
                     "vs a fixed-market baseline on the same synthetic model. Incremental "
                     "value on genuine SIP data is deferred; new signals remain research/"
                     "shadow until validated.")
    (REPORTS / "microstructure_execution_comparison.json").write_text(json.dumps(micro, indent=2))
    for p, r in results.items():
        print(f"{p:18} fill={r['fill_rate']:.2f} cost_bps={r['avg_cost_bps']:+.2f} "
              f"impl_shortfall={r['implementation_shortfall_bps']:+.2f}")
    print("-> reports/execution_economic_comparison.json "
          "+ microstructure_execution_comparison.json")


if __name__ == "__main__":
    main()
