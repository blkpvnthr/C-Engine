"""Capital / cost sensitivity (§21) and stress-regime (§22) certification.

Both run the production incumbent (−0.5% exit + momentum entry) on genuine 1-minute bars under
varied conditions and report how its net economics move. They are deliberately unforgiving:

* **Capital / cost sensitivity (§21)** — sweep the per-side cost from 0 to 16 bps on each
  symbol's sealed holdout and report net expectancy / profit factor / Sharpe at each level, plus
  the **break-even cost** (the cost at which expectancy crosses zero). A strategy is cost-robust
  only if it stays positive at the realistic cost; if it is negative even at ZERO cost it has no
  economic edge to protect and no capacity, independent of capital.

* **Stress regimes (§22)** — run the incumbent on fixed, genuinely-stressed calendar windows
  (the 2020 COVID crash, the 2022 bear market) and a calm control window, and report expectancy
  and drawdown per regime. Real market stress, not synthetic shocks.

True market-impact capacity (how size degrades fills) is NOT claimed here — we do not model
depth-based impact from the L1 data — so capacity beyond cost sensitivity is reported as
not-assessable (INSUFFICIENT_DATA) rather than fabricated.
"""

from __future__ import annotations

import datetime as _dt
from zoneinfo import ZoneInfo

from cengine.certification.contracts import (
    DATA_LABEL_HISTORICAL_REPLAY,
    CertificationResult,
    CertificationVerdict,
    safe_ticker,
)
from cengine.certification.holdout import derive_symbol_holdout
from cengine.optimization.economic import evaluate_candidate
from cengine.optimization.exits.experiment_matrix import (
    _entries_and_atr,
    _load_1m,
    _simulate,
)

SENSITIVITY_VERSION = "certification-sensitivity-1.0.0"
_ET = ZoneInfo("America/New_York")
_INCUMBENT_STOP = 0.005
_COST_SWEEP = [0.0, 0.0002, 0.0004, 0.0008, 0.0016]      # 0 / 2 / 4 / 8 / 16 bps per side
_REALISTIC_COST = 0.0004

# fixed, genuinely-stressed calendar windows (ET dates) + a calm control
_STRESS_WINDOWS = {
    "covid_crash_2020": ("2020-02-19", "2020-04-07"),
    "bear_market_2022": ("2022-01-03", "2022-10-14"),
    "calm_2021": ("2021-04-01", "2021-08-01"),
}


def _ns(date_str: str) -> int:
    d = _dt.date.fromisoformat(date_str)
    return int(_dt.datetime(d.year, d.month, d.day, tzinfo=_ET).timestamp() * 1e9)


def _config(config):
    if config is None:
        from cengine.lakehouse.config import LakehouseConfig
        return LakehouseConfig.from_env()
    return config


def _run(bars, entries, atr, cost_frac):
    rets = _simulate(bars, entries, atr, "fixed", {"stop": _INCUMBENT_STOP},
                     asset_class="large_cap", cost_frac=cost_frac)
    return evaluate_candidate(rets, label="incumbent", data_label=DATA_LABEL_HISTORICAL_REPLAY)


# --------------------------------------------------------------------------- §21 cost sensitivity


def certify_capital_sensitivity(symbols: list, *, config=None) -> list:
    config = _config(config)
    results: list = []
    cost_robust_syms = 0
    scored = 0
    for sym in symbols:
        s = safe_ticker(sym)
        h = derive_symbol_holdout(s, config=config)
        if h is None:
            results.append(CertificationResult(
                check=f"capital_sensitivity.{s}", component="strategy",
                verdict=CertificationVerdict.INSUFFICIENT_DATA,
                summary=f"{s}: no holdout window", data_label=DATA_LABEL_HISTORICAL_REPLAY))
            continue
        lo, hi = h.holdout
        bars = _load_1m(config, s, start_ns=lo, end_ns=hi)
        if bars is None:
            results.append(CertificationResult(
                check=f"capital_sensitivity.{s}", component="strategy",
                verdict=CertificationVerdict.INSUFFICIENT_DATA,
                summary=f"{s}: insufficient holdout bars", data_label=DATA_LABEL_HISTORICAL_REPLAY))
            continue
        entries, atr = _entries_and_atr(bars)
        curve = []
        for cf in _COST_SWEEP:
            ev = _run(bars, entries, atr, cf)
            curve.append({"cost_bps": round(cf * 1e4, 1),
                          "net_expectancy": round(ev.metrics.get("net_expectancy", 0.0), 6),
                          "profit_factor": round(ev.metrics.get("profit_factor", 0.0), 3),
                          "sharpe": round(ev.metrics.get("sharpe", 0.0), 3)})
        # break-even cost: largest cost with expectancy still > 0 (None if negative even at 0)
        positive = [p for p in curve if p["net_expectancy"] > 0]
        break_even = max((p["cost_bps"] for p in positive), default=None)
        at_realistic = next(p for p in curve if abs(p["cost_bps"] - _REALISTIC_COST * 1e4) < 1e-6)
        cost_robust = at_realistic["net_expectancy"] > 0
        scored += 1
        if cost_robust:
            cost_robust_syms += 1
        results.append(CertificationResult(
            check=f"capital_sensitivity.{s}", component="strategy",
            verdict=CertificationVerdict.CERTIFIED if cost_robust
            else CertificationVerdict.NOT_CERTIFIED,
            summary=(f"{s}: expectancy@0bps={curve[0]['net_expectancy']:+.5f}, "
                     f"@{_REALISTIC_COST*1e4:.0f}bps={at_realistic['net_expectancy']:+.5f}; "
                     f"break-even cost={break_even} bps "
                     f"({'cost-robust' if cost_robust else 'no positive edge at realistic cost'})"),
            sample_size=len(entries), data_label=DATA_LABEL_HISTORICAL_REPLAY,
            metrics={"cost_curve": curve, "break_even_cost_bps": break_even,
                     "negative_at_zero_cost": curve[0]["net_expectancy"] <= 0},
            notes="§21 cost sensitivity on the sealed holdout; true market-impact capacity "
                  "not modeled (reported separately as not-assessable)"))

    # capacity note — honest: we do not model depth-based market impact
    results.append(CertificationResult(
        check="capital_sensitivity.market_impact_capacity", component="strategy",
        verdict=CertificationVerdict.INSUFFICIENT_DATA,
        summary="true size/impact capacity not assessable from L1 bars; only cost sensitivity "
                "is evidenced",
        notes="capacity requires depth/impact modeling or live fills — not fabricated"))

    roll = (CertificationVerdict.CERTIFIED if scored and cost_robust_syms == scored
            else CertificationVerdict.NOT_CERTIFIED if scored
            else CertificationVerdict.INSUFFICIENT_DATA)
    results.append(CertificationResult(
        check="capital_sensitivity.portfolio", component="strategy", verdict=roll,
        summary=(f"{cost_robust_syms}/{scored} symbols retain positive expectancy at the "
                 f"realistic {_REALISTIC_COST*1e4:.0f}bps cost"),
        sample_size=scored, data_label=DATA_LABEL_HISTORICAL_REPLAY,
        metrics={"cost_robust": cost_robust_syms, "scored": scored}))
    return results


# --------------------------------------------------------------------------- §22 stress regimes


def certify_stress(symbols: list, *, config=None) -> list:
    config = _config(config)
    results: list = []
    for sym in symbols:
        s = safe_ticker(sym)
        per_regime = {}
        for regime, (d0, d1) in _STRESS_WINDOWS.items():
            bars = _load_1m(config, s, start_ns=_ns(d0), end_ns=_ns(d1))
            if bars is None:
                per_regime[regime] = {"status": "INSUFFICIENT_DATA"}
                continue
            entries, atr = _entries_and_atr(bars)
            if len(entries) < 30:
                per_regime[regime] = {"status": "INSUFFICIENT_DATA", "n_entries": len(entries)}
                continue
            ev = _run(bars, entries, atr, _REALISTIC_COST)
            per_regime[regime] = {
                "status": "COMPLETED", "n": ev.n,
                "net_expectancy": round(ev.metrics.get("net_expectancy", 0.0), 6),
                "profit_factor": round(ev.metrics.get("profit_factor", 0.0), 3),
                "sharpe": round(ev.metrics.get("sharpe", 0.0), 3),
                "max_drawdown": round(abs(ev.metrics.get("max_drawdown", 0.0)), 4)}
        completed = [v for v in per_regime.values() if v.get("status") == "COMPLETED"]
        if not completed:
            verdict = CertificationVerdict.INSUFFICIENT_DATA
            summ = f"{s}: no stress regime had sufficient data"
        else:
            # stress-robust = positive expectancy AND drawdown within the frozen ceiling in EVERY
            # completed stressed regime (the crash/bear are the real test)
            robust = all(v["net_expectancy"] > 0 and v["max_drawdown"] <= 0.25
                         for r, v in per_regime.items()
                         if v.get("status") == "COMPLETED" and r != "calm_2021")
            verdict = (CertificationVerdict.CERTIFIED if robust
                       else CertificationVerdict.NOT_CERTIFIED)
            summ = (f"{s}: " + ", ".join(
                f"{r}={v['net_expectancy']:+.5f}(DD{v['max_drawdown']:.0%})"
                for r, v in per_regime.items() if v.get("status") == "COMPLETED"))
        results.append(CertificationResult(
            check=f"stress.{s}", component="strategy", verdict=verdict, summary=summ,
            data_label=DATA_LABEL_HISTORICAL_REPLAY, metrics={"per_regime": per_regime},
            notes="§22 stress on genuine crash/bear/calm windows at realistic cost"))
    return results
