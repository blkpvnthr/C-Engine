"""Certification backtest harness + strategy economic certification (§8 / §25 / §26).

Reproduces the production contract on genuine 1-minute bars with conservative fills — reusing
the audited M6B replay machinery (entry contract ``_entries_and_atr`` §26, exit simulator
``_simulate`` with stop-before-target/gap-through-at-open, −0.5% incumbent §25) so certification
scores the SAME code the platform runs, not a re-implementation. It then:

1. runs the incumbent on an OOS window,
2. builds the §8 baseline hierarchy on the identical window,
3. scores everything against the FROZEN economic gates (§20),

and renders an honest :class:`CertificationResult`. The strategy driver scores the **sealed
holdout** (accessed once per symbol through the :class:`HoldoutSeal`, for scoring only) so the
verdict is a genuine out-of-sample claim.

Honest by construction: results are ``historical_replay`` (simulated on real bars), never a
live-profitability claim; and if the incumbent's net economics do not clear the gates, the
verdict is NOT_CERTIFIED — the harness never massages a pass.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from cengine.certification.baselines import evaluate_baselines
from cengine.certification.contracts import (
    DATA_LABEL_HISTORICAL_REPLAY,
    CertificationResult,
    CertificationVerdict,
    safe_ticker,
)
from cengine.certification.gates import FrozenEconomicGates, default_gates, evaluate_gates
from cengine.certification.holdout import HoldoutSeal
from cengine.optimization.economic import evaluate_candidate
from cengine.optimization.exits.experiment_matrix import (
    _entries_and_atr,
    _load_1m,
    _simulate,
)

BACKTEST_VERSION = "certification-backtest-1.0.0"
_INCUMBENT_STOP = 0.005          # the mandatory −0.5% protective exit (§25)
_MIN_ENTRIES = 30
_DEFAULT_COST_FRAC = 0.0004      # realistic per-side cost (4 bps)


@dataclass
class WindowBacktest:
    symbol: str
    window: str
    n_entries: int
    incumbent: Optional[dict] = None          # CandidateEvaluation.to_dict()
    baselines: dict = field(default_factory=dict)
    gate_result: dict = field(default_factory=dict)
    beats: dict = field(default_factory=dict)
    note: str = ""

    def to_dict(self) -> dict:
        return {"symbol": self.symbol, "window": self.window, "n_entries": self.n_entries,
                "incumbent": self.incumbent, "baselines": self.baselines,
                "gate_result": self.gate_result, "beats": self.beats, "note": self.note}


def run_window_backtest(symbol: str, window_name: str, start_ns: int, end_ns: int, *,
                        config=None, cost_frac: float = _DEFAULT_COST_FRAC,
                        gates: Optional[FrozenEconomicGates] = None) -> WindowBacktest:
    """Run the incumbent + baselines + frozen gates on one symbol/window."""
    sym = safe_ticker(symbol)
    gates = gates or default_gates()
    if config is None:
        from cengine.lakehouse.config import LakehouseConfig
        config = LakehouseConfig.from_env()
    bars = _load_1m(config, sym, start_ns=start_ns, end_ns=end_ns)
    if bars is None:
        return WindowBacktest(symbol=sym, window=window_name, n_entries=0,
                              note="insufficient 1Min bars for window")
    entries, atr_frac = _entries_and_atr(bars)
    if len(entries) < _MIN_ENTRIES:
        return WindowBacktest(symbol=sym, window=window_name, n_entries=len(entries),
                              note=f"insufficient entries ({len(entries)} < {_MIN_ENTRIES})")

    returns = _simulate(bars, entries, atr_frac, "fixed", {"stop": _INCUMBENT_STOP},
                        asset_class="large_cap", cost_frac=cost_frac)
    incumbent = evaluate_candidate(returns, label="incumbent_exit_00",
                                   data_label=DATA_LABEL_HISTORICAL_REPLAY)
    baselines = evaluate_baselines(bars, entries, atr_frac, cost_frac=cost_frac, symbol=sym)

    inc_score = incumbent.score
    beats = {}
    for name, ev in baselines.items():
        if inc_score is not None and ev.score is not None:
            beats[name] = bool(inc_score > ev.score)
    gate_result = evaluate_gates(
        incumbent.metrics, n_trades=incumbent.n,
        effective_sample=incumbent.metrics.get("effective_sample"),
        beats_buy_and_hold=beats.get("buy_and_hold"),
        beats_random_placebo=beats.get("random_placebo"),
        gates=gates,
    )
    return WindowBacktest(
        symbol=sym, window=window_name, n_entries=len(entries),
        incumbent=incumbent.to_dict(),
        baselines={k: v.to_dict() for k, v in baselines.items()},
        gate_result=gate_result, beats=beats,
        note=("historical-replay on genuine 1m bars; conservative fills; "
              "stop-before-target; costs both sides; NOT live performance"),
    )


def _verdict_from(bt: WindowBacktest) -> CertificationVerdict:
    if bt.n_entries < _MIN_ENTRIES or not bt.gate_result:
        return CertificationVerdict.INSUFFICIENT_DATA
    if bt.gate_result.get("insufficient"):
        return CertificationVerdict.INSUFFICIENT_DATA
    return (CertificationVerdict.CERTIFIED if bt.gate_result.get("passed")
            else CertificationVerdict.NOT_CERTIFIED)


def certify_strategy_economics(symbols: list, *, config=None, seal: Optional[HoldoutSeal] = None,
                               cost_frac: float = _DEFAULT_COST_FRAC,
                               gates: Optional[FrozenEconomicGates] = None) -> list:
    """Economic certification of the production incumbent on each symbol's SEALED HOLDOUT.

    The holdout window is obtained through the seal (access-logged, scoring-only). Returns a
    list[CertificationResult] — one per symbol plus a portfolio-level roll-up.
    """
    seal = seal or HoldoutSeal()
    gates = gates or default_gates()
    results: list = []
    passed = 0
    scored = 0
    for sym in symbols:
        s = safe_ticker(sym)
        try:
            lo, hi = seal.open(s, reason="economic_certification_scoring")
        except ValueError as exc:
            results.append(CertificationResult(
                check=f"strategy_economics.{s}", component="strategy",
                verdict=CertificationVerdict.INSUFFICIENT_DATA,
                summary=f"{s}: {exc}", data_label=DATA_LABEL_HISTORICAL_REPLAY))
            continue
        bt = run_window_backtest(s, "holdout", lo, hi, config=config, cost_frac=cost_frac,
                                 gates=gates)
        verdict = _verdict_from(bt)
        if verdict in (CertificationVerdict.CERTIFIED, CertificationVerdict.NOT_CERTIFIED):
            scored += 1
        if verdict == CertificationVerdict.CERTIFIED:
            passed += 1
        m = (bt.incumbent or {}).get("metrics", {})
        summary = (f"{s} holdout: {bt.note.split(';')[0]}"
                   if bt.n_entries < _MIN_ENTRIES else
                   f"{s} holdout n={bt.n_entries} expectancy={m.get('net_expectancy', 0):.5f} "
                   f"PF={m.get('profit_factor', 0):.2f} Sharpe={m.get('sharpe', 0):.2f} "
                   f"DD={abs(m.get('max_drawdown', 0)):.1%} "
                   f"-> {verdict.value}"
                   + (f"; fails: {bt.gate_result.get('failures')}" if bt.gate_result.get("failures")
                      else ""))
        results.append(CertificationResult(
            check=f"strategy_economics.{s}", component="strategy", verdict=verdict,
            summary=summary, sample_size=bt.n_entries,
            data_label=DATA_LABEL_HISTORICAL_REPLAY,
            metrics=bt.to_dict(),
            gate_refs=[bt.gate_result.get("gate_fingerprint", "")],
            notes="OOS sealed-holdout economic certification of the −0.5% incumbent entry+exit",
        ))

    # portfolio-level roll-up across symbols
    if scored == 0:
        roll = CertificationVerdict.INSUFFICIENT_DATA
        rsum = "no symbol produced a scoreable holdout backtest"
    elif passed == scored:
        roll = CertificationVerdict.CERTIFIED
        rsum = f"all {scored} symbols clear the frozen economic gates on the sealed holdout"
    elif passed == 0:
        roll = CertificationVerdict.NOT_CERTIFIED
        rsum = f"0/{scored} symbols clear the frozen economic gates on the sealed holdout"
    else:
        roll = CertificationVerdict.NOT_CERTIFIED
        rsum = f"only {passed}/{scored} symbols clear the gates — not a portfolio-wide edge"
    results.append(CertificationResult(
        check="strategy_economics.portfolio", component="strategy", verdict=roll,
        summary=rsum, sample_size=scored, data_label=DATA_LABEL_HISTORICAL_REPLAY,
        metrics={"symbols_scored": scored, "symbols_passed": passed,
                 "holdout_access_audit": seal.audit()},
        gate_refs=[gates.fingerprint()],
    ))
    return results
