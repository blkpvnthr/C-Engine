"""Certification CLI — ``python -m cengine.certification <command>``.

Commands:
  audit            engineering audit (M7A) component status matrix
  data             historical data integrity (M7B)
  native           native + Metal certification (§23)
  failure          failure-injection certification (§24)
  protective-exit  mandatory −0.5% exit contract (§25) + recovery reproduction
  entry            entry contract (§26)
  strategy         strategy economic certification on the sealed holdout (§8)
  sensitivity      capital / cost sensitivity (§21)
  stress           stress regimes (§22)
  portfolio        portfolio certification
  execution        execution certification
  learning         ML / bandit / RL / autonomous-opt certification
  holdout          print the sealed-holdout manifest
  full             run the whole suite -> report + dashboard (use --quick to skip heavy passes)
  report           run full suite and write the §36 report + dashboard + JSON artifact
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from cengine.certification.orchestration import CANONICAL_SYMBOLS


def _print_results(results) -> int:
    worst = 0
    for r in results:
        v = r.verdict.value
        print(f"[{v.upper():16}] {r.check}: {r.summary}")
        if v == "not_certified":
            worst = max(worst, 1)
    return worst


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m cengine.certification",
                                description="C-Engine M7 certification")
    p.add_argument("command", help="certification command")
    p.add_argument("--symbols", default=",".join(CANONICAL_SYMBOLS),
                   help="comma-separated tickers")
    p.add_argument("--quick", action="store_true", help="skip heavy economic/stress passes")
    p.add_argument("--out", default="reports", help="output dir for report/report artifacts")
    args = p.parse_args(argv)
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    cmd = args.command

    if cmd == "audit":
        from cengine.certification.audit import run_engineering_audit
        _, results = run_engineering_audit()
        return _print_results(results)
    if cmd == "data":
        from cengine.certification.data_integrity import run_data_integrity_certification
        return _print_results(run_data_integrity_certification())
    if cmd == "native":
        from cengine.certification.native_cert import certify_native_metal
        return _print_results(certify_native_metal())
    if cmd == "failure":
        from cengine.certification.failure_injection import certify_failure_injection
        return _print_results(certify_failure_injection())
    if cmd == "protective-exit":
        from cengine.certification.protective_exit_cert import certify_protective_exit
        return _print_results(certify_protective_exit(symbols))
    if cmd == "entry":
        from cengine.certification.backtest import certify_entry_contract
        return _print_results(certify_entry_contract(symbols))
    if cmd == "strategy":
        from cengine.certification.backtest import certify_strategy_economics
        return _print_results(certify_strategy_economics(symbols))
    if cmd == "sensitivity":
        from cengine.certification.sensitivity import certify_capital_sensitivity
        return _print_results(certify_capital_sensitivity(symbols))
    if cmd == "stress":
        from cengine.certification.sensitivity import certify_stress
        return _print_results(certify_stress(symbols))
    if cmd == "portfolio":
        from cengine.certification.portfolio_cert import certify_portfolio
        return _print_results(certify_portfolio(symbols))
    if cmd == "execution":
        from cengine.certification.execution_cert import certify_execution
        return _print_results(certify_execution())
    if cmd == "learning":
        from cengine.certification.learning_cert import certify_learning
        return _print_results(certify_learning())
    if cmd == "holdout":
        import json

        from cengine.certification.holdout import build_holdout_manifest
        print(json.dumps(build_holdout_manifest(symbols).to_dict(), indent=2, default=str))
        return 0
    if cmd in ("full", "report"):
        from cengine.certification.orchestration import run_full_certification
        from cengine.certification.report import (
            generate_dashboard,
            generate_json,
            generate_markdown_report,
        )
        suite = run_full_certification(symbols, quick=args.quick)
        print(generate_dashboard(suite))
        if cmd == "report":
            out = Path(args.out)
            out.mkdir(parents=True, exist_ok=True)
            (out / "m7_certification_report.md").write_text(generate_markdown_report(suite))
            (out / "m7_certification_dashboard.md").write_text(generate_dashboard(suite))
            (out / "m7_certification.json").write_text(generate_json(suite))
            print(f"\nwrote report + dashboard + JSON to {out}/")
        return 0

    print(f"unknown command: {cmd}", file=sys.stderr)
    print(__doc__)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
