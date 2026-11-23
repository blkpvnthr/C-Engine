"""Standalone autonomous-optimization CLI (Milestone 6B).

    python -m cengine.optimization <command> [options]

All commands are NON-ORDERING, research/shadow-only, and never activate a live policy, expand
risk, or relax the production -0.5% protective exit. Lazy imports keep each command independent.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Optional

from cengine.lakehouse.config import LakehouseConfig


def _print(obj) -> None:
    print(json.dumps(obj, default=str, indent=2))


def cmd_attribute(args, config):
    from cengine.optimization.attribution import PerformanceAttributionEngine
    _print(PerformanceAttributionEngine(config).attribute())
    return 0


def cmd_data_sufficiency(args, config):
    from cengine.optimization.data_sufficiency import DataSufficiencyController
    _print(DataSufficiencyController(config).assess())
    return 0


def cmd_counterfactual(args, config):
    from cengine.optimization.exits.counterfactual import analyze
    _print(analyze(config, args.symbol, max_events=args.max_events).to_dict())
    return 0


def cmd_exit_matrix(args, config):
    from cengine.optimization.exits.experiment_matrix import run_exit_matrix
    res = run_exit_matrix(config, args.symbol, risk_normalized=args.risk_normalized)
    _print(res.to_dict())
    return 0


def cmd_optimize(args, config):
    from cengine.optimization.orchestration import run_autonomous_cycle
    _print(run_autonomous_cycle(config, symbol=args.symbol))
    return 0


def cmd_memory(args, config):
    from cengine.optimization.memory import OptimizationMemory
    _print(OptimizationMemory(config).effectiveness())
    return 0


def cmd_validation(args, config):
    from cengine.optimization.validation import oos_windows
    w = oos_windows(config, args.symbol)
    _print(w.to_dict() if w else {"error": "no data for symbol"})
    return 0


def cmd_emergency_check(args, config):
    import _cengine_native as n
    d = n.evaluate_emergency_exit(args.unrealized_return)
    _print({"triggered": d.triggered, "threshold": d.threshold,
            "overrides_tighter_stop": d.overrides_tighter_stop,
            "note": "native candidate emergency exit (INACTIVE; M7 certifies)"})
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="cengine.optimization",
                                description="Autonomous performance-driven ML/RL optimization")
    p.add_argument("--root", default=None, help="lakehouse root (CENGINE_LAKEHOUSE_ROOT)")
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("attribute", help="performance attribution diagnostics").set_defaults(
        func=cmd_attribute)
    sub.add_parser("data-sufficiency", help="eligibility + collection tasks").set_defaults(
        func=cmd_data_sufficiency)
    sub.add_parser("memory", help="optimization-memory effectiveness").set_defaults(
        func=cmd_memory)

    cf = sub.add_parser("counterfactual", help="post -0.5%-exit recovery on genuine 1m bars")
    cf.add_argument("--symbol", required=True)
    cf.add_argument("--max-events", type=int, default=2000)
    cf.set_defaults(func=cmd_counterfactual)

    em = sub.add_parser("exit-matrix", help="EXIT-00..15 on genuine 1m bars")
    em.add_argument("--symbol", required=True)
    em.add_argument("--risk-normalized", action="store_true")
    em.set_defaults(func=cmd_exit_matrix)

    op = sub.add_parser("optimize", help="run one autonomous optimization cycle (OOS)")
    op.add_argument("--symbol", default="AAPL")
    op.set_defaults(func=cmd_optimize)

    va = sub.add_parser("validation-windows", help="show train/validation/holdout split")
    va.add_argument("--symbol", required=True)
    va.set_defaults(func=cmd_validation)

    eg = sub.add_parser("emergency-check", help="native candidate emergency-exit check (inactive)")
    eg.add_argument("--unrealized-return", type=float, required=True)
    eg.set_defaults(func=cmd_emergency_check)
    return p


def main(argv: Optional[list] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.root:
        os.environ["CENGINE_LAKEHOUSE_ROOT"] = args.root
    config = LakehouseConfig.from_env()
    config.ensure_dirs()
    return args.func(args, config)


if __name__ == "__main__":
    raise SystemExit(main())
