"""Continuous runtime entrypoint — ``python -m cengine.runtime <command>``.

Commands:
  status   print the current operating state, tick plan, worker health and acquisition gap
  once     run a single bounded tick (plan + one governance invariant check + due off-session work)
  run      run the persistent 24/7 supervisory loop (Ctrl-C to stop)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import signal
import sys


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m cengine.runtime",
                                description="C-Engine continuous runtime")
    p.add_argument("command", choices=["status", "once", "run"], help="runtime command")
    p.add_argument("--symbols", default="", help="comma-separated tickers (optional)")
    p.add_argument("--tick-interval", type=float, default=30.0)
    args = p.parse_args(argv)

    from cengine.runtime.service import ContinuousRuntimeConfig, ContinuousRuntimeService
    cfg = ContinuousRuntimeConfig(tick_interval_s=args.tick_interval)
    if args.symbols:
        cfg.symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    svc = ContinuousRuntimeService(cfg)

    if args.command == "status":
        print(json.dumps(svc.status(), indent=2, default=str))
        return 0
    if args.command == "once":
        import time
        now = time.time_ns()
        gov = svc.governance_tick(now)
        plan = svc.plan_tick(now)
        out = svc.dispatch(plan, now)
        print(json.dumps({"governance": gov, "plan": plan.to_dict(), "dispatched": out},
                         indent=2, default=str))
        return 0
    # run
    stop = asyncio.Event()

    def _sig(*_a):
        stop.set()

    try:
        signal.signal(signal.SIGINT, _sig)
        signal.signal(signal.SIGTERM, _sig)
    except ValueError:
        pass

    async def _run():
        return await svc.run(stop)

    result = asyncio.run(_run())
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
