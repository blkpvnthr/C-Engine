"""Standalone learning-platform CLI (Milestone 6, §34).

    python -m cengine.learning <command> [options]

All commands are NON-ORDERING and never submit live orders. Promotion enforces explicit
authorization + maturity checks. Training/evaluation/replay operate purely on the local
lakehouse. Imports are lazy inside each command so a missing optional dependency only
affects the command that needs it.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Optional

from cengine.lakehouse.config import LakehouseConfig


def _print(obj) -> None:
    print(json.dumps(obj, default=str, indent=2))


def cmd_hardware(args, config):
    from cengine.learning.hardware import inspect_hardware
    _print(inspect_hardware().to_dict())
    return 0


def cmd_profile_datasets(args, config):
    from cengine.learning.datasets.eligibility import assess_all
    _print(assess_all(config))
    return 0


def cmd_validate_datasets(args, config):
    from cengine.learning import FEATURE_SCHEMA_VERSION, LABEL_DEFINITION_VERSION
    from cengine.learning.datasets.certification import certify_training_dataset
    cert = certify_training_dataset(
        config, args.layer, args.dataset,
        feature_schema_version=FEATURE_SCHEMA_VERSION,
        label_definition_version=LABEL_DEFINITION_VERSION)
    _print(cert.to_dict())
    return 0 if cert.valid else 1


def cmd_train_supervised(args, config):
    from cengine.learning.supervised.base import TrainConfig
    from cengine.learning.supervised.trainer import train_forward_return_model
    rep = train_forward_return_model(
        config, horizon_bars=args.horizon,
        train_cfg=TrainConfig(epochs=args.epochs, batch_size=args.batch_size),
        classification=args.classification, register=not args.no_register)
    _print(rep.to_dict())
    return 0


def cmd_train_bandit(args, config):
    try:
        from cengine.learning.contextual_bandits import dataset as _ds  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        print(f"contextual bandit module unavailable: {exc}", file=sys.stderr)
        return 2
    print("contextual-bandit training requires gold offline_rl reward data; "
          "run profile-datasets to check eligibility (currently insufficient without "
          "genuine execution outcomes).")
    return 0


def cmd_train_offline_rl(args, config):
    try:
        from cengine.learning.offline_rl import transitions as _t  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        print(f"offline_rl module unavailable: {exc}", file=sys.stderr)
        return 2
    print("offline-RL training requires eligible gold offline_rl transitions with action "
          "support; run profile-datasets (currently insufficient without genuine "
          "execution outcomes).")
    return 0


def cmd_replay(args, config):
    from cengine.learning.environment.replay import ReplayConfig, ReplayEnvironment
    env = ReplayEnvironment(config, replay_config=ReplayConfig())

    def momentum_policy(ctx, pos_open):
        mom5, _mom20, vol = ctx
        if pos_open:
            return 0 if mom5 < 0 else 1
        return 1 if (mom5 > 0.001 and vol < 0.01) else 0
    res = env.run(args.symbol, momentum_policy)
    _print(res.to_dict())
    return 0


def cmd_registry(args, config):
    from cengine.learning.registry.model_registry import LearningModelRegistry
    reg = LearningModelRegistry(config)
    _print({"models": reg.list_latest()})
    return 0


def cmd_compare_models(args, config):
    from cengine.learning.registry.model_registry import LearningModelRegistry
    reg = LearningModelRegistry(config)
    _print({"champion": reg.champion(args.model_type),
            "challenger": reg.challenger(args.model_type)})
    return 0


def cmd_shadow(args, config):
    from cengine.learning.registry.maturity import ModelMaturity
    from cengine.learning.registry.model_registry import LearningModelRegistry
    reg = LearningModelRegistry(config)
    card = reg.transition(args.model_id, ModelMaturity.SHADOW, reason="cli_shadow")
    _print(card.to_dict())
    return 0


def cmd_promote(args, config):
    from cengine.learning.registry.maturity import ModelMaturity, PromotionError
    from cengine.learning.registry.model_registry import LearningModelRegistry
    reg = LearningModelRegistry(config)
    try:
        card = reg.transition(args.model_id, ModelMaturity(args.to),
                              approved=args.approve, reason=args.reason or "cli_promote")
    except PromotionError as exc:
        print(f"promotion refused: {exc}", file=sys.stderr)
        return 2
    _print(card.to_dict())
    return 0


def cmd_rollback(args, config):
    from cengine.learning.registry.model_registry import LearningModelRegistry
    reg = LearningModelRegistry(config)
    card = reg.rollback(args.model_id, to_reference=args.to_reference,
                        reason=args.reason or "cli_rollback")
    _print(card.to_dict())
    return 0


def cmd_daily_cycle(args, config):
    from cengine.learning.orchestration.daily_cycle import run_daily_cycle
    _print(run_daily_cycle(config))
    return 0


def cmd_benchmark(args, config):
    from cengine.learning.benchmarks.gpu_benchmarks import run_all_benchmarks
    _print(run_all_benchmarks())
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="cengine.learning",
                                description="Local adaptive ML/RL learning platform (M6)")
    p.add_argument("--root", default=None, help="lakehouse root (CENGINE_LAKEHOUSE_ROOT)")
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("hardware", help="MLX/Metal capability report").set_defaults(func=cmd_hardware)
    sub.add_parser("profile-datasets", help="dataset eligibility report").set_defaults(
        func=cmd_profile_datasets)

    v = sub.add_parser("validate-datasets", help="point-in-time dataset certification")
    v.add_argument("--layer", default="bronze")
    v.add_argument("--dataset", default="market_bars")
    v.set_defaults(func=cmd_validate_datasets)

    t = sub.add_parser("train-supervised", help="train the forward-return model")
    t.add_argument("--horizon", type=int, default=3)
    t.add_argument("--epochs", type=int, default=40)
    t.add_argument("--batch-size", type=int, default=4096)
    t.add_argument("--classification", action="store_true")
    t.add_argument("--no-register", action="store_true")
    t.set_defaults(func=cmd_train_supervised)

    sub.add_parser("train-bandit", help="contextual bandit (needs reward data)").set_defaults(
        func=cmd_train_bandit)
    sub.add_parser("train-offline-rl", help="offline RL (needs transitions)").set_defaults(
        func=cmd_train_offline_rl)

    r = sub.add_parser("replay", help="deterministic historical replay (simulated)")
    r.add_argument("--symbol", required=True)
    r.set_defaults(func=cmd_replay)

    sub.add_parser("registry", help="list registered models").set_defaults(func=cmd_registry)

    c = sub.add_parser("compare-models", help="champion vs challenger")
    c.add_argument("--model-type", default="mlx_quantile_regressor")
    c.set_defaults(func=cmd_compare_models)

    s = sub.add_parser("shadow", help="move a model to SHADOW")
    s.add_argument("--model-id", required=True)
    s.set_defaults(func=cmd_shadow)

    pr = sub.add_parser("promote", help="maturity transition (explicit approval for live)")
    pr.add_argument("--model-id", required=True)
    pr.add_argument("--to", required=True, help="target maturity state")
    pr.add_argument("--approve", action="store_true", help="explicit operator approval")
    pr.add_argument("--reason", default="")
    pr.set_defaults(func=cmd_promote)

    rb = sub.add_parser("rollback", help="roll a model back to baseline/last-approved")
    rb.add_argument("--model-id", required=True)
    rb.add_argument("--to-reference", default="baseline")
    rb.add_argument("--reason", default="")
    rb.set_defaults(func=cmd_rollback)

    sub.add_parser("daily-cycle", help="run the daily adaptive learning cycle").set_defaults(
        func=cmd_daily_cycle)
    sub.add_parser("benchmark", help="CPU vs MLX GPU benchmarks").set_defaults(func=cmd_benchmark)
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
