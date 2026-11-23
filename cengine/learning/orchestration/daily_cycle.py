"""Daily adaptive learning cycle (Milestone 6, §25 / addendum §14).

The post-session workflow, failure-isolated and idempotent:

  lakehouse materialization (M5 pipeline: silver->gold->quality->attribution) ->
  dataset eligibility assessment -> challenger training on ELIGIBLE tasks only ->
  independent OOS evaluation + baseline-gated promotion (handled inside the trainer) ->
  model-registry publication + shadow deployment -> optional drift check ->
  learning decision evidence.

Retraining is gated on data sufficiency and mature outcomes; a task with
INSUFFICIENT_DATA is skipped (never trained on fabricated data). A challenger that fails
to beat the baseline keeps the incumbent champion / deterministic baseline. Nothing here
submits orders, promotes to live, or touches the protective exit; it runs OFF the trading-
critical path (invoked post-session or from a detached loop).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

from cengine.lakehouse.config import LakehouseConfig
from cengine.learning.datasets.eligibility import EligibilityState, assess_all
from cengine.learning.supervised.base import TrainConfig
from cengine.learning.supervised.trainer import train_forward_return_model

DAILY_CYCLE_VERSION = "daily-learning-cycle-1.0.0"

# tasks this cycle can actually train today, mapped to their trainer entrypoints.
_TRAINABLE = {
    "forward_return_prediction": dict(classification=False),
    "probability_positive_return": dict(classification=True),
}


@dataclass
class DailyCycleReport:
    version: str = DAILY_CYCLE_VERSION
    started_ns: int = 0
    completed_ns: int = 0
    materialization: dict = field(default_factory=dict)
    eligibility: dict = field(default_factory=dict)
    trained: list = field(default_factory=list)
    skipped: list = field(default_factory=list)
    errors: list = field(default_factory=list)

    def to_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items()}
        d["duration_s"] = round((self.completed_ns - self.started_ns) / 1e9, 3) \
            if self.completed_ns else None
        return d


class DailyLearningCycle:
    def __init__(self, config: Optional[LakehouseConfig] = None, *,
                 decision_sink=None, train_cfg: Optional[TrainConfig] = None) -> None:
        self.config = config or LakehouseConfig.from_env()
        self.sink = decision_sink
        self.train_cfg = train_cfg or TrainConfig(epochs=40, batch_size=4096)

    def _stage(self, report: DailyCycleReport, name: str, fn):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 — stage isolation; the cycle never aborts
            report.errors.append({"stage": name, "error": f"{type(exc).__name__}:{exc}"})
            return None

    def run(self, *, now_ns: Optional[int] = None, materialize: bool = True,
            run_drift: bool = True) -> dict:
        now_ns = now_ns if now_ns is not None else time.time_ns()
        report = DailyCycleReport(started_ns=now_ns)

        # 1. lakehouse materialization (M5 pipeline) — gold/quality/attribution refreshed
        if materialize:
            report.materialization = self._stage(report, "materialize", self._materialize) or {}

        # 2. dataset eligibility assessment (honest; no fabricated examples)
        elig = self._stage(report, "eligibility", lambda: assess_all(self.config)) or {}
        report.eligibility = {t["task"]: t["eligibility"] for t in elig.get("tasks", [])}
        eligible_states = {EligibilityState.TRAINING_ELIGIBLE.value,
                           EligibilityState.VALIDATION_ELIGIBLE.value}

        # 3. train each trainable + eligible task (challenger); gate promotion on baseline
        for task, kw in _TRAINABLE.items():
            state = report.eligibility.get(task)
            if state not in eligible_states:
                report.skipped.append({"task": task, "eligibility": state,
                                       "reason": "not_training_eligible"})
                continue
            res = self._stage(report, f"train:{task}",
                              lambda kw=kw: train_forward_return_model(
                                  self.config, train_cfg=self.train_cfg, now_ns=now_ns, **kw))
            if res is None:
                continue
            report.trained.append(res.to_dict())
            self._emit_training(res, now_ns)

        # 4. optional drift check against the champion's reference (best-effort/optional)
        if run_drift:
            self._stage(report, "drift", lambda: self._drift(now_ns))

        report.completed_ns = time.time_ns()
        self._record_run(report)
        return report.to_dict()

    def _materialize(self) -> dict:
        from cengine.lakehouse.pipeline import LakehousePipeline
        out = LakehousePipeline(self.config).run(
            decision_sink=self.sink, compact=False)
        return {k: ("error" if isinstance(v, dict) and "error" in v else "ok")
                for k, v in out.items() if isinstance(v, dict)}

    def _emit_training(self, res, now_ns: int) -> None:
        if self.sink is None:
            return
        try:
            self.sink.record_training_result(
                now_ns=now_ns, model_id=res.model_id, model_version=res.model_version,
                payload={"task": res.task, "device": res.device,
                         "beats_baseline": res.beats_baseline,
                         "promoted_to": res.promoted_to,
                         "metrics": res.champion_metrics})
            if res.beats_baseline:
                self.sink.record_model_promotion(
                    now_ns=now_ns, model_id=res.model_id, model_version=res.model_version,
                    from_maturity="validated", to_maturity=res.promoted_to, approved=False,
                    reason="auto_shadow_beats_baseline")
        except Exception:  # noqa: BLE001
            pass

    def _drift(self, now_ns: int) -> Optional[dict]:
        """Optional drift check — enabled once monitoring is available."""
        try:
            from cengine.learning.monitoring import drift as _drift_mod  # noqa: F401
        except Exception:  # noqa: BLE001
            return None
        # monitoring integration is wired by the champion/challenger harness; here we only
        # confirm availability without fabricating a reference distribution.
        return {"monitoring_available": True}

    def _record_run(self, report: DailyCycleReport) -> None:
        import json
        path = self.config.checkpoints / "learning_cycle_runs.jsonl"
        try:
            self.config.checkpoints.mkdir(parents=True, exist_ok=True)
            with open(path, "a") as fh:
                fh.write(json.dumps({"version": report.version,
                                     "started_ns": report.started_ns,
                                     "completed_ns": report.completed_ns,
                                     "trained": len(report.trained),
                                     "skipped": len(report.skipped),
                                     "errors": len(report.errors)}, default=str) + "\n")
        except OSError:
            pass


def run_daily_cycle(config: Optional[LakehouseConfig] = None, *, decision_sink=None,
                    now_ns: Optional[int] = None) -> dict:
    return DailyLearningCycle(config, decision_sink=decision_sink).run(now_ns=now_ns)
