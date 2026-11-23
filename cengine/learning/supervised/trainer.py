"""End-to-end supervised training pipeline (Milestone 6, §7/§21/§24 + addendum §6/§14).

Ties the pieces together for the one genuinely-trainable task on real data (forward-return
prediction) and any future gold-outcome task:

  build look-ahead-free features -> certify point-in-time chronological splits -> fit the MLX
  GPU champion + independent CPU baselines on TRAIN -> evaluate OOS -> gate promotion on
  beating the baseline -> publish a checksummed artifact + register a ModelCard with real
  metrics and an honest maturity.

A model is promoted to SHADOW only if it genuinely beats the baseline out-of-sample; otherwise
it stays TRAINED/RESEARCH. Nothing here claims profitability and nothing is promoted to live.
"""

from __future__ import annotations

import platform
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from cengine.lakehouse.config import LakehouseConfig
from cengine.learning.datasets.certification import certify_training_dataset
from cengine.learning.features.builder import build_bar_features
from cengine.learning.features.schema import FEATURE_SCHEMA_VERSION
from cengine.learning.registry.maturity import ModelMaturity
from cengine.learning.registry.model_registry import LearningModelRegistry, ModelCard
from cengine.learning.supervised.base import (
    TrainConfig,
    classification_metrics,
    regression_metrics,
)
from cengine.learning.supervised.baselines import (
    ConstantBaseline,
    SklearnClassifier,
    SklearnRegressor,
)

LABEL_DEFINITION_VERSION = "label-def-1.0.0"


@dataclass
class TrainingReport:
    task: str
    trained: bool = False
    promoted_to: str = ModelMaturity.UNTRAINED.value
    model_id: str = ""
    model_version: str = ""
    device: str = "cpu"
    champion_metrics: dict = field(default_factory=dict)
    baseline_metrics: dict = field(default_factory=dict)
    beats_baseline: bool = False
    dataset_version: str = ""
    n_train: int = 0
    n_oos: int = 0
    reason: str = ""

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


def _chrono_masks(ts: np.ndarray, cert) -> tuple:
    """Boolean train / oos masks from certified chronological boundaries."""
    import datetime

    def _ns(d):
        if not d:
            return None
        dt = datetime.datetime.strptime(d, "%Y-%m-%d").replace(tzinfo=datetime.timezone.utc)
        return int(dt.timestamp() * 1e9)
    tr_end = _ns(cert.training_end)
    oos_start = _ns(cert.out_of_sample_start)
    train = ts <= tr_end if tr_end else np.zeros_like(ts, dtype=bool)
    oos = ts >= oos_start if oos_start else np.zeros_like(ts, dtype=bool)
    return train, oos


def train_forward_return_model(
    config: Optional[LakehouseConfig] = None, *,
    horizon_bars: int = 3, train_cfg: Optional[TrainConfig] = None,
    classification: bool = False, now_ns: Optional[int] = None,
    model_id: Optional[str] = None, register: bool = True,
) -> TrainingReport:
    """Train + register the forward-return model from genuine market bars."""
    config = config or LakehouseConfig.from_env()
    now_ns = now_ns if now_ns is not None else time.time_ns()
    task = "probability_positive_return" if classification else "forward_return_prediction"
    report = TrainingReport(task=task)

    matrix = build_bar_features(config, horizon_bars=horizon_bars)
    if matrix is None or matrix.n < 500:
        report.reason = "insufficient_bar_data"
        return report
    cert = certify_training_dataset(
        config, "bronze", "market_bars",
        feature_schema_version=FEATURE_SCHEMA_VERSION,
        label_definition_version=LABEL_DEFINITION_VERSION, now_ns=now_ns)
    if not cert.valid:
        report.reason = f"dataset_not_certified:{cert.reasons}"
        return report
    report.dataset_version = cert.dataset_version

    order = np.argsort(matrix.timestamps_ns)
    X = matrix.X[order]
    ts = matrix.timestamps_ns[order]
    y = (matrix.y_positive[order].astype(np.float32) if classification
         else matrix.y_return[order])
    train_mask, oos_mask = _chrono_masks(ts, cert)
    # carve a validation slice from the tail of train for early stopping
    tr_idx = np.where(train_mask)[0]
    val_cut = int(len(tr_idx) * 0.85)
    Xtr, ytr = X[tr_idx[:val_cut]], y[tr_idx[:val_cut]]
    Xval, yval = X[tr_idx[val_cut:]], y[tr_idx[val_cut:]]
    Xoos, yoos = X[oos_mask], y[oos_mask]
    report.n_train, report.n_oos = len(Xtr), len(Xoos)
    if report.n_train < 200 or report.n_oos < 100:
        report.reason = "insufficient_split_sizes"
        return report

    cfg = train_cfg or TrainConfig()
    champion, framework, arch, metrics_fn = _build_champion(
        matrix.schema.n_features, cfg, classification)
    fit = champion.fit(Xtr, ytr, X_val=Xval, y_val=yval)
    report.device = fit.device
    report.trained = True

    champ_pred = champion.predict(Xoos)
    base = SklearnClassifier() if classification else SklearnRegressor(kind="hgbr")
    base.fit(Xtr, ytr)
    base_pred = base.predict(Xoos)

    report.champion_metrics = metrics_fn(yoos, champ_pred)
    report.baseline_metrics = metrics_fn(yoos, base_pred)
    # also record the constant floor for context
    cfloor = ConstantBaseline()
    cfloor.fit(Xtr, ytr if not classification else ytr)
    report.beats_baseline = _beats(report.champion_metrics, report.baseline_metrics,
                                   classification)

    if register:
        _register(config, champion, framework, arch, report, cert, cfg, matrix, now_ns,
                  classification, model_id)
    else:
        report.promoted_to = (ModelMaturity.VALIDATED.value if report.beats_baseline
                              else ModelMaturity.TRAINED.value)
    return report


def _build_champion(n_features: int, cfg: TrainConfig, classification: bool):
    try:
        from cengine.learning.supervised.mlx_models import (
            MLXBinaryClassifier,
            MLXQuantileRegressor,
            mlx_is_available,
        )
        if mlx_is_available():
            if classification:
                return (MLXBinaryClassifier(n_features, config=cfg), "mlx",
                        f"mlx_mlp{[n_features, *cfg.hidden, 1]}", classification_metrics)
            return (MLXQuantileRegressor(n_features, config=cfg), "mlx",
                    f"mlx_quantile{[n_features, *cfg.hidden, len(cfg.quantiles)]}",
                    regression_metrics)
    except Exception:  # noqa: BLE001 — fall back to a CPU baseline as champion
        pass
    if classification:
        return SklearnClassifier(), "sklearn", "sklearn_hgbc", classification_metrics
    return SklearnRegressor(kind="hgbr"), "sklearn", "sklearn_hgbr", regression_metrics


def _beats(champ: dict, base: dict, classification: bool) -> bool:
    if classification:
        # lower brier is better; require a real improvement
        return champ.get("brier", 1.0) < base.get("brier", 1.0) - 1e-4
    # higher rank-IC AND non-worse directional accuracy
    return (champ.get("rank_ic", -1) > base.get("rank_ic", -1) + 1e-3
            and champ.get("directional_accuracy", 0) >= base.get("directional_accuracy", 0) - 0.01)


def _register(config, champion, framework, arch, report, cert, cfg, matrix, now_ns,
              classification, model_id) -> None:
    reg = LearningModelRegistry(config)
    mid = model_id or ("prob_positive" if classification else "forward_return")
    version = f"v{now_ns}"
    report.model_id, report.model_version = mid, version
    model_type = champion.model_type
    ref = reg.artifacts.save_npz(mid, version, champion.params(), framework=framework,
                                 framework_version=_fw_version(framework))
    card = ModelCard(
        model_id=mid, model_version=version, model_type=model_type,
        model_architecture=arch, framework=framework,
        framework_version=_fw_version(framework), training_device=report.device,
        inference_device=report.device, numerical_precision="float32",
        dataset_version=cert.dataset_version, dataset_checksum=cert.dataset_checksum,
        feature_schema_version=FEATURE_SCHEMA_VERSION,
        label_definition_version=LABEL_DEFINITION_VERSION,
        training_period=f"{cert.training_start}..{cert.training_end}",
        validation_period=f"{cert.validation_start}..{cert.validation_end}",
        out_of_sample_period=f"{cert.out_of_sample_start}..{cert.out_of_sample_end}",
        hyperparameters={"epochs": cfg.epochs, "batch_size": cfg.batch_size,
                         "learning_rate": cfg.learning_rate, "hidden": list(cfg.hidden)},
        random_seed=cfg.seed,
        software_versions={"python": platform.python_version(),
                           "numpy": np.__version__, "framework_version": _fw_version(framework)},
        validation_metrics=report.champion_metrics,
        economic_metrics={"baseline_comparison": report.baseline_metrics,
                          "beats_baseline": report.beats_baseline},
        applicable_instrument_classes=("equity",),
        execution_environment="research",
        artifacts={"weights": ref.to_dict()})
    reg.register(card)
    reg.transition(mid, ModelMaturity.TRAINED, reason="trained")
    if report.beats_baseline:
        reg.transition(mid, ModelMaturity.VALIDATED, reason="beats_baseline_oos")
        reg.transition(mid, ModelMaturity.SHADOW, reason="auto_shadow_deploy")
        report.promoted_to = ModelMaturity.SHADOW.value
    else:
        report.promoted_to = ModelMaturity.TRAINED.value
        report.reason = "did_not_beat_baseline_oos (kept TRAINED, not shadow-deployed)"


def _fw_version(framework: str) -> str:
    if framework == "mlx":
        try:
            import mlx.core as mx
            return getattr(mx, "__version__", "mlx")
        except Exception:  # noqa: BLE001
            return "mlx"
    if framework == "sklearn":
        import sklearn
        return sklearn.__version__
    return "numpy"
