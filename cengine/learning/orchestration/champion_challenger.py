"""Champion/challenger shadow evaluation (Milestone 6, §24).

Runs the current champion, a newly-trained challenger, and (implicitly) the deterministic
baseline over IDENTICAL observations, recording prediction/ranking differences and —
when realized outcomes are available — estimated financial impact. It is SHADOW-ONLY: it
never generates duplicate real orders, and a challenger that fails validation stays inactive.
Evidence is emitted through the nonblocking decision sink.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from cengine.lakehouse.config import LakehouseConfig
from cengine.learning.evaluation import DATA_LABEL_REAL, DATA_LABEL_SYNTHETIC, economic_metrics
from cengine.learning.inference.service import LearningInferenceService
from cengine.learning.registry.model_registry import LearningModelRegistry

CHAMPION_CHALLENGER_VERSION = "champion-challenger-1.0.0"


@dataclass
class ChampionChallengerReport:
    version: str = CHAMPION_CHALLENGER_VERSION
    task: str = ""
    model_type: str = ""
    champion: Optional[str] = None
    challenger: Optional[str] = None
    n: int = 0
    mean_abs_prediction_diff: Optional[float] = None
    rank_correlation: Optional[float] = None
    sign_disagreement_rate: Optional[float] = None
    champion_economic: dict = field(default_factory=dict)
    challenger_economic: dict = field(default_factory=dict)
    data_label: str = DATA_LABEL_SYNTHETIC
    recommendation: str = "retain_champion"
    notes: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


class ChampionChallengerHarness:
    def __init__(self, config: Optional[LakehouseConfig] = None, *,
                 registry: Optional[LearningModelRegistry] = None,
                 inference: Optional[LearningInferenceService] = None,
                 decision_sink=None) -> None:
        self.config = config or LakehouseConfig.from_env()
        self.registry = registry or LearningModelRegistry(self.config)
        self.inference = inference or LearningInferenceService(self.config, registry=self.registry)
        self.sink = decision_sink

    def evaluate(self, task: str, model_type: str, feature_matrix: np.ndarray, symbols, *,
                 feature_schema_version: str, realized_returns: Optional[np.ndarray] = None,
                 now_ns: int = 0, data_is_real: bool = False) -> ChampionChallengerReport:
        rep = ChampionChallengerReport(task=task, model_type=model_type)
        champ = self.registry.champion(model_type)
        chall = self.registry.challenger(model_type)
        rep.champion = (champ or {}).get("model_version")
        rep.challenger = (chall or {}).get("model_version")
        rep.data_label = DATA_LABEL_REAL if data_is_real else DATA_LABEL_SYNTHETIC
        if champ is None and chall is None:
            rep.notes.append("no_models_registered")
            return rep

        cp = self._preds(task, champ, feature_matrix, symbols, feature_schema_version, now_ns)
        hp = self._preds(task, chall, feature_matrix, symbols, feature_schema_version, now_ns)
        rep.n = int(len(feature_matrix))
        if cp is not None and hp is not None:
            diff = np.abs(cp - hp)
            rep.mean_abs_prediction_diff = float(np.mean(diff))
            rep.sign_disagreement_rate = float(np.mean((cp > 0) != (hp > 0)))
            rep.rank_correlation = _rank_corr(cp, hp)

        # estimated financial impact when realized outcomes are available (select-if-positive)
        if realized_returns is not None and len(realized_returns) == len(feature_matrix):
            if cp is not None:
                rep.champion_economic = self._econ(cp, realized_returns, rep.data_label)
            if hp is not None:
                rep.challenger_economic = self._econ(hp, realized_returns, rep.data_label)
            rep.recommendation = self._recommend(rep)

        self._emit(rep, now_ns)
        return rep

    def _preds(self, task, card, X, symbols, fsv, now_ns):
        if card is None:
            return None
        res = self.inference.predict_batch(task, X, symbols, fsv, now_ns=now_ns,
                                           model_id=card.get("model_id"))
        vals = [r.prediction for r in res]
        if any(v is None for v in vals):
            return None
        return np.asarray(vals, dtype=np.float64)

    @staticmethod
    def _econ(pred: np.ndarray, realized: np.ndarray, data_label: str) -> dict:
        # a prediction-driven long-if-positive paper strategy: realized return when pred>0.
        taken = realized[pred > 0]
        if len(taken) == 0:
            return {"n": 0, "note": "no_positive_predictions"}
        m = economic_metrics(taken, data_label=data_label)
        return {"n": m.n, "net_return": m.net_return, "net_expectancy": m.net_expectancy,
                "sharpe": m.sharpe, "win_rate": m.win_rate, "profit_factor": m.profit_factor,
                "data_label": m.data_label}

    @staticmethod
    def _recommend(rep: ChampionChallengerReport) -> str:
        ce = rep.challenger_economic.get("net_expectancy")
        pe = rep.champion_economic.get("net_expectancy")
        if ce is None:
            return "retain_champion"
        if pe is None:
            return "challenger_candidate (no champion economics)"
        # never auto-promote; only a recommendation for independent review
        return "challenger_outperforms_shadow" if ce > pe else "retain_champion"

    def _emit(self, rep: ChampionChallengerReport, now_ns: int) -> None:
        if self.sink is None:
            return
        try:
            self.sink.record_model_disagreement(
                symbol_id=None, now_ns=now_ns,
                payload={"task": rep.task, "champion": rep.champion,
                         "challenger": rep.challenger,
                         "mean_abs_prediction_diff": rep.mean_abs_prediction_diff,
                         "sign_disagreement_rate": rep.sign_disagreement_rate,
                         "recommendation": rep.recommendation,
                         "data_label": rep.data_label})
        except Exception:  # noqa: BLE001
            pass


def _rank_corr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3:
        return 0.0
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    ra -= ra.mean()
    rb -= rb.mean()
    d = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / d) if d > 0 else 0.0
