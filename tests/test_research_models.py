import numpy as np

from cengine.portfolio_optimization import (
    AllocationConstraints,
    PortfolioRiskLimits,
    PortfolioRiskManager,
    RegimeMarkowitzAllocator,
)
from cengine.prediction import NextSessionTideModel


def test_regime_markowitz_advice_is_bounded_and_risk_checked():
    covariance = np.array([[0.04, 0.01], [0.01, 0.03]])
    advice = RegimeMarkowitzAllocator(
        AllocationConstraints(gross_limit=1.0, per_asset_limit=0.7, risk_aversion=2.0)
    ).optimize(("SPY", "TLT"), np.array([0.08, 0.03]), covariance, 0.8)
    assert sum(abs(weight) for weight in advice.weights) <= 1.00001
    PortfolioRiskManager().validate(
        advice,
        covariance,
        PortfolioRiskLimits(1.0, 0.7, 1.0),
    )


def test_knn_next_session_tide_model_is_advisory_and_deterministic():
    features = np.array([[-2.0, -1.0], [-1.0, -2.0], [1.0, 2.0], [2.0, 1.0]])
    labels = np.array([-1, -1, 1, 1])
    model = NextSessionTideModel("knn", n_neighbors=1)
    model.fit(features, labels)
    prediction = model.predict(np.array([1.5, 1.5]))
    assert prediction.direction == 1
    assert prediction.model == "knn"
