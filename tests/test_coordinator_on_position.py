"""Regression: StrategyCoordinator.on_position must no-op for strategies it does
not manage, not raise.

Exposed by the first genuine fill of a regime-intelligence (bridge) paper order:
reconcile_execution -> strategies.on_position(REGIME_INTELLIGENCE, ...) raised
KeyError because the legacy coordinator only manages turtle/dual_ma/apo/di_bb,
crashing the reconciliation loop (and the engine) on every non-legacy fill.
Non-legacy order positions are tracked by the native state + protective exit, not
here, so the coordinator must ignore them.
"""

from __future__ import annotations

from strategies import StrategyCoordinator, StrategyName


class _FakeStrategy:
    def __init__(self, name: StrategyName) -> None:
        self.name = name
        self.positions: list[tuple[str, int]] = []

    def on_position(self, symbol: str, quantity: int) -> None:
        self.positions.append((symbol, quantity))


def test_on_position_noops_for_unmanaged_strategy():
    coord = StrategyCoordinator((_FakeStrategy(StrategyName.TURTLE),))
    # Must NOT raise for strategies the coordinator does not manage.
    coord.on_position(StrategyName.REGIME_INTELLIGENCE, "SPY", 1)
    coord.on_position(StrategyName.PORTFOLIO, "QQQ", 2)


def test_on_position_still_delegates_to_managed_strategy():
    fake = _FakeStrategy(StrategyName.TURTLE)
    coord = StrategyCoordinator((fake,))

    coord.on_position(StrategyName.TURTLE, "SPY", 5)
    assert fake.positions == [("SPY", 5)]

    # An unmanaged strategy is ignored, not misrouted to the managed one.
    coord.on_position(StrategyName.REGIME_INTELLIGENCE, "QQQ", 9)
    assert fake.positions == [("SPY", 5)]
