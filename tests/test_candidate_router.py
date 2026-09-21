import pytest

from cengine.execution_policy import ConfiguredExecutionPolicy
from cengine.strategy_state import StrategyStateBook
from order_manager import OrderType, TimeInForce
from run import CandidateRouter
from strategies import SignalSide, StrategyName, TradeCandidate


class Manager:
    def __init__(self):
        self.intents = []

    async def submit(self, intent):
        self.intents.append(intent)


class EntryQuantity:
    def quantity_for(self, candidate):
        return 5


class ExitQuantity:
    def quantity_to_close(self, candidate):
        return 3


def candidate(side):
    return TradeCandidate(StrategyName.TURTLE, "AAPL", side, 1, 100, "test")


@pytest.mark.asyncio
async def test_exit_wins_over_opposed_entry_signal():
    manager = Manager()
    router = CandidateRouter(
        order_manager=manager,
        entry_quantity_policy=EntryQuantity(),
        exit_quantity_provider=ExitQuantity(),
        execution_policy=ConfiguredExecutionPolicy(OrderType.MARKET, TimeInForce.DAY),
        strategy_state=StrategyStateBook(),
    )
    submitted, conflicted = await router.route(
        (candidate(SignalSide.EXIT_LONG), candidate(SignalSide.LONG))
    )
    assert submitted == 1
    assert not conflicted
    assert manager.intents[0].quantity == 3
    assert manager.intents[0].side.value == "sell"
