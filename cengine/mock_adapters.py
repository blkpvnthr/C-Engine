"""Native-adapter module for isolated mock execution (directive §9).

Selected via ``ENGINE_NATIVE_ADAPTER_MODULE=cengine.mock_adapters``. It pairs the
REAL native C++ RiskEngine (pure computation, no network) with the in-memory
:class:`cengine.mock_broker.SimulatedBroker`, so the full order path can be
exercised with no broker connectivity and no live orders.

This never touches live/paper endpoints. It is for certification harnesses and
tests only; nothing selects it by default.
"""

from __future__ import annotations

from cengine.mock_broker import SimulatedBroker
from native_adapters import build_risk_engine  # real native C++ risk boundary

__all__ = ["build_execution_venue", "build_risk_engine"]


def build_execution_venue() -> SimulatedBroker:
    """Return an isolated in-memory venue. Fill prices/plans may be configured by
    the harness after construction via ``set_price`` / ``set_plan``."""
    return SimulatedBroker()
