"""C-Engine local adaptive ML/RL data engineering platform (Milestone 5).

An entirely local Parquet + PyArrow + DuckDB lakehouse (bronze/silver/gold) that
accumulates market observations, engine decisions, execution experience and realized
outcomes for subsequent supervised / contextual-bandit / offline-RL learning.

It is ANALYTICAL only: the authoritative transactional financial journal
(cengine/journal.py) stays separate, Parquet is never the order-management database,
and no analytical failure can block the mandatory protective exit or broker
reconciliation (ingestion is nonblocking + bounded). No MongoDB/Supabase/cloud.
"""

from cengine.lakehouse.config import LakehouseConfig
from cengine.lakehouse.envelope import (
    ENVELOPE_SCHEMA_VERSION,
    EventEnvelope,
    envelope_from_decision_row,
)
from cengine.lakehouse.maintenance import LakehouseMaintenance
from cengine.lakehouse.pipeline import LakehousePipeline
from cengine.lakehouse.storage_monitor import StorageMonitor

__all__ = [
    "ENVELOPE_SCHEMA_VERSION",
    "EventEnvelope",
    "LakehouseConfig",
    "LakehouseMaintenance",
    "LakehousePipeline",
    "StorageMonitor",
    "envelope_from_decision_row",
]
