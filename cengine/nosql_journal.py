"""Daily MongoDB document storage for verified audit-journal records."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any, Protocol
from zoneinfo import ZoneInfo


class CollectionLike(Protocol):
    def create_index(self, keys: list[tuple[str, int]], **kwargs: Any) -> Any: ...

    def update_one(self, query: dict[str, Any], update: dict[str, Any], *, upsert: bool) -> Any: ...


class DatabaseLike(Protocol):
    def __getitem__(self, name: str) -> CollectionLike: ...


class ClientLike(Protocol):
    def __getitem__(self, name: str) -> DatabaseLike: ...

    def close(self) -> None: ...


class MongoDailyJournalStore:
    """Idempotent daily collections keyed by the journal's global sequence."""

    def __init__(
        self,
        uri: str,
        database: str,
        timezone_name: str,
        timeout_ms: int,
        *,
        client_factory: Callable[..., ClientLike] | None = None,
    ) -> None:
        if not uri.strip() or not database.strip():
            raise ValueError("MongoDB URI and database are required")
        if timeout_ms <= 0:
            raise ValueError("MongoDB timeout must be positive")
        self._zone = ZoneInfo(timezone_name)
        client_options: dict[str, Any] = {
            "serverSelectionTimeoutMS": timeout_ms,
            "connectTimeoutMS": timeout_ms,
            "retryWrites": True,
            "w": "majority",
            "journal": True,
        }
        if client_factory is None:
            try:
                from pymongo import MongoClient
                from pymongo.server_api import ServerApi
            except ImportError as exc:
                raise RuntimeError("install cengine[nosql] for MongoDB journaling") from exc
            client_factory = MongoClient
            client_options["server_api"] = ServerApi("1")
        self._client: ClientLike = client_factory(uri, **client_options)
        self._database: DatabaseLike = self._client[database]
        self._indexed: set[str] = set()

    def write(self, record: dict[str, Any]) -> None:
        collection_name = self._collection_name(int(record["timestamp_ns"]))
        collection = self._database[collection_name]
        if collection_name not in self._indexed:
            collection.create_index([("sequence", 1)], unique=True, name="journal_sequence")
            collection.create_index(
                [("timestamp_ns", 1), ("kind", 1)],
                name="journal_time_kind",
            )
            self._indexed.add(collection_name)
        document = dict(record)
        document["trading_date"] = collection_name.removeprefix("journal_")
        # An identical replay is a no-op. A different checksum with the same
        # sequence violates the unique index and fails closed.
        collection.update_one(
            {"sequence": record["sequence"], "sha256": record["sha256"]},
            {"$setOnInsert": document},
            upsert=True,
        )

    def close(self) -> None:
        self._client.close()

    def _collection_name(self, timestamp_ns: int) -> str:
        observed = datetime.fromtimestamp(timestamp_ns / 1_000_000_000, tz=timezone.utc).astimezone(
            self._zone
        )
        return f"journal_{observed:%Y%m%d}"
