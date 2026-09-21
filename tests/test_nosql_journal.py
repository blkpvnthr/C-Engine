from datetime import datetime, timezone

from cengine.journal import AuditJournal
from cengine.nosql_journal import MongoDailyJournalStore


class Collection:
    def __init__(self):
        self.indexes = []
        self.documents = {}

    def create_index(self, keys, **kwargs):
        self.indexes.append((keys, kwargs))

    def update_one(self, query, update, *, upsert):
        assert upsert
        sequence = query["sequence"]
        document = update["$setOnInsert"]
        existing = self.documents.get(sequence)
        if existing is not None and existing["sha256"] != query["sha256"]:
            raise RuntimeError("duplicate sequence with a different checksum")
        self.documents.setdefault(sequence, document)


class Database:
    def __init__(self):
        self.collections = {}

    def __getitem__(self, name):
        return self.collections.setdefault(name, Collection())


class Client:
    def __init__(self):
        self.databases = {}
        self.closed = False

    def __getitem__(self, name):
        return self.databases.setdefault(name, Database())

    def close(self):
        self.closed = True


def test_daily_nosql_journal_is_idempotent_and_date_partitioned(tmp_path):
    client = Client()

    def factory(*args, **kwargs):
        assert kwargs["retryWrites"]
        assert kwargs["w"] == "majority"
        assert kwargs["journal"] is True
        return client

    store = MongoDailyJournalStore(
        "mongodb://example", "cengine", "America/New_York", 1000, client_factory=factory
    )
    journal = AuditJournal(tmp_path / "audit.jsonl", sinks=(store,))
    # 2026-09-22 01:00 UTC is still the prior New York trading date.
    timestamp_ns = int(datetime(2026, 9, 22, 1, 0, tzinfo=timezone.utc).timestamp() * 1_000_000_000)
    journal.append("portfolio_metrics", timestamp_ns, {"equity_ticks": 100})
    collection = client.databases["cengine"].collections["journal_20260921"]
    assert collection.documents[1]["payload"]["equity_ticks"] == 100
    assert collection.documents[1]["trading_date"] == "20260921"
    # Reopening backfills the same sequence without duplication.
    AuditJournal(tmp_path / "audit.jsonl", sinks=(store,))
    assert len(collection.documents) == 1


def test_nosql_store_closes_client():
    client = Client()
    store = MongoDailyJournalStore(
        "mongodb://example",
        "cengine",
        "UTC",
        1000,
        client_factory=lambda *args, **kwargs: client,
    )
    store.close()
    assert client.closed
