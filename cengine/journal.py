"""Append-only, checksummed JSONL event journal."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, is_dataclass
from pathlib import Path
from threading import RLock
from typing import Any, Iterator


class AuditJournal:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._sequence = self._last_sequence()
        self._lock = RLock()

    def append(self, kind: str, timestamp_ns: int, payload: Any) -> int:
        if timestamp_ns <= 0 or not kind:
            raise ValueError("journal kind and authoritative timestamp are required")
        value = (
            asdict(payload) if is_dataclass(payload) and not isinstance(payload, type) else payload
        )
        with self._lock:
            self._sequence += 1
            body = {
                "sequence": self._sequence,
                "kind": kind,
                "timestamp_ns": timestamp_ns,
                "payload": value,
            }
            encoded = json.dumps(body, sort_keys=True, separators=(",", ":"), default=str)
            record = {**body, "sha256": hashlib.sha256(encoded.encode()).hexdigest()}
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True, default=str) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            return self._sequence

    def records(self) -> Iterator[dict[str, Any]]:
        if not self.path.exists():
            return
        with self.path.open(encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                checksum = record.pop("sha256")
                encoded = json.dumps(record, sort_keys=True, separators=(",", ":"), default=str)
                if hashlib.sha256(encoded.encode()).hexdigest() != checksum:
                    raise ValueError(f"journal checksum failure at sequence {record['sequence']}")
                record["sha256"] = checksum
                yield record

    def _last_sequence(self) -> int:
        last = 0
        for record in self.records() or ():
            expected = last + 1
            if record["sequence"] != expected:
                raise ValueError(f"journal sequence gap: expected {expected}")
            last = expected
        return last
