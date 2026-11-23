"""Atomic, checksummed model-artifact store (Milestone 6, §18/§22).

Mirrors the lakehouse ParquetWriter publication recipe (temp file + fsync + os.replace +
sha256 content checksum) so model blobs are published atomically and never half-written.
Artifacts are verified against their checksum on load; corrupted or incompatible blobs are
REJECTED rather than allowed to influence financial decisions. Blobs live under
``<lakehouse_root>/_learning_models/`` and are NEVER committed to Git.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from cengine.lakehouse.config import LakehouseConfig


class ArtifactVerificationError(RuntimeError):
    """Raised when an artifact fails checksum/compatibility verification on load."""


@dataclass(frozen=True)
class ArtifactRef:
    path: str
    checksum: str
    bytes: int
    kind: str                 # 'npz' | 'json' | 'bin'
    framework: str = ""
    framework_version: str = ""

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


class ArtifactStore:
    def __init__(self, config: Optional[LakehouseConfig] = None) -> None:
        self.config = config or LakehouseConfig.from_env()
        self._dir = self.config.root / "_learning_models"
        self._dir.mkdir(parents=True, exist_ok=True)

    def recover_incomplete(self) -> int:
        removed = 0
        for tmp in self._dir.rglob("*.tmp"):
            try:
                tmp.unlink()
                removed += 1
            except OSError:
                pass
        return removed

    def _publish_bytes(self, model_id: str, model_version: str, suffix: str,
                       data: bytes, kind: str, *, framework: str = "",
                       framework_version: str = "") -> ArtifactRef:
        checksum = hashlib.sha256(data).hexdigest()
        sub = self._dir / model_id
        sub.mkdir(parents=True, exist_ok=True)
        final = sub / f"{model_version}-{checksum[:16]}{suffix}"
        if final.exists():  # idempotent: identical content already published
            return ArtifactRef(str(final), checksum, len(data), kind, framework,
                               framework_version)
        fd, tmp = tempfile.mkstemp(suffix=".tmp", dir=str(sub))
        try:
            with os.fdopen(fd, "wb") as fh:
                fh.write(data)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, final)
        except Exception:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
        return ArtifactRef(str(final), checksum, len(data), kind, framework,
                           framework_version)

    def save_json(self, model_id: str, model_version: str, obj: dict, *,
                  name: str = "params") -> ArtifactRef:
        data = json.dumps(obj, sort_keys=True, default=str).encode()
        return self._publish_bytes(model_id, model_version, f".{name}.json", data, "json")

    def save_npz(self, model_id: str, model_version: str, arrays: dict, *,
                 framework: str = "", framework_version: str = "",
                 name: str = "weights") -> ArtifactRef:
        """Serialize a dict of numpy arrays atomically (portable across CPU/MLX)."""
        import io

        import numpy as np

        buf = io.BytesIO()
        np.savez(buf, **{k: np.asarray(v) for k, v in arrays.items()})
        data = buf.getvalue()
        return self._publish_bytes(model_id, model_version, f".{name}.npz", data, "npz",
                                   framework=framework, framework_version=framework_version)

    def load_bytes(self, ref: ArtifactRef) -> bytes:
        p = Path(ref.path)
        if not p.exists():
            raise ArtifactVerificationError(f"artifact missing: {ref.path}")
        data = p.read_bytes()
        actual = hashlib.sha256(data).hexdigest()
        if actual != ref.checksum:
            raise ArtifactVerificationError(
                f"checksum mismatch for {ref.path}: expected {ref.checksum[:12]} "
                f"got {actual[:12]} (corrupted/incompatible artifact rejected)")
        return data

    def load_json(self, ref: ArtifactRef) -> dict:
        return json.loads(self.load_bytes(ref).decode())

    def load_npz(self, ref: ArtifactRef) -> dict:
        import io

        import numpy as np

        data = self.load_bytes(ref)
        with np.load(io.BytesIO(data), allow_pickle=False) as npz:
            return {k: npz[k] for k in npz.files}

    def verify(self, ref: ArtifactRef) -> bool:
        try:
            self.load_bytes(ref)
            return True
        except ArtifactVerificationError:
            return False
