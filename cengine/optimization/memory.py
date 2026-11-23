"""Persistent optimization memory (Milestone 6B, §10).

The feature that distinguishes autonomous optimization from ordinary scheduled retraining:
the controller remembers every prior experiment (config, dataset, regime, cost, economic
outcome, decision, failure reason) and uses it to avoid redundant work, focus on promising
search regions, deprioritize repeatedly-failing configs, resume interrupted sessions, and
score the effectiveness of its own experiment-selection process (the outer loop).

Stored as an append-only JSONL ledger under the lakehouse root (local, gitignored). The
optimizer may improve its selection POLICY via this evidence; it never modifies its own
executable code.
"""

from __future__ import annotations

import json
import os
import time
from typing import Optional

from cengine.lakehouse.config import LakehouseConfig
from cengine.optimization.contracts import config_fingerprint

OPTIMIZATION_MEMORY_VERSION = "optimization-memory-1.0.0"


class OptimizationMemory:
    def __init__(self, config: Optional[LakehouseConfig] = None) -> None:
        self.config = config or LakehouseConfig.from_env()
        self._dir = self.config.root / "_optimization"
        self._dir.mkdir(parents=True, exist_ok=True)
        self._path = self._dir / "optimization_memory.jsonl"

    # -- persistence ---------------------------------------------------------
    def record(self, *, hypothesis_id: Optional[str], search_space: str, config: dict,
               score: Optional[float], economic_metrics: dict, decision: str,
               rejection_reason: str = "", data_label: str = "synthetic",
               regime: str = "all", cost_s: float = 0.0,
               now_ns: Optional[int] = None) -> dict:
        row = {"fingerprint": config_fingerprint({"space": search_space, **config}),
               "hypothesis_id": hypothesis_id, "search_space": search_space,
               "config": config, "score": score, "economic_metrics": economic_metrics,
               "decision": decision, "rejection_reason": rejection_reason,
               "data_label": data_label, "regime": regime, "cost_s": cost_s,
               "recorded_ns": now_ns if now_ns is not None else time.time_ns(),
               "version": OPTIMIZATION_MEMORY_VERSION}
        with open(self._path, "a") as fh:
            fh.write(json.dumps(row, default=str) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        return row

    def all(self) -> list:
        if not self._path.exists():
            return []
        out = []
        for line in self._path.read_text().splitlines():
            if line.strip():
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        return out

    # -- selection intelligence ----------------------------------------------
    def seen_fingerprints(self) -> set:
        return {r["fingerprint"] for r in self.all() if r.get("fingerprint")}

    def is_duplicate(self, search_space: str, config: dict) -> bool:
        return config_fingerprint({"space": search_space, **config}) in self.seen_fingerprints()

    def dedupe(self, search_space: str, configs: list) -> list:
        """Drop configs already evaluated (any data label)."""
        seen = self.seen_fingerprints()
        out, local = [], set()
        for c in configs:
            fp = config_fingerprint({"space": search_space, **c})
            if fp in seen or fp in local:
                continue
            local.add(fp)
            out.append(c)
        return out

    def history_for(self, search_space: str) -> list:
        """(config, score) history for a space — feeds the TPE sampler."""
        return [{"config": r["config"], "score": r["score"]}
                for r in self.all()
                if r.get("search_space") == search_space and r.get("score") is not None]

    def repeated_failures(self, search_space: str, *, min_count: int = 3) -> list:
        """Config fingerprints that have been rejected repeatedly — deprioritize."""
        counts: dict = {}
        for r in self.all():
            if r.get("search_space") == search_space and r.get("decision") == "rejected":
                fp = r.get("fingerprint")
                counts[fp] = counts.get(fp, 0) + 1
        return [fp for fp, c in counts.items() if c >= min_count]

    # -- outer-loop effectiveness (§10) --------------------------------------
    def effectiveness(self) -> dict:
        rows = self.all()
        attempted = len(rows)
        completed = sum(1 for r in rows if r.get("decision") not in ("deferred", "error", None))
        improvements = sum(1 for r in rows if r.get("decision") == "eligible_for_shadow")
        retained = sum(1 for r in rows
                       if r.get("decision") == "eligible_for_shadow"
                       and (r.get("shadow_retained") is True))
        total_cost = sum(float(r.get("cost_s", 0.0)) for r in rows)
        return {"experiments_attempted": attempted, "experiments_completed": completed,
                "improvements_identified": improvements, "improvements_retained": retained,
                "compute_seconds_total": round(total_cost, 2),
                "compute_per_improvement_s": (round(total_cost / improvements, 2)
                                              if improvements else None),
                "duplicate_configs_in_memory":
                    attempted - len(self.seen_fingerprints()),
                "version": OPTIMIZATION_MEMORY_VERSION}
