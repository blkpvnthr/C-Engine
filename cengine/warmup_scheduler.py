"""Continuous, bounded, persistent historical warm-up scheduler (directive item 4).

Replaces the one-shot 500-instrument warm-up ceiling with a background scheduler
that keeps processing the WHOLE eligible master universe across successive bounded
batches until coverage is complete, then maintains it (membership growth, stale
refresh, bounded retry). A *batch* limit is acceptable; a permanent *coverage*
ceiling is not.

Design
------
* Prioritized acquisition (positions -> pending -> references -> actionable ->
  preliminary -> remaining), reusing :func:`historical_warmup.prioritize`.
* Batched multi-symbol requests (the underlying ``HistoricalWarmup`` paginates and
  fetches incrementally; this scheduler decides *which* symbols, *in what order*).
* Persistent progress (``data/warmup_progress.json``) so a restart resumes instead
  of restarting coverage; records per-symbol status, attempt count and timestamps.
* Bounded retry with attempt caps; rate-limit / failure backoff is applied by the
  caller between ``run_once`` batches (interval widened on failure).
* Membership updates: new master members are picked up automatically on the next
  ``next_batch``; a membership-version change re-opens previously-failed symbols.
* Stale refresh: symbols warmed longer than ``refresh_after_s`` ago become eligible
  again so native buffers do not silently drift.
* Restart recovery: progress is loaded on construction.

It is pure orchestration over an injected ``HistoricalWarmup`` and clock, so it is
fully unit-testable offline with no network. The caller runs ``run_once`` off the
hot path (a worker thread); mandatory protective exits never wait on it.
"""

from __future__ import annotations

import datetime
import json
import os
from typing import Any, Callable, Optional

from cengine.historical_warmup import WarmupStatus, prioritize


class ContinuousWarmupScheduler:
    def __init__(
        self,
        warmup: Any,
        *,
        universe_fn: Callable[[], list[str]],
        priority_fn: Optional[Callable[[], dict]] = None,
        batch_size: int = 100,
        state_path: Optional[str] = None,
        max_attempts: int = 3,
        refresh_after_s: float = 6 * 3600.0,
        start_5m_days: int = 20,
        start_1m_days: int = 4,
    ) -> None:
        self._warmup = warmup
        self._universe_fn = universe_fn
        self._priority_fn = priority_fn or (lambda: {})
        self._batch = max(1, int(batch_size))
        self._state_path = state_path
        self._max_attempts = max(1, int(max_attempts))
        self._refresh_after_s = float(refresh_after_s)
        self._start_5m_days = int(start_5m_days)
        self._start_1m_days = int(start_1m_days)
        self._progress: dict[str, Any] = {"membership_version": None, "symbols": {}}
        self._batches_run = 0
        self._load_progress()

    # -- persistence ---------------------------------------------------------
    def _load_progress(self) -> None:
        if not self._state_path or not os.path.exists(self._state_path):
            return
        try:
            with open(self._state_path) as f:
                doc = json.load(f)
            if isinstance(doc, dict) and isinstance(doc.get("symbols"), dict):
                self._progress = doc
        except Exception:  # noqa: BLE001 — corrupt progress must not block warm-up
            self._progress = {"membership_version": None, "symbols": {}}

    def _save_progress(self) -> None:
        if not self._state_path:
            return
        try:
            tmp = f"{self._state_path}.tmp"
            with open(tmp, "w") as f:
                json.dump(self._progress, f)
            os.replace(tmp, self._state_path)
        except Exception:  # noqa: BLE001 — persistence is best-effort
            pass

    # -- membership / eligibility -------------------------------------------
    def set_membership_version(self, version: Any) -> None:
        """Record the membership version; a change re-opens FAILED symbols so a
        refreshed snapshot gets another acquisition attempt."""
        prev = self._progress.get("membership_version")
        self._progress["membership_version"] = version
        if prev is not None and version is not None and prev != version:
            for rec in self._progress["symbols"].values():
                if rec.get("status") == WarmupStatus.FAILED.value:
                    rec["attempts"] = 0

    def _eligible(self, sym: str, now: float) -> bool:
        rec = self._progress["symbols"].get(sym)
        if rec is None:
            return True
        status = rec.get("status")
        if status == WarmupStatus.WARMED.value:
            warmed_at = float(rec.get("warmed_at", 0.0) or 0.0)
            return (now - warmed_at) >= self._refresh_after_s  # stale -> re-warm
        if status == WarmupStatus.FAILED.value:
            return int(rec.get("attempts", 0)) < self._max_attempts
        # INCOMPLETE / DOWNLOADING / unknown -> retry while under the attempt cap
        return int(rec.get("attempts", 0)) < self._max_attempts

    def next_batch(self, now: float = 0.0) -> list[str]:
        """The next prioritized, deduplicated, eligible batch (bounded by batch_size)."""
        prio = self._priority_fn() or {}
        remaining = self._universe_fn() or []
        order = prioritize(
            positions=prio.get("positions", ()), pending=prio.get("pending", ()),
            references=prio.get("references", ()), actionable=prio.get("actionable", ()),
            preliminary=prio.get("preliminary", ()), remaining=remaining)
        batch: list[str] = []
        for s in order:
            if self._eligible(s, now):
                batch.append(s)
            if len(batch) >= self._batch:
                break
        return batch

    # -- one bounded batch ---------------------------------------------------
    def _starts(self, now: float):
        # The injected clock is monotonic/seconds-from-arbitrary; use wall dates for
        # the REST window. (now is used only for eligibility/staleness bookkeeping.)
        today = datetime.date.today()
        start_5m = (today - datetime.timedelta(days=self._start_5m_days)).isoformat()
        start_1m = (today - datetime.timedelta(days=self._start_1m_days)).isoformat()
        return start_5m, start_1m

    def run_once(self, now: float = 0.0) -> dict:
        """Warm ONE bounded batch, persist progress, and return a batch summary.
        Synchronous; the caller runs it off the hot path. Never raises."""
        batch = self.next_batch(now)
        if not batch:
            return {"batch": 0, "complete": self.is_complete(), **self.coverage()}
        start_5m, start_1m = self._starts(now)
        try:
            self._warmup.warm(batch, start_5m=start_5m, start_1m=start_1m)
        except Exception:  # noqa: BLE001 — a batch failure must not abort the loop
            for s in batch:
                rec = self._progress["symbols"].setdefault(s, {"attempts": 0})
                rec["attempts"] = int(rec.get("attempts", 0)) + 1
                rec["status"] = WarmupStatus.FAILED.value
                rec["last_attempt"] = now
            self._batches_run += 1
            self._save_progress()
            return {"batch": len(batch), "complete": False, "error": True,
                    **self.coverage()}
        for s in batch:
            st = self._warmup.status(s)
            rec = self._progress["symbols"].setdefault(s, {"attempts": 0})
            rec["attempts"] = int(rec.get("attempts", 0)) + 1
            rec["status"] = st.value
            rec["last_attempt"] = now
            if st == WarmupStatus.WARMED:
                rec["warmed_at"] = now
        self._batches_run += 1
        self._save_progress()
        return {"batch": len(batch), "complete": self.is_complete(), **self.coverage()}

    # -- reporting -----------------------------------------------------------
    def is_complete(self) -> bool:
        """True when every current master member is WARMED (not stale) or terminally
        FAILED (attempt cap reached). New members or staleness re-open the loop."""
        return not self.next_batch(now=0.0) and bool(self._progress["symbols"])

    def coverage(self, now: float = 0.0) -> dict:
        """Exact, non-conflated counts over the master universe."""
        universe = set(self._universe_fn() or [])
        counts = {s.value: 0 for s in WarmupStatus}
        for rec in self._progress["symbols"].values():
            status = rec.get("status", WarmupStatus.NOT_REQUESTED.value)
            counts[status] = counts.get(status, 0) + 1
        processed = set(self._progress["symbols"])
        warmed = counts.get(WarmupStatus.WARMED.value, 0)
        bars = getattr(self._warmup, "_bars_ingested", {})
        return {
            "universe": len(universe),
            "processed": len(processed),
            "pending": len(universe - processed),
            "warmed": warmed,
            "incomplete": counts.get(WarmupStatus.INCOMPLETE.value, 0),
            "failed": counts.get(WarmupStatus.FAILED.value, 0),
            "stale": sum(
                1 for rec in self._progress["symbols"].values()
                if rec.get("status") == WarmupStatus.WARMED.value
                and (now - float(rec.get("warmed_at", 0.0) or 0.0)) >= self._refresh_after_s),
            "batches_run": self._batches_run,
            "bars_ingested": dict(bars) if isinstance(bars, dict) else {},
            "by_status": {k: v for k, v in counts.items() if v},
        }
