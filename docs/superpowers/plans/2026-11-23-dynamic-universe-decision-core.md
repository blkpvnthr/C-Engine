# Dynamic Universe — Decision Core Implementation Plan (Plan 1 of N)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the pure, deterministic decision core for dynamic universe rotation — config, permanent-baseline registry, 4-layer model, candidate lifecycle, and the rotation scheduler — with zero engine/IO coupling and 100% unit-test coverage.

**Architecture:** A new `cengine/dynamic_universe/` package. Every module here is pure (no network, no broker, no asyncio, no live filesystem except the baseline-config read, which is isolated in `BaselineRegistry` and fully tempdir-testable). Later plans wire this core into the live engine (`manager.py`, decision persistence, observability, learned policy, replay).

**Tech Stack:** Python 3.11+, stdlib `dataclasses`/`enum`/`pathlib`, `pytest`. Reuses the existing pure `universe` module (`build_universe`, `normalize_symbol`, `load/write_canonical_universe`, `UniverseResult`, `SymbolInfo`).

## Global Constraints

- Baseline is config-driven in `baseline_universe.txt`, seeded with the exact spec-30. `UPRO` is **not** baseline. Permanence = membership, not continuous subscription. Only operator edits change baseline; invalid updates are rejected without overwriting last-good.
- `ACTIVE_UNIVERSE_SIZE = 30`; the active window never exceeds 30 unique tickers.
- Slot targets are *targets*, not quotas: `baseline_target_slots=20`, `opportunity_target_slots=10`; fill unused slots with eligible baseline.
- Reuse the existing pure `universe` module for normalize/dedupe/validate/classify — never reimplement symbol validation.
- Everything in this plan is pure/deterministic; no `time.time()`/`random` inside scheduler logic — timestamps are passed in (monotonic `now_s: float`) so tests are reproducible.
- All code: type hints + docstrings, match surrounding style (frozen dataclasses, `from __future__ import annotations`).

---

### Task 1: `UniverseRotationConfig`

**Files:**
- Create: `cengine/dynamic_universe/__init__.py`
- Create: `cengine/dynamic_universe/config.py`
- Test: `tests/test_dynamic_universe_config.py`

**Interfaces:**
- Produces: `UniverseRotationConfig` frozen dataclass with fields
  `active_universe_size:int=30`, `baseline_target_slots:int=20`,
  `opportunity_target_slots:int=10`, `refresh_interval_seconds:float=60.0`,
  `opportunity_refresh_seconds:float=300.0`, `baseline_max_revisit_seconds:float=300.0`,
  `candidate_min_residence_seconds:float=180.0`, `protect_open_positions:bool=True`,
  `protect_pending_orders:bool=True`, `hysteresis_margin:float=0.05`.
  Classmethod `from_env(getenv=os.getenv) -> UniverseRotationConfig` reading
  `CENGINE_UNIVERSE_*` overrides. Method `validate() -> None` raising `ValueError`
  on incoherent config (negative intervals, size<1, target slots sum guidance).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_dynamic_universe_config.py
import pytest
from cengine.dynamic_universe.config import UniverseRotationConfig


def test_defaults_match_spec():
    c = UniverseRotationConfig()
    assert c.active_universe_size == 30
    assert c.baseline_target_slots == 20
    assert c.opportunity_target_slots == 10
    assert c.refresh_interval_seconds == 60.0
    assert c.opportunity_refresh_seconds == 300.0
    assert c.baseline_max_revisit_seconds == 300.0
    assert c.candidate_min_residence_seconds == 180.0
    assert c.protect_open_positions is True
    assert c.protect_pending_orders is True


def test_from_env_overrides():
    env = {
        "CENGINE_UNIVERSE_ACTIVE_SIZE": "25",
        "CENGINE_UNIVERSE_BASELINE_SLOTS": "15",
        "CENGINE_UNIVERSE_OPPORTUNITY_SLOTS": "10",
        "CENGINE_UNIVERSE_MIN_RESIDENCE_S": "90",
    }
    c = UniverseRotationConfig.from_env(getenv=lambda k, d=None: env.get(k, d))
    assert c.active_universe_size == 25
    assert c.baseline_target_slots == 15
    assert c.candidate_min_residence_seconds == 90.0


def test_validate_rejects_bad_values():
    with pytest.raises(ValueError):
        UniverseRotationConfig(active_universe_size=0).validate()
    with pytest.raises(ValueError):
        UniverseRotationConfig(candidate_min_residence_seconds=-1).validate()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_dynamic_universe_config.py -v`
Expected: FAIL (`ModuleNotFoundError: cengine.dynamic_universe`)

- [ ] **Step 3: Write minimal implementation**

```python
# cengine/dynamic_universe/__init__.py
"""Dynamic universe management & opportunity rotation (selection/rotation layer).

Pure decision core: config, permanent baseline registry, 4-layer model,
candidate lifecycle, and the rotation scheduler. No socket ownership, no second
data connection — the active universe computed here is fed to the existing
subscription owner (universe_sync / adaptive controller) by the manager layer.
"""
```

```python
# cengine/dynamic_universe/config.py
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class UniverseRotationConfig:
    """Allocation targets and cadences for dynamic universe rotation.

    Slot counts are *targets*, not rigid quotas: unused opportunity slots are
    filled with eligible baseline securities, and protected positions/orders
    always take precedence (see the rotation scheduler).
    """

    active_universe_size: int = 30
    baseline_target_slots: int = 20
    opportunity_target_slots: int = 10
    refresh_interval_seconds: float = 60.0
    opportunity_refresh_seconds: float = 300.0
    baseline_max_revisit_seconds: float = 300.0
    candidate_min_residence_seconds: float = 180.0
    protect_open_positions: bool = True
    protect_pending_orders: bool = True
    hysteresis_margin: float = 0.05

    @classmethod
    def from_env(
        cls, getenv: Callable[[str, str | None], str | None] = os.getenv
    ) -> "UniverseRotationConfig":
        def _int(key: str, default: int) -> int:
            raw = getenv(key, None)
            return int(raw) if raw not in (None, "") else default

        def _float(key: str, default: float) -> float:
            raw = getenv(key, None)
            return float(raw) if raw not in (None, "") else default

        def _bool(key: str, default: bool) -> bool:
            raw = getenv(key, None)
            if raw in (None, ""):
                return default
            return str(raw).strip().lower() not in ("0", "false", "no")

        cfg = cls(
            active_universe_size=_int("CENGINE_UNIVERSE_ACTIVE_SIZE", 30),
            baseline_target_slots=_int("CENGINE_UNIVERSE_BASELINE_SLOTS", 20),
            opportunity_target_slots=_int("CENGINE_UNIVERSE_OPPORTUNITY_SLOTS", 10),
            refresh_interval_seconds=_float("CENGINE_UNIVERSE_REFRESH_S", 60.0),
            opportunity_refresh_seconds=_float("CENGINE_UNIVERSE_OPP_REFRESH_S", 300.0),
            baseline_max_revisit_seconds=_float("CENGINE_UNIVERSE_REVISIT_S", 300.0),
            candidate_min_residence_seconds=_float("CENGINE_UNIVERSE_MIN_RESIDENCE_S", 180.0),
            protect_open_positions=_bool("CENGINE_UNIVERSE_PROTECT_POSITIONS", True),
            protect_pending_orders=_bool("CENGINE_UNIVERSE_PROTECT_ORDERS", True),
            hysteresis_margin=_float("CENGINE_UNIVERSE_HYSTERESIS", 0.05),
        )
        cfg.validate()
        return cfg

    def validate(self) -> None:
        if self.active_universe_size < 1:
            raise ValueError("active_universe_size must be >= 1")
        for name in (
            "refresh_interval_seconds",
            "opportunity_refresh_seconds",
            "baseline_max_revisit_seconds",
            "candidate_min_residence_seconds",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be >= 0")
        if self.baseline_target_slots < 0 or self.opportunity_target_slots < 0:
            raise ValueError("slot targets must be >= 0")
        if not (0.0 <= self.hysteresis_margin < 1.0):
            raise ValueError("hysteresis_margin must be in [0, 1)")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_dynamic_universe_config.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add cengine/dynamic_universe/__init__.py cengine/dynamic_universe/config.py tests/test_dynamic_universe_config.py
git commit -m "dynamic-universe: rotation config with env overrides + validation"
```

---

### Task 2: `BaselineRegistry` + `baseline_universe.txt`

**Files:**
- Create: `cengine/dynamic_universe/baseline.py`
- Create: `baseline_universe.txt` (repo-root seed, exact spec-30)
- Test: `tests/test_dynamic_universe_baseline.py`

**Interfaces:**
- Consumes: `universe.build_universe`, `universe.normalize_symbol`, `universe.write_canonical_universe`.
- Produces:
  - `BASELINE_UNIVERSE: tuple[str, ...]` — the spec-30 constant.
  - `BaselineRegistry(path: str|Path = BASELINE_TXT)`.
    - `.symbols() -> tuple[str, ...]` ordered baseline membership.
    - `.contains(symbol) -> bool`.
    - `.is_baseline(symbol) -> bool` (alias of contains).
    - `.reload() -> bool` (re-read file; keep last-good on invalid; returns True iff changed).
    - `.apply_operator_update(symbols, *, actor="operator", now_s=None) -> BaselineChange` — validates, atomically rewrites file, appends audit; raises `ValueError` on invalid (empty/all-malformed) without touching the live file or last-good.
    - `.audit_log() -> tuple[BaselineChange, ...]`.
  - `BaselineChange` frozen dataclass `(timestamp_s, actor, added, removed, resulting)`.
- **Permanence contract:** `BaselineRegistry` is the ONLY writer of `baseline_universe.txt`. The scanner/opportunity finder never calls `apply_operator_update`.

- [ ] **Step 1: Create the seed file**

```text
# baseline_universe.txt  (repo root)
# Permanent baseline universe (operator-managed). 30 symbols.
# The Opportunity Finder NEVER edits this file. UPRO is intentionally NOT here
# (dynamic-only). universe.txt remains the mutable working/active universe.
'ANT', 'GPN', 'DVA', 'HUM', 'UNH', 'HCA', 'UHS', 'GIS', 'WMT', 'COST', 'NVDA', 'AMD', 'AVGO', 'AAPL', 'MU', 'TSLA', 'SPY', 'QQQ', 'DIA', 'IWM', 'XLP', 'XLV', 'GLD', 'SQQQ', 'SOXS', 'SPXU', 'SDOW', 'TZA', 'TQQQ', 'SOXL'
```

- [ ] **Step 2: Write the failing test**

```python
# tests/test_dynamic_universe_baseline.py
import pytest
from pathlib import Path
from cengine.dynamic_universe.baseline import (
    BASELINE_UNIVERSE, BaselineRegistry, BaselineChange,
)

SPEC_30 = (
    "ANT", "GPN", "DVA", "HUM", "UNH", "HCA", "UHS", "GIS", "WMT", "COST",
    "NVDA", "AMD", "AVGO", "AAPL", "MU", "TSLA", "SPY", "QQQ", "DIA", "IWM",
    "XLP", "XLV", "GLD", "SQQQ", "SOXS", "SPXU", "SDOW", "TZA", "TQQQ", "SOXL",
)


def _seed(tmp_path: Path, symbols=SPEC_30) -> Path:
    p = tmp_path / "baseline_universe.txt"
    p.write_text(", ".join(f"'{s}'" for s in symbols) + "\n", encoding="utf-8")
    return p


def test_constant_is_exact_spec_30():
    assert BASELINE_UNIVERSE == SPEC_30
    assert len(BASELINE_UNIVERSE) == 30
    assert "UPRO" not in BASELINE_UNIVERSE


def test_loads_all_30(tmp_path):
    reg = BaselineRegistry(_seed(tmp_path))
    assert reg.symbols() == SPEC_30
    assert reg.contains("nvda") is True       # normalization
    assert reg.is_baseline("UPRO") is False


def test_operator_update_adds_and_audits(tmp_path):
    reg = BaselineRegistry(_seed(tmp_path))
    change = reg.apply_operator_update(SPEC_30 + ("JEPQ",), actor="alice", now_s=100.0)
    assert isinstance(change, BaselineChange)
    assert change.added == ("JEPQ",)
    assert "JEPQ" in reg.symbols()
    assert reg.audit_log()[-1].actor == "alice"
    # persisted: a fresh registry over the same file sees JEPQ
    assert "JEPQ" in BaselineRegistry(reg.path).symbols()


def test_invalid_update_rejected_keeps_last_good(tmp_path):
    reg = BaselineRegistry(_seed(tmp_path))
    before = reg.symbols()
    with pytest.raises(ValueError):
        reg.apply_operator_update([], actor="bob")          # empty -> reject
    with pytest.raises(ValueError):
        reg.apply_operator_update(["!!!", "???"], actor="bob")  # all malformed
    assert reg.symbols() == before                          # unchanged
    # file on disk still the good one
    assert BaselineRegistry(reg.path).symbols() == before


def test_opportunity_finder_cannot_mutate_via_reload(tmp_path):
    # Reload only re-reads; it never writes. Simulate an external bad write.
    p = _seed(tmp_path)
    reg = BaselineRegistry(p)
    p.write_text("", encoding="utf-8")       # external truncation (bad)
    assert reg.reload() is False             # keep last good
    assert reg.symbols() == SPEC_30
```

- [ ] **Step 3: Run test to verify it fails**

Run: `python -m pytest tests/test_dynamic_universe_baseline.py -v`
Expected: FAIL (`ModuleNotFoundError`)

- [ ] **Step 4: Write minimal implementation**

```python
# cengine/dynamic_universe/baseline.py
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import universe

logger = logging.getLogger(__name__)

# Permanent baseline (operator-managed). Exactly the spec-30. UPRO is NOT here.
BASELINE_UNIVERSE: tuple[str, ...] = (
    "ANT", "GPN", "DVA", "HUM", "UNH", "HCA", "UHS", "GIS", "WMT", "COST",
    "NVDA", "AMD", "AVGO", "AAPL", "MU", "TSLA", "SPY", "QQQ", "DIA", "IWM",
    "XLP", "XLV", "GLD", "SQQQ", "SOXS", "SPXU", "SDOW", "TZA", "TQQQ", "SOXL",
)

BASELINE_TXT = universe.PROJECT_ROOT / "baseline_universe.txt"


@dataclass(frozen=True)
class BaselineChange:
    timestamp_s: float
    actor: str
    added: tuple[str, ...]
    removed: tuple[str, ...]
    resulting: tuple[str, ...]


def _clean(symbols: Iterable[str]) -> tuple[str, ...]:
    """Normalize/dedupe/validate via the shared pure universe pipeline.

    max_symbols=0 means 'no cap' (baseline size is operator-chosen).
    """
    result = universe.build_universe(list(symbols), max_symbols=0)
    return tuple(result.symbols)


class BaselineRegistry:
    """Authoritative, permanent baseline membership.

    The ONLY writer of ``baseline_universe.txt``. The Opportunity Finder must
    never call :meth:`apply_operator_update`. Permanence applies to *membership*,
    not to continuous subscription: a baseline symbol may rotate out of the
    active window but is always eligible for reinsertion.
    """

    def __init__(self, path: str | Path = BASELINE_TXT) -> None:
        self._path = Path(path)
        self._lock = threading.RLock()
        self._audit: list[BaselineChange] = []
        if not self._path.exists():
            # First run: seed with the spec-30 constant.
            universe.write_canonical_universe(BASELINE_UNIVERSE, self._path)
        self._symbols = _clean(universe.load_canonical_universe(self._path))
        if not self._symbols:
            # Defensive: an existing-but-unusable file falls back to the constant.
            self._symbols = _clean(BASELINE_UNIVERSE)

    @property
    def path(self) -> Path:
        return self._path

    def symbols(self) -> tuple[str, ...]:
        with self._lock:
            return self._symbols

    def contains(self, symbol: str) -> bool:
        return universe.normalize_symbol(symbol) in set(self.symbols())

    is_baseline = contains

    def reload(self) -> bool:
        """Re-read the file; keep last-good on invalid. Returns True iff changed."""
        try:
            fresh = _clean(universe.load_canonical_universe(self._path))
        except Exception as exc:  # noqa: BLE001 - keep last good
            logger.warning("baseline reload failed (keeping last good): %s", exc)
            return False
        if not fresh:
            return False
        with self._lock:
            if fresh == self._symbols:
                return False
            self._symbols = fresh
            return True

    def apply_operator_update(
        self,
        symbols: Sequence[str],
        *,
        actor: str = "operator",
        now_s: float | None = None,
    ) -> BaselineChange:
        """Validate + atomically rewrite the baseline. Reject-invalid-keep-last-good."""
        cleaned = _clean(symbols)
        if not cleaned:
            raise ValueError(
                "refusing baseline update with 0 usable symbols "
                "(last-good baseline left untouched)"
            )
        ts = float(now_s) if now_s is not None else 0.0
        with self._lock:
            old = set(self._symbols)
            new = set(cleaned)
            # Atomic file write BEFORE mutating in-memory state; on failure the
            # exception propagates and in-memory state is unchanged.
            universe.write_canonical_universe(cleaned, self._path)
            change = BaselineChange(
                timestamp_s=ts,
                actor=actor,
                added=tuple(s for s in cleaned if s not in old),
                removed=tuple(s for s in self._symbols if s not in new),
                resulting=cleaned,
            )
            self._symbols = cleaned
            self._audit.append(change)
            logger.info(
                "baseline updated by %s: +%s -%s (now %d)",
                actor, change.added, change.removed, len(cleaned),
            )
            return change

    def audit_log(self) -> tuple[BaselineChange, ...]:
        with self._lock:
            return tuple(self._audit)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_dynamic_universe_baseline.py -v`
Expected: PASS (5 passed)

- [ ] **Step 6: Commit**

```bash
git add cengine/dynamic_universe/baseline.py baseline_universe.txt tests/test_dynamic_universe_baseline.py
git commit -m "dynamic-universe: permanent BaselineRegistry (config-driven, spec-30, audited)"
```

---

### Task 3: `UniverseLayers` 4-layer snapshot

**Files:**
- Create: `cengine/dynamic_universe/layers.py`
- Test: `tests/test_dynamic_universe_layers.py`

**Interfaces:**
- Produces: `UniverseLayers` frozen dataclass:
  - fields: `baseline: tuple[str,...]`, `opportunity: tuple[str,...]`, `held: tuple[str,...]`, `pending: tuple[str,...]`, `active: tuple[str,...]`, `version: int`.
  - classmethod `build(baseline, opportunity, held, pending, active, version) -> UniverseLayers` — normalizes/dedupes each layer via `universe.normalize_symbol`, preserving order.
  - property `master -> tuple[str,...]` = dedup(baseline + opportunity + held + pending), order-preserving.
  - method `counts() -> dict[str,int]` = `{baseline, opportunity, master, active, held, pending}`.
- **Invariant:** `active` ⊆ `master`; `len(active)` unconstrained here (the scheduler enforces the ≤30 cap — layers is a passive snapshot).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_dynamic_universe_layers.py
from cengine.dynamic_universe.layers import UniverseLayers


def test_master_is_dedup_union_order_preserving():
    L = UniverseLayers.build(
        baseline=["SPY", "QQQ", "AAPL"],
        opportunity=["AAPL", "NVDA"],      # AAPL duplicate across layers
        held=["TSLA"],
        pending=["QQQ"],                   # dup of baseline
        active=["SPY", "AAPL", "TSLA"],
        version=1,
    )
    assert L.master == ("SPY", "QQQ", "AAPL", "NVDA", "TSLA")
    assert L.counts() == {
        "baseline": 3, "opportunity": 2, "master": 5,
        "active": 3, "held": 1, "pending": 1,
    }


def test_normalization_applied():
    L = UniverseLayers.build(
        baseline=["spy", " qqq "], opportunity=["'nvda'"],
        held=[], pending=[], active=["spy"], version=2,
    )
    assert L.baseline == ("SPY", "QQQ")
    assert L.opportunity == ("NVDA",)
    assert L.active == ("SPY",)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_dynamic_universe_layers.py -v`
Expected: FAIL (`ModuleNotFoundError`)

- [ ] **Step 3: Write minimal implementation**

```python
# cengine/dynamic_universe/layers.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import universe


def _norm_seq(symbols: Iterable[str]) -> tuple[str, ...]:
    out: list[str] = []
    seen: set[str] = set()
    for s in symbols:
        n = universe.normalize_symbol(s)
        if n and n not in seen:
            seen.add(n)
            out.append(n)
    return tuple(out)


@dataclass(frozen=True)
class UniverseLayers:
    """Immutable snapshot of the four universe layers (spec §3).

    * baseline    — permanent operator-managed membership
    * opportunity — dynamically discovered/ranked candidates
    * master      — dedup union of baseline + opportunity + held + pending
    * active      — current scanning window (<=30, enforced by the scheduler)
    """

    baseline: tuple[str, ...]
    opportunity: tuple[str, ...]
    held: tuple[str, ...]
    pending: tuple[str, ...]
    active: tuple[str, ...]
    version: int

    @classmethod
    def build(
        cls,
        baseline: Iterable[str],
        opportunity: Iterable[str],
        held: Iterable[str],
        pending: Iterable[str],
        active: Iterable[str],
        version: int,
    ) -> "UniverseLayers":
        return cls(
            baseline=_norm_seq(baseline),
            opportunity=_norm_seq(opportunity),
            held=_norm_seq(held),
            pending=_norm_seq(pending),
            active=_norm_seq(active),
            version=int(version),
        )

    @property
    def master(self) -> tuple[str, ...]:
        return _norm_seq(self.baseline + self.opportunity + self.held + self.pending)

    def counts(self) -> dict[str, int]:
        return {
            "baseline": len(self.baseline),
            "opportunity": len(self.opportunity),
            "master": len(self.master),
            "active": len(self.active),
            "held": len(self.held),
            "pending": len(self.pending),
        }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_dynamic_universe_layers.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add cengine/dynamic_universe/layers.py tests/test_dynamic_universe_layers.py
git commit -m "dynamic-universe: 4-layer UniverseLayers snapshot (baseline/opportunity/master/active)"
```

---

### Task 4: Candidate lifecycle state machine

**Files:**
- Create: `cengine/dynamic_universe/lifecycle.py`
- Test: `tests/test_dynamic_universe_lifecycle.py`

**Interfaces:**
- Produces:
  - `LifecycleState(str, Enum)`: `DISCOVERED, ELIGIBLE, RANKED, ACTIVE, ROTATED_OUT, REASSESSMENT, EXPIRED`.
  - `CandidateState` frozen dataclass: `symbol`, `state`, `is_baseline`, `last_eval_s`, `entered_active_s`, `rotated_out_s`, `last_score`, `reason`.
  - `LifecycleTracker(config: UniverseRotationConfig)` with:
    - `.observe(symbol, *, is_baseline, eligible, ranked, score, now_s) -> CandidateState` — advances DISCOVERED→ELIGIBLE→RANKED.
    - `.mark_active(symbol, now_s, reason) -> None` — → ACTIVE, stamps `entered_active_s`.
    - `.mark_rotated_out(symbol, now_s, reason) -> None` — → ROTATED_OUT, stamps `rotated_out_s`.
    - `.can_demote(symbol, now_s) -> bool` — False while within `candidate_min_residence_seconds` (baseline residence governed separately; emergency bypass via `force`).
    - `.needs_revisit(symbol, now_s) -> bool` — True for a baseline symbol whose `last_eval_s` is older than `baseline_max_revisit_seconds`.
    - `.reassess(symbol, now_s) -> None` — ROTATED_OUT→REASSESSMENT.
    - `.expire(symbol, now_s, reason) -> None` — →EXPIRED (never allowed for baseline; raises `ValueError` if `is_baseline`).
    - `.get(symbol) -> CandidateState | None`; `.all() -> dict[str, CandidateState]`.
- Consumes: `UniverseRotationConfig` (Task 1).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_dynamic_universe_lifecycle.py
import pytest
from cengine.dynamic_universe.config import UniverseRotationConfig
from cengine.dynamic_universe.lifecycle import LifecycleTracker, LifecycleState


def _tracker():
    return LifecycleTracker(UniverseRotationConfig(
        candidate_min_residence_seconds=180.0,
        baseline_max_revisit_seconds=300.0,
    ))


def test_discovery_to_ranked_progression():
    t = _tracker()
    s = t.observe("NVDA", is_baseline=False, eligible=False, ranked=False, score=0.0, now_s=0.0)
    assert s.state is LifecycleState.DISCOVERED
    s = t.observe("NVDA", is_baseline=False, eligible=True, ranked=False, score=0.1, now_s=1.0)
    assert s.state is LifecycleState.ELIGIBLE
    s = t.observe("NVDA", is_baseline=False, eligible=True, ranked=True, score=0.7, now_s=2.0)
    assert s.state is LifecycleState.RANKED
    assert s.last_eval_s == 2.0


def test_min_residence_blocks_demotion():
    t = _tracker()
    t.observe("NVDA", is_baseline=False, eligible=True, ranked=True, score=0.7, now_s=0.0)
    t.mark_active("NVDA", now_s=0.0, reason="promoted")
    assert t.can_demote("NVDA", now_s=100.0) is False   # within 180s
    assert t.can_demote("NVDA", now_s=200.0) is True     # past residence


def test_baseline_revisit_deadline():
    t = _tracker()
    t.observe("SPY", is_baseline=True, eligible=True, ranked=True, score=0.5, now_s=0.0)
    assert t.needs_revisit("SPY", now_s=100.0) is False
    assert t.needs_revisit("SPY", now_s=400.0) is True   # > 300s since last eval


def test_baseline_cannot_expire():
    t = _tracker()
    t.observe("SPY", is_baseline=True, eligible=True, ranked=True, score=0.5, now_s=0.0)
    with pytest.raises(ValueError):
        t.expire("SPY", now_s=1000.0, reason="stale")


def test_rotated_out_then_reassess():
    t = _tracker()
    t.observe("NVDA", is_baseline=False, eligible=True, ranked=True, score=0.7, now_s=0.0)
    t.mark_active("NVDA", now_s=0.0, reason="promoted")
    t.mark_rotated_out("NVDA", now_s=300.0, reason="weaker score")
    assert t.get("NVDA").state is LifecycleState.ROTATED_OUT
    t.reassess("NVDA", now_s=600.0)
    assert t.get("NVDA").state is LifecycleState.REASSESSMENT
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_dynamic_universe_lifecycle.py -v`
Expected: FAIL (`ModuleNotFoundError`)

- [ ] **Step 3: Write minimal implementation**

```python
# cengine/dynamic_universe/lifecycle.py
from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum

from .config import UniverseRotationConfig


class LifecycleState(str, Enum):
    DISCOVERED = "discovered"
    ELIGIBLE = "eligible"
    RANKED = "ranked"
    ACTIVE = "active"
    ROTATED_OUT = "rotated_out"
    REASSESSMENT = "reassessment"
    EXPIRED = "expired"


@dataclass(frozen=True)
class CandidateState:
    symbol: str
    state: LifecycleState
    is_baseline: bool
    last_eval_s: float
    entered_active_s: float | None = None
    rotated_out_s: float | None = None
    last_score: float = 0.0
    reason: str = ""


class LifecycleTracker:
    """Per-symbol candidate lifecycle with residence/revisit/hysteresis rules.

    Pure: all time is passed in as ``now_s`` so behavior is reproducible.
    Baseline membership is permanent — baseline symbols can be ROTATED_OUT or
    REASSESSMENT but never EXPIRED.
    """

    def __init__(self, config: UniverseRotationConfig) -> None:
        self._cfg = config
        self._states: dict[str, CandidateState] = {}

    def get(self, symbol: str) -> CandidateState | None:
        return self._states.get(symbol)

    def all(self) -> dict[str, CandidateState]:
        return dict(self._states)

    def observe(
        self,
        symbol: str,
        *,
        is_baseline: bool,
        eligible: bool,
        ranked: bool,
        score: float,
        now_s: float,
    ) -> CandidateState:
        cur = self._states.get(symbol)
        if ranked:
            new_state = LifecycleState.RANKED
        elif eligible:
            new_state = LifecycleState.ELIGIBLE
        else:
            new_state = LifecycleState.DISCOVERED
        # Do not regress an ACTIVE symbol back to RANKED via observe(); active
        # transitions are explicit (mark_active/mark_rotated_out).
        if cur is not None and cur.state is LifecycleState.ACTIVE:
            new_state = LifecycleState.ACTIVE
        state = CandidateState(
            symbol=symbol,
            state=new_state,
            is_baseline=is_baseline,
            last_eval_s=now_s,
            entered_active_s=cur.entered_active_s if cur else None,
            rotated_out_s=cur.rotated_out_s if cur else None,
            last_score=score,
            reason=cur.reason if cur else "",
        )
        self._states[symbol] = state
        return state

    def mark_active(self, symbol: str, now_s: float, reason: str) -> None:
        cur = self._states.get(symbol)
        base = cur.is_baseline if cur else False
        self._states[symbol] = replace(
            cur or CandidateState(symbol, LifecycleState.ACTIVE, base, now_s),
            state=LifecycleState.ACTIVE,
            entered_active_s=now_s,
            reason=reason,
            last_eval_s=now_s,
        )

    def mark_rotated_out(self, symbol: str, now_s: float, reason: str) -> None:
        cur = self._states.get(symbol)
        base = cur.is_baseline if cur else False
        self._states[symbol] = replace(
            cur or CandidateState(symbol, LifecycleState.ROTATED_OUT, base, now_s),
            state=LifecycleState.ROTATED_OUT,
            rotated_out_s=now_s,
            reason=reason,
        )

    def can_demote(self, symbol: str, now_s: float, *, force: bool = False) -> bool:
        if force:
            return True
        cur = self._states.get(symbol)
        if cur is None or cur.entered_active_s is None:
            return True
        return (now_s - cur.entered_active_s) >= self._cfg.candidate_min_residence_seconds

    def needs_revisit(self, symbol: str, now_s: float) -> bool:
        cur = self._states.get(symbol)
        if cur is None or not cur.is_baseline:
            return False
        return (now_s - cur.last_eval_s) >= self._cfg.baseline_max_revisit_seconds

    def reassess(self, symbol: str, now_s: float) -> None:
        cur = self._states.get(symbol)
        if cur is None:
            return
        self._states[symbol] = replace(
            cur, state=LifecycleState.REASSESSMENT, last_eval_s=now_s
        )

    def expire(self, symbol: str, now_s: float, reason: str) -> None:
        cur = self._states.get(symbol)
        if cur is not None and cur.is_baseline:
            raise ValueError(f"baseline symbol {symbol!r} may never EXPIRE")
        base = cur.is_baseline if cur else False
        self._states[symbol] = replace(
            cur or CandidateState(symbol, LifecycleState.EXPIRED, base, now_s),
            state=LifecycleState.EXPIRED,
            reason=reason,
            last_eval_s=now_s,
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_dynamic_universe_lifecycle.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add cengine/dynamic_universe/lifecycle.py tests/test_dynamic_universe_lifecycle.py
git commit -m "dynamic-universe: candidate lifecycle state machine (residence/revisit/no-baseline-expiry)"
```

---

### Task 5: Rotation scheduler (the 13-step cycle)

**Files:**
- Create: `cengine/dynamic_universe/rotation.py`
- Test: `tests/test_dynamic_universe_rotation.py`

**Interfaces:**
- Consumes: `UniverseRotationConfig`, `LifecycleTracker`, `LifecycleState` (Tasks 1, 4).
- Produces:
  - `RankedSymbol` frozen dataclass `(symbol: str, score: float, eligible: bool)`.
  - `RotationResult` frozen dataclass:
    `active: tuple[str,...]`, `added: tuple[str,...]`, `removed: tuple[str,...]`,
    `protected: tuple[str,...]`, `active_baseline: tuple[str,...]`,
    `active_opportunity: tuple[str,...]`, `reasons: dict[str,str]`,
    `degraded: bool`, `dropped_mandatory: tuple[str,...]`,
    `stale_baseline: tuple[str,...]` (baseline past revisit not placed this cycle).
  - `UniverseRotationScheduler(config)`:
    - `.rotate(*, baseline, opportunities, protected, references, previous_active, tracker, now_s) -> RotationResult`
      implementing the spec §6 steps: reserve protected+references, pick baseline by (not-already-active-recency i.e. staleness) + rank, pick top opportunities, dedup, fill remaining with eligible baseline, enforce `active_universe_size`, compute add/remove vs `previous_active`, record reasons; set `degraded=True` and populate `dropped_mandatory` when protected+references alone exceed the cap.
      - `baseline`: `Sequence[RankedSymbol]` (full baseline w/ current scores; may be ineligible)
      - `opportunities`: `Sequence[RankedSymbol]` (non-baseline, ranked desc)
      - `protected`: `Iterable[str]` (positions + pending + incomplete execs + unresolved recon)
      - `references`: `Iterable[str]` (mandatory reference symbols, e.g. SPY/QQQ for regime)
      - `previous_active`: `Iterable[str]`
    - Updates `tracker` (observe/mark_active/mark_rotated_out) as a side effect.

**Algorithm precedence (fixed):** protected → references → stale baseline needing revisit → top opportunities → remaining baseline by staleness+rank → fill leftovers with any eligible baseline. Everything deduped; hard cap `active_universe_size`. Protected is never dropped unless it alone exceeds the cap (then `degraded=True`, excess in `dropped_mandatory`, newest-protected kept deterministically by input order).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_dynamic_universe_rotation.py
from cengine.dynamic_universe.config import UniverseRotationConfig
from cengine.dynamic_universe.lifecycle import LifecycleTracker
from cengine.dynamic_universe.rotation import (
    UniverseRotationScheduler, RankedSymbol, RotationResult,
)

SPEC_BASELINE = [f"B{i:02d}" for i in range(30)]   # 30 synthetic baseline symbols


def _ranked(symbols, base_score=0.5):
    return [RankedSymbol(s, base_score - i * 0.001, True) for i, s in enumerate(symbols)]


def _sched(**kw):
    cfg = UniverseRotationConfig(**kw)
    return UniverseRotationScheduler(cfg), LifecycleTracker(cfg)


def test_active_never_exceeds_cap():
    sched, tracker = _sched(active_universe_size=30, baseline_target_slots=20,
                            opportunity_target_slots=10)
    opps = _ranked([f"O{i:02d}" for i in range(50)], base_score=0.9)
    res = sched.rotate(
        baseline=_ranked(SPEC_BASELINE), opportunities=opps,
        protected=[], references=[], previous_active=[], tracker=tracker, now_s=0.0,
    )
    assert isinstance(res, RotationResult)
    assert len(res.active) == 30
    assert len(set(res.active)) == 30          # unique
    assert res.degraded is False


def test_protected_always_retained():
    sched, tracker = _sched(active_universe_size=30)
    held = ["POS1", "POS2", "POS3"]            # not in baseline/opps
    res = sched.rotate(
        baseline=_ranked(SPEC_BASELINE),
        opportunities=_ranked([f"O{i:02d}" for i in range(50)], base_score=0.9),
        protected=held, references=[], previous_active=[], tracker=tracker, now_s=0.0,
    )
    for s in held:
        assert s in res.active
    assert len(res.active) == 30


def test_fills_with_baseline_when_no_opportunities():
    sched, tracker = _sched(active_universe_size=30, opportunity_target_slots=10)
    res = sched.rotate(
        baseline=_ranked(SPEC_BASELINE), opportunities=[],   # none qualified
        protected=[], references=[], previous_active=[], tracker=tracker, now_s=0.0,
    )
    assert len(res.active) == 30
    assert set(res.active) == set(SPEC_BASELINE)   # all 30 baseline fill the window


def test_strong_opportunity_promoted_over_baseline():
    sched, tracker = _sched(active_universe_size=30, baseline_target_slots=20,
                            opportunity_target_slots=10)
    strong = _ranked([f"O{i:02d}" for i in range(10)], base_score=0.99)
    res = sched.rotate(
        baseline=_ranked(SPEC_BASELINE, base_score=0.3), opportunities=strong,
        protected=[], references=[], previous_active=[], tracker=tracker, now_s=0.0,
    )
    assert set(res.active_opportunity) == {s.symbol for s in strong}
    assert len(res.active_baseline) == 20
    assert len(res.active) == 30


def test_starvation_resistance_baseline_revisit():
    # A baseline symbol past its revisit deadline must be scheduled even when
    # opportunities would otherwise fill the window.
    sched, tracker = _sched(active_universe_size=30, baseline_target_slots=20,
                            opportunity_target_slots=10, baseline_max_revisit_seconds=300.0)
    # Seed: B29 last evaluated long ago, everything else recent.
    for s in SPEC_BASELINE:
        tracker.observe(s, is_baseline=True, eligible=True, ranked=True, score=0.5, now_s=0.0)
    tracker.observe("B29", is_baseline=True, eligible=True, ranked=True, score=0.01, now_s=-1000.0)
    strong = _ranked([f"O{i:02d}" for i in range(30)], base_score=0.99)
    res = sched.rotate(
        baseline=_ranked(SPEC_BASELINE, base_score=0.02),  # all baseline weak
        opportunities=strong, protected=[], references=[],
        previous_active=[], tracker=tracker, now_s=400.0,
    )
    assert "B29" in res.active                # revisit guarantee beats weak score
    assert len(res.active) == 30


def test_degraded_when_protected_exceeds_cap():
    sched, tracker = _sched(active_universe_size=5)
    held = [f"POS{i}" for i in range(8)]       # 8 protected > cap of 5
    res = sched.rotate(
        baseline=_ranked(SPEC_BASELINE), opportunities=[], protected=held,
        references=[], previous_active=[], tracker=tracker, now_s=0.0,
    )
    assert res.degraded is True
    assert len(res.active) == 5
    assert len(res.dropped_mandatory) == 3
    # deterministic: first-5 protected kept by input order
    assert list(res.active) == held[:5]


def test_add_remove_delta_vs_previous():
    sched, tracker = _sched(active_universe_size=30)
    prev = SPEC_BASELINE[:30]
    res = sched.rotate(
        baseline=_ranked(SPEC_BASELINE),
        opportunities=_ranked(["NEWA", "NEWB"], base_score=0.99),
        protected=[], references=[], previous_active=prev, tracker=tracker, now_s=0.0,
    )
    assert set(res.added) == set(res.active) - set(prev)
    assert set(res.removed) == set(prev) - set(res.active)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_dynamic_universe_rotation.py -v`
Expected: FAIL (`ModuleNotFoundError`)

- [ ] **Step 3: Write minimal implementation**

```python
# cengine/dynamic_universe/rotation.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import universe

from .config import UniverseRotationConfig
from .lifecycle import LifecycleTracker


@dataclass(frozen=True)
class RankedSymbol:
    symbol: str
    score: float
    eligible: bool = True


@dataclass(frozen=True)
class RotationResult:
    active: tuple[str, ...]
    added: tuple[str, ...]
    removed: tuple[str, ...]
    protected: tuple[str, ...]
    active_baseline: tuple[str, ...]
    active_opportunity: tuple[str, ...]
    reasons: dict[str, str]
    degraded: bool
    dropped_mandatory: tuple[str, ...]
    stale_baseline: tuple[str, ...]


class UniverseRotationScheduler:
    """Priority-aware, starvation-resistant round-robin selection (spec §6).

    Pure and deterministic: no wall-clock, no randomness. ``now_s`` is supplied
    by the caller. Precedence (highest first):

        protected -> references -> baseline needing revisit -> top opportunities
        -> remaining baseline by staleness+rank -> fill with eligible baseline

    Everything deduped; the active window never exceeds ``active_universe_size``.
    """

    def __init__(self, config: UniverseRotationConfig) -> None:
        self._cfg = config

    def rotate(
        self,
        *,
        baseline: Sequence[RankedSymbol],
        opportunities: Sequence[RankedSymbol],
        protected: Iterable[str],
        references: Iterable[str],
        previous_active: Iterable[str],
        tracker: LifecycleTracker,
        now_s: float,
    ) -> RotationResult:
        cap = self._cfg.active_universe_size
        norm = universe.normalize_symbol

        baseline_syms = {norm(r.symbol) for r in baseline}
        reasons: dict[str, str] = {}
        active: list[str] = []
        seen: set[str] = set()

        def _add(sym: str, reason: str) -> bool:
            s = norm(sym)
            if not s or s in seen or len(active) >= cap:
                return False
            seen.add(s)
            active.append(s)
            reasons[s] = reason
            return True

        # ---- observe all candidates into the lifecycle tracker (eval stamp) ----
        for r in baseline:
            tracker.observe(norm(r.symbol), is_baseline=True, eligible=r.eligible,
                            ranked=r.eligible, score=r.score, now_s=now_s)
        for r in opportunities:
            tracker.observe(norm(r.symbol), is_baseline=False, eligible=r.eligible,
                            ranked=r.eligible, score=r.score, now_s=now_s)

        # ---- Step 1-4: reserve protected + mandatory references ----
        protected_list = [norm(s) for s in protected if norm(s)]
        references_list = [norm(s) for s in references if norm(s)]
        dropped_mandatory: list[str] = []

        mandatory = list(dict.fromkeys(protected_list + references_list))
        if len(mandatory) > cap:
            # Degraded: cannot even fit mandatory monitoring. Keep first `cap`
            # deterministically (input order = protected-before-references), the
            # rest are reported, never silently dropped.
            for s in mandatory[:cap]:
                _add(s, "protected")
            dropped_mandatory = mandatory[cap:]
            return RotationResult(
                active=tuple(active),
                added=tuple(sorted(set(active) - set(norm(x) for x in previous_active))),
                removed=tuple(sorted(set(norm(x) for x in previous_active) - set(active))),
                protected=tuple(protected_list),
                active_baseline=tuple(s for s in active if s in baseline_syms),
                active_opportunity=tuple(s for s in active if s not in baseline_syms),
                reasons=reasons,
                degraded=True,
                dropped_mandatory=tuple(dropped_mandatory),
                stale_baseline=tuple(),
            )

        for s in protected_list:
            _add(s, "protected")
        for s in references_list:
            _add(s, "reference")

        # ---- Step 5a: baseline needing revisit (starvation resistance) ----
        stale_baseline: list[str] = []
        baseline_by_sym = {norm(r.symbol): r for r in baseline}
        for sym in baseline_by_sym:
            if tracker.needs_revisit(sym, now_s):
                if not _add(sym, "baseline_revisit"):
                    if sym not in seen:
                        stale_baseline.append(sym)

        # ---- Step 6: top opportunities (respect opportunity target slot guidance) ----
        opp_sorted = sorted(
            (r for r in opportunities if r.eligible and norm(r.symbol) not in baseline_syms),
            key=lambda r: r.score, reverse=True,
        )
        opp_budget = min(self._cfg.opportunity_target_slots, cap - len(active))
        placed_opp = 0
        for r in opp_sorted:
            if placed_opp >= opp_budget or len(active) >= cap:
                break
            if _add(r.symbol, "opportunity"):
                placed_opp += 1

        # ---- Step 7-8: remaining baseline by staleness then rank ----
        def _staleness(sym: str) -> float:
            st = tracker.get(sym)
            return (now_s - st.last_eval_s) if st else now_s

        baseline_sorted = sorted(
            baseline,
            key=lambda r: (-_staleness(norm(r.symbol)), -r.score),
        )
        for r in baseline_sorted:
            if len(active) >= cap:
                break
            _add(r.symbol, "baseline")

        # ---- Step 8 (fill): any leftover slots with eligible baseline, then opps ----
        for r in baseline_sorted:
            if len(active) >= cap:
                break
            _add(r.symbol, "baseline_fill")
        for r in opp_sorted:
            if len(active) >= cap:
                break
            _add(r.symbol, "opportunity_fill")

        # ---- Step 9-10: enforce cap already done; compute add/remove delta ----
        prev = [norm(x) for x in previous_active if norm(x)]
        prev_set = set(prev)
        active_set = set(active)
        added = tuple(s for s in active if s not in prev_set)
        removed = tuple(s for s in prev if s not in active_set)

        # ---- lifecycle side effects: promotions & demotions ----
        for s in added:
            tracker.mark_active(s, now_s, reasons.get(s, "promoted"))
        for s in removed:
            tracker.mark_rotated_out(s, now_s, "rotated_out")

        return RotationResult(
            active=tuple(active),
            added=added,
            removed=removed,
            protected=tuple(protected_list),
            active_baseline=tuple(s for s in active if s in baseline_syms),
            active_opportunity=tuple(s for s in active if s not in baseline_syms),
            reasons=reasons,
            degraded=False,
            dropped_mandatory=tuple(),
            stale_baseline=tuple(stale_baseline),
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_dynamic_universe_rotation.py -v`
Expected: PASS (7 passed)

- [ ] **Step 5: Property test — active ≤ cap for random pools**

Add to `tests/test_dynamic_universe_rotation.py`:

```python
import itertools

def test_property_cap_and_uniqueness_over_many_pools():
    sched, _ = _sched(active_universe_size=30)
    for seed in range(200):
        tracker = LifecycleTracker(UniverseRotationConfig(active_universe_size=30))
        n_opp = (seed * 7) % 60
        n_prot = (seed * 3) % 6
        opps = _ranked([f"O{i:02d}" for i in range(n_opp)], base_score=0.9)
        prot = [f"P{i}" for i in range(n_prot)]
        res = sched.rotate(
            baseline=_ranked(SPEC_BASELINE), opportunities=opps, protected=prot,
            references=["SPY"], previous_active=[], tracker=tracker, now_s=float(seed),
        )
        assert len(res.active) <= 30
        assert len(set(res.active)) == len(res.active)       # unique
        for p in prot:
            assert p in res.active                           # protection held
```

Run: `python -m pytest tests/test_dynamic_universe_rotation.py -v`
Expected: PASS (8 passed)

- [ ] **Step 6: Commit**

```bash
git add cengine/dynamic_universe/rotation.py tests/test_dynamic_universe_rotation.py
git commit -m "dynamic-universe: priority-aware starvation-resistant rotation scheduler (spec §6)"
```

---

### Task 6: Package regression gate

**Files:**
- Test: (runs existing suite; no new files)

- [ ] **Step 1: Run the new package's tests together**

Run: `python -m pytest tests/test_dynamic_universe_*.py -v`
Expected: PASS (all config/baseline/layers/lifecycle/rotation tests)

- [ ] **Step 2: Run the full regression suite (no existing test regressed)**

Run: `python -m pytest -q`
Expected: PASS (pre-existing pass count + the new dynamic-universe tests). If any
pre-existing test fails, STOP and investigate before proceeding — the decision
core is pure and must not affect any existing module.

- [ ] **Step 3: Commit (if any incidental fix was required)**

```bash
git commit -am "dynamic-universe: decision-core regression gate green" || echo "nothing to commit"
```

---

## Self-Review

**Spec coverage (this plan = decision core only):**
- §2 baseline permanence → Task 2 (BaselineRegistry, config-driven, spec-30, UPRO excluded). ✓
- §3 four-layer model → Task 3 (UniverseLayers: baseline/opportunity/master/active). ✓
- §5 dynamic slot allocation + UNIVERSE_CONFIG → Task 1 (config), Task 5 (slot targets as guidance, fill-with-baseline). ✓
- §6 continuous rotation algorithm (steps 1-10, the pure subset) → Task 5. Steps 11-13 (apply subscriptions, publish atomically, repeat) are the manager/engine layer → **Plan 2**. ✓ (explicitly deferred)
- §8 opportunity lifecycle states + hysteresis/min-residence/revisit → Task 4. ✓
- §12 validation: baseline preservation, active ≤30, promotion/removal, baseline rotation, starvation prevention, insufficient candidates → Tasks 2,5 (+ property test). Remaining §12 items (protection of live positions/orders end-to-end, streaming failures, reconnection, restart recovery, ML/RL persistence, replay) depend on later plans. ✓ (scoped)

**Deferred to later plans (NOT gaps in this plan):** §4 opportunity-finder wiring (reuse scanner ranking) → Plan 2; §6 steps 11-13 + §7 live protection + §11 operational → Plan 2 (manager/integration); §9 persistence + learned policy → Plans 3 & 5; §10 observability → Plan 4; §12 replay comparison → Plan 6.

**Placeholder scan:** none — every code step contains complete code.

**Type consistency:** `RankedSymbol`/`RotationResult`/`CandidateState`/`LifecycleState`/`UniverseRotationConfig`/`UniverseLayers`/`BaselineRegistry`/`BaselineChange` names and fields are used identically across tasks. `rotate(...)` keyword params match the test call sites. `now_s: float` is threaded consistently. ✓
