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

    Candidate registration (:meth:`LifecycleTracker.observe` with
    ``stamp_eval=False``) records membership/score/state WITHOUT advancing the
    evaluation timestamp — only active placement refreshes it, so a rotated-out
    baseline symbol grows stale and the revisit guarantee eventually re-schedules
    it (starvation resistance).
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

        # ---- register candidates (membership/score/state) WITHOUT stamping eval ----
        for r in baseline:
            tracker.observe(norm(r.symbol), is_baseline=True, eligible=r.eligible,
                            ranked=r.eligible, score=r.score, now_s=now_s,
                            stamp_eval=False)
        for r in opportunities:
            tracker.observe(norm(r.symbol), is_baseline=False, eligible=r.eligible,
                            ranked=r.eligible, score=r.score, now_s=now_s,
                            stamp_eval=False)

        # ---- Step 1-4: reserve protected + mandatory references ----
        protected_list = [norm(s) for s in protected if norm(s)]
        references_list = [norm(s) for s in references if norm(s)]

        mandatory = list(dict.fromkeys(protected_list + references_list))
        if len(mandatory) > cap:
            # Degraded: cannot even fit mandatory monitoring. Keep first `cap`
            # deterministically (input order = protected-before-references), the
            # rest are reported, never silently dropped.
            for s in mandatory[:cap]:
                _add(s, "protected")
            dropped_mandatory = mandatory[cap:]
            prev_set = {norm(x) for x in previous_active if norm(x)}
            return RotationResult(
                active=tuple(active),
                added=tuple(s for s in active if s not in prev_set),
                removed=tuple(s for s in prev_set if s not in set(active)),
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

        # ---- Step 4b: hysteresis / minimum residence — retain incumbents that
        # entered the active window too recently to demote (anti-churn, spec §8).
        # An incumbent that is no longer eligible is NOT held (do not trap an
        # invalid/deteriorated symbol); emergency demotion uses the tracker's
        # force path upstream.
        eligible_now = {
            norm(r.symbol)
            for r in list(baseline) + list(opportunities)
            if r.eligible
        }
        for sym in (norm(x) for x in previous_active):
            if len(active) >= cap:
                break
            if sym in seen or sym not in eligible_now:
                continue
            if not tracker.can_demote(sym, now_s):
                _add(sym, "residence_hold")

        # ---- Step 5a: baseline needing revisit (starvation resistance) ----
        stale_baseline: list[str] = []
        baseline_by_sym = {norm(r.symbol): r for r in baseline}
        for sym in baseline_by_sym:
            if tracker.needs_revisit(sym, now_s):
                if not _add(sym, "baseline_revisit") and sym not in seen:
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

        # ---- lifecycle side effects: refresh eval for ALL active (placed this
        # cycle), stamp entries for newcomers, demote the rotated-out ----
        for s in active:
            tracker.mark_active(s, now_s, reasons.get(s, "active"))
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
