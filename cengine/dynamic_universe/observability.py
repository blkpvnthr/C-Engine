"""Terminal observability for the dynamic universe manager (spec §10).

Pure rendering — takes the manager's ``dashboard_stats()`` dict and returns the
``[DYNAMIC UNIVERSE MANAGER]`` block for the existing terminal dashboard. Kept
free of engine/clock coupling so it is unit-testable and safe to call from any
dashboard consumer. ``render`` never raises on a partial/missing stats dict.
"""

from __future__ import annotations

import time
from typing import Any, Optional


def _hhmmss(epoch_s: Optional[float]) -> str:
    if not epoch_s or epoch_s <= 0:
        return "--:--:--"
    try:
        return time.strftime("%H:%M:%S", time.localtime(epoch_s))
    except (OSError, ValueError, OverflowError):
        return "--:--:--"


def render_dynamic_universe_block(stats: Optional[dict[str, Any]]) -> str:
    """Render the terminal block. Returns a disabled stub when stats is None."""
    if not stats:
        return "[DYNAMIC UNIVERSE MANAGER]\n  (disabled)"
    g = stats.get
    active = g("active", 0)
    cap = g("active_cap", 30)
    lines = [
        "[DYNAMIC UNIVERSE MANAGER]",
        f"Baseline universe:    {g('baseline', 0):>6}",
        f"Dynamic candidates:   {g('dynamic_candidates', 0):>6}",
        f"Master universe:      {g('master', 0):>6}",
        f"Active subscriptions: {active:>3}/{cap}",
        f"Protected positions:  {g('protected', 0):>6}",
        f"Active baseline:      {g('active_baseline', 0):>6}",
        f"Active opportunities: {g('active_opportunity', 0):>6}",
        f"Pending rotation:     {g('pending_rotation', 0):>6}",
        f"Last rotation:      {_hhmmss(g('last_rotation_s')):>8}",
        f"Next rotation:      {_hhmmss(g('next_rotation_s')):>8}",
        f"Coverage status:    {g('coverage', 'PENDING'):>8}",
    ]
    stale = g("stale_baseline", 0)
    if stale:
        lines.append(f"Stale baseline:       {stale:>6}  {g('stale_baseline_symbols', [])}")
    return "\n".join(lines)
