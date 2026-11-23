"""M8 — Autonomous alpha discovery and economic optimization.

An orchestration + research layer over the existing M1–M7 stack. It builds **no** new trading
engine, learning platform, optimizer or lakehouse — it composes the audited certification
harness (`cengine.certification`), the governed optimizer (`cengine.optimization`), the learning
stack (`cengine.learning`), the continuous runtime (`cengine.runtime`) and the native/Metal
substrate, adding:

* a **capital-constrained portfolio replay accountant** that models executable portfolio equity
  (chronological capital, concurrent positions, sizing, cash conservation, time-indexed drawdown)
  — the P0 correction to the M7 per-trade compounded "drawdown";
* governed **alpha-discovery research** (entry-quality ablation, multi-timeframe, regime
  conditioning, strategy-family comparison, supervised opportunity filtering) driven by genuine
  economic performance;
* an honest **validation protocol** that treats the consumed M7 holdout as spent and never lets
  development experiments touch independent certification evidence.

The guiding principle is unchanged from CLAUDE.md: falsification-first, net-performance-only,
out-of-sample-is-king, no fabrication. A positive economic result is a research objective, not a
guaranteed deliverable; negative results are preserved, not reinterpreted.
"""

ALPHA_VERSION = "alpha-discovery-1.0.0"

__all__ = ["ALPHA_VERSION"]
