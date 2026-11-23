"""Forward paper-trading certification framework (defined, inactive, non-fabricating).

Paper certification is a FORWARD exercise: it can only be earned by running the strategy on live
market data in a paper account over a genuine elapsed window and recording the realized outcomes.
This module defines the gate and ingests real paper sessions when they exist — it NEVER fabricates
prospective paper evidence, and with zero recorded sessions it reports NOT_CERTIFIED (not a pass,
not a silent skip).

Requirement to even BEGIN paper certification: a research-validated strategy (positive OOS
holdout economics). Since the current incumbent is NOT economically certified, the platform is
not yet paper-eligible — reported honestly, not worked around.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from cengine.certification.contracts import (
    DATA_LABEL_REAL,
    CertificationResult,
    CertificationVerdict,
)

PAPER_FRAMEWORK_VERSION = "certification-paper-framework-1.0.0"


@dataclass(frozen=True)
class PaperCertificationRequirements:
    min_forward_sessions: int = 30          # genuine elapsed trading sessions
    min_net_expectancy: float = 0.0         # positive realized net expectancy
    min_profit_factor: float = 1.10
    max_drawdown: float = 0.25
    require_consistency_with_backtest: bool = True   # paper must not contradict the backtest
    require_research_validated_first: bool = True     # must have a certified OOS edge to start

    def to_dict(self) -> dict:
        from dataclasses import asdict
        return asdict(self)


@dataclass
class PaperSession:
    """One genuine paper-trading session's realized outcome. Only constructed from REAL fills."""
    session_date: str
    net_return: float
    trades: int
    data_label: str = DATA_LABEL_REAL      # paper fills are realized (real), not simulated

    def to_dict(self) -> dict:
        return {"session_date": self.session_date, "net_return": self.net_return,
                "trades": self.trades, "data_label": self.data_label}


@dataclass
class PaperTradingFramework:
    """Accumulates genuine forward paper sessions and renders an honest certification verdict."""

    requirements: PaperCertificationRequirements = field(
        default_factory=PaperCertificationRequirements)
    sessions: list = field(default_factory=list)       # list[PaperSession] — REAL only
    research_validated: bool = False                    # upstream OOS economic certification

    def record_session(self, session: PaperSession) -> None:
        """Record a GENUINE paper session. Rejects anything not labeled real (no fabrication)."""
        if session.data_label != DATA_LABEL_REAL:
            raise ValueError("paper sessions must be genuine realized outcomes (data_label=real)")
        self.sessions.append(session)

    def status(self) -> CertificationResult:
        n = len(self.sessions)
        if not self.research_validated:
            return CertificationResult(
                check="paper.certification", component="paper_trading",
                verdict=CertificationVerdict.NOT_CERTIFIED,
                summary="paper certification cannot begin: no research-validated (OOS-certified) "
                        "strategy edge exists to forward-test",
                sample_size=n,
                notes="requires a certified sealed-holdout economic edge first")
        if n < self.requirements.min_forward_sessions:
            return CertificationResult(
                check="paper.certification", component="paper_trading",
                verdict=CertificationVerdict.INSUFFICIENT_DATA,
                summary=(f"paper window not elapsed: {n}/{self.requirements.min_forward_sessions} "
                         f"genuine forward sessions recorded"),
                sample_size=n, data_label=DATA_LABEL_REAL,
                notes="no prospective paper evidence is fabricated to fill the window")
        # genuine evaluation over real recorded sessions
        import numpy as np
        rets = np.array([s.net_return for s in self.sessions], dtype=float)
        expectancy = float(rets.mean())
        wins = rets[rets > 0].sum()
        losses = -rets[rets < 0].sum()
        pf = float(wins / losses) if losses > 0 else float("inf")
        # simple equity-curve drawdown
        eq = np.cumsum(rets)
        peak = np.maximum.accumulate(eq)
        dd = float(np.max(peak - eq)) if len(eq) else 0.0
        passed = (expectancy > self.requirements.min_net_expectancy
                  and pf >= self.requirements.min_profit_factor
                  and dd <= self.requirements.max_drawdown)
        return CertificationResult(
            check="paper.certification", component="paper_trading",
            verdict=CertificationVerdict.CERTIFIED if passed
            else CertificationVerdict.NOT_CERTIFIED,
            summary=(f"{n} genuine paper sessions: expectancy={expectancy:+.5f}, PF={pf:.2f}, "
                     f"DD={dd:.3f} -> {'PASS' if passed else 'FAIL'}"),
            sample_size=n, data_label=DATA_LABEL_REAL,
            metrics={"expectancy": expectancy, "profit_factor": pf, "max_drawdown": dd})


def paper_status(*, research_validated: bool = False,
                 sessions: Optional[list] = None) -> CertificationResult:
    """Convenience: current paper-certification status (defaults to inactive/NOT_CERTIFIED)."""
    fw = PaperTradingFramework(research_validated=research_validated, sessions=sessions or [])
    return fw.status()
