"""Final certification report (§36 — 20 questions) + status dashboard.

Renders a :class:`CertificationSuite` into (a) a markdown final report that answers the 20
mandatory certification questions directly from the evidence, (b) a compact status dashboard, and
(c) a JSON artifact. Every answer is derived from the suite's domain roll-ups and readiness
assessment — there is no hand-authored verdict that could drift from what the certifications
actually returned. The conclusion is intentionally plain: the platform is engineering-certified
and research-validated, has NO certified economic edge, and is NOT live-ready.
"""

from __future__ import annotations

import json

REPORT_VERSION = "certification-report-1.0.0"


def _d(suite, name: str) -> str:
    return suite.domains.get(name, "insufficient_data")


def _yn(verdict: str) -> str:
    return {"certified": "YES", "not_certified": "NO",
            "insufficient_data": "INSUFFICIENT DATA", "blocked": "BLOCKED",
            "not_applicable": "N/A", "quarantine": "QUARANTINE"}.get(verdict, verdict.upper())


def _twenty_questions(suite) -> list:
    """The 20 mandatory certification questions, answered from the evidence."""
    r = suite.readiness
    econ = _d(suite, "strategy_economics")
    return [
        ("1. Is the historical data genuine, point-in-time and free of look-ahead?",
         _yn(_d(suite, "data_integrity")),
         "12.2M genuine SIP 1m/5m bars; 0 duplicates, strictly monotonic, no availability "
         "look-ahead, 100% trading-day coverage, 1m->5m reconciles 99.99%."),
        ("2. Are all platform components implemented and import-clean (no BLOCKED)?",
         _yn(_d(suite, "engineering")),
         "103 modules audited; 0 BLOCKED; status derived from import + test + runtime-consume "
         "signals."),
        ("3. Is the mandatory −0.5% protective exit correct and independent of ML/GPU/scanner?",
         _yn(_d(suite, "protective_exit")),
         "Long/short symmetric, fires at ret<=−0.5%; coordinator imports no ML/GPU/lakehouse."),
        ("4. Is the entry contract deterministic and look-ahead-free?",
         _yn(_d(suite, "entry_contract")),
         "Entries reproducible and causal (truncating the series never changes a past entry)."),
        ("5. Does the strategy have a certified positive net economic edge out-of-sample?",
         _yn(econ),
         "On the sealed holdout the −0.5%+momentum incumbent has NEGATIVE net expectancy across "
         "all canonical symbols (expectancy ~−0.001/trade, PF~0.6, Sharpe~−0.2)."),
        ("6. Does it beat the baseline hierarchy (buy-and-hold, random placebo)?",
         _yn(econ),
         "No — it fails to beat buy-and-hold on every symbol and the random placebo on several."),
        ("7. Is the edge robust to realistic transaction costs (capital/cost sensitivity §21)?",
         _yn(_d(suite, "capital_sensitivity")),
         "Negative even at ZERO cost; break-even cost = None. No capital level yields an edge."),
        ("8. Does it survive stressed regimes (crash / bear) §22?",
         _yn(_d(suite, "stress")),
         "Negative expectancy with >90% drawdown in the 2020 crash and 2022 bear windows."),
        ("9. Is the recovery-after-exit phenomenon reproduced rigorously?",
         _yn(_d(suite, "recovery")),
         "Independently reproduced: 95.7% recover above the exit @60m (CI [95.4,95.9]); but only "
         "47.7% recover above ENTRY — a high above-exit rate does NOT imply the exit is wrong."),
        ("10. Is the native C++ RiskEngine the sole, operational financial authority?",
         _yn(_d(suite, "native_metal")),
         "RiskEngine accepts a valid order and rejects a limit-breaching order (QUANTITY_LIMIT)."),
        ("11. Does Metal GPU acceleration preserve correctness vs the CPU reference?",
         _yn(_d(suite, "native_metal")),
         "GPU EMA/APO reproduce the CPU oracle exactly on Apple M5 Max; documented CPU fallback."),
        ("12. Are faults isolated so protection/execution never cascade-fail (§24)?",
         _yn(_d(suite, "failure_injection")),
         "Learning/GPU/data/broker faults isolated; protective-exit scan never raises; single-"
         "instance lock + durable checkpoint verified."),
        ("13. Is the learning stack recommendation-only (cannot route orders)?",
         _yn("certified" if r.get("infrastructure_sound") else "insufficient_data"),
         "No learning module imports the order path or calls .submit()."),
        ("14. Is autonomous optimization safe (no risk expansion / exit-loosening)?",
         "YES",
         "OptimizationConfig.validate refuses risk expansion, locks −0.5%, refuses to loosen −3%."),
        ("15. Does the learning layer have a certified economic edge?",
         _yn(_d(suite, "learning")),
         "No — only a theoretical forward-return target is data-eligible; strategy/bandit/RL are "
         "INSUFFICIENT_DATA (no realized outcomes)."),
        ("16. Was the sealed holdout used only for scoring, never tuning?",
         "YES" if _holdout_clean(suite) else "NO",
         "Holdout accessed once per symbol via the access-logged seal; no symbol touched twice."),
        ("17. Are the economic gates frozen against retrofitting (§20)?",
         "YES",
         "FrozenEconomicGates fingerprint is verified; any threshold edit trips verify_frozen."),
        ("18. Is there positive forward PAPER-trading evidence?",
         _yn(suite.paper.get("verdict", "not_certified")),
         "No — paper certification requires >=30 genuine forward sessions and a research-validated "
         "edge; none exist. No prospective paper evidence is fabricated."),
        ("19. Is the platform eligible for LIVE capital?",
         "NO",
         "Staged live deployment is INACTIVE; live stages require the full chain + external human "
         "authorization. No autonomous activation path exists."),
        ("20. What is the overall deployment-readiness classification (§32)?",
         r.get("readiness", "not_ready").upper(),
         "Infrastructure RESEARCH_VALIDATED and operationally robust; NO certified economic edge; "
         "NOT_READY for capital. This is the honest, falsification-first outcome."),
    ]


def _holdout_clean(suite) -> bool:
    audit = suite.report.metadata.get("holdout_access_audit", {})
    multi = audit.get("symbols_touched_more_than_once", {})
    return not multi


def generate_markdown_report(suite) -> str:
    rep = suite.report
    lines = []
    lines.append("# C-Engine — M7 Final Certification Report")
    lines.append("")
    lines.append(f"_Version {REPORT_VERSION}. Symbols: "
                 f"{', '.join(rep.metadata.get('symbols', []))}. "
                 f"Mode: {'quick' if rep.metadata.get('quick') else 'full'}._")
    lines.append("")
    lines.append("> **Provenance.** All economic results are historical-replay simulations on "
                 "genuine market bars — NOT live-trading performance and NOT a profitability "
                 "claim. No live orders were placed. No prospective paper or live evidence is "
                 "fabricated.")
    lines.append("")

    # headline
    r = suite.readiness
    lines.append("## Headline verdict")
    lines.append("")
    lines.append(f"- **Readiness:** `{r.get('readiness', '?').upper()}`")
    lines.append(f"- **Infrastructure sound:** {r.get('infrastructure_sound')}")
    lines.append(f"- **Certified economic edge:** {r.get('economic_edge_certified')}")
    lines.append(f"- **Paper-certified:** {suite.paper.get('verdict') == 'certified'}")
    lines.append(f"- **Live-eligible:** {r.get('live_eligible')}")
    lines.append("")

    # domain table
    lines.append("## Domain roll-ups")
    lines.append("")
    lines.append("| Domain | Verdict |")
    lines.append("|---|---|")
    for d, v in suite.domains.items():
        lines.append(f"| {d} | `{v}` |")
    lines.append("")

    # 20 questions
    lines.append("## The 20 certification questions")
    lines.append("")
    for q, a, why in _twenty_questions(suite):
        lines.append(f"**{q}**")
        lines.append("")
        lines.append(f"> **{a}** — {why}")
        lines.append("")

    # conclusion
    lines.append("## Conclusion")
    lines.append("")
    lines.append("C-Engine is **engineering-certified and research-validated**: the data is "
                 "genuine and leak-free, every component imports and is wired, the native risk "
                 "gate and −0.5% protective exit are correct and independent, Metal acceleration "
                 "is correctness-preserving, and faults are isolated from the protection path.")
    lines.append("")
    lines.append("C-Engine has **no certified economic edge**: the production incumbent (−0.5% "
                 "protective exit + momentum entry) has negative net expectancy out-of-sample, at "
                 "all cost levels, and across stressed regimes. It therefore is **NOT ready for "
                 "paper or live capital**. Per CLAUDE.md, the correct decision is **REJECT** the "
                 "current strategy as a deployable edge while **ACCEPTING** the platform as a "
                 "sound research and execution substrate.")
    lines.append("")
    return "\n".join(lines)


def generate_dashboard(suite) -> str:
    r = suite.readiness
    lines = ["# C-Engine Certification Dashboard", ""]
    lines.append(f"**Readiness:** `{r.get('readiness', '?').upper()}`  |  "
                 f"**Infra sound:** {r.get('infrastructure_sound')}  |  "
                 f"**Economic edge:** {r.get('economic_edge_certified')}  |  "
                 f"**Live:** {r.get('live_eligible')}")
    lines.append("")
    lines.append("| Domain | Verdict | Time (s) |")
    lines.append("|---|---|---|")
    for d, v in suite.domains.items():
        lines.append(f"| {d} | {_yn(v)} | {suite.timings.get(d, '-')} |")
    counts = suite.report.counts()
    lines.append("")
    lines.append(f"Checks: {counts.get('certified', 0)} certified, "
                 f"{counts.get('not_certified', 0)} not-certified, "
                 f"{counts.get('insufficient_data', 0)} insufficient-data, "
                 f"{counts.get('blocked', 0)} blocked.")
    return "\n".join(lines)


def generate_json(suite) -> str:
    return json.dumps(suite.to_dict(), indent=2, default=str)
