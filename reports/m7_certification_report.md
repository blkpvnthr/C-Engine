# C-Engine — M7 Final Certification Report

_Version certification-report-1.0.0. Symbols: AAPL, MSFT, NVDA, SPY, QQQ. Mode: full._

> **Provenance.** All economic results are historical-replay simulations on genuine market bars — NOT live-trading performance and NOT a profitability claim. No live orders were placed. No prospective paper or live evidence is fabricated.

## Headline verdict

- **Readiness:** `RESEARCH_VALIDATED`
- **Infrastructure sound:** True
- **Certified economic edge:** False
- **Paper-certified:** False
- **Live-eligible:** False

## Domain roll-ups

| Domain | Verdict |
|---|---|
| engineering | `certified` |
| data_integrity | `certified` |
| native_metal | `certified` |
| failure_injection | `certified` |
| protective_exit | `certified` |
| entry_contract | `certified` |
| execution | `insufficient_data` |
| learning | `insufficient_data` |
| strategy_economics | `not_certified` |
| recovery | `certified` |
| capital_sensitivity | `not_certified` |
| stress | `not_certified` |
| portfolio | `not_certified` |

## The 20 certification questions

**1. Is the historical data genuine, point-in-time and free of look-ahead?**

> **YES** — 12.2M genuine SIP 1m/5m bars; 0 duplicates, strictly monotonic, no availability look-ahead, 100% trading-day coverage, 1m->5m reconciles 99.99%.

**2. Are all platform components implemented and import-clean (no BLOCKED)?**

> **YES** — 103 modules audited; 0 BLOCKED; status derived from import + test + runtime-consume signals.

**3. Is the mandatory −0.5% protective exit correct and independent of ML/GPU/scanner?**

> **YES** — Long/short symmetric, fires at ret<=−0.5%; coordinator imports no ML/GPU/lakehouse.

**4. Is the entry contract deterministic and look-ahead-free?**

> **YES** — Entries reproducible and causal (truncating the series never changes a past entry).

**5. Does the strategy have a certified positive net economic edge out-of-sample?**

> **NO** — On the sealed holdout the −0.5%+momentum incumbent has NEGATIVE net expectancy across all canonical symbols (expectancy ~−0.001/trade, PF~0.6, Sharpe~−0.2).

**6. Does it beat the baseline hierarchy (buy-and-hold, random placebo)?**

> **NO** — No — it fails to beat buy-and-hold on every symbol and the random placebo on several.

**7. Is the edge robust to realistic transaction costs (capital/cost sensitivity §21)?**

> **NO** — Negative even at ZERO cost; break-even cost = None. No capital level yields an edge.

**8. Does it survive stressed regimes (crash / bear) §22?**

> **NO** — Negative expectancy with >90% drawdown in the 2020 crash and 2022 bear windows.

**9. Is the recovery-after-exit phenomenon reproduced rigorously?**

> **YES** — Independently reproduced: 95.7% recover above the exit @60m (CI [95.4,95.9]); but only 47.7% recover above ENTRY — a high above-exit rate does NOT imply the exit is wrong.

**10. Is the native C++ RiskEngine the sole, operational financial authority?**

> **YES** — RiskEngine accepts a valid order and rejects a limit-breaching order (QUANTITY_LIMIT).

**11. Does Metal GPU acceleration preserve correctness vs the CPU reference?**

> **YES** — GPU EMA/APO reproduce the CPU oracle exactly on Apple M5 Max; documented CPU fallback.

**12. Are faults isolated so protection/execution never cascade-fail (§24)?**

> **YES** — Learning/GPU/data/broker faults isolated; protective-exit scan never raises; single-instance lock + durable checkpoint verified.

**13. Is the learning stack recommendation-only (cannot route orders)?**

> **YES** — No learning module imports the order path or calls .submit().

**14. Is autonomous optimization safe (no risk expansion / exit-loosening)?**

> **YES** — OptimizationConfig.validate refuses risk expansion, locks −0.5%, refuses to loosen −3%.

**15. Does the learning layer have a certified economic edge?**

> **INSUFFICIENT DATA** — No — only a theoretical forward-return target is data-eligible; strategy/bandit/RL are INSUFFICIENT_DATA (no realized outcomes).

**16. Was the sealed holdout used only for scoring, never tuning?**

> **YES** — Holdout accessed once per symbol via the access-logged seal; no symbol touched twice.

**17. Are the economic gates frozen against retrofitting (§20)?**

> **YES** — FrozenEconomicGates fingerprint is verified; any threshold edit trips verify_frozen.

**18. Is there positive forward PAPER-trading evidence?**

> **NO** — No — paper certification requires >=30 genuine forward sessions and a research-validated edge; none exist. No prospective paper evidence is fabricated.

**19. Is the platform eligible for LIVE capital?**

> **NO** — Staged live deployment is INACTIVE; live stages require the full chain + external human authorization. No autonomous activation path exists.

**20. What is the overall deployment-readiness classification (§32)?**

> **RESEARCH_VALIDATED** — Infrastructure RESEARCH_VALIDATED and operationally robust; NO certified economic edge; NOT_READY for capital. This is the honest, falsification-first outcome.

## Conclusion

C-Engine is **engineering-certified and research-validated**: the data is genuine and leak-free, every component imports and is wired, the native risk gate and −0.5% protective exit are correct and independent, Metal acceleration is correctness-preserving, and faults are isolated from the protection path.

C-Engine has **no certified economic edge**: the production incumbent (−0.5% protective exit + momentum entry) has negative net expectancy out-of-sample, at all cost levels, and across stressed regimes. It therefore is **NOT ready for paper or live capital**. Per CLAUDE.md, the correct decision is **REJECT** the current strategy as a deployable edge while **ACCEPTING** the platform as a sound research and execution substrate.
