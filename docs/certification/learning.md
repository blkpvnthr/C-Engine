# C-Engine M7 Certification — Learning

**Scope:** supervised ML, contextual bandits, offline RL, and autonomous optimization.
**Verdict:** engineering **CERTIFIED**; provenance discipline **CERTIFIED**; recommendation-only **CERTIFIED**; autonomous-optimization safety **CERTIFIED**; data-eligibility **INSUFFICIENT_DATA**; bandit/RL **INSUFFICIENT_DATA**.

The governed optimization loop **retains the incumbent**: it certified **no economic improvement** — the correct falsification-first outcome.

**CLI:** `python -m cengine.certification learning`

---

## 1. Summary

The learning subsystem is engineered correctly, isolated correctly, and governed correctly. Every structural guarantee a learning stack must satisfy before it is allowed anywhere near capital is in place and verified. What is **not** in place is the one thing no amount of engineering can substitute for: **materialized, genuine, realized outcomes** on which a learner could be trained and validated.

Consequently the learning certification splits cleanly:

| Dimension | Verdict | Meaning |
| --- | --- | --- |
| Learning infrastructure | **CERTIFIED** | Pipelines, interfaces, governance wired and tested |
| Provenance discipline | **CERTIFIED** | Synthetic/replay data can never back a profitability claim |
| Recommendation-only isolation | **CERTIFIED** | No learning module can place or influence an order |
| Autonomous-optimization safety | **CERTIFIED** | Optimizer cannot expand risk; locks the protective exits |
| Data-eligibility | **INSUFFICIENT_DATA** | Only a theoretical target is validation-eligible |
| Bandit / offline RL | **INSUFFICIENT_DATA** | No realized-outcome layers; no live sessions |

This is an honest ledger. The learning stack is **research-validated substrate**, not a certified source of edge. No learning component has produced, or is permitted to produce, a profitability claim.

---

## 2. Platform Context

- Local-first native quant stack on **Apple M5 Max** (48 GiB unified memory), **Python 3.11**, **MLX/Metal** GPU.
- Data at certification time: **12.2M genuine Alpaca SIP bars** in a local Parquet + DuckDB lakehouse.
  - 1Min: 6.76M bars / 5 canonical symbols (AAPL, MSFT, NVDA, SPY, QQQ), 2020-01-01 .. 2026-11-21.
  - 5Min: 5.45M bars / 31 symbols.
  - `price_scale = 10000` ticks.
- Data integrity (M7B) is independently **CERTIFIED**: no duplicate bars, strictly monotonic timestamps, valid OHLC, non-negative volume, and **no look-ahead** (`availability_ts >= event_ts`). The learning stack therefore trains on leak-free, availability-correct bars — a necessary (not sufficient) precondition.

All economic figures referenced anywhere in this document are **historical-replay simulations on genuine bars**, not live performance.

---

## 3. Learning Infrastructure — CERTIFIED

The infrastructure certification asserts that the learning pipeline exists, is wired end to end, and is exercised by tests — **independently of whether any task is currently trainable**. Certifying the plumbing is distinct from certifying that there is water in the pipe; we certify only the plumbing here.

**What is certified:**

- Supervised-learning, contextual-bandit, and offline-RL task interfaces are defined and importable under the isolated-subprocess engineering audit (M7A: 103 modules audited, 0 BLOCKED).
- Feature/target assembly draws from the certified lakehouse (leak-free, availability-correct bars).
- The optimization configuration path (`OptimizationConfig`) and the governed loop are present and test-covered.
- Off-session, the **ContinuousOptimizationService** runs **shadow-only** — it may propose, never dispose. It is scheduled below protection on the runtime GPU-priority ladder; research work (priority ≥ P6) is suspended under GPU pressure and bounded in-session, while **P0 protection is never gated**.

**What infrastructure certification deliberately does NOT assert:**

- That any learning task has sufficient data to train or validate (see §6).
- That any learned policy has an edge (it does not — see §7).
- That a learner may act (it may not — see §5).

---

## 4. Provenance Discipline — CERTIFIED

Provenance is the firewall between simulation and truth. Every result carries an origin tag, and the certification verifies that the origin controls what the result is allowed to claim.

| Provenance | May back a profitability claim? |
| --- | --- |
| `synthetic` | **No** |
| `historical_replay` | **No** |
| `real` (genuine realized outcomes) | **Yes** |

- Synthetic and historical-replay results **can never** back a profitability claim. Only genuine `real` realized outcomes can.
- This is enforced as a discipline across the learning stack, not left to convention: a replay-sourced metric is structurally barred from being promoted into an economic claim.
- Because there are currently **no materialized realized-outcome layers** and **no live sessions**, there is at present **no `real`-provenance learning evidence in existence**. By construction, the learning stack therefore cannot assert profitability today — and does not.

This is why "learning is CERTIFIED" never contradicts "there is no certified edge": the provenance rule makes the former incapable of implying the latter.

---

## 5. Recommendation-Only Isolation — CERTIFIED

Learning is advisory. It can inform, never execute.

- **No learning module imports the order path, and none calls `.submit()`.** This is verified structurally, not promised.
- The learning subsystem is strictly upstream of the execution boundary. Its outputs are recommendations consumed (or ignored) by governed components; they are never orders.
- This composes with the broader failure-isolation guarantee (§24): **faults in learning / discovery / GPU / data never cascade into protection or execution**. A learning component can crash, hang, or emit garbage without placing a single share at risk.
- The sole financial authority remains the **native `RiskEngine`** (accepts a valid 10-share order; rejects a 50k-share order with `QUANTITY_LIMIT_EXCEEDED`), which imports no ML/GPU/lakehouse/scanner module. Learning sits entirely outside this authority.

---

## 6. Autonomous-Optimization Safety — CERTIFIED

The optimizer is allowed to search, but not to make the system more dangerous. `OptimizationConfig.validate` enforces a one-way ratchet on risk:

| Candidate change | Optimizer verdict |
| --- | --- |
| Expand autonomous risk | **Refused** |
| Loosen the -0.5% production protective exit | **Refused (locked)** |
| Loosen the -3% native emergency threshold | **Refused** |
| Tighten the -0.5% exit / -3% emergency | **Allowed** |

- Risk can only be held constant or **tightened**, never widened, by any autonomous process.
- This mirrors the native-side invariant: `clamp_emergency_threshold` never widens risk — the result is always at least as tight as the current setting; tightening is allowed, loosening refused.
- The protective exits the optimizer is forbidden to touch are themselves certified (§25): the `ProtectiveExitCoordinator` fires at `ret <= -0.5%` (fires at -0.5%, holds at -0.4%), is long/short symmetric, and imports no ML/GPU/lakehouse/scanner module — protection is independent of learning and discovery.

The optimizer therefore cannot "learn its way" into a riskier posture. Its degrees of freedom are bounded by frozen safety invariants it has no authority to relax.

---

## 7. Governed Loop Outcome — Incumbent Retained

The governed optimization loop ran and produced the scientifically correct result: **it certified no economic improvement and retained the incumbent.**

- The production incumbent (mandatory -0.5% protective exit + momentum entry) has **negative net expectancy** on the sealed holdout for **every** canonical symbol (expectancy ≈ -0.001/trade, profit factor ≈ 0.6, Sharpe ≈ -0.2, drawdown > 90%). It fails to beat buy-and-hold on every symbol and the random placebo on 3/5.
- Against a **frozen** economic bar — `FrozenEconomicGates`: expectancy > 0, PF ≥ 1.10, Sharpe > 0, drawdown ≤ 25%, must beat buy-and-hold **and** random placebo — the optimizer found **no candidate** that cleared the bar.
- The gates are frozen by fingerprint; `verify_frozen()` trips on any threshold drift (anti-retrofitting). The optimizer cannot move the goalposts to manufacture a "winner."
- Retaining the incumbent when nothing beats it is **falsification-first behaving correctly**: the burden of proof is on the challenger, and no challenger met it.

> Retaining the incumbent is not a bug or a null result to be explained away — it is the loop refusing to promote a strategy that did not earn promotion.

---

## 8. Data-Eligibility — INSUFFICIENT_DATA

This is the binding constraint on the entire learning program.

| Task class | Eligibility |
| --- | --- |
| Theoretical `forward_return` target | **Validation-eligible** (1 task) |
| 6 strategy-relevant / bandit / RL tasks | **INSUFFICIENT_DATA** |

- Only a **theoretical `forward_return` target** is currently validation-eligible.
- The **6 strategy-relevant / bandit / RL tasks are INSUFFICIENT_DATA** because there are **no materialized realized-outcome layers** and **no live sessions** from which to construct genuine labels or reward signals.
- A forward-return target computed from bars is a *theoretical* target — it is not a realized trading outcome and, per the provenance rule (§4), cannot be promoted into a profitability claim.

Without realized outcomes there is nothing of `real` provenance to train the strategy-relevant learners on, and nothing to validate them against. This is reported as **INSUFFICIENT_DATA**, not as failure and not as fabricated capacity.

---

## 9. Bandit / Offline RL — INSUFFICIENT_DATA

- Contextual bandits and offline RL both require **logged decisions with realized rewards** (bandit) or **trajectories with realized returns** (offline RL).
- Neither exists: there are **no materialized realized-outcome layers and no live sessions**.
- Accordingly, `bandit_rl` is **INSUFFICIENT_DATA**. The interfaces are present and audited (§3), but there is no genuine data to populate them.
- No synthetic or historical-replay substitute is permitted to stand in for the missing realized data (§4). We report the gap rather than paper over it.

---

## 10. What Would Move These Verdicts

The path from INSUFFICIENT_DATA to a trainable (and only then, possibly certifiable) learning program is gated, not fabricated:

1. Accumulate **genuine realized-outcome layers** and **live/forward sessions** of `real` provenance.
2. Materialize bandit reward logs / RL trajectories from those realized outcomes.
3. Re-run the governed loop; any candidate must clear the **frozen economic gates** and beat buy-and-hold **and** the random placebo.
4. Paper certification additionally requires **≥ 30 genuine elapsed forward sessions** *and* a research-validated edge — **none exist today**, and no prospective paper evidence is fabricated.
5. Live remains **INACTIVE** and hard-gated behind the full chain plus an **external human authorization token the software cannot mint**. There is no autonomous activation path.

Until then, the honest verdicts stand: learning infrastructure and governance are **CERTIFIED**; the data to learn an edge is **INSUFFICIENT**.

---

## 11. Conclusion

- Learning **infrastructure**, **provenance discipline**, **recommendation-only isolation**, and **autonomous-optimization safety** are **CERTIFIED**.
- **Data-eligibility** and **bandit/RL** are **INSUFFICIENT_DATA**: only a theoretical forward-return target is validation-eligible; there are no realized-outcome layers and no live sessions.
- The governed loop **retained the incumbent**, certifying **no economic improvement** — correct falsification-first behavior against frozen, fingerprinted gates.
- Consistent with overall M7 readiness (**RESEARCH_VALIDATED**, `live_eligible=False`): the learning stack is a **sound research substrate with no certified economic edge**, not a deployable alpha source.

---

### Provenance Note

All economic results referenced in this document are **historical-replay simulations on genuine bars** — **not** live performance and **not** a profitability claim. **No live orders were placed.**
