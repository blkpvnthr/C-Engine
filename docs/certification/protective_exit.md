# Protective-Exit Certification (§25)

**Milestone:** M7 Certification
**Verdict:** **CERTIFIED** (protective-exit contract) · **CERTIFIED (measurement only)** (recovery reproduction)
**CLI:** `python -m cengine.certification protective-exit`

This section certifies the mandatory `-0.5%` protective exit as a *mechanism*: that
its unrealized-return math is correct and long/short symmetric, that it fires at
exactly the right boundary, and that it is **structurally independent** of the
learning, GPU, lakehouse, and scanner layers. It also records an independent
reproduction of the exit's recovery statistics on genuine bars.

It makes **no** economic claim. The `-0.5%` exit is a **risk backstop, not an
economic edge** — strategy economics are separately **NOT_CERTIFIED** on the
sealed holdout (0/5 symbols), including at zero cost (§21) and under stress
(§22). A correct, independent backstop is necessary for safe deployment; it is not
by itself profitable.

---

## 1. The contract

The protective exit is the one piece of the strategy stack that is *mandatory* and
*non-negotiable*. Two guarantees define it:

1. **Correct, symmetric trigger.** The `ProtectiveExitCoordinator` computes
   unrealized return with long/short symmetric math and fires when that return
   reaches `-0.5%` — at the boundary it fires; just inside it holds.
2. **Independence from speculation.** The coordinator imports **no** ML / GPU /
   lakehouse / scanner module. Protection does not depend on anything that learns
   or discovers, so a fault or a bad recommendation upstream cannot reach it.

A separate native `-3%` emergency candidate exists but is **INACTIVE by default**,
and the clamp that governs it can only ever *tighten* risk, never loosen it.

---

## 2. The `-0.5%` trigger — correct and symmetric

The coordinator's unrealized-return math is **long/short symmetric**: a long
position losing 0.5% and a short position losing 0.5% are treated identically.
The fire condition is `ret <= -0.5%` — a `<=` comparison on the trigger, not a
strict `<`.

| Unrealized return | Decision | Why |
|---|---|---|
| `-0.4%` | **HOLD** | Inside the band; not yet at the trigger |
| `-0.5%` | **FIRE** | At the trigger — `ret <= -0.5%` is satisfied |

Certified properties of the trigger:

- **Symmetric** — the loss math is identical for long and short positions; neither
  side gets a wider or narrower band.
- **At-boundary firing** — `-0.5%` fires (the `<=` includes the boundary), `-0.4%`
  holds. There is no gap at the trigger where a loss at exactly `-0.5%` would be
  missed.
- **Deterministic** — the decision is a function of position side, entry, and the
  current mark; nothing stochastic enters the protective path.

---

## 3. Independence from learning, GPU, data, and discovery

The central structural guarantee of §25: **protection has no speculative
dependencies.**

The `ProtectiveExitCoordinator` **imports no ML / GPU / lakehouse / scanner
module**. This is verified as an import-level fact, not a runtime hope: the module
that decides whether to protect capital has no code path into the layers that
learn or discover.

| Layer | Imported by the coordinator? | Consequence |
|---|---|---|
| ML / learning | **No** | A learning fault or bad recommendation cannot reach protection |
| GPU / Metal | **No** | GPU unavailability never gates the protective exit |
| Lakehouse / data | **No** | An empty or broken data source cannot stall protection |
| Scanner / discovery | **No** | A discovery fault cannot cascade into protection |

This is what lets §24 (failure injection) certify that faults in learning,
discovery, GPU, or data **never cascade into protection or execution** — the
isolation is built into the import graph, so the containment is structural rather
than best-effort. It is also why the continuous-runtime P0–P9 ladder can safely
suspend research (≥P6) under GPU pressure: **P0 protection is never gated**,
because it never needed the GPU or the learner in the first place.

---

## 4. Native `-3%` emergency candidate — pure, inactive, never-loosen

Alongside the active `-0.5%` exit, the native extension exposes a `-3%`
emergency-exit threshold candidate. Its certified properties:

- **Pure** — a deterministic candidate computation with no side effects.
- **INACTIVE by default** — it does not fire on its own. The active protective
  authority is, and remains, the `-0.5%` exit. The `-3%` candidate is a wider
  emergency floor, not a replacement for the mandatory exit.
- **One-directional on risk** — `clamp_emergency_threshold` **never widens risk**.
  Its result is always `>= current`: tightening the emergency threshold is
  allowed, loosening it is **refused**.

This mirrors the safety invariant enforced across the stack (autonomous
optimization, §20 frozen gates): machinery may tighten risk, never loosen it. The
production `-0.5%` exit is locked, and the `-3%` emergency cannot be loosened by
any automated path.

```
# clamp_emergency_threshold never loosens the floor
clamped = clamp_emergency_threshold(current, proposed)
assert clamped >= current          # tighten OK, loosen refused
```

---

## 5. Recovery reproduction (measurement only)

A recurring claim about the `-0.5%` exit is that most exits "come back" — that a
large fraction of positions stopped out at `-0.5%` recover shortly afterward. §25
reproduces that measurement with an **independent reimplementation** on genuine 1m
bars, and then interprets it honestly.

### Method

- **Independent reimplementation** of the exit/recovery measurement (not a reuse of
  the production code path), run on genuine 1-minute bars.
- **20,572 exit events** evaluated.
- Two distinct recovery questions measured, with a **moving-block-bootstrap** 95%
  confidence interval on the headline rate (block bootstrap preserves the serial
  correlation in minute bars; a naive i.i.d. bootstrap would understate the
  interval).

### Results

| Measurement | Window | Rate | 95% CI (moving-block bootstrap) |
|---|---|---|---|
| Recover **above the exit level** | 60 min | **95.7%** | **[95.4%, 95.9%]** |
| Recover **above the entry price** | 60 min | **47.7%** | — |

The `95.7%` above-exit figure **reproduces the ~95.3%** result that motivated the
original claim — the measurement replicates.

### Honest interpretation

The two rows above answer different questions, and only one is decision-relevant.

- **Above-exit recovery (95.7%)** asks: after the stop fired at `-0.5%`, did price
  later trade back above that `-0.5%` level within an hour? Almost always, yes.
  But this is **not** evidence that the exit was wrong. By construction, a position
  stopped out near a local trough will usually see price tick back above the stop
  level — that says little about whether holding would have been profitable.
- **Above-entry recovery (47.7%)** asks the decision-relevant question: did price
  recover above the **original entry** within an hour? Only **47.7%** of the time —
  a coin flip, not a recovery.

A high above-exit rate does **NOT** prove the exit is economically wrong. The
above-exit measurement ignores everything that makes a stop valuable or costly:

- **Downside avoided** — the losses on the positions that did *not* recover are not
  credited to the exit.
- **Capital reuse** — capital freed by the exit can be redeployed; sitting in a
  stopped-out position has an opportunity cost the above-exit rate does not capture.
- **Opportunity cost** — "recovered above the exit level" is not the same as
  "would have been better to hold."

§25 therefore certifies the recovery statistic as a **measurement**, not as a
verdict on the exit. The exit's *economic* worth is governed by §21 (sensitivity),
§22 (stress), and the sealed-holdout strategy report — all of which remain
**NOT_CERTIFIED**.

---

## 6. What this certifies — and what it does not

**Certified (§25):**

- The `ProtectiveExitCoordinator` unrealized-return math is **long/short
  symmetric** and fires at `ret <= -0.5%` (`-0.5%` fires, `-0.4%` holds).
- The coordinator **imports no ML / GPU / lakehouse / scanner module** —
  protection is independent of learning and discovery.
- The native `-3%` emergency candidate is **pure** and **INACTIVE by default**;
  `clamp_emergency_threshold` **never widens risk** (result `>= current`;
  tightening allowed, loosening refused).
- The recovery statistic reproduces **independently**: **95.7%** above-exit at
  60 min (block-bootstrap CI **[95.4%, 95.9%]**), **47.7%** above-entry, over
  **20,572** events — a replicated measurement, honestly interpreted.

**Explicitly not claimed here:**

- **No economic edge.** The `-0.5%` exit is a risk backstop, not an edge. Strategy
  economics are **NOT_CERTIFIED** on the sealed holdout (0/5 symbols), at zero cost
  (§21), and under stress (§22, >90% drawdown in the 2020 and 2022 windows).
- **No claim that the exit is "right" or "wrong" economically.** The 95.7%
  above-exit recovery rate is a measurement, not a judgment; the decision-relevant
  above-entry rate is only 47.7%.
- **No paper or live readiness.** Readiness is **RESEARCH_VALIDATED**; live is
  **INACTIVE** and hard-gated; `live_eligible = False`.

---

## 7. How to reproduce

```
python -m cengine.certification protective-exit
```

Related sections:

- `python -m cengine.certification native` — §23 native `RiskEngine` authority and
  the pure, inactive `-3%` emergency candidate.
- `python -m cengine.certification failure` — §24 fault isolation (faults in
  learning / discovery / GPU / data never cascade into protection or execution).
- `python -m cengine.certification sensitivity` / `stress` — §21/§22 economics
  (both **NOT_CERTIFIED**), which govern whether the exit is profitable.
- `python -m cengine.certification full` / `report` — the complete chain.

---

## Provenance

All economic results referenced in this document are **historical-replay
simulations on genuine Alpaca SIP bars** — they are **not live performance** and
**not a profitability claim**. The trigger math, symmetry, independence, and
emergency-clamp results are mechanical correctness checks; the recovery statistics
are measurements on genuine bars, not performance. **No live orders were placed.**
