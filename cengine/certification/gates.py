"""Frozen economic certification gates (§20).

The gates a strategy must clear to be economically certified are defined ONCE, here, and
**frozen** by fingerprint. ``FROZEN_GATES_FINGERPRINT`` is an order-independent hash of the
default thresholds; :func:`verify_frozen` recomputes it and raises if it drifts. The point is
anti-retrofitting: you cannot quietly loosen a gate to make a failing strategy pass, because
any edit to the thresholds changes the fingerprint and forces a visible, deliberate update to
this constant (and a git diff that a reviewer will see). This mirrors CLAUDE.md "out-of-sample
is king / rejection over false confidence": the bar is set before the result is known.

These are deployment-grade gates applied to **net, cost-inclusive** per-decision returns on an
**out-of-sample** window. They are intentionally strict — a certification is a claim that the
evidence supports deployment, and we would rather return NOT_CERTIFIED than overclaim.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from cengine.certification.contracts import config_fingerprint

GATES_VERSION = "certification-economic-gates-1.0.0"


@dataclass(frozen=True)
class FrozenEconomicGates:
    """Mandatory economic thresholds for strategy/portfolio economic certification.

    Every field is a hard gate: a candidate violating ANY is NOT_CERTIFIED regardless of how
    strong its other metrics are. Returns are net of realistic costs; the window is OOS.
    """

    # sample sufficiency — below this, the honest verdict is INSUFFICIENT_DATA, not a pass/fail
    min_trades: int = 30
    min_effective_sample: float = 20.0
    # per-decision economic floors (net of costs)
    min_net_expectancy: float = 0.0        # strictly-positive expectancy required (> this)
    min_profit_factor: float = 1.10
    min_sharpe: float = 0.0                 # strictly-positive per-decision Sharpe required
    min_win_rate: float = 0.0              # no floor: a low-WR positive-expectancy edge is valid
    # risk ceilings
    max_drawdown: float = 0.25
    # relative gates — must out-perform the appropriate baselines on the SAME window
    must_beat_buy_and_hold: bool = True
    must_beat_random_placebo: bool = True
    # OOS discipline
    require_positive_holdout: bool = True   # the sealed-holdout result itself must clear the gates

    def fingerprint(self) -> str:
        return config_fingerprint(asdict(self))

    def to_dict(self) -> dict:
        d = asdict(self)
        d["fingerprint"] = self.fingerprint()
        d["version"] = GATES_VERSION
        return d


# The frozen fingerprint of the default gates. If you change ANY default above, this test-
# enforced constant will no longer match and `verify_frozen()` will raise — forcing a conscious,
# reviewable edit here rather than a silent loosening elsewhere.
FROZEN_GATES_FINGERPRINT = "22136c42daa0c109"


def default_gates() -> FrozenEconomicGates:
    return FrozenEconomicGates()


def verify_frozen(gates: FrozenEconomicGates | None = None) -> str:
    """Return the gates fingerprint, raising if the default gates have drifted from frozen."""
    g = gates or default_gates()
    fp = g.fingerprint()
    if gates is None and fp != FROZEN_GATES_FINGERPRINT:
        raise AssertionError(
            f"economic gates drifted from frozen fingerprint: {fp} != "
            f"{FROZEN_GATES_FINGERPRINT}. If this change is intentional, update "
            f"FROZEN_GATES_FINGERPRINT in cengine/certification/gates.py so the change is "
            f"explicit and reviewable (§20 anti-retrofitting)."
        )
    return fp


def evaluate_gates(metrics: dict, *, n_trades: int, effective_sample: float | None,
                   beats_buy_and_hold: bool | None, beats_random_placebo: bool | None,
                   gates: FrozenEconomicGates | None = None) -> dict:
    """Score a metrics dict against the frozen gates.

    Returns ``{passed, insufficient, failures, checked}``. ``insufficient`` is distinct from
    ``not passed``: if the sample floor is not met we cannot honestly render a pass/fail, so we
    flag insufficiency and the caller maps it to INSUFFICIENT_DATA.
    """
    g = gates or default_gates()
    failures: list = []

    # sample sufficiency first — gates below are only meaningful with enough evidence
    insufficient = False
    if n_trades < g.min_trades:
        insufficient = True
        failures.append(f"insufficient_trades:{n_trades}<{g.min_trades}")
    if effective_sample is not None and effective_sample < g.min_effective_sample:
        insufficient = True
        failures.append(f"insufficient_effective_sample:{effective_sample:.1f}"
                        f"<{g.min_effective_sample}")

    def _m(key: str) -> float:
        v = metrics.get(key)
        return float(v) if v is not None else 0.0

    if _m("net_expectancy") <= g.min_net_expectancy:
        failures.append(f"expectancy_not_positive:{_m('net_expectancy'):.6f}")
    if _m("profit_factor") < g.min_profit_factor:
        failures.append(f"profit_factor_below:{_m('profit_factor'):.3f}<{g.min_profit_factor}")
    if _m("sharpe") <= g.min_sharpe:
        failures.append(f"sharpe_not_positive:{_m('sharpe'):.3f}")
    if _m("win_rate") < g.min_win_rate:
        failures.append(f"win_rate_below:{_m('win_rate'):.3f}<{g.min_win_rate}")
    if abs(_m("max_drawdown")) > g.max_drawdown:
        failures.append(f"drawdown_exceeds:{abs(_m('max_drawdown')):.3f}>{g.max_drawdown}")
    if g.must_beat_buy_and_hold and beats_buy_and_hold is False:
        failures.append("does_not_beat_buy_and_hold")
    if g.must_beat_random_placebo and beats_random_placebo is False:
        failures.append("does_not_beat_random_placebo")

    passed = (not failures) and (not insufficient)
    return {"passed": passed, "insufficient": insufficient, "failures": failures,
            "gate_fingerprint": g.fingerprint()}
