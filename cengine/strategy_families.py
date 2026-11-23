"""Five-family adaptive strategy architecture (milestone §7-13).

Consolidates existing and new strategy capabilities under five families, each with
explicit metadata (entry conditions, regime compatibility, data/indicator needs,
holding horizon, invalidation, estimated costs, liquidity + exposure requirements,
versions) and an EXPLICIT eligibility state. Existing proven strategies are MAPPED
to families (not duplicated); the one missing family (MULTI_DAY_SWING) is new and
ships RESEARCH-only.

Safety:
  * A family/strategy is NEVER automatically EXECUTION-eligible. New or unvalidated
    capabilities default to RESEARCH; promotion to EXECUTION is an explicit, gated
    decision (``mark_execution_eligible``) — this module never flips it on its own.
  * The strict elective-long entry contract (§13) is implemented ONCE here with
    explicit parentheses and mirrors the native ``screen_market`` kernel contract;
    it is a precondition every family evaluator must pass before an executable
    recommendation is produced.

The family evaluators are deterministic CONDITION gates over native feature rows +
a regime assessment; they emit advisory :class:`StrategyRecommendation` objects.
They do not place orders and do not bypass the exposure registry or native
RiskEngine — those remain the execution authority.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from cengine.regime_engine import RegimeDirection, TrendStructure
from cengine.strategy_maturity import MaturityState

FEATURE_VERSION = "features-1.0.0"


class StrategyFamily(str, Enum):
    MOMENTUM_CONTINUATION = "momentum_continuation"
    VOLATILITY_BREAKOUT = "volatility_breakout"
    TREND_PULLBACK = "trend_pullback"
    MEAN_REVERSION = "mean_reversion"
    MULTI_DAY_SWING = "multi_day_swing"


class EligibilityState(str, Enum):
    DISABLED = "disabled"
    RESEARCH = "research"        # evaluated + logged, never routed to execution
    EXECUTION = "execution"      # explicitly validated + enabled for elective entry


class ExposureDirection(str, Enum):
    LONG = "long"
    SHORT = "short"
    NEUTRAL = "neutral"


class HoldingHorizon(str, Enum):
    INTRADAY = "intraday"
    SWING = "multi_day"


@dataclass
class StrategyFamilySpec:
    family: StrategyFamily
    entry_conditions: list[str]
    compatible_directions: set              # RegimeDirection values the family favors
    compatible_trends: set                  # TrendStructure values the family favors
    required_indicators: list[str]
    holding_horizon: HoldingHorizon
    invalidation_conditions: list[str]
    est_transaction_cost_bps: float
    min_rel_volume: float
    exposure_requirement: str               # human-readable exposure note
    maps_to: str                            # existing strategy or "NEW"
    strategy_version: str
    config_version: str
    eligibility: EligibilityState           # default posture (see safety note)
    has_regression_tests: bool
    description: str = ""


# --- family specifications --------------------------------------------------
# Existing strategies are MAPPED (reused), not reimplemented. MULTI_DAY_SWING is
# new and ships RESEARCH-only. Default eligibility is conservative: only families
# backed by regression tests AND already order-path-proven may be EXECUTION by
# default; everything else is RESEARCH until explicitly validated + enabled.
FAMILY_SPECS: dict[StrategyFamily, StrategyFamilySpec] = {
    StrategyFamily.MOMENTUM_CONTINUATION: StrategyFamilySpec(
        family=StrategyFamily.MOMENTUM_CONTINUATION,
        entry_conditions=[
            "DI+5m > DI-5m (directional)",
            "price above >=1 valid reference (VWAP / SMA20 / BB-mid)",
            "developing 1m > previous 1m close",
            "ADX >= trend threshold (structure present)",
            "non-redundant momentum: rel-volume OR momentum acceleration OR HTF agreement",
        ],
        compatible_directions={RegimeDirection.BULLISH},
        compatible_trends={TrendStructure.PERSISTENT, TrendStructure.EXPANSION,
                           TrendStructure.RECOVERY},
        required_indicators=["pdi", "mdi", "adx", "vwap", "sma20", "bb_mid",
                             "short_mom", "rel_volume"],
        holding_horizon=HoldingHorizon.INTRADAY,
        invalidation_conditions=["DI+ <= DI-", "price loses all references",
                                 "1m momentum turns negative"],
        est_transaction_cost_bps=6.0, min_rel_volume=0.8,
        exposure_requirement="long (common equity / broad / sector; geared via exposure auth)",
        maps_to="DualMAStrategy (momentum continuation; order-path proven)",
        strategy_version="momcont-1.0.0", config_version="momcont-cfg-1",
        eligibility=EligibilityState.RESEARCH, has_regression_tests=True,
        description="Continuation of an established intraday uptrend."),

    StrategyFamily.VOLATILITY_BREAKOUT: StrategyFamilySpec(
        family=StrategyFamily.VOLATILITY_BREAKOUT,
        entry_conditions=[
            "Donchian/opening-range breakout of prior channel high",
            "ATR expansion OR Bollinger-compression release",
            "relative volume confirmation",
            "developing 1m > previous 1m close (momentum confirmation)",
            "strict entry contract satisfied",
        ],
        compatible_directions={RegimeDirection.BULLISH},
        compatible_trends={TrendStructure.EXPANSION, TrendStructure.COMPRESSION,
                           TrendStructure.PERSISTENT},
        required_indicators=["donchian_high", "atr_pct", "bb_width", "rel_volume",
                             "pdi", "mdi", "vwap"],
        holding_horizon=HoldingHorizon.INTRADAY,
        invalidation_conditions=["close back inside the broken channel (false breakout)",
                                 "volume fails to confirm", "ATR contracts"],
        est_transaction_cost_bps=7.0, min_rel_volume=1.0,
        exposure_requirement="long (common equity / broad / sector; geared via exposure auth)",
        maps_to="TurtleStrategy (Donchian breakout; order-path proven)",
        strategy_version="volbreak-1.0.0", config_version="volbreak-cfg-1",
        eligibility=EligibilityState.RESEARCH, has_regression_tests=True,
        description="Breakout on volatility expansion from compression/range."),

    StrategyFamily.TREND_PULLBACK: StrategyFamilySpec(
        family=StrategyFamily.TREND_PULLBACK,
        entry_conditions=[
            "higher-timeframe trend is up (HTF agreement)",
            "price pulled back toward VWAP/SMA then RECOVERING",
            "seller exhaustion OR decelerating negative momentum",
            "positive momentum recovery confirmed",
            "strict entry contract satisfied on recovery",
        ],
        compatible_directions={RegimeDirection.BULLISH},
        compatible_trends={TrendStructure.RECOVERY, TrendStructure.PERSISTENT,
                           TrendStructure.RANGE_BOUND},
        required_indicators=["htf_direction", "vwap", "sma20", "short_mom",
                             "momentum_accel", "pdi", "mdi"],
        holding_horizon=HoldingHorizon.INTRADAY,
        invalidation_conditions=["HTF trend flips down", "no recovery confirmation",
                                 "price breaks support"],
        est_transaction_cost_bps=6.0, min_rel_volume=0.6,
        exposure_requirement="long (common equity / broad / sector)",
        maps_to="DIBBStrategy (DI/BB pullback; has tests, OFF by default)",
        strategy_version="pullback-1.0.0", config_version="pullback-cfg-1",
        eligibility=EligibilityState.RESEARCH, has_regression_tests=True,
        description="Buy a recovery within an established uptrend; requires confirmation."),

    StrategyFamily.MEAN_REVERSION: StrategyFamilySpec(
        family=StrategyFamily.MEAN_REVERSION,
        entry_conditions=[
            "volatility-adjusted displacement below the mean (BB lower / RSI / CCI)",
            "at/near support",
            "seller exhaustion",
            "POSITIVE reversal confirmation (never buy solely because oversold)",
            "strict entry contract satisfied on the reversal",
        ],
        compatible_directions={RegimeDirection.NEUTRAL, RegimeDirection.BULLISH},
        compatible_trends={TrendStructure.RANGE_BOUND, TrendStructure.RECOVERY,
                           TrendStructure.EXHAUSTION},
        required_indicators=["bb_dist", "rsi", "cci", "short_mom", "momentum_accel"],
        holding_horizon=HoldingHorizon.INTRADAY,
        invalidation_conditions=["no reversal confirmation", "support breaks",
                                 "displacement keeps extending"],
        est_transaction_cost_bps=6.0, min_rel_volume=0.5,
        exposure_requirement="long (common equity / broad / sector)",
        maps_to="APOMeanReversionStrategy (recovery-confirmed variant; order-path proven)",
        strategy_version="meanrev-1.0.0", config_version="meanrev-cfg-1",
        eligibility=EligibilityState.RESEARCH, has_regression_tests=True,
        description="Fade an exhausted displacement ONLY on confirmed positive reversal."),

    StrategyFamily.MULTI_DAY_SWING: StrategyFamilySpec(
        family=StrategyFamily.MULTI_DAY_SWING,
        entry_conditions=[
            "15m + hourly + daily trends aligned up",
            "relative sector strength",
            "directional persistence",
            "acceptable overnight exposure + no adverse known corporate event",
            "economically viable AFTER the mandatory -0.5% protective exit",
        ],
        compatible_directions={RegimeDirection.BULLISH},
        compatible_trends={TrendStructure.PERSISTENT},
        required_indicators=["trend_15m", "trend_1h", "trend_1d", "sector_rel_strength",
                             "realized_vol"],
        holding_horizon=HoldingHorizon.SWING,
        invalidation_conditions=["any timeframe trend flips down",
                                 "sector relative strength deteriorates",
                                 "not viable under -0.5% protective exit"],
        est_transaction_cost_bps=8.0, min_rel_volume=0.8,
        exposure_requirement="long, overnight-capable (common equity / broad / sector)",
        maps_to="NEW (not previously implemented)",
        strategy_version="swing-0.1.0", config_version="swing-cfg-0",
        eligibility=EligibilityState.RESEARCH, has_regression_tests=True,
        description="Multi-day swing; viability under the intraday -0.5% stop must be "
                    "demonstrated before any execution eligibility."),
}


# --- strict elective-long entry contract (§13) ------------------------------
def passes_entry_contract(row: dict) -> tuple[bool, str]:
    """The strict elective-long entry contract, with EXPLICIT parentheses around the
    alternative price-reference clause (§13):

        DI+5m > DI-5m
        AND ( price > VWAP  OR  price > SMA20_5m  OR  price > BB_middle_5m )
        AND developing_1m_price > previous_completed_1m_close

    This mirrors the native ``screen_market`` SCREEN_ENTRY_QUALIFIED contract.
    Applies to all elective longs INCLUDING inverse ETFs (the contract is on the
    traded instrument's own price action; underlying-exposure authorization is a
    separate exposure-registry concern)."""
    di_plus = row.get("di_plus_5m", row.get("pdi"))
    di_minus = row.get("di_minus_5m", row.get("mdi"))
    price = row.get("price")
    vwap = row.get("vwap")
    sma20 = row.get("sma20_5m", row.get("sma20"))
    bb_mid = row.get("bb_middle_5m", row.get("bb_mid"))
    dev_1m = row.get("developing_1m_price", row.get("developing_1m"))
    prev_1m = row.get("previous_completed_1m_close", row.get("prev_1m_close"))

    if di_plus is None or di_minus is None or price is None or dev_1m is None \
            or prev_1m is None:
        return False, "insufficient_observations"
    if not (di_plus > di_minus):
        return False, "di_plus_not_greater_than_di_minus"
    ref_ok = ((vwap is not None and price > vwap)
              or (sma20 is not None and price > sma20)
              or (bb_mid is not None and price > bb_mid))
    if not ref_ok:
        return False, "price_below_all_valid_references"
    if not (dev_1m > prev_1m):
        return False, "developing_1m_not_above_previous_1m_close"
    return True, "entry_contract_satisfied"


# --- family registry --------------------------------------------------------
class StrategyFamilyRegistry:
    """Holds the five family specs + eligibility + maturity. Never auto-enables
    execution.

    Two INDEPENDENT dimensions are tracked per family:

    * ``eligibility`` (:class:`EligibilityState`) — governs the ADVISORY arbitration
      path; ``EXECUTION`` means authorized live elective entry and is default-off.
    * ``maturity`` (:class:`MaturityState`) — governs the EXECUTION BRIDGE and, in
      particular, isolated mock-broker authorization. Mock maturity is deliberately
      separate from ``EXECUTION`` eligibility so exercising the mock path can never
      grant paper/live routing (directive §4).
    """

    def __init__(self, specs: Optional[dict] = None) -> None:
        # deep-ish copy so per-instance eligibility changes don't leak globally
        self._specs = {f: _copy_spec(s) for f, s in (specs or FAMILY_SPECS).items()}
        # Maturity defaults to RESEARCH for every family; promotion is explicit.
        self._maturity: dict = {f: MaturityState.RESEARCH for f in self._specs}

    def spec(self, family: StrategyFamily) -> StrategyFamilySpec:
        return self._specs[family]

    def families(self) -> list[StrategyFamily]:
        return list(self._specs)

    def eligibility(self, family: StrategyFamily) -> EligibilityState:
        return self._specs[family].eligibility

    def is_execution_eligible(self, family: StrategyFamily) -> bool:
        return self._specs[family].eligibility == EligibilityState.EXECUTION

    def mark_execution_eligible(self, family: StrategyFamily, *, validated: bool) -> bool:
        """Explicitly promote a family to EXECUTION — ONLY if it has regression tests
        AND the caller asserts it has been validated. Returns the new state's
        eligibility. This is never called automatically."""
        spec = self._specs[family]
        if validated and spec.has_regression_tests:
            spec.eligibility = EligibilityState.EXECUTION
        return self.is_execution_eligible(family)

    def disable(self, family: StrategyFamily) -> None:
        self._specs[family].eligibility = EligibilityState.DISABLED

    # --- maturity dimension (execution bridge; independent of eligibility) ----
    def maturity(self, family: StrategyFamily) -> MaturityState:
        return self._maturity[family]

    def set_maturity(self, family: StrategyFamily, state: MaturityState) -> None:
        """Explicitly set a family's maturity. Never called automatically."""
        self._maturity[family] = state

    def mark_mock_execution_eligible(self, family: StrategyFamily, *,
                                     test_authorization: bool) -> MaturityState:
        """Grant ISOLATED mock-execution authorization — ONLY if the family has
        regression tests AND the caller explicitly asserts test authorization.

        This NEVER implies paper/live eligibility (see
        :func:`cengine.strategy_maturity.authorization_permits`) and NEVER touches
        the advisory ``EXECUTION`` eligibility state. A BLOCKED family is never
        promoted. Returns the resulting maturity.
        """
        spec = self._specs[family]
        if (test_authorization and spec.has_regression_tests
                and self._maturity[family] is not MaturityState.BLOCKED):
            self._maturity[family] = MaturityState.MOCK_EXECUTION
        return self._maturity[family]

    def mark_paper_eligible(self, family: StrategyFamily, *,
                            validated: bool) -> MaturityState:
        """Promote to PAPER_ELIGIBLE only with regression tests + explicit validation."""
        spec = self._specs[family]
        if (validated and spec.has_regression_tests
                and self._maturity[family] is not MaturityState.BLOCKED):
            self._maturity[family] = MaturityState.PAPER_ELIGIBLE
        return self._maturity[family]

    def mark_live_eligible(self, family: StrategyFamily, *,
                           validated: bool) -> MaturityState:
        """Promote to LIVE_ELIGIBLE only with regression tests + explicit validation."""
        spec = self._specs[family]
        if (validated and spec.has_regression_tests
                and self._maturity[family] is not MaturityState.BLOCKED):
            self._maturity[family] = MaturityState.LIVE_ELIGIBLE
        return self._maturity[family]

    def block(self, family: StrategyFamily) -> None:
        """Hard stop: BLOCKED overrides every authorization (directive §16)."""
        self._maturity[family] = MaturityState.BLOCKED


def _copy_spec(s: StrategyFamilySpec) -> StrategyFamilySpec:
    return StrategyFamilySpec(
        family=s.family, entry_conditions=list(s.entry_conditions),
        compatible_directions=set(s.compatible_directions),
        compatible_trends=set(s.compatible_trends),
        required_indicators=list(s.required_indicators),
        holding_horizon=s.holding_horizon,
        invalidation_conditions=list(s.invalidation_conditions),
        est_transaction_cost_bps=s.est_transaction_cost_bps,
        min_rel_volume=s.min_rel_volume, exposure_requirement=s.exposure_requirement,
        maps_to=s.maps_to, strategy_version=s.strategy_version,
        config_version=s.config_version, eligibility=s.eligibility,
        has_regression_tests=s.has_regression_tests, description=s.description)
