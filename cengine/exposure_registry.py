"""Instrument exposure registry: ETF classification + exposure accounting (§5-8).

Classifies the discovery universe by *economic exposure* — conventional equity,
broad/sector/industry/thematic ETFs, leveraged-long ETFs, inverse ETFs, and
leveraged-inverse ETFs — with the metadata the portfolio and risk layers need:
underlying benchmark, exposure group, direction, and verified **target daily
leverage** (negative for inverse products).

Exposure is NOT inferred from ticker naming (§6). Each seeded instrument carries
CURATED metadata derived from the issuer's published fund objective (the seed
lists and their categories are given explicitly in the directive, §5); the
metadata_source records this, and production should reconcile against issuer
prospectuses. Tradability / asset status is reconciled against Alpaca assets.

Safety invariant (§7): leveraged and inverse products NEVER auto-authorize
execution. `execution_authorized` defaults True only for common equity and
unlevered (1x) long ETFs; everything else requires an explicit, user-authorized
enablement. Target daily leverage is a one-day reset multiplier, NOT a guaranteed
realized or multi-day exposure (§8).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class InstrumentType(str, Enum):
    COMMON_EQUITY = "COMMON_EQUITY"
    BROAD_MARKET_ETF = "BROAD_MARKET_ETF"
    SECTOR_ETF = "SECTOR_ETF"
    INDUSTRY_ETF = "INDUSTRY_ETF"
    THEMATIC_ETF = "THEMATIC_ETF"
    LEVERAGED_LONG_ETF = "LEVERAGED_LONG_ETF"
    INVERSE_ETF = "INVERSE_ETF"
    LEVERAGED_INVERSE_ETF = "LEVERAGED_INVERSE_ETF"
    REFERENCE_ONLY = "REFERENCE_ONLY"
    UNCLASSIFIED = "UNCLASSIFIED"


class ExposureDirection(str, Enum):
    LONG = "LONG"
    INVERSE = "INVERSE"
    NEUTRAL = "NEUTRAL"


@dataclass
class ExposureInfo:
    symbol: str
    instrument_type: InstrumentType = InstrumentType.UNCLASSIFIED
    instrument_subtype: str = ""
    underlying_benchmark: str = ""
    underlying_exposure_group: str = ""       # clustering key for overlap/opposition
    exposure_direction: ExposureDirection = ExposureDirection.NEUTRAL
    target_daily_leverage: float = 0.0        # signed; negative for inverse
    daily_reset: bool = False
    inverse: bool = False
    leveraged: bool = False
    metadata_source: str = "curated_fund_objective"
    # reconciled against Alpaca
    tradable: bool = False
    asset_status: str = ""
    # safety gate — leveraged/inverse never auto-authorized (§7)
    execution_authorized: bool = False


# --- curated seed table (documented fund objectives, directive §5) -----------
# (symbol, type, underlying_benchmark, exposure_group, target_daily_leverage)
# leverage is signed; daily_reset/leveraged/inverse derived below.
_B, _S, _I = (InstrumentType.BROAD_MARKET_ETF, InstrumentType.SECTOR_ETF,
              InstrumentType.INDUSTRY_ETF)
_LL, _INV, _LINV = (InstrumentType.LEVERAGED_LONG_ETF, InstrumentType.INVERSE_ETF,
                    InstrumentType.LEVERAGED_INVERSE_ETF)

_SEED: list[tuple] = [
    # broad market (1x long)
    ("SPY", _B, "S&P 500", "SP500", 1.0), ("VOO", _B, "S&P 500", "SP500", 1.0),
    ("RSP", _B, "S&P 500 Equal Weight", "SP500", 1.0),
    ("SSO", _LL, "S&P 500", "SP500", 2.0), ("UPRO", _LL, "S&P 500", "SP500", 3.0),
    ("SPXL", _LL, "S&P 500", "SP500", 3.0),
    ("SH", _INV, "S&P 500", "SP500", -1.0), ("SDS", _LINV, "S&P 500", "SP500", -2.0),
    ("SPXU", _LINV, "S&P 500", "SP500", -3.0), ("SPXS", _LINV, "S&P 500", "SP500", -3.0),
    ("QQQ", _B, "Nasdaq-100", "NASDAQ100", 1.0),
    ("QLD", _LL, "Nasdaq-100", "NASDAQ100", 2.0), ("TQQQ", _LL, "Nasdaq-100", "NASDAQ100", 3.0),
    ("PSQ", _INV, "Nasdaq-100", "NASDAQ100", -1.0), ("QID", _LINV, "Nasdaq-100", "NASDAQ100", -2.0),
    ("SQQQ", _LINV, "Nasdaq-100", "NASDAQ100", -3.0),
    ("DIA", _B, "Dow Jones Industrial Average", "DJIA", 1.0),
    ("UDOW", _LL, "Dow Jones Industrial Average", "DJIA", 3.0),
    ("DOG", _INV, "Dow Jones Industrial Average", "DJIA", -1.0),
    ("DXD", _LINV, "Dow Jones Industrial Average", "DJIA", -2.0),
    ("SDOW", _LINV, "Dow Jones Industrial Average", "DJIA", -3.0),
    ("IWM", _B, "Russell 2000", "RUSSELL2000", 1.0),
    ("TNA", _LL, "Russell 2000", "RUSSELL2000", 3.0),
    ("RWM", _INV, "Russell 2000", "RUSSELL2000", -1.0),
    ("TWM", _LINV, "Russell 2000", "RUSSELL2000", -2.0),
    ("TZA", _LINV, "Russell 2000", "RUSSELL2000", -3.0),
    ("VTI", _B, "US Total Market", "TOTAL_MARKET", 1.0),
    ("SCHB", _B, "US Broad Market", "TOTAL_MARKET", 1.0),
    # sector / industry (1x) + leveraged/inverse counterparts
    ("XLK", _S, "Technology Select Sector", "TECHNOLOGY", 1.0),
    ("TECL", _LL, "Technology Select Sector", "TECHNOLOGY", 3.0),
    ("TECS", _LINV, "Technology Select Sector", "TECHNOLOGY", -3.0),
    ("XLF", _S, "Financial Select Sector", "FINANCIALS", 1.0),
    ("FAS", _LL, "Russell 1000 Financials", "FINANCIALS", 3.0),
    ("FAZ", _LINV, "Russell 1000 Financials", "FINANCIALS", -3.0),
    ("KRE", _I, "S&P Regional Banks", "REGIONAL_BANKS", 1.0),
    ("XLE", _S, "Energy Select Sector", "ENERGY", 1.0),
    ("ERX", _LL, "Energy Select Sector", "ENERGY", 2.0),
    ("ERY", _LINV, "Energy Select Sector", "ENERGY", -2.0),
    ("XOP", _I, "S&P Oil & Gas E&P", "OIL_GAS", 1.0),
    ("DRIP", _LINV, "S&P Oil & Gas E&P", "OIL_GAS", -2.0),
    ("XLV", _S, "Health Care Select Sector", "HEALTHCARE", 1.0),
    ("XLI", _S, "Industrial Select Sector", "INDUSTRIALS", 1.0),
    ("XLY", _S, "Consumer Discretionary Select Sector", "CONS_DISCRETIONARY", 1.0),
    ("XLP", _S, "Consumer Staples Select Sector", "CONS_STAPLES", 1.0),
    ("XLU", _S, "Utilities Select Sector", "UTILITIES", 1.0),
    ("XLB", _S, "Materials Select Sector", "MATERIALS", 1.0),
    ("XLRE", _S, "Real Estate Select Sector", "REAL_ESTATE", 1.0),
    ("XLC", _S, "Communication Services Select Sector", "COMMUNICATIONS", 1.0),
    ("SOXX", _I, "ICE Semiconductor", "SEMICONDUCTORS", 1.0),
    ("SMH", _I, "Solactive Semiconductor", "SEMICONDUCTORS", 1.0),
    ("SOXL", _LL, "ICE Semiconductor", "SEMICONDUCTORS", 3.0),
    ("SOXS", _LINV, "ICE Semiconductor", "SEMICONDUCTORS", -3.0),
    ("XBI", _I, "S&P Biotechnology", "BIOTECH", 1.0),
    ("IBB", _I, "Nasdaq Biotechnology", "BIOTECH", 1.0),
    ("LABU", _LL, "S&P Biotechnology", "BIOTECH", 3.0),
    ("LABD", _LINV, "S&P Biotechnology", "BIOTECH", -3.0),
]


def _build_info(symbol, itype, benchmark, group, lev) -> ExposureInfo:
    inverse = lev < 0
    leveraged = abs(lev) > 1.0
    direction = (ExposureDirection.INVERSE if inverse else
                 ExposureDirection.LONG if lev > 0 else ExposureDirection.NEUTRAL)
    daily_reset = leveraged or inverse  # geared/inverse ETFs reset daily
    # Safety: only common equity + unlevered long ETFs auto-authorize (§7).
    auto = (itype in (InstrumentType.BROAD_MARKET_ETF, InstrumentType.SECTOR_ETF,
                      InstrumentType.INDUSTRY_ETF, InstrumentType.THEMATIC_ETF)
            and not leveraged and not inverse and lev == 1.0)
    return ExposureInfo(
        symbol=symbol, instrument_type=itype, underlying_benchmark=benchmark,
        underlying_exposure_group=group, exposure_direction=direction,
        target_daily_leverage=float(lev), daily_reset=daily_reset,
        inverse=inverse, leveraged=leveraged, execution_authorized=auto)


def seed_exposure_table() -> dict[str, ExposureInfo]:
    return {sym: _build_info(sym, t, b, g, lev) for (sym, t, b, g, lev) in _SEED}


def seed_symbols() -> list[str]:
    return [row[0] for row in _SEED]


def effective_exposure(market_value: float, target_daily_leverage: float) -> float:
    """Signed leverage-adjusted underlying exposure (§8). Negative for inverse.
    This is a one-day-reset estimate, NOT a guaranteed realized/multi-day figure."""
    return float(market_value) * float(target_daily_leverage)


def detect_opposing(positions: dict[str, ExposureInfo]) -> list[tuple]:
    """Economically redundant/opposing positions: same exposure_group held with
    opposite signed leverage (e.g., TQQQ long + SQQQ). Returns (a, b, group)."""
    by_group: dict[str, list[ExposureInfo]] = {}
    for info in positions.values():
        if info.underlying_exposure_group:
            by_group.setdefault(info.underlying_exposure_group, []).append(info)
    out = []
    for group, infos in by_group.items():
        for i in range(len(infos)):
            for j in range(i + 1, len(infos)):
                a, b = infos[i], infos[j]
                if (a.target_daily_leverage * b.target_daily_leverage) < 0:
                    out.append((a.symbol, b.symbol, group))
    return out


def detect_overlap(positions: dict[str, ExposureInfo]) -> dict[str, list[str]]:
    """Group -> symbols sharing the same underlying exposure group (same-direction
    or opposing). Used for leverage-adjusted concentration accounting."""
    by_group: dict[str, list[str]] = {}
    for sym, info in positions.items():
        if info.underlying_exposure_group:
            by_group.setdefault(info.underlying_exposure_group, []).append(sym)
    return {g: sorted(s) for g, s in by_group.items() if len(s) > 1}


class ExposureRegistry:
    """Exposure metadata for the universe, reconciled against Alpaca assets."""

    def __init__(self) -> None:
        self._table: dict[str, ExposureInfo] = seed_exposure_table()

    def reconcile(self, assets_by_symbol: dict[str, dict]) -> dict:
        """Mark tradability / asset status from Alpaca; a seeded instrument absent
        from Alpaca is kept but flagged non-tradable (never dropped silently)."""
        present = 0
        for sym, info in self._table.items():
            a = assets_by_symbol.get(sym)
            if a is not None:
                info.tradable = bool(a.get("tradable", False))
                info.asset_status = a.get("status", "")
                present += 1
            else:
                info.tradable = False
                info.asset_status = "not_in_alpaca_assets"
        return {"seeded": len(self._table), "present_in_alpaca": present,
                "missing": len(self._table) - present}

    def get(self, symbol: str) -> ExposureInfo | None:
        return self._table.get(str(symbol).upper().strip())

    def all(self) -> dict[str, ExposureInfo]:
        return dict(self._table)

    def by_type(self, itype: InstrumentType) -> list[str]:
        return sorted(s for s, i in self._table.items() if i.instrument_type == itype)

    def inverse_symbols(self) -> list[str]:
        return sorted(s for s, i in self._table.items() if i.inverse)

    def leveraged_symbols(self) -> list[str]:
        return sorted(s for s, i in self._table.items() if i.leveraged)

    def execution_authorized_symbols(self) -> list[str]:
        return sorted(s for s, i in self._table.items() if i.execution_authorized)

    def authorize(self, symbol: str, authorized: bool = True) -> bool:
        """Explicit, user-authorized enablement of a leveraged/inverse product."""
        info = self.get(symbol)
        if info is None:
            return False
        info.execution_authorized = bool(authorized)
        return True

    def authorize_entry(self, symbol: str) -> tuple[bool, str]:
        """Elective-entry authorization for an instrument (§8/§10). A symbol known
        to the exposure registry is authorized only if execution_authorized is set
        (leveraged/inverse default False -> blocked until explicitly authorized).
        A symbol NOT in the registry is treated as a conventional equity and
        authorized (the engine already trades those); comprehensive gating of
        *unclassified ETFs* depends on dynamic ETF discovery (§6) and is noted as
        a follow-up. Exits are never gated — risk reduction always proceeds."""
        info = self.get(symbol)
        if info is None:
            return True, "not_in_exposure_registry_assumed_common_equity"
        if info.execution_authorized:
            return True, "execution_authorized"
        kind = info.instrument_type.value
        return False, f"geared_or_inverse_not_authorized:{kind}"

    def counts(self) -> dict:
        t = self._table.values()
        return {
            "total_seeded": len(self._table),
            "broad_market": sum(1 for i in t if i.instrument_type == _B),
            "sector_industry": sum(1 for i in t if i.instrument_type in (_S, _I)),
            "leveraged_long": sum(1 for i in t if i.instrument_type == _LL),
            "inverse": sum(1 for i in t if i.instrument_type == _INV),
            "leveraged_inverse": sum(1 for i in t if i.instrument_type == _LINV),
            "execution_authorized_default": sum(1 for i in t if i.execution_authorized),
            "tradable": sum(1 for i in t if i.tradable),
            "exposure_groups": len({i.underlying_exposure_group for i in t}),
        }
