"""Web-based discovery of US-listed securities for the scanner.

Reuses the existing web fetchers in ``scanner.py`` (S&P 500 via Wikipedia,
NASDAQ-listed) and validates/classifies the result through ``universe`` so the
scanner is not confined to a predetermined ticker list. It always folds in a
curated ETF seed (index, sector, leveraged-bull, inverse) so directional ETF
candidates remain discoverable even if a web source is unavailable.

Fetchers are injectable (defaulting to the ``scanner`` functions, imported
lazily so this module stays cheap/offline-testable) and the whole thing is
fail-safe: a source error degrades gracefully and never raises into the loop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional

import universe
from universe import InstrumentType, classify_symbol

# Curated directional ETF seed: index, sector, leveraged-long, inverse. Keeps
# bullish/inverse ETF candidates discoverable regardless of web availability.
DEFAULT_ETF_SEED: tuple[str, ...] = (
    # broad index
    "SPY", "QQQ", "DIA", "IWM",
    # sector / thematic
    "SMH", "SOXX", "XLK", "XLE", "XLF", "XLV", "XLI", "XLY", "XLP", "XLU", "QTUM",
    # leveraged long
    "TQQQ", "SOXL", "TECL", "SPXL", "UPRO", "TNA", "TSLL", "NVDL", "FNGU",
    # inverse / inverse-leveraged
    "SQQQ", "SOXS", "TECS", "SPXU", "SDS", "SH", "PSQ", "QID", "DOG", "SDOW",
    "RWM", "TZA", "DRIP",
)


@dataclass
class DiscoveryResult:
    symbols: list[str] = field(default_factory=list)   # validated, sorted US-listed set
    equities: list[str] = field(default_factory=list)
    etfs: list[str] = field(default_factory=list)      # index/leveraged/inverse ETFs
    source_counts: dict = field(default_factory=dict)
    error: Optional[str] = None


def _default_sp500_fetcher() -> list[str]:
    import scanner
    return scanner.get_sp500_symbols()


def _default_nasdaq_fetcher() -> list[str]:
    import scanner
    return scanner.get_nasdaq_symbols()


def _safe_fetch(fetcher: Callable[[], Iterable[str]], label: str,
                errors: list[str]) -> list[str]:
    try:
        return [str(s) for s in (fetcher() or [])]
    except Exception as exc:  # noqa: BLE001 - never raise into the scanner loop
        errors.append(f"{label}: {exc}")
        return []


def discover_us_listed(
    *,
    sp500_fetcher: Optional[Callable[[], Iterable[str]]] = None,
    nasdaq_fetcher: Optional[Callable[[], Iterable[str]]] = None,
    extra: Iterable[str] = (),
    max_symbols: Optional[int] = None,
    include_etf_seed: bool = True,
) -> DiscoveryResult:
    """Discover + validate + classify the US-listed candidate universe.

    Fail-safe: if a source raises, its contribution is skipped, ``error`` is set,
    and the remaining sources (plus the ETF seed) still produce a usable result.
    """
    errors: list[str] = []
    extra_list = list(extra)
    sp = _safe_fetch(sp500_fetcher or _default_sp500_fetcher, "sp500", errors)
    nd = _safe_fetch(nasdaq_fetcher or _default_nasdaq_fetcher, "nasdaq", errors)

    seed = list(DEFAULT_ETF_SEED) if include_etf_seed else []
    # Fetched symbols take priority for any cap; seed guarantees ETF coverage.
    combined = list(sp) + list(nd) + extra_list + seed

    cap = max_symbols if (isinstance(max_symbols, int) and max_symbols > 0) else len(combined) + 1
    result = universe.build_universe(combined, max_symbols=cap)
    symbols = sorted(result.symbols)

    etfs: list[str] = []
    equities: list[str] = []
    for sym in symbols:
        if classify_symbol(sym).instrument_type == InstrumentType.EQUITY:
            equities.append(sym)
        else:
            etfs.append(sym)

    return DiscoveryResult(
        symbols=symbols,
        equities=equities,
        etfs=etfs,
        source_counts={"sp500": len(sp), "nasdaq": len(nd),
                       "extra": len(extra_list), "seed": len(seed)},
        error="; ".join(errors) if errors else None,
    )
