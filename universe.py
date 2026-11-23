"""Deterministic, broker-independent universe construction and symbol validation.

Every function here is pure (no network, no broker, no filesystem) so it can be
unit-tested in isolation. The pipeline order is intentionally:

    normalize -> deduplicate -> validate -> classify -> cap -> log exclusions

Previously the hard-coded candidate list in run.py applied a symbol-count cap
(`.head(30)`) *before* deduplication, which let a duplicate "AMD" entry consume
a universe slot that a 31st distinct symbol should have had, and let the typo
"NVD" (not a real ticker) ride through untouched. Capping after validation and
deduplication, as implemented below, is what the acceptance criteria require:
no invalid or duplicate symbol should ever survive universe construction, and
the size limit should only ever remove genuinely distinct, valid candidates.

Validation here is necessarily limited: without broker/market-data credentials
there is no way to confirm a well-formed ticker actually trades. This module
catches (a) malformed symbols and (b) a curated denylist of known-bad entries
(e.g. "NVD", a typo for "NVDA"). It does not and cannot catch every possible
nonexistent-but-well-formed ticker; that is the data provider's job downstream.
"""

from __future__ import annotations

import os
import random
import re
import tempfile
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence


class InstrumentType(str, Enum):
    EQUITY = "equity"
    INDEX_ETF = "index_etf"
    LEVERAGED_ETF = "leveraged_etf"
    INVERSE_ETF = "inverse_etf"


@dataclass(frozen=True)
class SymbolInfo:
    symbol: str
    instrument_type: InstrumentType
    exposure_group: str
    leverage: float = 1.0


@dataclass(frozen=True)
class ExcludedSymbol:
    symbol: str
    reason: str


@dataclass(frozen=True)
class UniverseResult:
    symbols: list[str]
    excluded: list[ExcludedSymbol]
    classifications: dict[str, SymbolInfo]

    def exposure_groups(self) -> dict[str, list[str]]:
        groups: dict[str, list[str]] = {}
        for sym in self.symbols:
            group = self.classifications[sym].exposure_group
            groups.setdefault(group, []).append(sym)
        return groups


# Only ETFs that create correlated/leveraged/inverse exposure need a manual
# entry here. Anything not listed defaults to an ordinary equity with its own
# singleton exposure group (the conservative default: no false diversification
# credit, no leverage assumed).
SYMBOL_REGISTRY: dict[str, SymbolInfo] = {
    # Broad index / sector ETFs
    "SPY": SymbolInfo("SPY", InstrumentType.INDEX_ETF, "us_large_cap"),
    "QQQ": SymbolInfo("QQQ", InstrumentType.INDEX_ETF, "nasdaq100"),
    "DIA": SymbolInfo("DIA", InstrumentType.INDEX_ETF, "dow30"),
    "IWM": SymbolInfo("IWM", InstrumentType.INDEX_ETF, "russell2000"),
    "SMH": SymbolInfo("SMH", InstrumentType.INDEX_ETF, "semiconductors"),
    "SOXX": SymbolInfo("SOXX", InstrumentType.INDEX_ETF, "semiconductors"),
    "XLE": SymbolInfo("XLE", InstrumentType.INDEX_ETF, "energy_sector"),
    "XLK": SymbolInfo("XLK", InstrumentType.INDEX_ETF, "technology_sector"),
    "JEPQ": SymbolInfo("JEPQ", InstrumentType.INDEX_ETF, "nasdaq100_income"),

    # Leveraged long ETFs
    "TQQQ": SymbolInfo("TQQQ", InstrumentType.LEVERAGED_ETF, "nasdaq100", 3.0),
    "TECL": SymbolInfo("TECL", InstrumentType.LEVERAGED_ETF, "technology_sector", 3.0),
    "SOXL": SymbolInfo("SOXL", InstrumentType.LEVERAGED_ETF, "semiconductors", 3.0),
    "TSLL": SymbolInfo("TSLL", InstrumentType.LEVERAGED_ETF, "tsla", 2.0),

    # Inverse / inverse-leveraged ETFs
    "SQQQ": SymbolInfo("SQQQ", InstrumentType.INVERSE_ETF, "nasdaq100", -3.0),
    "TECS": SymbolInfo("TECS", InstrumentType.INVERSE_ETF, "technology_sector", -3.0),
    "SOXS": SymbolInfo("SOXS", InstrumentType.INVERSE_ETF, "semiconductors", -3.0),
    "DRIP": SymbolInfo("DRIP", InstrumentType.INVERSE_ETF, "energy_sector", -2.0),
}

# Curated denylist of known-invalid entries (typos, delisted, non-tickers)
# found in this codebase's hand-maintained candidate lists.
KNOWN_INVALID_SYMBOLS: dict[str, str] = {
    "NVD": "not a real ticker (did you mean NVDA?)",
    # Normalized form of "HPE/PC" (default_normalize maps "/" -> "-"); not a
    # real HPE share class, just stray data from an earlier screener export.
    "HPE-PC": "not a tradable symbol",
}

# 1-5 uppercase letters, optional single share-class suffix (e.g. BRK.B, BRK-B).
VALID_SYMBOL_RE = re.compile(r"^[A-Z]{1,5}([.\-][A-Z]{1,2})?$")


def default_normalize(symbol: str) -> str:
    return str(symbol).upper().strip().replace("/", "-")


# Equity sector classification, populated at runtime via
# register_equity_sectors() -- kept out of SYMBOL_REGISTRY (which is for
# ETF/leverage/inverse typing) and out of any network call (this module
# stays pure/offline-testable). Callers with a sector source (e.g.
# features.get_sp500_sector_map(), itself a present-day Wikipedia snapshot
# applied retroactively -- same documented limitation as index membership)
# inject it here. Unregistered equities fall back to a singleton group
# (their own symbol), which is the conservative default: no false
# diversification credit from an unknown grouping.
EQUITY_SECTOR_MAP: dict[str, str] = {}


def register_equity_sectors(
    mapping: dict[str, str],
    normalize: Callable[[str], str] = default_normalize,
) -> None:
    """Populate (or update) the equity sector map used by classify_symbol()
    for exposure-group assignment. Call with {} to clear."""
    EQUITY_SECTOR_MAP.clear()
    EQUITY_SECTOR_MAP.update({normalize(k): v for k, v in mapping.items()})


def classify_symbol(symbol: str) -> SymbolInfo:
    info = SYMBOL_REGISTRY.get(symbol)
    if info is not None:
        return info
    sector = EQUITY_SECTOR_MAP.get(symbol)
    if sector:
        return SymbolInfo(symbol, InstrumentType.EQUITY,
                          exposure_group=f"sector:{sector}", leverage=1.0)
    return SymbolInfo(symbol, InstrumentType.EQUITY, exposure_group=symbol, leverage=1.0)


def build_universe(
    candidates: Iterable[str],
    max_symbols: int,
    normalize: Callable[[str], str] = default_normalize,
) -> UniverseResult:
    """Normalize, dedupe, validate, classify, then cap at max_symbols.

    The cap is applied last so that invalid/duplicate entries never consume a
    universe slot that a distinct valid symbol should have had.
    """
    seen: set[str] = set()
    excluded: list[ExcludedSymbol] = []
    ordered: list[str] = []

    for raw in candidates:
        if raw is None:
            continue
        sym = normalize(raw)
        if not sym:
            continue

        if sym in seen:
            excluded.append(ExcludedSymbol(sym, "duplicate"))
            continue
        seen.add(sym)

        if sym in KNOWN_INVALID_SYMBOLS:
            excluded.append(ExcludedSymbol(sym, f"invalid: {KNOWN_INVALID_SYMBOLS[sym]}"))
            continue

        if not VALID_SYMBOL_RE.match(sym):
            excluded.append(ExcludedSymbol(sym, "invalid: malformed ticker format"))
            continue

        ordered.append(sym)

    capped = ordered[:max_symbols] if max_symbols > 0 else list(ordered)
    for sym in ordered[max_symbols:] if max_symbols > 0 else []:
        excluded.append(ExcludedSymbol(sym, f"excluded: exceeds max_symbols={max_symbols}"))

    classifications = {sym: classify_symbol(sym) for sym in capped}

    return UniverseResult(symbols=capped, excluded=excluded, classifications=classifications)


def log_excluded(excluded: Iterable[ExcludedSymbol], log_fn: Callable[[str], None] = print) -> None:
    """Log exclusions without failing the run."""
    for item in excluded:
        log_fn(f"[universe] excluded symbol={item.symbol!r} reason={item.reason}")


# ======================================================================
# Canonical trading universe: loading + batching + coverage
# ----------------------------------------------------------------------
# Everything above this line is pure (no filesystem, no network). The
# layer below is the ONE place allowed to read universe.txt and to drive
# batched market-data requests. Consumers (features.py, trade.py,
# scanner.py, run.py) import from here instead of maintaining their own
# competing loaders / fallback lists. The single rule this layer enforces:
# universe.txt is the only trading-symbol source, and it fails closed if
# that file is missing or empty -- never a hard-coded fallback universe.
# ======================================================================

PROJECT_ROOT = Path(__file__).resolve().parent
UNIVERSE_TXT = PROJECT_ROOT / "universe.txt"

# Deterministic batch size for Alpaca multi-symbol historical requests.
# Overridable via env so an operator can dial it down under rate pressure
# without a code change. For the 55-symbol universe at the default of 20
# this yields exactly three batches: 20, 20, 15.
ALPACA_SYMBOL_BATCH_SIZE = int(os.getenv("ALPACA_SYMBOL_BATCH_SIZE", "20"))

# Tokens that are never symbols even after normalization.
_NON_SYMBOL_TOKENS = {"", "NAN", "NONE", "NULL"}


def normalize_symbol(symbol: str | None) -> str:
    """Canonical symbol normalization shared by every consumer.

    Strips surrounding quotes/whitespace (universe.txt stores quoted,
    comma-separated tokens like ``'INTC'``), then applies
    :func:`default_normalize` (uppercase, trim, ``/`` -> ``-``). Returns an
    empty string for anything that is not a usable token so callers can
    filter it out uniformly.
    """
    if symbol is None:
        return ""
    text = str(symbol).strip().strip("'\"").strip()
    if not text:
        return ""
    normalized = default_normalize(text)
    return "" if normalized in _NON_SYMBOL_TOKENS else normalized


def iter_symbol_batches(
    symbols: list[str],
    batch_size: int = ALPACA_SYMBOL_BATCH_SIZE,
):
    """Yield deterministic, order-preserving, de-duplicated symbol batches.

    Normalization and de-duplication happen once, up front, preserving the
    first occurrence order. ``batch_size`` must be at least 1.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")

    normalized = list(
        dict.fromkeys(
            normalize_symbol(symbol)
            for symbol in symbols
            if normalize_symbol(symbol)
        )
    )

    for start in range(0, len(normalized), batch_size):
        yield normalized[start:start + batch_size]


def _parse_universe_text(text: str) -> list[str]:
    """Parse universe.txt content into a normalized, de-duplicated,
    order-preserving symbol list. Supports one-per-line, comma-separated,
    and quoted tokens, with ``#`` line comments."""
    symbols: list[str] = []
    for line in text.splitlines():
        content = line.split("#", 1)[0].replace(",", " ")
        for token in content.split():
            sym = normalize_symbol(token)
            if sym:
                symbols.append(sym)
    return list(dict.fromkeys(symbols))


def load_canonical_universe(path: str | Path = UNIVERSE_TXT) -> list[str]:
    """Load the canonical trading universe from universe.txt.

    This is the single source of trading symbols. It fails closed: a
    missing or empty file raises ``RuntimeError`` rather than silently
    falling back to a hard-coded list or a broad-market expansion. File
    order is preserved; symbols are normalized and de-duplicated.
    """
    p = Path(path)
    try:
        text = p.read_text(encoding="utf-8")
    except OSError as exc:
        raise RuntimeError(
            f"Canonical trading universe is unavailable (fail-closed): {p}"
        ) from exc

    symbols = _parse_universe_text(text)
    if not symbols:
        raise RuntimeError(
            f"Canonical trading universe is empty (fail-closed): {p}"
        )
    return symbols


def write_canonical_universe(
    symbols: Sequence[str],
    path: str | Path = UNIVERSE_TXT,
    *,
    min_symbols: int = 1,
) -> list[str]:
    """Atomically rewrite universe.txt with ``symbols`` in the given order.

    This is the single writer counterpart to :func:`load_canonical_universe`.
    The real-time scanner calls it to republish the ranked trading universe,
    so it is engineered to be safe against the loader's fail-closed contract:

      * symbols are normalized (:func:`normalize_symbol`) and de-duplicated,
        first occurrence wins, preserving the caller's (descending-rank) order;
      * it refuses to write fewer than ``min_symbols`` usable symbols, raising
        ``ValueError`` -- it never produces an empty file the loader would then
        reject (and it leaves any pre-existing good file untouched on refusal);
      * the write is atomic: content goes to a uniquely-named temp file in the
        *same directory*, is flushed+fsync'd, then ``os.replace``'d onto
        ``path`` so no reader ever observes a torn or empty file mid-write.

    The on-disk format matches the existing universe.txt (a single line of
    quoted, comma-separated tokens) so it round-trips through
    :func:`load_canonical_universe` and every existing consumer. Returns the
    normalized list actually written.
    """
    normalized = list(
        dict.fromkeys(
            normalize_symbol(sym) for sym in symbols if normalize_symbol(sym)
        )
    )
    if len(normalized) < max(1, int(min_symbols)):
        raise ValueError(
            "refusing to write universe.txt with "
            f"{len(normalized)} symbol(s) (min_symbols={min_symbols}); "
            "the loader is fail-closed on empty/short universes"
        )

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    content = ", ".join(f"'{sym}'" for sym in normalized) + "\n"

    fd, tmp_name = tempfile.mkstemp(
        dir=str(p.parent), prefix=".universe.", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, p)
    except BaseException:
        # Never leave a temp artifact behind, and never touch the live file
        # on failure.
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise
    return normalized


# ----------------------------------------------------------------------
# Coverage accounting
# ----------------------------------------------------------------------


@dataclass
class CoverageReport:
    """Per-run accounting of a batched market-data sweep.

    Distinguishes three non-success outcomes that must never be conflated:
      * a *failed request* (the batch errored out after retries) ->
        ``failed_symbols`` / ``failed_batches``;
      * a *successful request that returned no data for a symbol* ->
        ``no_data_symbols``;
      * a *symbol rejected as nontradable* -> ``nontradable_symbols``.

    ``missing`` is anything in the universe that did not come back with
    usable data, so complete coverage can never be reported after an
    incomplete request cycle.
    """

    universe_count: int
    batch_size: int
    batch_count: int = 0
    attempted_symbols: list[str] = field(default_factory=list)
    successful_symbols: list[str] = field(default_factory=list)
    no_data_symbols: list[str] = field(default_factory=list)
    nontradable_symbols: list[str] = field(default_factory=list)
    failed_symbols: list[str] = field(default_factory=list)
    failed_batches: list[dict[str, Any]] = field(default_factory=list)

    @property
    def attempted_symbol_count(self) -> int:
        return len(self.attempted_symbols)

    @property
    def successful_symbol_count(self) -> int:
        return len(self.successful_symbols)

    @property
    def missing_symbols(self) -> list[str]:
        successful = set(self.successful_symbols)
        # Preserve universe order via attempted order.
        seen: set[str] = set()
        missing: list[str] = []
        for sym in self.attempted_symbols:
            if sym not in successful and sym not in seen:
                seen.add(sym)
                missing.append(sym)
        return missing

    @property
    def missing_symbol_count(self) -> int:
        return len(self.missing_symbols)

    @property
    def failed_batch_count(self) -> int:
        return len(self.failed_batches)

    @property
    def is_complete(self) -> bool:
        """True only when every universe symbol came back with usable data
        and no batch failed. This is what callers check before claiming
        full coverage; anything else is an incomplete cycle."""
        return (
            self.failed_batch_count == 0
            and self.missing_symbol_count == 0
            and self.attempted_symbol_count >= self.universe_count
        )

    def record_nontradable(self, symbols: Iterable[str]) -> None:
        for sym in symbols:
            sym = normalize_symbol(sym)
            if sym and sym not in self.nontradable_symbols:
                self.nontradable_symbols.append(sym)

    def to_dict(self) -> dict[str, Any]:
        return {
            "universe_count": self.universe_count,
            "batch_size": self.batch_size,
            "batch_count": self.batch_count,
            "attempted_symbol_count": self.attempted_symbol_count,
            "successful_symbol_count": self.successful_symbol_count,
            "missing_symbol_count": self.missing_symbol_count,
            "failed_batch_count": self.failed_batch_count,
            "no_data_symbols": list(self.no_data_symbols),
            "nontradable_symbols": list(self.nontradable_symbols),
            "failed_symbols": list(self.failed_symbols),
            "missing_symbols": self.missing_symbols,
            "is_complete": self.is_complete,
        }

    def summary_lines(self, label: str = "universe") -> list[str]:
        return [
            (
                f"[{label}] loaded={self.universe_count} "
                f"batch_size={self.batch_size} batches={self.batch_count}"
            ),
            (
                f"[{label}] attempted={self.attempted_symbol_count} "
                f"successful={self.successful_symbol_count} "
                f"missing={self.missing_symbol_count} "
                f"failed_batches={self.failed_batch_count}"
            ),
        ]

    def log_summary(self, log_fn: Callable[[str], None] = print, label: str = "universe") -> None:
        for line in self.summary_lines(label=label):
            log_fn(line)
        if self.missing_symbol_count or self.failed_batch_count:
            log_fn(
                f"[{label}] INCOMPLETE COVERAGE "
                f"missing={self.missing_symbols} "
                f"failed_symbols={self.failed_symbols} "
                f"no_data={self.no_data_symbols} "
                f"nontradable={self.nontradable_symbols}"
            )


# ----------------------------------------------------------------------
# Rate-limit-aware batched fetch
# ----------------------------------------------------------------------


class PermanentRequestError(Exception):
    """Raised/surfaced for failures that must NOT be retried: authentication,
    entitlement, malformed-request, and other permanent 4xx responses."""


def _http_status(exc: BaseException) -> int | None:
    """Best-effort extraction of an HTTP status code from the many exception
    shapes Alpaca / requests / urllib3 can raise."""
    for attr in ("status_code", "code", "status"):
        val = getattr(exc, attr, None)
        if isinstance(val, int) and 100 <= val <= 599:
            return val
    resp = getattr(exc, "response", None)
    if resp is not None:
        val = getattr(resp, "status_code", None)
        if isinstance(val, int):
            return val
    match = re.search(r"\b(4\d\d|5\d\d)\b", str(exc))
    if match:
        return int(match.group(1))
    return None


def is_transient_error(exc: BaseException) -> bool:
    """Classify an exception as transient (safe to retry) vs permanent.

    Transient: HTTP 429 (rate limit) and 5xx (server) responses, plus
    connection/timeout errors. Permanent: 401/403 (auth/entitlement), 400
    (malformed), and every other 4xx -- these are never retried because a
    retry cannot change the outcome and only burns the rate budget.
    """
    if isinstance(exc, PermanentRequestError):
        return False

    status = _http_status(exc)
    if status is not None:
        if status == 429:
            return True
        if 500 <= status <= 599:
            return True
        if 400 <= status <= 499:
            return False
        return False

    # No decodable status code: treat genuine network faults as transient,
    # but anything else (including programming errors) as permanent so we do
    # not mask bugs behind retries.
    name = type(exc).__name__.lower()
    transient_markers = ("timeout", "connection", "connectionerror", "readtimeout")
    return any(marker in name for marker in transient_markers)


def _backoff_delay(
    attempt: int,
    base_delay: float,
    max_delay: float,
    rng: random.Random,
) -> float:
    """Bounded exponential backoff with full jitter in [50%, 100%] of the
    capped exponential term."""
    capped = min(max_delay, base_delay * (2 ** attempt))
    return capped * (0.5 + 0.5 * rng.random())


def fetch_with_retry(
    fetch_one_batch: Callable[[list[str]], Any],
    batch: list[str],
    *,
    max_retries: int = 4,
    base_delay: float = 0.5,
    max_delay: float = 30.0,
    sleep_fn: Callable[[float], None] = time.sleep,
    rng: random.Random | None = None,
    log_fn: Callable[[str], None] = print,
    label: str = "universe",
) -> Any:
    """Call ``fetch_one_batch(batch)`` with bounded exponential backoff on
    transient (429/5xx/network) failures only. Permanent failures (auth,
    entitlement, malformed 4xx) raise immediately without retry.

    ``max_retries`` is the number of *additional* attempts after the first,
    so total attempts = max_retries + 1.
    """
    rng = rng or random.Random()
    attempt = 0
    while True:
        try:
            return fetch_one_batch(batch)
        except Exception as exc:  # noqa: BLE001 - we re-raise after classifying
            if not is_transient_error(exc):
                log_fn(
                    f"[{label}] permanent failure (no retry) "
                    f"symbols={batch} error={exc}"
                )
                raise
            if attempt >= max_retries:
                log_fn(
                    f"[{label}] transient failure, retries exhausted "
                    f"({attempt}/{max_retries}) symbols={batch} error={exc}"
                )
                raise
            delay = _backoff_delay(attempt, base_delay, max_delay, rng)
            log_fn(
                f"[{label}] transient failure (retry {attempt + 1}/{max_retries} "
                f"in {delay:.2f}s) symbols={batch} error={exc}"
            )
            sleep_fn(delay)
            attempt += 1


def fetch_bars_batched(
    fetch_one_batch: Callable[[list[str]], Any],
    symbols: Sequence[str],
    *,
    batch_size: int = ALPACA_SYMBOL_BATCH_SIZE,
    max_retries: int = 4,
    base_delay: float = 0.5,
    max_delay: float = 30.0,
    sleep_fn: Callable[[float], None] = time.sleep,
    rng: random.Random | None = None,
    log_fn: Callable[[str], None] = print,
    label: str = "universe",
    symbol_column: str = "symbol",
):
    """Drive a batched market-data sweep over the full universe.

    ``fetch_one_batch`` is a caller-supplied callable that takes a list of
    symbols and returns a pandas DataFrame (with a ``symbol_column``) for
    that batch. This module owns batching, sequential ordering, retry
    policy, frame combination, and coverage accounting; the caller owns the
    broker-specific request details.

    Returns ``(combined_df, CoverageReport)``. Frames from every successful
    batch are concatenated (never overwritten); row order follows universe
    (batch) order. A symbol present in the universe but absent from the
    returned frame is recorded as ``no_data``; a batch that fails after
    retries records all of its symbols as ``failed`` and increments
    ``failed_batch_count`` -- so complete coverage is never reported after
    an incomplete cycle.
    """
    import pandas as pd

    batches = list(iter_symbol_batches(symbols, batch_size))
    normalized_all = [sym for batch in batches for sym in batch]

    report = CoverageReport(
        universe_count=len(normalized_all),
        batch_size=batch_size,
        batch_count=len(batches),
    )

    frames: list[Any] = []
    for index, batch in enumerate(batches):
        report.attempted_symbols.extend(batch)
        log_fn(f"[{label}] batch {index + 1}/{len(batches)} size={len(batch)}")

        try:
            df = fetch_with_retry(
                fetch_one_batch,
                batch,
                max_retries=max_retries,
                base_delay=base_delay,
                max_delay=max_delay,
                sleep_fn=sleep_fn,
                rng=rng,
                log_fn=log_fn,
                label=label,
            )
        except Exception as exc:  # noqa: BLE001 - recorded, not swallowed
            report.failed_batches.append(
                {"batch_index": index, "symbols": list(batch), "error": str(exc)}
            )
            report.failed_symbols.extend(batch)
            continue

        if df is None or getattr(df, "empty", True):
            report.no_data_symbols.extend(batch)
            continue

        if symbol_column in df.columns:
            present = {
                normalize_symbol(s)
                for s in df[symbol_column].astype(str).unique()
            }
        else:
            present = set(batch)  # cannot verify per-symbol; assume batch-level success

        frames.append(df)
        for sym in batch:
            if sym in present:
                report.successful_symbols.append(sym)
            else:
                report.no_data_symbols.append(sym)

    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return combined, report


# ----------------------------------------------------------------------
# Live-scan batch rotation
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class RotationBatch:
    cycle_id: int
    batch_index: int
    batch_count: int
    symbols: list[str]
    cycle_complete: bool


class UniverseScanRotation:
    """Deterministic rotating cursor over the universe for live scanning.

    Each :meth:`next_batch` call returns the next batch and advances the
    cursor, so consecutive calls never re-scan only the first batch
    (no first-batch starvation). A scan cycle is complete only after every
    universe symbol has been handed out exactly once; when the cursor wraps,
    the cycle id increments.
    """

    def __init__(
        self,
        symbols: Sequence[str],
        batch_size: int = ALPACA_SYMBOL_BATCH_SIZE,
    ) -> None:
        self._batches = list(iter_symbol_batches(symbols, batch_size))
        if not self._batches:
            raise ValueError("UniverseScanRotation requires at least one symbol")
        self.batch_size = batch_size
        self.symbols = [sym for batch in self._batches for sym in batch]
        self.cycle_id = 1
        self._next_index = 0
        self._attempted_this_cycle: set[str] = set()

    @property
    def batch_count(self) -> int:
        return len(self._batches)

    def next_batch(self) -> RotationBatch:
        if self._next_index >= len(self._batches):
            # Wrap: begin a new cycle.
            self.cycle_id += 1
            self._next_index = 0
            self._attempted_this_cycle = set()

        index = self._next_index
        batch = self._batches[index]
        self._next_index += 1
        self._attempted_this_cycle.update(batch)

        return RotationBatch(
            cycle_id=self.cycle_id,
            batch_index=index,
            batch_count=len(self._batches),
            symbols=list(batch),
            cycle_complete=self._next_index >= len(self._batches),
        )

    def current_cycle_complete(self) -> bool:
        return len(self._attempted_this_cycle) >= len(self.symbols)
