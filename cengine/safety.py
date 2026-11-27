"""Market/session/feed gates; no trading thresholds are invented here."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

BAR_CHANNEL = "bar"


class MarketClosedError(RuntimeError):
    """Raised by MarketSessionGate.require_open when the session is closed. A subclass
    of RuntimeError so existing fail-closed callers still catch it, but typed so the
    market-data path can treat an ordinary closed session as nonfatal."""


@dataclass(frozen=True, slots=True)
class MarketGateConfig:
    max_feed_age_ns: int
    max_bar_gap_ns: int

    def __post_init__(self) -> None:
        if self.max_feed_age_ns <= 0 or self.max_bar_gap_ns <= 0:
            raise ValueError("freshness limits must be explicitly positive")


class MarketSafetyGate:
    """Per-channel freshness and ordering gate.

    A real market-data feed (IEX or SIP) multiplexes independent channels
    (quote, trade, bar, and licensed index factors). Each channel is monotonic
    per symbol, but the
    channels are delivered interleaved and are NOT mutually monotonic, so
    ordering is enforced per ``(symbol, channel)`` rather than per symbol.
    Feed-liveness (``require_fresh``) uses the freshest observation across all
    channels for the symbol; the missing-bar gap gate applies to bars only.
    """

    def __init__(self, config: MarketGateConfig) -> None:
        self.config = config
        self._last_event_ns: dict[tuple[str, str], int] = {}
        self._last_bar_ns: dict[str, int] = {}
        self._freshest_ns: dict[str, int] = {}

    def observe(self, symbol: str, timestamp_ns: int, *, channel: str) -> None:
        if not channel:
            raise ValueError("market event channel is required")
        key = (symbol, channel)
        previous = self._last_event_ns.get(key)
        if previous is not None and timestamp_ns < previous:
            raise RuntimeError(f"out-of-order market event for {symbol} on {channel}")
        self._last_event_ns[key] = timestamp_ns
        newest = self._freshest_ns.get(symbol)
        if newest is None or timestamp_ns > newest:
            self._freshest_ns[symbol] = timestamp_ns
        if channel == BAR_CHANNEL:
            prior_bar = self._last_bar_ns.get(symbol)
            if prior_bar is not None and timestamp_ns - prior_bar > self.config.max_bar_gap_ns:
                raise RuntimeError(f"missing-bar gate tripped for {symbol}")
            self._last_bar_ns[symbol] = timestamp_ns

    def require_fresh(self, symbol: str, now_ns: int) -> None:
        observed = self._freshest_ns.get(symbol)
        if observed is None:
            raise RuntimeError(f"no market data for {symbol}")
        if now_ns < observed or now_ns - observed > self.config.max_feed_age_ns:
            raise RuntimeError(f"stale market data for {symbol}")


@dataclass(frozen=True, slots=True)
class SessionConfig:
    timezone_name: str
    open_minute: int
    close_minute: int
    weekdays: tuple[int, ...]
    # Exchange calendar overlays on top of the static weekday/hours window. A
    # naive weekday 09:30-16:00 clock is wrong on exchange holidays (fully
    # closed on a weekday) and half-days (early close), where the broker accepts
    # orders it can never fill. ``holidays`` are ISO ``YYYY-MM-DD`` full closures;
    # ``early_closes`` map an ISO date to its early close minute-of-day.
    holidays: frozenset[str] = frozenset()
    early_closes: tuple[tuple[str, int], ...] = ()

    def __post_init__(self) -> None:
        ZoneInfo(self.timezone_name)
        if not 0 <= self.open_minute < self.close_minute <= 24 * 60:
            raise ValueError("invalid session open/close minutes")
        if not self.weekdays or any(day < 0 or day > 6 for day in self.weekdays):
            raise ValueError("weekdays must use datetime weekday values 0..6")
        for day_iso, close_min in self.early_closes:
            if not self.open_minute < close_min <= self.close_minute:
                raise ValueError(
                    f"early close for {day_iso} must fall within the regular session"
                )


class MarketSessionGate:
    def __init__(self, config: SessionConfig) -> None:
        self.config = config
        self._zone = ZoneInfo(config.timezone_name)
        self._early_closes = dict(config.early_closes)

    def session_reason(self, timestamp_ns: int) -> str:
        """Return '' if the configured session is open at ``timestamp_ns``, else a
        human-readable closed reason. Nonfatal — never raises for an ordinary closed
        session, so market-data ingestion/research can continue 24/7.

        Honors the exchange calendar overlays: a date in ``holidays`` is fully
        closed, and a date in ``early_closes`` uses its earlier close minute in
        place of the regular close."""
        local = datetime.fromtimestamp(timestamp_ns / 1_000_000_000, tz=timezone.utc)
        local = local.astimezone(self._zone)
        minute = local.hour * 60 + local.minute
        if local.weekday() not in self.config.weekdays:
            return "market session is closed for configured weekday"
        day_iso = local.date().isoformat()
        if day_iso in self.config.holidays:
            return "market session is closed for exchange holiday"
        close_minute = self._early_closes.get(day_iso, self.config.close_minute)
        if not self.config.open_minute <= minute < close_minute:
            return "market session is outside configured hours"
        return ""

    def is_open(self, timestamp_ns: int) -> bool:
        """Nonfatal session check for the elective order-authorization boundary."""
        return self.session_reason(timestamp_ns) == ""

    def require_open(self, timestamp_ns: int) -> None:
        """Fatal session gate (retained for callers that must fail closed). Raises the
        typed :class:`MarketClosedError` so callers can distinguish an ordinary closed
        session from a genuine fault."""
        reason = self.session_reason(timestamp_ns)
        if reason:
            raise MarketClosedError(reason)
