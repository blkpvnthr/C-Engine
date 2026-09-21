"""Market/session/feed gates; no trading thresholds are invented here."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from zoneinfo import ZoneInfo


@dataclass(frozen=True, slots=True)
class MarketGateConfig:
    max_feed_age_ns: int
    max_bar_gap_ns: int

    def __post_init__(self) -> None:
        if self.max_feed_age_ns <= 0 or self.max_bar_gap_ns <= 0:
            raise ValueError("freshness limits must be explicitly positive")


class MarketSafetyGate:
    def __init__(self, config: MarketGateConfig) -> None:
        self.config = config
        self._last_event_ns: dict[str, int] = {}
        self._last_bar_ns: dict[str, int] = {}

    def observe(self, symbol: str, timestamp_ns: int, *, is_bar: bool = False) -> None:
        previous = self._last_event_ns.get(symbol)
        if previous is not None and timestamp_ns < previous:
            raise RuntimeError(f"out-of-order market event for {symbol}")
        self._last_event_ns[symbol] = timestamp_ns
        if is_bar:
            prior_bar = self._last_bar_ns.get(symbol)
            if prior_bar is not None and timestamp_ns - prior_bar > self.config.max_bar_gap_ns:
                raise RuntimeError(f"missing-bar gate tripped for {symbol}")
            self._last_bar_ns[symbol] = timestamp_ns

    def require_fresh(self, symbol: str, now_ns: int) -> None:
        observed = self._last_event_ns.get(symbol)
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

    def __post_init__(self) -> None:
        ZoneInfo(self.timezone_name)
        if not 0 <= self.open_minute < self.close_minute <= 24 * 60:
            raise ValueError("invalid session open/close minutes")
        if not self.weekdays or any(day < 0 or day > 6 for day in self.weekdays):
            raise ValueError("weekdays must use datetime weekday values 0..6")


class MarketSessionGate:
    def __init__(self, config: SessionConfig) -> None:
        self.config = config
        self._zone = ZoneInfo(config.timezone_name)

    def require_open(self, timestamp_ns: int) -> None:
        local = datetime.fromtimestamp(timestamp_ns / 1_000_000_000, tz=timezone.utc)
        local = local.astimezone(self._zone)
        minute = local.hour * 60 + local.minute
        if local.weekday() not in self.config.weekdays:
            raise RuntimeError("market session is closed for configured weekday")
        if not self.config.open_minute <= minute < self.config.close_minute:
            raise RuntimeError("market session is outside configured hours")
