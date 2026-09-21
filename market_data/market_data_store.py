#!/usr/bin/env python3
from __future__ import annotations

import argparse
import queue
import threading
import time
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Optional
from zoneinfo import ZoneInfo

import h5py
import numpy as np


SCHEMA_VERSION = 2
DEFAULT_PRICE_SCALE = 10_000
DEFAULT_CHUNK_ROWS = 4096
DEFAULT_FLUSH_INTERVAL_SECONDS = 1.0
DEFAULT_QUEUE_SIZE = 250_000
NY_TZ = ZoneInfo("America/New_York")

_UTF8 = h5py.string_dtype(encoding="utf-8")

QUOTE_DTYPE = np.dtype([
    ("sequence", "<u8"),
    ("timestamp_ns", "<u8"),
    ("received_ns", "<u8"),
    ("bid_price_ticks", "<i8"),
    ("bid_size", "<u8"),
    ("ask_price_ticks", "<i8"),
    ("ask_size", "<u8"),
    ("bid_exchange", _UTF8),
    ("ask_exchange", _UTF8),
    ("conditions", _UTF8),
    ("tape", _UTF8),
])

TRADE_DTYPE = np.dtype([
    ("sequence", "<u8"),
    ("timestamp_ns", "<u8"),
    ("received_ns", "<u8"),
    ("trade_id", "<u8"),
    ("price_ticks", "<i8"),
    ("size", "<u8"),
    ("exchange", _UTF8),
    ("conditions", _UTF8),
    ("tape", _UTF8),
])

BAR_DTYPE = np.dtype([
    ("sequence", "<u8"),
    ("timestamp_ns", "<u8"),
    ("received_ns", "<u8"),
    ("open_ticks", "<i8"),
    ("high_ticks", "<i8"),
    ("low_ticks", "<i8"),
    ("close_ticks", "<i8"),
    ("volume", "<u8"),
    ("trade_count", "<u8"),
    ("vwap_ticks", "<i8"),
    ("has_vwap", "u1"),
])

FACTOR_DTYPE = np.dtype([
    ("sequence", "<u8"),
    ("timestamp_ns", "<u8"),
    ("received_ns", "<u8"),
    ("value_ticks", "<i8"),
    ("provider", _UTF8),
])

ANALYTICS_SUMMARY_DTYPE = np.dtype([
    ("timestamp_ns", "<u8"),
    ("model_version", _UTF8),
    ("name", _UTF8),
    ("value", "<f8"),
])


class MarketDataStoreError(RuntimeError):
    pass


class PersistenceQueueFull(MarketDataStoreError):
    """Archival persistence fell behind; data must not be silently discarded."""


def _trading_date_from_ns(timestamp_ns: int) -> date:
    dt = datetime.fromtimestamp(timestamp_ns / 1_000_000_000, tz=timezone.utc)
    return dt.astimezone(NY_TZ).date()


def _daily_path(root: Path, trading_date: date) -> Path:
    return (
        root
        / f"{trading_date.year:04d}"
        / f"{trading_date.month:02d}"
        / f"market_{trading_date.isoformat()}.h5"
    )


def create_daily_file(
    root: Path,
    trading_date: date,
    *,
    price_scale: int = DEFAULT_PRICE_SCALE,
    provider: str,
    exclusive: bool = True,
) -> Path:
    """Create a valid daily HDF5 container.

    exclusive=True refuses to overwrite an existing trading-day file.
    """
    if not provider or not provider.strip():
        raise ValueError("provider must be explicitly identified")

    path = _daily_path(root, trading_date)
    path.parent.mkdir(parents=True, exist_ok=True)

    mode = "x" if exclusive else "a"
    with h5py.File(path, mode, libver="latest") as h5:
        h5.attrs.setdefault("schema_version", SCHEMA_VERSION)
        h5.attrs.setdefault("trading_date", trading_date.isoformat())
        h5.attrs.setdefault("layout", "one_file_per_trading_day")
        h5.attrs.setdefault("price_scale", int(price_scale))
        h5.attrs.setdefault("provider", provider)
        h5.attrs.setdefault("created_ns", time.time_ns())
        h5.attrs["last_opened_ns"] = time.time_ns()

        h5.require_group("metadata")
        h5.require_group("equities")
        h5.require_group("factors")
        h5.require_group("analytics")

    return path


@dataclass(slots=True)
class WriterStats:
    submitted: int = 0
    written: int = 0
    flushes: int = 0
    rotations: int = 0
    failures: int = 0
    queue_high_watermark: int = 0


class DailyHDF5Writer:
    """Buffered, append-only daily HDF5 writer.

    The writer owns a dedicated persistence queue and background thread.
    Queue overflow is an explicit error: archival market data is never
    silently dropped.

    Supported event shapes are the normalized QuoteEvent, TradeEvent,
    BarEvent, and IndexEvent produced by alpaca_sip_stream.py.
    """

    def __init__(
        self,
        root: str | Path = "market_data",
        *,
        price_scale: int = DEFAULT_PRICE_SCALE,
        chunk_rows: int = DEFAULT_CHUNK_ROWS,
        compression: Optional[str] = "gzip",
        compression_opts: int = 4,
        flush_interval_seconds: float = DEFAULT_FLUSH_INTERVAL_SECONDS,
        queue_size: int = DEFAULT_QUEUE_SIZE,
        provider: str,
    ) -> None:
        if price_scale <= 0:
            raise ValueError("price_scale must be positive")
        if chunk_rows <= 0:
            raise ValueError("chunk_rows must be positive")
        if flush_interval_seconds <= 0:
            raise ValueError("flush_interval_seconds must be positive")
        if queue_size <= 0:
            raise ValueError("queue_size must be positive")
        if not provider or not provider.strip():
            raise ValueError("provider must be explicitly identified")

        self.root = Path(root)
        self.price_scale = int(price_scale)
        self.chunk_rows = int(chunk_rows)
        self.compression = compression
        self.compression_opts = compression_opts
        self.flush_interval_seconds = float(flush_interval_seconds)
        self.provider = provider

        self._queue: queue.Queue[Any] = queue.Queue(maxsize=queue_size)
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._h5: Optional[h5py.File] = None
        self._trading_date: Optional[date] = None
        self._last_flush_monotonic = time.monotonic()
        self._failure: Optional[BaseException] = None
        self.stats = WriterStats()

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._failure = None
        self._thread = threading.Thread(
            target=self._run,
            name="daily-hdf5-writer",
            daemon=True,
        )
        self._thread.start()

    def submit(self, event: Any) -> None:
        self.raise_if_failed()
        if self._thread is None or not self._thread.is_alive():
            raise MarketDataStoreError("DailyHDF5Writer is not running")

        try:
            self._queue.put_nowait(event)
        except queue.Full as exc:
            raise PersistenceQueueFull(
                "HDF5 persistence queue is full; refusing to silently drop "
                "market data"
            ) from exc

        self.stats.submitted += 1
        self.stats.queue_high_watermark = max(
            self.stats.queue_high_watermark,
            self._queue.qsize(),
        )

    def flush(self, timeout: Optional[float] = None) -> None:
        self.raise_if_failed()
        deadline = None if timeout is None else time.monotonic() + timeout

        while self._queue.unfinished_tasks:
            self.raise_if_failed()
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError("timed out waiting for HDF5 writer queue")
            time.sleep(0.005)

        if self._h5 is not None:
            self._h5.flush()
            self.stats.flushes += 1

    def close(self, *, drain: bool = True) -> None:
        if self._thread is None:
            return

        if drain:
            self.flush()

        self._stop.set()
        self._thread.join(timeout=10.0)

        if self._thread.is_alive():
            raise MarketDataStoreError("HDF5 writer thread did not stop")

        self._thread = None
        self.raise_if_failed()

    def raise_if_failed(self) -> None:
        if self._failure is not None:
            raise MarketDataStoreError(
                f"HDF5 writer failed: {self._failure}"
            ) from self._failure

    def __enter__(self) -> "DailyHDF5Writer":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close(drain=exc is None)

    def _run(self) -> None:
        try:
            while not self._stop.is_set() or not self._queue.empty():
                try:
                    event = self._queue.get(timeout=0.1)
                except queue.Empty:
                    self._periodic_flush()
                    continue

                try:
                    self._append_event(event)
                    self.stats.written += 1
                finally:
                    self._queue.task_done()

                self._periodic_flush()

            if self._h5 is not None:
                self._h5.flush()
                self.stats.flushes += 1
        except BaseException as exc:
            self.stats.failures += 1
            self._failure = exc
        finally:
            if self._h5 is not None:
                try:
                    self._h5.flush()
                    self._h5.close()
                finally:
                    self._h5 = None
                    self._trading_date = None

    def _periodic_flush(self) -> None:
        now = time.monotonic()
        if (
            self._h5 is not None
            and now - self._last_flush_monotonic >= self.flush_interval_seconds
        ):
            self._h5.flush()
            self.stats.flushes += 1
            self._last_flush_monotonic = now

    def _rotate_for_timestamp(self, timestamp_ns: int) -> None:
        trading_date = _trading_date_from_ns(int(timestamp_ns))
        if self._h5 is not None and trading_date == self._trading_date:
            return

        if self._h5 is not None:
            self._h5.flush()
            self._h5.close()
            self._h5 = None
            self.stats.rotations += 1

        path = create_daily_file(
            self.root,
            trading_date,
            price_scale=self.price_scale,
            provider=self.provider,
            exclusive=False,
        )
        self._h5 = h5py.File(path, "a", libver="latest")
        self._trading_date = trading_date
        self._last_flush_monotonic = time.monotonic()

    def _dataset(self, path: str, dtype: np.dtype) -> h5py.Dataset:
        assert self._h5 is not None
        if path in self._h5:
            return self._h5[path]

        parent, name = path.rsplit("/", 1)
        group = self._h5.require_group(parent)

        kwargs: dict[str, Any] = {
            "shape": (0,),
            "maxshape": (None,),
            "dtype": dtype,
            "chunks": (self.chunk_rows,),
            "shuffle": True,
            "fletcher32": True,
        }
        if self.compression:
            kwargs["compression"] = self.compression
            if self.compression == "gzip":
                kwargs["compression_opts"] = self.compression_opts

        ds = group.create_dataset(name, **kwargs)
        ds.attrs["append_only"] = True
        ds.attrs["schema_version"] = SCHEMA_VERSION
        return ds

    @staticmethod
    def _append_row(ds: h5py.Dataset, row: tuple[Any, ...]) -> None:
        idx = ds.shape[0]
        ds.resize((idx + 1,))
        ds[idx] = row

    def _append_event(self, event: Any) -> None:
        event_type = type(event).__name__
        timestamp_ns = int(event.timestamp_ns)
        self._rotate_for_timestamp(timestamp_ns)
        assert self._h5 is not None

        symbol = str(event.symbol).strip().upper()
        if not symbol:
            raise ValueError("event symbol is empty")

        if event_type == "QuoteEvent":
            ds = self._dataset(
                f"/equities/{symbol}/quotes",
                QUOTE_DTYPE,
            )
            self._append_row(ds, (
                int(event.sequence),
                timestamp_ns,
                int(event.received_ns),
                int(event.bid_price_ticks),
                int(event.bid_size),
                int(event.ask_price_ticks),
                int(event.ask_size),
                str(event.bid_exchange),
                str(event.ask_exchange),
                ",".join(event.conditions),
                str(event.tape),
            ))
            return

        if event_type == "TradeEvent":
            ds = self._dataset(
                f"/equities/{symbol}/trades",
                TRADE_DTYPE,
            )
            self._append_row(ds, (
                int(event.sequence),
                timestamp_ns,
                int(event.received_ns),
                int(event.trade_id),
                int(event.price_ticks),
                int(event.size),
                str(event.exchange),
                ",".join(event.conditions),
                str(event.tape),
            ))
            return

        if event_type == "BarEvent":
            ds = self._dataset(
                f"/equities/{symbol}/bars",
                BAR_DTYPE,
            )
            has_vwap = event.vwap_ticks is not None
            self._append_row(ds, (
                int(event.sequence),
                timestamp_ns,
                int(event.received_ns),
                int(event.open_ticks),
                int(event.high_ticks),
                int(event.low_ticks),
                int(event.close_ticks),
                int(event.volume),
                int(event.trade_count),
                int(event.vwap_ticks or 0),
                int(has_vwap),
            ))
            return

        if event_type == "IndexEvent":
            if symbol not in {"VIX", "VXN"}:
                raise ValueError(
                    f"unsupported factor {symbol!r}; expected VIX or VXN"
                )
            ds = self._dataset(
                f"/factors/{symbol}/observations",
                FACTOR_DTYPE,
            )
            self._append_row(ds, (
                int(event.sequence),
                timestamp_ns,
                int(event.received_ns),
                int(event.value_ticks),
                str(event.provider),
            ))
            return

        raise TypeError(f"unsupported market-data event type: {event_type}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a daily HDF5 market-data container."
    )
    parser.add_argument(
        "date",
        help="Trading date in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--provider",
        required=True,
        help="Explicit provenance label for data stored in this file.",
    )
    parser.add_argument("--root", default="market_data")
    parser.add_argument(
        "--allow-existing",
        action="store_true",
        help="Open/create instead of refusing an existing daily file.",
    )
    args = parser.parse_args()

    trading_date = date.fromisoformat(args.date)
    path = create_daily_file(
        Path(args.root),
        trading_date,
        provider=args.provider,
        exclusive=not args.allow_existing,
    )
    print(path)


if __name__ == "__main__":
    main()
