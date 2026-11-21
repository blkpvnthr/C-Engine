#!/usr/bin/env python3
"""Sustenance — unified autonomous trading application entry point.

This is the single operational entry point for the integrated system. It is the
orchestration / configuration / lifecycle layer on top of the existing native
C++ engine (``run.py`` / ``_cengine_native``) and the migrated real-time IEX
market scanner (``market_scan/``).

Responsibilities (Phase 1/2 of the integration):

  * load Alpaca credentials from ``.env`` (via :func:`run.load_runtime_env`);
  * resolve the operating mode from CLI + ``.env`` (paper default; live stays
    gated by the engine's existing triple-confirmation env vars);
  * load the canonical ``universe.txt`` into the in-memory
    :class:`canonical_universe.CanonicalUniverseManager` and seed the engine's
    subscription universe from it;
  * auto-start the migrated market scanner as an in-process daemon (the sole
    runtime publisher of ``universe.txt`` + ``live_context_<mode>.json``);
  * delegate the full engine lifecycle (broker reconcile, market data, strategy
    coordination, native risk, OrderManager, shutdown) to :func:`run.main`.

In paper/live the engine's own :class:`protective_exit.ProtectiveExitCoordinator`
(wired into ``run.MarketEngine``) is the single authoritative protective-exit
path and EXECUTES the mandatory −0.5% close through the native OrderManager. The
thread-based :class:`ProtectiveExitMonitor` here is used ONLY in ``--shadow``
(where no engine runs) and is detection/recommendation only.

Modes::

    python3 trade.py                 # paper autonomous (default)
    python3 trade.py --paper
    python3 trade.py --live          # requires the engine's live gates in .env
    python3 trade.py --shadow        # scanner + universe + protective detection,
                                     #   NO engine order path
    python3 trade.py --validate      # offline readiness checks, no network/broker
    python3 trade.py --status        # read-only account + universe snapshot

``--backtest`` / ``--replay`` / ``--research`` are recognised but defer to a
later integration milestone (they exit with a clear, non-silent message).

SAFETY: this module never forces live trading on and never submits live orders
during development/testing. Live requires the engine's explicit triple-gate env
confirmations. Protective-exit *execution* happens inside the engine's single
OrderManager path (never a parallel broker writer).
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import threading
from dataclasses import dataclass
from pathlib import Path

import run  # existing native-engine composition root

logger = logging.getLogger("trade")

REPO_ROOT = Path(__file__).resolve().parent
UNIVERSE_PATH = REPO_ROOT / "universe.txt"

# Modes that must never establish live broker-writing connectivity.
_OFFLINE_MODES = {"validate"}
_HISTORICAL_MODES = {"backtest", "replay", "research"}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CliOptions:
    mode: str  # paper | live | shadow | validate | status | backtest | replay | research


def parse_args(argv: list[str]) -> CliOptions:
    """Parse argv into a single resolved mode. Paper is the default."""
    parser = argparse.ArgumentParser(
        prog="trade.py",
        description="Sustenance unified autonomous trading engine.",
    )
    group = parser.add_mutually_exclusive_group()
    for flag, mode in (
        ("--paper", "paper"),
        ("--live", "live"),
        ("--shadow", "shadow"),
        ("--validate", "validate"),
        ("--status", "status"),
        ("--backtest", "backtest"),
        ("--replay", "replay"),
        ("--research", "research"),
    ):
        group.add_argument(
            flag, dest="mode", action="store_const", const=mode,
            help=f"run in {mode} mode",
        )
    parser.set_defaults(mode="paper")
    ns = parser.parse_args(argv)
    return CliOptions(mode=ns.mode)


# ---------------------------------------------------------------------------
# Scanner host adapter
# ---------------------------------------------------------------------------
class _BrokerAdapter:
    """Minimal broker surface the market scanner needs from its host.

    The scanner only reads ``api_key``/``secret_key`` (for its own IEX stream +
    historical client) and the two read-only methods below (to keep open
    positions / pending orders subscribed). It never submits orders through this
    adapter -- all execution stays on the engine's single OrderManager path.
    Both read methods are fail-safe and called off the scanner's event loop.
    """

    def __init__(self, api_key: str, secret_key: str, *, paper: bool) -> None:
        self.api_key = api_key
        self.secret_key = secret_key
        self._paper = paper
        self._client = None  # lazy: constructed on first broker read (no network at init)

    def _trading_client(self):
        if self._client is None:
            from alpaca.trading.client import TradingClient
            self._client = TradingClient(self.api_key, self.secret_key, paper=self._paper)
        return self._client

    def get_all_open_positions(self) -> list:
        try:
            return list(self._trading_client().get_all_positions())
        except Exception as exc:  # noqa: BLE001 - never raise into the scanner loop
            logger.debug("broker positions read failed: %s", exc)
            return []

    def list_open_orders(self) -> list:
        try:
            from alpaca.trading.enums import QueryOrderStatus
            from alpaca.trading.requests import GetOrdersRequest
            req = GetOrdersRequest(status=QueryOrderStatus.OPEN)
            return list(self._trading_client().get_orders(filter=req))
        except Exception as exc:  # noqa: BLE001
            logger.debug("broker open-orders read failed: %s", exc)
            return []


class _ScannerHostConfig:
    def __init__(self, *, paper: bool) -> None:
        self.paper = paper


class ScannerHost:
    """The object the scanner calls ``executor`` -- just a credential + state
    provider. Mirrors the attributes ``market_scan.scanner.MarketScanner`` reads
    (``.broker`` and ``.config.paper``); deliberately exposes no order path."""

    def __init__(self, api_key: str, secret_key: str, *, paper: bool) -> None:
        self.broker = _BrokerAdapter(api_key, secret_key, paper=paper)
        self.config = _ScannerHostConfig(paper=paper)


# ---------------------------------------------------------------------------
# Independent protective-exit monitor (−0.5% detection)
# ---------------------------------------------------------------------------
class ProtectiveExitMonitor:
    """Scanner-independent, strategy-independent position loss monitor.

    Polls open positions directly from the broker on a fixed cadence and surfaces
    a protective-exit recommendation for any position whose unrealised return is
    at or below the configured threshold (default −0.5%). This runs regardless of
    whether the scanner is healthy, the universe is fresh, or any entry strategy
    is enabled -- satisfying the spec's independence requirement for *detection*.

    MILESTONE NOTE: this monitor currently emits recommendations (logged); it
    does NOT yet submit exits. Routing protective exits through the native risk
    gate + OrderManager is the next integration milestone. It therefore never
    places an order and is safe to run in paper/shadow.
    """

    def __init__(
        self,
        broker: _BrokerAdapter,
        *,
        threshold: float = -0.005,
        interval_s: float = 5.0,
        on_breach=None,
    ) -> None:
        self.broker = broker
        self.threshold = threshold
        self.interval_s = interval_s
        self._on_breach = on_breach or self._log_recommendation
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @staticmethod
    def unrealized_return(position) -> float | None:
        """Fractional unrealised return for an alpaca-py position, or None."""
        plpc = getattr(position, "unrealized_plpc", None)
        if plpc is not None:
            try:
                return float(plpc)  # alpaca reports this as a fraction already
            except (TypeError, ValueError):
                pass
        # Fallback from avg entry + current price.
        try:
            entry = float(position.avg_entry_price)
            price = float(position.current_price)
            if entry:
                return (price - entry) / entry
        except (TypeError, ValueError, AttributeError):
            return None
        return None

    def scan_once(self) -> list[tuple[str, float]]:
        """Return (symbol, return) for every position at/below threshold."""
        breaches: list[tuple[str, float]] = []
        for pos in self.broker.get_all_open_positions():
            ret = self.unrealized_return(pos)
            sym = str(getattr(pos, "symbol", "") or "").upper()
            if ret is not None and sym and ret <= self.threshold:
                breaches.append((sym, ret))
        return breaches

    def _log_recommendation(self, symbol: str, ret: float) -> None:
        logger.warning(
            "[protective-exit] RECOMMEND EXIT %s: unrealised %.3f%% <= %.3f%% "
            "(detection-only this milestone; not submitted)",
            symbol, ret * 100.0, self.threshold * 100.0,
        )

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                for sym, ret in self.scan_once():
                    self._on_breach(sym, ret)
            except Exception as exc:  # noqa: BLE001 - monitor must never die
                logger.error("protective-exit monitor error: %s", exc)
            self._stop.wait(self.interval_s)

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._loop, name="protective-exit", daemon=True
        )
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def resolve_credentials() -> tuple[str | None, str | None]:
    """Return (api_key, secret_key) from the mirrored env, or (None, None)."""
    key = os.environ.get("ALPACA_API_KEY") or os.environ.get("APCA_API_KEY_ID")
    secret = os.environ.get("ALPACA_SECRET_KEY") or os.environ.get("APCA_API_SECRET_KEY")
    return (key or None, secret or None)


def is_live_endpoint() -> bool:
    """True iff ALPACA_TRADING_BASE_URL is the live Alpaca endpoint.

    Mirrors run.py's canonical comparison to ``https://api.alpaca.markets`` but
    tolerates a trailing ``/v2`` (the form the real .env carries) and trailing
    slashes. The paper host contains ``paper-`` and never matches.
    """
    url = os.environ.get("ALPACA_TRADING_BASE_URL", "").rstrip("/")
    if url.endswith("/v2"):
        url = url[: -len("/v2")]
    return url == "https://api.alpaca.markets"


def seed_engine_universe(manager) -> tuple[str, ...]:
    """Seed ``CENGINE_SYMBOLS`` from the canonical universe unless an operator
    override is already present. Returns the symbol tuple the engine will use.

    An explicit ``CENGINE_SYMBOLS`` override also pins the subscription set: it
    sets ``CENGINE_DYNAMIC_UNIVERSE=0`` so the engine's dynamic resubscription
    does not override the operator's chosen symbols. Canonical seeding leaves
    dynamic resubscription enabled (its default)."""
    override = os.environ.get("CENGINE_SYMBOLS", "").strip()
    if override:
        logger.info("CENGINE_SYMBOLS override present; pinning symbols "
                    "(dynamic universe resubscription disabled)")
        os.environ["CENGINE_DYNAMIC_UNIVERSE"] = "0"
        return tuple(s.strip().upper() for s in override.split(",") if s.strip())
    symbols = manager.symbols()
    os.environ["CENGINE_SYMBOLS"] = ",".join(symbols)
    return symbols


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------
def run_validate() -> int:
    """Offline readiness checks. No network, no broker, no orders. Returns exit code."""
    checks: list[tuple[str, bool, str]] = []

    def check(name: str, fn) -> None:
        try:
            ok, detail = fn()
        except Exception as exc:  # noqa: BLE001
            ok, detail = False, f"{type(exc).__name__}: {exc}"
        checks.append((name, ok, detail))

    def _native():
        import _cengine_native  # noqa: F401
        return True, "_cengine_native importable"

    def _scanner_pkg():
        import market_scan.scanner  # noqa: F401
        return True, "market_scan.scanner graph importable"

    def _scan_deps():
        import importlib.util as u
        missing = [m for m in ("alpaca", "pandas", "requests") if not u.find_spec(m)]
        return (not missing), ("all present" if not missing else f"missing: {missing}")

    def _creds():
        key, secret = resolve_credentials()
        return (bool(key and secret),
                "ALPACA_API_KEY/SECRET present" if (key and secret) else "MISSING")

    def _universe():
        from canonical_universe import CanonicalUniverseManager
        mgr = CanonicalUniverseManager(UNIVERSE_PATH)
        syms = mgr.symbols()
        preview = ", ".join(syms[:8]) + ("…" if len(syms) > 8 else "")
        return True, f"{len(syms)} symbols (v{mgr.version()}): {preview}"

    def _scan_config():
        from market_scan.config import MarketScanConfig
        cfg = MarketScanConfig.from_env()
        return True, (f"enabled={cfg.enabled} feed={cfg.data_feed} "
                      f"rank={cfg.rank_refresh_seconds}s loss={cfg.loss_removal_threshold}")

    check("native module (_cengine_native)", _native)
    check("market_scan package", _scanner_pkg)
    check("scanner runtime deps", _scan_deps)
    check("canonical universe.txt", _universe)
    check("market scan config", _scan_config)
    check("alpaca credentials (.env)", _creds)

    print("\nSUSTENANCE — readiness validation")
    print("-" * 60)
    critical_ok = True
    for name, ok, detail in checks:
        mark = "PASS" if ok else "FAIL"
        print(f"  [{mark}] {name:34} {detail}")
        # Credentials are not required to pass offline validation (CI has none).
        if not ok and name != "alpaca credentials (.env)":
            critical_ok = False
    print("-" * 60)
    print("RESULT:", "READY" if critical_ok else "NOT READY")
    return 0 if critical_ok else 1


def run_status() -> int:
    """Read-only snapshot: canonical universe + (if creds present) account."""
    from canonical_universe import CanonicalUniverseManager

    mgr = CanonicalUniverseManager(UNIVERSE_PATH)
    print("\nSUSTENANCE — status")
    print("-" * 60)
    print(f"  Universe file      : {UNIVERSE_PATH}")
    print(f"  Universe version   : {mgr.version()}")
    print(f"  Trading universe   : {len(mgr.symbols())} symbols")
    print(f"  Trading env        : {'LIVE' if is_live_endpoint() else 'PAPER'}")

    key, secret = resolve_credentials()
    if not (key and secret):
        print("  Account            : (no credentials in .env; skipping broker read)")
        print("-" * 60)
        return 0
    try:
        from alpaca.trading.client import TradingClient
        client = TradingClient(key, secret, paper=not is_live_endpoint())
        acct = client.get_account()
        print(f"  Account equity     : {getattr(acct, 'equity', '?')}")
        print(f"  Cash               : {getattr(acct, 'cash', '?')}")
        print(f"  Buying power       : {getattr(acct, 'buying_power', '?')}")
        positions = client.get_all_positions()
        print(f"  Open positions     : {len(positions)}")
    except Exception as exc:  # noqa: BLE001
        print(f"  Account            : broker read failed: {exc}")
    print("-" * 60)
    return 0


def _start_scanner(host: ScannerHost):
    """Start the market scanner daemon; returns the handle or None if disabled."""
    try:
        import market_scan
        handle = market_scan.start_market_scanner(host)
        if handle is None:
            logger.info("market scanner not started (disabled or lock held)")
        return handle
    except Exception as exc:  # noqa: BLE001 - never block the engine on scanner init
        logger.error("market scanner start failed: %s", exc, exc_info=True)
        return None


def run_autonomous(mode: str) -> int:
    """paper | live | shadow autonomous run."""
    from canonical_universe import CanonicalUniverseManager

    paper = mode != "live"
    key, secret = resolve_credentials()
    if not (key and secret):
        print("ERROR: Alpaca credentials missing from .env "
              "(set ALPACA_API_KEY / ALPACA_SECRET_KEY).", file=sys.stderr)
        return 1
    if mode == "live" and not is_live_endpoint():
        print("ERROR: --live requested but ALPACA_TRADING_BASE_URL is not the live "
              "endpoint. Configure the live endpoint + the engine's live gates in "
              ".env and retry.", file=sys.stderr)
        return 1

    manager = CanonicalUniverseManager(UNIVERSE_PATH)
    symbols = seed_engine_universe(manager)
    logger.info("canonical universe v%d seeded %d symbols into the engine",
                manager.version(), len(symbols))

    host = ScannerHost(key, secret, paper=paper)
    scanner_handle = _start_scanner(host)
    shadow_monitor: ProtectiveExitMonitor | None = None

    try:
        if mode == "shadow":
            # No engine runs in shadow, so the thread monitor is the detection
            # path here (recommend-only; it never submits).
            shadow_monitor = ProtectiveExitMonitor(
                host.broker,
                threshold=float(os.environ.get("MARKET_SCAN_LOSS_THRESHOLD", "-0.005")),
                interval_s=float(os.environ.get("MARKET_SCAN_LOSS_MONITOR_SECONDS", "5")),
            )
            shadow_monitor.start()
            print("\n[shadow] scanner + canonical universe + protective detection "
                  "running; engine order path disabled. Ctrl-C to stop.\n")
            _run_shadow_forever()
            return 0
        # paper / live: the engine's authoritative ProtectiveExitCoordinator (in
        # run.MarketEngine) owns and EXECUTES protective exits through the native
        # OrderManager path. We do NOT start a second (thread) monitor here --
        # one authoritative coordinator only. Delegate the lifecycle to run.main().
        run.main()
        return 0
    except KeyboardInterrupt:
        return 0
    finally:
        if shadow_monitor is not None:
            shadow_monitor.stop()
        if scanner_handle is not None:
            try:
                scanner_handle.stop()
            except Exception as exc:  # noqa: BLE001
                logger.warning("scanner stop error: %s", exc)


def _run_shadow_forever() -> None:
    stop = threading.Event()

    def _sig(*_a):
        stop.set()
    try:
        import signal
        signal.signal(signal.SIGINT, _sig)
        signal.signal(signal.SIGTERM, _sig)
    except Exception:  # noqa: BLE001
        pass
    while not stop.is_set():
        stop.wait(1.0)


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=os.environ.get("TRADE_LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    opts = parse_args(sys.argv[1:] if argv is None else argv)

    # Load .env for every mode except where we explicitly want a pristine env.
    run.load_runtime_env()

    if opts.mode == "validate":
        return run_validate()
    if opts.mode == "status":
        return run_status()
    if opts.mode in _HISTORICAL_MODES:
        print(f"[{opts.mode}] mode is recognised but not yet implemented in this "
              f"integration milestone. Use --paper / --shadow / --validate.",
              file=sys.stderr)
        return 2
    return run_autonomous(opts.mode)


if __name__ == "__main__":
    raise SystemExit(main())
