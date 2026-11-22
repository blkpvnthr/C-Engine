"""Tests for the unified trade.py entry point (pure/host helpers, no network)."""

from __future__ import annotations

import os

import pytest

import trade


# ------------------------------------------------------------------- arg parsing
def test_default_mode_is_paper():
    assert trade.parse_args([]).mode == "paper"


@pytest.mark.parametrize("flag,mode", [
    ("--paper", "paper"), ("--live", "live"), ("--shadow", "shadow"),
    ("--validate", "validate"), ("--status", "status"),
    ("--backtest", "backtest"), ("--replay", "replay"), ("--research", "research"),
])
def test_each_flag_resolves(flag, mode):
    assert trade.parse_args([flag]).mode == mode


def test_modes_are_mutually_exclusive():
    with pytest.raises(SystemExit):
        trade.parse_args(["--paper", "--live"])


# --------------------------------------------------------------------- live/env
def test_is_live_endpoint(monkeypatch):
    monkeypatch.setenv("ALPACA_TRADING_BASE_URL", "https://api.alpaca.markets/v2")
    assert trade.is_live_endpoint() is True
    monkeypatch.setenv("ALPACA_TRADING_BASE_URL", "https://paper-api.alpaca.markets/v2")
    assert trade.is_live_endpoint() is False


def test_resolve_credentials_prefers_alpaca_then_apca(monkeypatch):
    monkeypatch.delenv("ALPACA_API_KEY", raising=False)
    monkeypatch.delenv("ALPACA_SECRET_KEY", raising=False)
    monkeypatch.setenv("APCA_API_KEY_ID", "k")
    monkeypatch.setenv("APCA_API_SECRET_KEY", "s")
    assert trade.resolve_credentials() == ("k", "s")
    monkeypatch.setenv("ALPACA_API_KEY", "K")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "S")
    assert trade.resolve_credentials() == ("K", "S")


# ------------------------------------------------------------- universe seeding
class _FakeManager:
    def __init__(self, symbols): self._s = tuple(symbols)
    def symbols(self): return self._s
    def version(self): return 1


def test_seed_engine_universe_sets_env(monkeypatch):
    monkeypatch.delenv("CENGINE_SYMBOLS", raising=False)
    syms = trade.seed_engine_universe(_FakeManager(["NVDA", "AMD"]))
    assert syms == ("NVDA", "AMD")
    assert os.environ["CENGINE_SYMBOLS"] == "NVDA,AMD"


def test_seed_engine_universe_respects_operator_override(monkeypatch):
    monkeypatch.setenv("CENGINE_SYMBOLS", "SPY, QQQ")
    monkeypatch.setenv("CENGINE_DYNAMIC_UNIVERSE", "1")  # owned by monkeypatch -> auto-revert
    syms = trade.seed_engine_universe(_FakeManager(["NVDA", "AMD"]))
    assert syms == ("SPY", "QQQ")  # override wins, canonical not applied
    assert os.environ["CENGINE_SYMBOLS"] == "SPY, QQQ"
    # An explicit override pins symbols: dynamic resubscription is disabled.
    assert os.environ["CENGINE_DYNAMIC_UNIVERSE"] == "0"


def test_seed_engine_universe_canonical_leaves_dynamic_enabled(monkeypatch):
    monkeypatch.delenv("CENGINE_SYMBOLS", raising=False)
    monkeypatch.delenv("CENGINE_DYNAMIC_UNIVERSE", raising=False)
    trade.seed_engine_universe(_FakeManager(["NVDA", "AMD"]))
    # Canonical seeding does not pin; dynamic stays at its enabled default.
    assert os.environ.get("CENGINE_DYNAMIC_UNIVERSE", "1") != "0"


# ----------------------------------------------------- protective-exit monitor
class _Pos:
    def __init__(self, symbol, plpc=None, entry=None, price=None):
        self.symbol = symbol
        if plpc is not None:
            self.unrealized_plpc = plpc
        if entry is not None:
            self.avg_entry_price = entry
        if price is not None:
            self.current_price = price


class _FakeBroker:
    def __init__(self, positions): self._p = positions
    def get_all_open_positions(self): return self._p
    def list_open_orders(self): return []


def test_unrealized_return_from_plpc():
    assert trade.ProtectiveExitMonitor.unrealized_return(_Pos("X", plpc="-0.01")) == -0.01


def test_unrealized_return_from_prices():
    r = trade.ProtectiveExitMonitor.unrealized_return(_Pos("X", entry="100", price="99"))
    assert r == pytest.approx(-0.01)


def test_unrealized_return_none_when_unavailable():
    assert trade.ProtectiveExitMonitor.unrealized_return(_Pos("X")) is None


def test_scan_once_flags_only_breaches():
    broker = _FakeBroker([
        _Pos("WIN", plpc="0.02"),       # up, ignored
        _Pos("FLAT", plpc="-0.004"),    # above threshold, ignored
        _Pos("LOSS", plpc="-0.006"),    # below -0.005, flagged
        _Pos("EDGE", plpc="-0.005"),    # at threshold, flagged (<=)
    ])
    mon = trade.ProtectiveExitMonitor(broker, threshold=-0.005)
    breaches = dict(mon.scan_once())
    assert set(breaches) == {"LOSS", "EDGE"}


def test_monitor_callback_invoked(monkeypatch):
    broker = _FakeBroker([_Pos("LOSS", plpc="-0.02")])
    seen = []
    mon = trade.ProtectiveExitMonitor(
        broker, threshold=-0.005, on_breach=lambda s, r: seen.append((s, r))
    )
    for sym, ret in mon.scan_once():
        mon._on_breach(sym, ret)
    assert seen == [("LOSS", -0.02)]


# --------------------------------------------------------- broker adapter maps
def test_broker_adapter_maps_method_names(monkeypatch):
    adapter = trade._BrokerAdapter("k", "s", paper=True)

    class _Client:
        def get_all_positions(self): return ["p1", "p2"]
        def get_orders(self, filter=None): return ["o1"]

    adapter._client = _Client()
    assert adapter.get_all_open_positions() == ["p1", "p2"]
    assert adapter.list_open_orders() == ["o1"]


def test_broker_adapter_failsafe_on_error():
    adapter = trade._BrokerAdapter("k", "s", paper=True)

    class _Boom:
        def get_all_positions(self): raise RuntimeError("network")
        def get_orders(self, filter=None): raise RuntimeError("network")

    adapter._client = _Boom()
    assert adapter.get_all_open_positions() == []
    assert adapter.list_open_orders() == []


def test_scanner_host_exposes_contract():
    host = trade.ScannerHost("k", "s", paper=True)
    assert host.broker.api_key == "k"
    assert host.broker.secret_key == "s"
    assert host.config.paper is True
    # The host deliberately exposes NO order-submission method.
    assert not hasattr(host, "submit_order")
    assert not hasattr(host.broker, "submit_order")
