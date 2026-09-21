import pytest

from cengine.execution import ExecutionConfigurationError, trading_base_url


def test_live_endpoint_needs_both_exact_gates(monkeypatch):
    monkeypatch.setenv("ALPACA_TRADING_BASE_URL", "https://api.alpaca.markets")
    monkeypatch.delenv("CENGINE_ENABLE_LIVE", raising=False)
    monkeypatch.delenv("CENGINE_LIVE_CONFIRMATION", raising=False)
    with pytest.raises(ExecutionConfigurationError):
        trading_base_url()


def test_default_endpoint_is_paper(monkeypatch):
    monkeypatch.delenv("ALPACA_TRADING_BASE_URL", raising=False)
    assert "paper-api" in trading_base_url()


def test_live_remains_blocked_without_reconciliation_certification(monkeypatch):
    monkeypatch.setenv("ALPACA_TRADING_BASE_URL", "https://api.alpaca.markets")
    monkeypatch.setenv("CENGINE_ENABLE_LIVE", "YES_I_ACCEPT_LIVE_TRADING_RISK")
    monkeypatch.setenv("CENGINE_LIVE_CONFIRMATION", "LIVE_ORDERS_MAY_LOSE_MONEY")
    monkeypatch.delenv("CENGINE_LIVE_RECONCILIATION_CERTIFIED", raising=False)
    with pytest.raises(ExecutionConfigurationError):
        trading_base_url()
