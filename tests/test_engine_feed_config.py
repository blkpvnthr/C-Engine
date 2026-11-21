"""Engine-level feed configuration: IEX default, explicit SIP, provenance."""

import pytest

import run
from market_data.feed import FeedConfigurationError

_FEED_ENV = (
    "CENGINE_ALPACA_FEED",
    "CENGINE_ALPACA_URL",
    "CENGINE_ALPACA_SANDBOX",
    "CENGINE_ARCHIVE_PROVIDER_LABEL",
)


@pytest.fixture(autouse=True)
def _clear_feed_env(monkeypatch):
    for name in _FEED_ENV:
        monkeypatch.delenv(name, raising=False)


def test_config_from_env_defaults_to_iex():
    config = run.config_from_env()
    assert config.alpaca_feed == "iex"
    assert config.alpaca_url == "wss://stream.data.alpaca.markets/v2/iex"
    assert config.archive_provider_label == "alpaca-iex"


def test_config_from_env_explicit_sip(monkeypatch):
    monkeypatch.setenv("CENGINE_ALPACA_FEED", "sip")
    config = run.config_from_env()
    assert config.alpaca_feed == "sip"
    assert config.alpaca_url.endswith("/v2/sip")
    assert config.archive_provider_label == "alpaca-sip"


def test_config_from_env_provider_label_override(monkeypatch):
    monkeypatch.setenv("CENGINE_ARCHIVE_PROVIDER_LABEL", "alpaca-iex+index")
    config = run.config_from_env()
    assert config.alpaca_feed == "iex"
    assert config.archive_provider_label == "alpaca-iex+index"


def test_config_from_env_rejects_inconsistent_url_and_feed(monkeypatch):
    monkeypatch.setenv("CENGINE_ALPACA_URL", "wss://stream.data.alpaca.markets/v2/iex")
    monkeypatch.setenv("CENGINE_ALPACA_FEED", "sip")
    with pytest.raises(FeedConfigurationError):
        run.config_from_env()


def test_config_from_env_rejects_bad_url_host(monkeypatch):
    monkeypatch.setenv("CENGINE_ALPACA_URL", "wss://evil.example.com/v2/iex")
    with pytest.raises(FeedConfigurationError):
        run.config_from_env()
