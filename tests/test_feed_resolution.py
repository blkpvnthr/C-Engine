"""Tests for the single authoritative market-data feed resolver."""

import pytest

from market_data.feed import (
    DEFAULT_FEED,
    FeedConfigurationError,
    build_url,
    feed_from_url,
    provider_label_for,
    resolve_feed,
)


def test_default_feed_is_iex_not_sip():
    resolved = resolve_feed(env={})
    assert resolved.name == "iex"
    assert DEFAULT_FEED == "iex"
    assert resolved.url == "wss://stream.data.alpaca.markets/v2/iex"
    assert resolved.provider_label == "alpaca-iex"
    assert resolved.is_consolidated is False


def test_omitted_feed_never_resolves_to_sip():
    # An empty / whitespace feed env must still land on IEX, never SIP.
    for value in ("", "   "):
        resolved = resolve_feed(env={"CENGINE_ALPACA_FEED": value})
        assert resolved.name == "iex"


def test_explicit_sip_override_is_supported():
    resolved = resolve_feed(env={"CENGINE_ALPACA_FEED": "sip"})
    assert resolved.name == "sip"
    assert resolved.url.endswith("/v2/sip")
    assert resolved.provider_label == "alpaca-sip"
    assert resolved.is_consolidated is True


def test_delayed_sip_is_consolidated_and_labeled():
    resolved = resolve_feed(env={"CENGINE_ALPACA_FEED": "delayed_sip"})
    assert resolved.name == "delayed_sip"
    assert resolved.provider_label == "alpaca-delayed-sip"
    assert resolved.is_consolidated is True


def test_sandbox_host_selected():
    resolved = resolve_feed(env={"CENGINE_ALPACA_SANDBOX": "1"})
    assert resolved.sandbox is True
    assert resolved.host == "stream.data.sandbox.alpaca.markets"
    assert resolved.url == "wss://stream.data.sandbox.alpaca.markets/v2/iex"


def test_unknown_feed_rejected():
    with pytest.raises(FeedConfigurationError, match="unknown market-data feed"):
        resolve_feed(env={"CENGINE_ALPACA_FEED": "nasdaq_totalview"})


def test_explicit_url_override_derives_feed_and_provenance():
    resolved = resolve_feed(
        env={"CENGINE_ALPACA_URL": "wss://stream.data.alpaca.markets/v2/sip"}
    )
    assert resolved.name == "sip"
    assert resolved.provider_label == "alpaca-sip"
    assert resolved.url == "wss://stream.data.alpaca.markets/v2/sip"


def test_explicit_url_sandbox_detected():
    resolved = resolve_feed(
        env={"CENGINE_ALPACA_URL": "wss://stream.data.sandbox.alpaca.markets/v2/iex"}
    )
    assert resolved.name == "iex"
    assert resolved.sandbox is True


@pytest.mark.parametrize(
    "url, match",
    [
        ("https://stream.data.alpaca.markets/v2/iex", "wss scheme"),
        ("ws://stream.data.alpaca.markets/v2/iex", "wss scheme"),
        ("wss://evil.example.com/v2/iex", "not an Alpaca streaming host"),
        ("wss://stream.data.alpaca.markets/v2/bogus", "not a known Alpaca feed"),
    ],
)
def test_invalid_explicit_url_rejected(url, match):
    with pytest.raises(FeedConfigurationError, match=match):
        resolve_feed(env={"CENGINE_ALPACA_URL": url})


def test_url_and_feed_name_inconsistency_rejected():
    with pytest.raises(FeedConfigurationError, match="refusing to connect"):
        resolve_feed(
            env={
                "CENGINE_ALPACA_URL": "wss://stream.data.alpaca.markets/v2/iex",
                "CENGINE_ALPACA_FEED": "sip",
            }
        )


def test_url_consistent_with_feed_name_is_accepted():
    resolved = resolve_feed(
        env={
            "CENGINE_ALPACA_URL": "wss://stream.data.alpaca.markets/v2/iex",
            "CENGINE_ALPACA_FEED": "iex",
        }
    )
    assert resolved.name == "iex"


def test_explicit_provider_label_preserved():
    resolved = resolve_feed(
        env={"CENGINE_ARCHIVE_PROVIDER_LABEL": "alpaca-iex+licensed_index"}
    )
    assert resolved.name == "iex"
    assert resolved.provider_label == "alpaca-iex+licensed_index"


def test_explicit_arguments_win_over_environment():
    resolved = resolve_feed(env={"CENGINE_ALPACA_FEED": "sip"}, feed="iex")
    assert resolved.name == "iex"


def test_feed_from_url_helpers():
    assert feed_from_url("wss://stream.data.alpaca.markets/v2/iex") == "iex"
    assert feed_from_url("wss://stream.data.alpaca.markets/v2/other") is None
    assert build_url("iex", sandbox=False) == "wss://stream.data.alpaca.markets/v2/iex"
    assert provider_label_for("delayed_sip") == "alpaca-delayed-sip"
