"""Authoritative Alpaca market-data feed resolution.

A single resolver, shared by every entry point (``run.py``, ``start_engine.py``,
and programmatic engine construction), decides which Alpaca market-data feed the
engine streams from. Centralizing the decision guarantees that the live
WebSocket endpoint, the archive provenance label, and the startup feed
verification can never disagree.

Policy
------
* **IEX is the default.** When no feed is configured the resolver selects
  ``iex`` (``wss://stream.data.alpaca.markets/v2/iex``). An omitted feed never
  resolves to SIP.
* **SIP stays supported but explicit.** Entitled accounts may select ``sip``
  (or ``delayed_sip``) deliberately. The resolver never *falls back* to SIP when
  IEX fails — a failed IEX connection is surfaced, not silently downgraded to a
  different data source.
* **Explicit URL overrides are validated.** ``CENGINE_ALPACA_URL`` (or an
  injected ``url``) is honored only after its scheme (``wss``), host (a known
  Alpaca streaming host), and feed path are validated and found consistent with
  any separately declared feed name. The feed name is then *derived* from the
  URL so the provenance label matches the endpoint actually connected to.

Provenance
----------
Each resolved feed carries an archive ``provider_label`` (e.g. ``alpaca-iex``)
and an ``is_consolidated`` flag. Only the SIP family is consolidated
(NBBO / full-tape). IEX is a single exchange: its quotes are exchange-specific
top-of-book, not the National Best Bid and Offer, and its trades are IEX prints,
not consolidated market volume. Downstream consumers use ``is_consolidated`` to
decide whether a signal that *requires* market-wide coverage is eligible.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Mapping, Optional
from urllib.parse import urlparse

# Feed name -> URL path under the Alpaca streaming host.
FEED_PATHS: dict[str, str] = {
    "iex": "v2/iex",
    "sip": "v2/sip",
    "delayed_sip": "v2/delayed_sip",
    "boats": "v1beta1/boats",
    "overnight": "v1beta1/overnight",
    "test": "v2/test",
}

# Feeds that carry consolidated (NBBO / full-tape) data. Everything else is a
# single venue or a restricted stream and must NOT be presented as market-wide.
CONSOLIDATED_FEEDS: frozenset[str] = frozenset({"sip", "delayed_sip"})

DEFAULT_FEED = "iex"

LIVE_HOST = "stream.data.alpaca.markets"
SANDBOX_HOST = "stream.data.sandbox.alpaca.markets"
_KNOWN_HOSTS: frozenset[str] = frozenset({LIVE_HOST, SANDBOX_HOST})

# URL path (normalized, no leading slash) -> feed name.
_PATH_TO_FEED: dict[str, str] = {path: name for name, path in FEED_PATHS.items()}

_TRUE_TOKENS: frozenset[str] = frozenset({"1", "true", "yes", "on"})


class FeedConfigurationError(RuntimeError):
    """A requested market-data feed or endpoint is invalid or inconsistent."""


def provider_label_for(feed: str) -> str:
    """Return the canonical archive provenance label for a feed name."""
    return "alpaca-" + feed.strip().lower().replace("_", "-")


@dataclass(frozen=True, slots=True)
class ResolvedFeed:
    """The single authoritative description of the selected market-data feed."""

    name: str
    url: str
    provider_label: str
    sandbox: bool

    @property
    def is_consolidated(self) -> bool:
        """True only for feeds that carry consolidated NBBO / full-tape data."""
        return self.name in CONSOLIDATED_FEEDS

    @property
    def host(self) -> str:
        return SANDBOX_HOST if self.sandbox else LIVE_HOST


def feed_from_url(url: str) -> Optional[str]:
    """Derive the feed name from a streaming URL, or None if unrecognized.

    Returns None rather than raising so callers (e.g. the stream's health
    reporting) can label a custom endpoint without failing.
    """
    try:
        parsed = urlparse(url.strip())
    except (AttributeError, ValueError):
        return None
    path = parsed.path.strip("/")
    return _PATH_TO_FEED.get(path)


def _validate_url(url: str) -> tuple[str, bool]:
    """Validate an explicit streaming URL and return (feed_name, sandbox).

    Enforces the ``wss`` scheme, a known Alpaca streaming host, and a recognized
    feed path. The feed name is derived from the path so provenance can never
    drift from the endpoint actually connected to.
    """
    parsed = urlparse(url.strip())
    if parsed.scheme != "wss":
        raise FeedConfigurationError(
            f"market-data URL must use the wss scheme, got {parsed.scheme or '<none>'!r}: {url!r}"
        )
    host = parsed.hostname or ""
    if host not in _KNOWN_HOSTS:
        raise FeedConfigurationError(
            f"market-data URL host {host!r} is not an Alpaca streaming host "
            f"({sorted(_KNOWN_HOSTS)}): {url!r}"
        )
    path = parsed.path.strip("/")
    feed = _PATH_TO_FEED.get(path)
    if feed is None:
        raise FeedConfigurationError(
            f"market-data URL path {path!r} is not a known Alpaca feed "
            f"({sorted(FEED_PATHS)}): {url!r}"
        )
    return feed, host == SANDBOX_HOST


def build_url(feed: str, *, sandbox: bool) -> str:
    """Construct the streaming URL for a validated feed name."""
    host = SANDBOX_HOST if sandbox else LIVE_HOST
    return f"wss://{host}/{FEED_PATHS[feed]}"


def resolve_feed(
    *,
    env: Optional[Mapping[str, str]] = None,
    feed: Optional[str] = None,
    url: Optional[str] = None,
    sandbox: Optional[bool] = None,
    provider_label: Optional[str] = None,
) -> ResolvedFeed:
    """Resolve the authoritative market-data feed.

    Precedence (explicit arguments win over environment):

    1. An explicit full ``url`` (argument or ``CENGINE_ALPACA_URL``) is validated
       and its feed name is derived from the path. A separately declared feed
       name that disagrees with the URL path is a hard error.
    2. Otherwise the feed name (argument or ``CENGINE_ALPACA_FEED``) selects a
       feed, defaulting to IEX. The sandbox flag (argument or
       ``CENGINE_ALPACA_SANDBOX``) picks the host.

    The archive ``provider_label`` defaults to the canonical label for the feed
    (``alpaca-iex``) but an explicit label (argument or
    ``CENGINE_ARCHIVE_PROVIDER_LABEL``) is preserved for custom provenance.
    """
    environ = os.environ if env is None else env

    explicit_url = url if url is not None else environ.get("CENGINE_ALPACA_URL")
    declared_feed = feed if feed is not None else environ.get("CENGINE_ALPACA_FEED")
    if declared_feed and declared_feed.strip():
        declared_feed = declared_feed.strip().lower()
    else:
        declared_feed = None

    if explicit_url and explicit_url.strip():
        url_feed, url_sandbox = _validate_url(explicit_url)
        if declared_feed and declared_feed != url_feed:
            raise FeedConfigurationError(
                f"explicit market-data URL selects feed {url_feed!r} but "
                f"CENGINE_ALPACA_FEED requests {declared_feed!r}; refusing to "
                "connect to one feed while labeling provenance as another"
            )
        resolved_name = url_feed
        resolved_sandbox = url_sandbox
        resolved_url = explicit_url.strip()
    else:
        resolved_name = declared_feed or DEFAULT_FEED
        if resolved_name not in FEED_PATHS:
            raise FeedConfigurationError(
                f"unknown market-data feed {resolved_name!r}; expected one of "
                f"{sorted(FEED_PATHS)}"
            )
        if sandbox is not None:
            resolved_sandbox = bool(sandbox)
        else:
            resolved_sandbox = (
                str(environ.get("CENGINE_ALPACA_SANDBOX", "")).strip().lower() in _TRUE_TOKENS
            )
        resolved_url = build_url(resolved_name, sandbox=resolved_sandbox)

    if provider_label is not None:
        label = provider_label
    else:
        label = environ.get("CENGINE_ARCHIVE_PROVIDER_LABEL")
    label = label.strip() if label and label.strip() else provider_label_for(resolved_name)

    return ResolvedFeed(
        name=resolved_name,
        url=resolved_url,
        provider_label=label,
        sandbox=resolved_sandbox,
    )


__all__ = [
    "CONSOLIDATED_FEEDS",
    "DEFAULT_FEED",
    "FEED_PATHS",
    "FeedConfigurationError",
    "LIVE_HOST",
    "ResolvedFeed",
    "SANDBOX_HOST",
    "build_url",
    "feed_from_url",
    "provider_label_for",
    "resolve_feed",
]
