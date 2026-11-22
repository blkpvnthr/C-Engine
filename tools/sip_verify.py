"""Alpaca SIP / Algo Trader Plus entitlement verification (read-only, non-trading).

Verifies, WITHOUT exposing credentials and WITHOUT submitting any order, whether
the configured account actually has SIP market-data entitlements:

  * SIP historical bars REST     (GET /v2/stocks/bars?feed=sip)
  * SIP latest quote/trade REST  (GET /v2/stocks/{sym}/quotes|trades/latest?feed=sip)
  * IEX baseline                 (proves the keys work at all)
  * SIP WebSocket auth + a bar subscription acknowledgement

A Basic account requesting feed=sip is rejected (HTTP 403 / WS error 402/403);
Algo Trader Plus returns 200 / authenticated. Entitlement is DERIVED from actual
responses, never assumed from payment (directive §4).

Run:  ./.venv/bin/python tools/sip_verify.py
Writes reports/alpaca_subscription_validation.json. Exit 0 always (verification,
not a gate); the JSON carries the verdict. Network failures are reported as
BLOCKED, not as a lack of entitlement.
"""

from __future__ import annotations

import datetime
import json
import os
import urllib.error
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
REPORTS = REPO / "reports"
DATA_BASE = "https://data.alpaca.markets"
SIP_WS = "wss://stream.data.alpaca.markets/v2/sip"


def _load_env() -> None:
    for name in (".env", "atlas-credentials.env"):
        p = REPO / name
        if not p.exists():
            continue
        for line in p.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def _creds() -> tuple[str, str]:
    key = os.environ.get("ALPACA_API_KEY") or os.environ.get("APCA_API_KEY_ID", "")
    sec = os.environ.get("ALPACA_SECRET_KEY") or os.environ.get("APCA_API_SECRET_KEY", "")
    return key, sec


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _rest(path: str, key: str, sec: str, timeout: float = 15.0) -> dict:
    """GET a data endpoint; return {ok, status, note} — never the credentials."""
    req = urllib.request.Request(DATA_BASE + path, method="GET")
    req.add_header("APCA-API-KEY-ID", key)
    req.add_header("APCA-API-SECRET-KEY", sec)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read(4096).decode("utf-8", "replace")
            has_data = any(tok in body for tok in ('"bars"', '"quote"', '"trade"'))
            return {"ok": True, "status": resp.status, "note": "ok" if has_data
                    else "200 but no data payload", "sample_keys": _keys(body)}
    except urllib.error.HTTPError as e:
        detail = e.read(512).decode("utf-8", "replace") if e.fp else ""
        return {"ok": False, "status": e.code,
                "note": f"HTTP {e.code}: {detail[:160]}"}
    except Exception as e:  # noqa: BLE001 — network / DNS / TLS
        return {"ok": False, "status": None, "note": f"{type(e).__name__}: {e}"}


def _keys(body: str) -> list[str]:
    try:
        obj = json.loads(body)
        return sorted(obj.keys())[:8] if isinstance(obj, dict) else []
    except Exception:  # noqa: BLE001
        return []


def _ws_auth(key: str, sec: str, timeout: float = 20.0) -> dict:
    """Connect to the SIP WebSocket, authenticate, subscribe to one bar."""
    try:
        import asyncio

        import websockets
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "note": f"websockets import failed: {e}"}

    async def run() -> dict:
        try:
            async with websockets.connect(SIP_WS, open_timeout=timeout,
                                          close_timeout=5) as ws:
                hello = json.loads(await asyncio.wait_for(ws.recv(), timeout))
                await ws.send(json.dumps({"action": "auth", "key": key,
                                          "secret": sec}))
                auth = json.loads(await asyncio.wait_for(ws.recv(), timeout))
                msg = auth[0] if isinstance(auth, list) and auth else {}
                if msg.get("T") == "error":
                    return {"ok": False, "status": msg.get("code"),
                            "note": f"auth error: {msg.get('msg')}",
                            "hello": hello}
                authed = msg.get("T") == "success" and msg.get("msg") == "authenticated"
                sub_ack = None
                if authed:
                    await ws.send(json.dumps({"action": "subscribe",
                                              "bars": ["AAPL"]}))
                    sub = json.loads(await asyncio.wait_for(ws.recv(), timeout))
                    sub_ack = sub[0] if isinstance(sub, list) and sub else sub
                return {"ok": bool(authed), "status": msg.get("T"),
                        "note": "authenticated" if authed else str(msg),
                        "subscription_ack": sub_ack}
        except Exception as e:  # noqa: BLE001
            return {"ok": False, "status": None, "note": f"{type(e).__name__}: {e}"}

    try:
        return asyncio.run(run())
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "status": None, "note": f"{type(e).__name__}: {e}"}


def main() -> int:
    REPORTS.mkdir(exist_ok=True)
    _load_env()
    key, sec = _creds()
    report: dict = {"recorded_utc": _now(), "data_base": DATA_BASE,
                    "sip_ws": SIP_WS, "credentials_present": bool(key and sec)}

    if not (key and sec):
        report["verdict"] = "BLOCKED"
        report["reason"] = "No ALPACA_API_KEY/SECRET in .env; cannot verify."
        _write(report)
        return 0

    # A recent completed trading day for historical bars (avoid weekend-empty).
    start = (datetime.date(2026, 11, 14)).isoformat()
    tests = {
        "sip_historical_bars": _rest(
            f"/v2/stocks/bars?symbols=AAPL&timeframe=1Day&feed=sip&limit=1&start={start}",
            key, sec),
        "sip_latest_quote": _rest("/v2/stocks/AAPL/quotes/latest?feed=sip", key, sec),
        "sip_latest_trade": _rest("/v2/stocks/AAPL/trades/latest?feed=sip", key, sec),
        "iex_baseline_quote": _rest("/v2/stocks/AAPL/quotes/latest?feed=iex", key, sec),
        "sip_websocket_auth": _ws_auth(key, sec),
    }
    report["tests"] = tests

    net_ok = any(t.get("status") is not None for t in tests.values())
    sip_rest_ok = tests["sip_historical_bars"]["ok"] or tests["sip_latest_quote"]["ok"]
    sip_ws_ok = tests["sip_websocket_auth"]["ok"]
    sip_forbidden = any(tests[k].get("status") == 403
                        for k in ("sip_historical_bars", "sip_latest_quote",
                                  "sip_latest_trade"))

    if not net_ok:
        report["verdict"] = "BLOCKED"
        report["reason"] = ("No network egress to Alpaca from this environment; "
                            "run `! ./.venv/bin/python tools/sip_verify.py` locally.")
    elif sip_rest_ok and sip_ws_ok:
        report["verdict"] = "SIP_ENTITLED"
        report["reason"] = "SIP REST + WebSocket both authorized (Algo Trader Plus active)."
    elif sip_rest_ok and not sip_ws_ok:
        report["verdict"] = "SIP_REST_ONLY"
        report["reason"] = ("SIP REST authorized but WS auth/subscribe failed "
                            "(possible connection-limit or transient issue).")
    elif sip_forbidden:
        report["verdict"] = "NOT_ENTITLED"
        report["reason"] = ("SIP returned HTTP 403 — account lacks SIP entitlement "
                            "(subscription not active). Payment != activation.")
    else:
        report["verdict"] = "INCONCLUSIVE"
        report["reason"] = "Mixed/unexpected responses; see per-test notes."

    report["entitlement_summary"] = {
        "network_reachable": net_ok,
        "sip_rest_authorized": sip_rest_ok,
        "sip_websocket_authorized": sip_ws_ok,
        "iex_baseline_ok": tests["iex_baseline_quote"]["ok"],
    }
    _write(report)
    return 0


def _write(report: dict) -> None:
    out = REPORTS / "alpaca_subscription_validation.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"verdict={report.get('verdict')} -> {out}")
    print(f"reason: {report.get('reason')}")


if __name__ == "__main__":
    raise SystemExit(main())
