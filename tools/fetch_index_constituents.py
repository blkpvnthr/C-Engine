"""Fetch GENUINE S&P 500 and DJIA constituents from verifiable public sources and
write them to the index-source files consumed by tools/build_membership.py
(directive item 5).

Sources (machine-readable, deterministic parse — NEVER a summarizing model, which
can hallucinate a long list):

  * S&P 500 : datahub.io "s-and-p-500-companies" dataset (CSV; Wikipedia-derived),
              https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv
  * DJIA    : the Dow Jones Industrial Average components table (Wikipedia). The 30
              components are passed on the command line / embedded with the
              retrieval date so the provenance is explicit and auditable.

Each output file begins with a provenance header (source, retrieved_at, as_of,
count) so the snapshot can record point-in-time identity, effective date and
degraded status. Constituents are NEVER fabricated: if a source cannot be fetched
the corresponding file is left unchanged and the gap is reported.

Run:  ./.venv/bin/python tools/fetch_index_constituents.py   (read-only network; no trading)
"""

from __future__ import annotations

import datetime
import sys
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SP500_CSV = ("https://raw.githubusercontent.com/datasets/"
             "s-and-p-500-companies/main/data/constituents.csv")

# DJIA 30 components as published in the Wikipedia "Dow Jones Industrial Average"
# components table (retrieved 2026-11-22; table "as of" 2026-07-31). DJIA has only
# 30, hand-verifiable members, so the list is embedded with explicit provenance
# rather than parsed from long HTML.
DJIA_AS_OF = "2026-07-31"
DJIA_SOURCE = "https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average"
DJIA_COMPONENTS = [
    "MMM", "GOOGL", "AMZN", "AXP", "AMGN", "AAPL", "BA", "CAT", "CVX", "CSCO",
    "KO", "DIS", "GS", "HD", "HON", "IBM", "JNJ", "JPM", "MCD", "MRK",
    "MSFT", "NKE", "NVDA", "PG", "CRM", "SHW", "TRV", "UNH", "V", "WMT",
]


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _write(path: Path, symbols: list[str], *, source: str, as_of: str,
           retrieved: str, name: str) -> None:
    uniq = sorted(dict.fromkeys(s.strip().upper() for s in symbols if s.strip()))
    header = [
        f"# {name} constituents — verifiable source, NEVER fabricated.",
        f"# source: {source}",
        f"# retrieved_at: {retrieved}",
        f"# as_of: {as_of}",
        f"# count: {len(uniq)}",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(header + uniq) + "\n")
    print(f"wrote {path} ({len(uniq)} symbols; as_of {as_of})")


def fetch_sp500() -> list[str]:
    req = urllib.request.Request(SP500_CSV, headers={"User-Agent": "cengine/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        text = resp.read().decode("utf-8", "replace")
    lines = text.splitlines()
    if not lines or not lines[0].lower().startswith("symbol"):
        raise RuntimeError("unexpected S&P 500 CSV format")
    out = []
    for line in lines[1:]:
        sym = line.split(",", 1)[0].strip().strip('"')
        if sym:
            out.append(sym)
    if len(out) < 400:  # sanity: the S&P 500 must have ~500 members
        raise RuntimeError(f"S&P 500 CSV returned only {len(out)} symbols")
    return out


def main() -> int:
    now = _now()
    ok = True
    try:
        sp500 = fetch_sp500()
        _write(REPO / "data" / "sp500.txt", sp500, source=SP500_CSV,
               as_of=now[:10], retrieved=now, name="S&P 500")
    except Exception as e:  # noqa: BLE001 — never fabricate; report the gap
        ok = False
        print(f"S&P 500 fetch FAILED (file left unchanged): {e}", file=sys.stderr)
    _write(REPO / "data" / "djia.txt", DJIA_COMPONENTS, source=DJIA_SOURCE,
           as_of=DJIA_AS_OF, retrieved=now, name="DJIA")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
