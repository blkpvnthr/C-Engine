"""Master-universe membership acquisition and the stable symbol registry (§14-17).

Builds the deduplicated master universe

    MASTER = NASDAQ_LISTED ∪ SP500 ∪ DJIA ∪ CONFIGURED

from an AUTHORITATIVE, genuinely-acquirable source — the Alpaca assets API
(`/v2/assets`) — reconciled for tradability, fractionability, exchange, and asset
status. "NASDAQ-listed" means supported equities whose listing exchange is NASDAQ,
NOT the Nasdaq-100 index (§14).

S&P 500 and DJIA constituents are INDEX memberships, not exchange memberships, so
they require independent constituent sources (§15). They are pluggable here
(explicit symbol lists supplied by the caller / a configured file); this module
never fabricates index constituents. When a constituent source is not configured,
the corresponding membership flag is simply absent and the snapshot records the
gap honestly — it never infers index membership from exchange listing, and never
silently reverts to a small ETF fallback universe (§17).

Discovery membership and executable tradability are tracked separately: a symbol
can be in the discovery universe yet be non-tradable or non-fractionable.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

ALPACA_TRADING_BASE = "https://api.alpaca.markets"


@dataclass
class SymbolInfo:
    symbol: str
    symbol_id: int = -1          # stable integer id (registry index)
    asset_id: str = ""           # Alpaca asset UUID
    generation: int = 1
    exchange: str = ""
    asset_class: str = "us_equity"
    asset_status: str = ""       # active / inactive
    tradable: bool = False
    fractionable: bool = False
    sector: str = ""             # not provided by Alpaca; left empty, never faked
    industry: str = ""
    nasdaq_member: bool = False
    sp500_member: bool = False
    djia_member: bool = False
    configured_member: bool = False

    def in_master(self) -> bool:
        return (self.nasdaq_member or self.sp500_member or self.djia_member
                or self.configured_member)


@dataclass
class MembershipSnapshot:
    membership_version: int = 0
    membership_effective_date: str = ""
    retrieved_at: str = ""
    source: dict = field(default_factory=dict)
    counts: dict = field(default_factory=dict)
    changes: dict = field(default_factory=dict)
    degraded: bool = False
    note: str = ""


def _norm(sym: str) -> str:
    return str(sym).upper().strip()


def fetch_alpaca_assets(key: str, sec: str, base: str = ALPACA_TRADING_BASE,
                        timeout: float = 40.0) -> list[dict]:
    """GET active US-equity assets from Alpaca (authoritative supported universe).
    Raises on network/HTTP error so the caller can fall back to cache (§17)."""
    base = base.rstrip("/")
    if base.endswith("/v2"):
        base = base[:-3]
    url = f"{base}/v2/assets?status=active&asset_class=us_equity"
    req = urllib.request.Request(url, method="GET")
    req.add_header("APCA-API-KEY-ID", key)
    req.add_header("APCA-API-SECRET-KEY", sec)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8", "replace"))


class MembershipRegistry:
    """Stable symbol registry with id/generation and group membership (§16)."""

    def __init__(self) -> None:
        self._by_symbol: dict[str, SymbolInfo] = {}
        self._order: list[str] = []            # id -> symbol (stable)
        self.snapshot_meta = MembershipSnapshot()

    # -- construction --------------------------------------------------------
    def build(self, assets: list[dict], *, sp500: list[str] | None = None,
              djia: list[str] | None = None, configured: list[str] | None = None,
              now_iso: str = "", effective_date: str = "",
              sources: dict | None = None) -> MembershipSnapshot:
        prev = dict(self._by_symbol)  # for generation/change tracking
        sp500_set = {_norm(s) for s in (sp500 or [])}
        djia_set = {_norm(s) for s in (djia or [])}
        cfg_set = {_norm(s) for s in (configured or [])}

        self._by_symbol = {}
        for a in assets:
            sym = _norm(a.get("symbol", ""))
            if not sym:
                continue
            exch = a.get("exchange", "")
            info = SymbolInfo(
                symbol=sym, asset_id=a.get("id", ""), exchange=exch,
                asset_class=a.get("class", "us_equity"),
                asset_status=a.get("status", ""),
                tradable=bool(a.get("tradable", False)),
                fractionable=bool(a.get("fractionable", False)),
                nasdaq_member=(exch == "NASDAQ"),
                sp500_member=(sym in sp500_set),
                djia_member=(sym in djia_set),
                configured_member=(sym in cfg_set))
            self._by_symbol[sym] = info

        # Index/configured members not present in the Alpaca asset list are still
        # recorded (so membership is not lost), flagged non-tradable until matched.
        for sym in (sp500_set | djia_set | cfg_set):
            if sym not in self._by_symbol:
                self._by_symbol[sym] = SymbolInfo(
                    symbol=sym, asset_status="not_in_alpaca_assets",
                    sp500_member=(sym in sp500_set), djia_member=(sym in djia_set),
                    configured_member=(sym in cfg_set))

        # Assign stable ids (sorted for determinism) + generations (bump on a
        # status/exchange change vs the previous snapshot; new symbols start at 1).
        self._order = sorted(self._by_symbol)
        new_syms, changed_syms, removed_syms = [], [], []
        for i, sym in enumerate(self._order):
            info = self._by_symbol[sym]
            info.symbol_id = i
            old = prev.get(sym)
            if old is None:
                info.generation = 1
                if prev:
                    new_syms.append(sym)
            elif (old.asset_status != info.asset_status
                  or old.exchange != info.exchange
                  or old.tradable != info.tradable):
                info.generation = old.generation + 1
                changed_syms.append(sym)
            else:
                info.generation = old.generation
        removed_syms = [s for s in prev if s not in self._by_symbol]

        snap = MembershipSnapshot(
            membership_version=self.snapshot_meta.membership_version + 1,
            membership_effective_date=effective_date or now_iso,
            retrieved_at=now_iso,
            source=sources or {"universe": "alpaca_assets"},
            counts=self.counts(),
            changes={"added": sorted(new_syms)[:50], "added_count": len(new_syms),
                     "changed": sorted(changed_syms)[:50], "changed_count": len(changed_syms),
                     "removed": sorted(removed_syms)[:50], "removed_count": len(removed_syms)},
            degraded=False)
        self.snapshot_meta = snap
        return snap

    # -- queries -------------------------------------------------------------
    def counts(self) -> dict:
        vals = self._by_symbol.values()
        master = [i for i in vals if i.in_master()]
        return {
            "alpaca_supported": len(self._by_symbol),
            "tradable": sum(1 for i in vals if i.tradable),
            "fractionable": sum(1 for i in vals if i.fractionable),
            "nasdaq_member": sum(1 for i in vals if i.nasdaq_member),
            "sp500_member": sum(1 for i in vals if i.sp500_member),
            "djia_member": sum(1 for i in vals if i.djia_member),
            "configured_member": sum(1 for i in vals if i.configured_member),
            # deduplicated combined master universe (each symbol counted ONCE)
            "master_universe": len(master),
            "master_tradable": sum(1 for i in master if i.tradable),
        }

    def get(self, symbol: str) -> SymbolInfo | None:
        return self._by_symbol.get(_norm(symbol))

    def master_symbols(self) -> list[str]:
        return [s for s in self._order if self._by_symbol[s].in_master()]

    def tradable_master_symbols(self) -> list[str]:
        return [s for s in self._order
                if self._by_symbol[s].in_master() and self._by_symbol[s].tradable]

    def group(self, name: str) -> list[str]:
        attr = {"nasdaq": "nasdaq_member", "sp500": "sp500_member",
                "djia": "djia_member", "configured": "configured_member"}[name]
        return [s for s in self._order if getattr(self._by_symbol[s], attr)]

    def __len__(self) -> int:
        return len(self._by_symbol)

    # -- persistence ---------------------------------------------------------
    def persist(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "membership_version": self.snapshot_meta.membership_version,
            "membership_effective_date": self.snapshot_meta.membership_effective_date,
            "retrieved_at": self.snapshot_meta.retrieved_at,
            "source": self.snapshot_meta.source,
            "counts": self.snapshot_meta.counts,
            "changes": self.snapshot_meta.changes,
            "degraded": self.snapshot_meta.degraded,
            "symbols": {s: {"symbol_id": i.symbol_id, "generation": i.generation,
                            "exchange": i.exchange, "tradable": i.tradable,
                            "fractionable": i.fractionable,
                            "asset_status": i.asset_status,
                            "nasdaq": i.nasdaq_member, "sp500": i.sp500_member,
                            "djia": i.djia_member, "configured": i.configured_member}
                        for s, i in self._by_symbol.items()},
        }
        path.write_text(json.dumps(payload, indent=2))

    def load_cache(self, path: str | Path) -> bool:
        """Populate from a persisted snapshot (degraded recovery, §17). Returns
        True if a cache was loaded."""
        path = Path(path)
        if not path.exists():
            return False
        doc = json.loads(path.read_text())
        self._by_symbol = {}
        for sym, d in (doc.get("symbols") or {}).items():
            self._by_symbol[sym] = SymbolInfo(
                symbol=sym, symbol_id=d.get("symbol_id", -1),
                generation=d.get("generation", 1), exchange=d.get("exchange", ""),
                asset_status=d.get("asset_status", ""),
                tradable=d.get("tradable", False),
                fractionable=d.get("fractionable", False),
                nasdaq_member=d.get("nasdaq", False), sp500_member=d.get("sp500", False),
                djia_member=d.get("djia", False), configured_member=d.get("configured", False))
        self._order = sorted(self._by_symbol)
        self.snapshot_meta = MembershipSnapshot(
            membership_version=doc.get("membership_version", 0),
            membership_effective_date=doc.get("membership_effective_date", ""),
            retrieved_at=doc.get("retrieved_at", ""),
            source=doc.get("source", {}), counts=doc.get("counts", {}),
            degraded=True, note="loaded from cache (degraded recovery)")
        return True


def _read_symbol_file(path: str | Path) -> list[str]:
    p = Path(path)
    if not p.exists():
        return []
    out = []
    for line in p.read_text().splitlines():
        s = line.split("#", 1)[0].strip()
        if s:
            out.append(_norm(s))
    return out
