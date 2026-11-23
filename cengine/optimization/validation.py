"""Out-of-sample validation windows (Milestone 6B, §38).

Chronological train / optimization-validation / UNTOUCHED-certification splits with embargo,
plus purged walk-forward folds. The optimizer is only ever handed train + validation windows;
the final certification holdout is reserved and must never be accessed for feature selection,
tuning, reward optimization, or experiment selection (Milestone 7 owns it).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import duckdb

from cengine.lakehouse.config import LakehouseConfig

VALIDATION_VERSION = "oos-validation-1.0.0"
_DAY_NS = 86_400 * 1_000_000_000


@dataclass(frozen=True)
class OOSWindows:
    train: tuple          # (start_ns, end_ns)
    validation: tuple
    holdout: tuple        # UNTOUCHED certification set (optimizer must not access)
    embargo_days: int
    version: str = VALIDATION_VERSION

    def to_dict(self) -> dict:
        return {"train": list(self.train), "validation": list(self.validation),
                "holdout_RESERVED": list(self.holdout), "embargo_days": self.embargo_days,
                "version": self.version,
                "note": "holdout is reserved for Milestone 7; optimizer uses train+validation only"}


def _data_span(config: LakehouseConfig, symbol: str, timeframe: str = "1Min") -> Optional[tuple]:
    base = config.dataset_path("bronze", "market_bars")
    if not base.exists() or not any(base.rglob("*.parquet")):
        return None
    glob = str(base / "**" / "*.parquet").replace("'", "''")
    con = duckdb.connect()
    try:
        r = con.execute(
            f"SELECT min(event_timestamp_ns), max(event_timestamp_ns) FROM "
            f"read_parquet('{glob}', union_by_name=true) WHERE symbol_id='{symbol}' "
            f"AND timeframe='{timeframe}'").fetchone()
    finally:
        con.close()
    if not r or r[0] is None:
        return None
    return int(r[0]), int(r[1])


def oos_windows(config: Optional[LakehouseConfig], symbol: str, *,
                train_frac: float = 0.6, validation_frac: float = 0.2,
                embargo_days: int = 3, timeframe: str = "1Min") -> Optional[OOSWindows]:
    """Chronological train/validation/holdout from a symbol's actual data span."""
    config = config or LakehouseConfig.from_env()
    span = _data_span(config, symbol, timeframe)
    if span is None:
        return None
    lo, hi = span
    total = hi - lo
    embargo = embargo_days * _DAY_NS
    train_end = lo + int(total * train_frac)
    val_start = train_end + embargo
    val_end = lo + int(total * (train_frac + validation_frac))
    holdout_start = val_end + embargo
    return OOSWindows(train=(lo, train_end), validation=(val_start, val_end),
                      holdout=(holdout_start, hi), embargo_days=embargo_days)


def purged_walk_forward(config: Optional[LakehouseConfig], symbol: str, *,
                        n_folds: int = 4, embargo_days: int = 2,
                        timeframe: str = "1Min") -> list:
    """Expanding-window purged walk-forward folds (train grows; each test is future + embargoed).

    The final segment is withheld as the certification holdout and is NOT returned as a fold.
    """
    config = config or LakehouseConfig.from_env()
    span = _data_span(config, symbol, timeframe)
    if span is None:
        return []
    lo, hi = span
    embargo = embargo_days * _DAY_NS
    usable = int((hi - lo) * 0.8)   # reserve the last 20% as the untouched holdout
    seg = usable // (n_folds + 1)
    folds = []
    for k in range(1, n_folds + 1):
        train_end = lo + seg * k
        test_start = train_end + embargo
        test_end = lo + seg * (k + 1)
        if test_end <= test_start:
            continue
        folds.append({"fold": k, "train": (lo, train_end),
                      "test": (test_start, test_end)})
    return folds
