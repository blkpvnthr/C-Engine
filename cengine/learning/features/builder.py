"""Point-in-time feature/label matrix construction (Milestone 6, §5/§7).

Builds a look-ahead-free training matrix straight from the lakehouse Parquet via DuckDB
window functions. Every feature at bar t uses ONLY bars <= t within the same (symbol,
session); the forward-return LABEL uses bars > t and is the prediction target, never an
input. Partitioning by (symbol, session) prevents overnight-gap contamination of
intraday returns and prevents windows from spanning instruments.

This is the genuinely trainable supervised pipeline on real market data. It predicts the
THEORETICAL forward bar return, which is explicitly distinct from a realized strategy
return under the mandatory -0.5% protective exit (§7) — the net-of-exit return requires
genuine execution outcomes (the gold datasets), handled separately by gold_training_frame.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import duckdb
import numpy as np

from cengine.lakehouse.config import LakehouseConfig
from cengine.learning.features.schema import BAR_FEATURE_COLUMNS, FeatureSchema, bar_feature_schema


@dataclass
class BarFeatureMatrix:
    schema: FeatureSchema
    X: np.ndarray                 # (n, n_features) float32, look-ahead-free
    y_return: np.ndarray          # (n,) forward log return over horizon_bars
    y_positive: np.ndarray        # (n,) int {0,1} forward return > 0
    timestamps_ns: np.ndarray     # (n,) int64 feature (decision) time = bar close
    symbols: np.ndarray           # (n,) object symbol_id
    horizon_bars: int = 3
    feature_columns: tuple = BAR_FEATURE_COLUMNS

    @property
    def n(self) -> int:
        return int(self.X.shape[0])


def _glob(config: LakehouseConfig, layer: str, dataset: str) -> Optional[str]:
    base = config.dataset_path(layer, dataset)
    if not base.exists() or not any(base.rglob("*.parquet")):
        return None
    return str(base / "**" / "*.parquet").replace("'", "''")


_FEATURE_SQL = """
WITH base AS (
    SELECT symbol_id,
           event_timestamp_ns AS ts,
           CAST(to_timestamp(event_timestamp_ns/1e9) AS DATE) AS session,
           CAST(bar_close_ticks AS DOUBLE) AS close,
           CAST(bar_high_ticks AS DOUBLE) AS high,
           CAST(bar_low_ticks  AS DOUBLE) AS low,
           CAST(volume AS DOUBLE) AS volume,
           CAST(vwap_ticks AS DOUBLE) AS vwap
    FROM read_parquet('{glob}', union_by_name=true)
    WHERE bar_close_ticks > 0 AND timeframe = '{timeframe}'
),
r AS (
    SELECT *,
        ln(close / NULLIF(lag(close,1)  OVER w, 0)) AS ret_1_raw
    FROM base
    WINDOW w AS (PARTITION BY symbol_id, session ORDER BY ts)
),
f AS (
    SELECT symbol_id, ts, session, close, high, low, volume, vwap, ret_1_raw,
        ret_1_raw AS ret_1,
        ln(close / NULLIF(lag(close,5)  OVER w, 0)) AS ret_5,
        ln(close / NULLIF(lag(close,15) OVER w, 0)) AS ret_15,
        close / NULLIF(lag(close,10) OVER w, 0) - 1 AS mom_10,
        close / NULLIF(lag(close,20) OVER w, 0) - 1 AS mom_20,
        stddev_samp(ret_1_raw) OVER (PARTITION BY symbol_id, session ORDER BY ts
            ROWS BETWEEN 10 PRECEDING AND CURRENT ROW) AS vol_10,
        stddev_samp(ret_1_raw) OVER (PARTITION BY symbol_id, session ORDER BY ts
            ROWS BETWEEN 20 PRECEDING AND CURRENT ROW) AS vol_20,
        close / NULLIF(avg(close) OVER (PARTITION BY symbol_id, session ORDER BY ts
            ROWS BETWEEN 10 PRECEDING AND CURRENT ROW), 0) - 1 AS sma_ratio_10,
        close / NULLIF(avg(close) OVER (PARTITION BY symbol_id, session ORDER BY ts
            ROWS BETWEEN 20 PRECEDING AND CURRENT ROW), 0) - 1 AS sma_ratio_20,
        avg((high - low) / NULLIF(close,0)) OVER (PARTITION BY symbol_id, session ORDER BY ts
            ROWS BETWEEN 10 PRECEDING AND CURRENT ROW) AS range_10,
        avg(greatest(ret_1_raw, 0)) OVER (PARTITION BY symbol_id, session ORDER BY ts
            ROWS BETWEEN 14 PRECEDING AND CURRENT ROW) AS avg_gain_14,
        avg(greatest(-ret_1_raw, 0)) OVER (PARTITION BY symbol_id, session ORDER BY ts
            ROWS BETWEEN 14 PRECEDING AND CURRENT ROW) AS avg_loss_14,
        volume / NULLIF(avg(volume) OVER (PARTITION BY symbol_id, session ORDER BY ts
            ROWS BETWEEN 10 PRECEDING AND CURRENT ROW), 0) AS volume_ratio_10,
        close / NULLIF(vwap, 0) - 1 AS vwap_dist,
        lead(close, {horizon}) OVER w AS fwd_close
    FROM r
    WINDOW w AS (PARTITION BY symbol_id, session ORDER BY ts)
)
SELECT symbol_id, ts,
       ret_1, ret_5, ret_15, mom_10, mom_20, vol_10, vol_20,
       sma_ratio_10, sma_ratio_20, range_10,
       CASE WHEN avg_loss_14 = 0 OR avg_loss_14 IS NULL THEN 100.0
            ELSE 100.0 - 100.0 / (1.0 + avg_gain_14 / avg_loss_14) END AS rsi_14,
       volume_ratio_10,
       COALESCE(vwap_dist, 0.0) AS vwap_dist,
       ln(fwd_close / NULLIF(close, 0)) AS fwd_ret
FROM f
WHERE fwd_close IS NOT NULL
  AND ret_1 IS NOT NULL AND ret_5 IS NOT NULL AND ret_15 IS NOT NULL
  AND mom_10 IS NOT NULL AND mom_20 IS NOT NULL
  AND vol_10 IS NOT NULL AND vol_20 IS NOT NULL
  AND sma_ratio_10 IS NOT NULL AND sma_ratio_20 IS NOT NULL
  AND range_10 IS NOT NULL AND volume_ratio_10 IS NOT NULL
ORDER BY ts, symbol_id
"""


def build_bar_features(config: Optional[LakehouseConfig] = None, *,
                       horizon_bars: int = 3,
                       timeframe: str = "5Min") -> Optional[BarFeatureMatrix]:
    """Build the look-ahead-free bar feature/label matrix from bronze market_bars.

    Returns None when no market_bars exist. ``timeframe`` selects a SINGLE bar resolution
    (default '5Min') — mixing resolutions would corrupt the rolling windows. horizon_bars is
    the forward-return horizon in bars of that timeframe.
    """
    config = config or LakehouseConfig.from_env()
    glob = _glob(config, "bronze", "market_bars")
    if glob is None:
        return None
    if timeframe not in ("1Min", "5Min", "15Min", "30Min", "1Hour", "1Day"):
        raise ValueError(f"unsupported timeframe {timeframe!r}")
    sql = _FEATURE_SQL.format(glob=glob, horizon=int(horizon_bars), timeframe=timeframe)
    con = duckdb.connect()
    try:
        tbl = con.sql(sql).to_arrow_table()
    finally:
        con.close()
    if tbl.num_rows == 0:
        return None
    cols = tbl.column_names
    data = {c: tbl.column(c).to_numpy(zero_copy_only=False) for c in cols}
    schema = bar_feature_schema()
    X = np.column_stack([np.asarray(data[c], dtype=np.float32)
                         for c in schema.columns])
    y_return = np.asarray(data["fwd_ret"], dtype=np.float32)
    # guard against any residual non-finite values from degenerate windows
    finite = np.isfinite(X).all(axis=1) & np.isfinite(y_return)
    X, y_return = X[finite], y_return[finite]
    ts = np.asarray(data["ts"], dtype=np.int64)[finite]
    syms = np.asarray(data["symbol_id"], dtype=object)[finite]
    y_pos = (y_return > 0).astype(np.int64)
    return BarFeatureMatrix(schema=schema, X=X, y_return=y_return, y_positive=y_pos,
                            timestamps_ns=ts, symbols=syms, horizon_bars=int(horizon_bars))


def gold_training_frame(config: Optional[LakehouseConfig] = None, *,
                        realized_only: bool = True):
    """Training frame from the GOLD supervised dataset (realized strategy outcomes).

    Returns a pyarrow.Table (or None if unavailable). Labels are genuine realized
    net returns under the mandatory exit; censored rows are excluded when
    realized_only. This is distinct from build_bar_features (theoretical forward return)
    and only yields data once genuine execution lifecycles exist in the lakehouse.
    """
    config = config or LakehouseConfig.from_env()
    glob = _glob(config, "gold", "supervised")
    if glob is None:
        return None
    where = "WHERE label_status = 'realized'" if realized_only else ""
    con = duckdb.connect()
    try:
        tbl = con.sql(f"SELECT * FROM read_parquet('{glob}', union_by_name=true) "
                      f"{where} ORDER BY decision_ts").to_arrow_table()
    finally:
        con.close()
    return tbl if tbl.num_rows else None
