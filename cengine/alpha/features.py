"""Point-in-time entry features for alpha-discovery research (M8 §5 / §6 / §10).

Vectorized, **look-ahead-free** features computed on genuine 1-minute bars and aligned to the
momentum-entry indices produced by the existing entry contract. Every feature at bar ``i`` uses
only information available through the close of bar ``i`` (the decision bar; execution is the next
bar's open), so a feature can never peek at the outcome it is meant to predict.

Higher-timeframe context (§6) is built from **completed** higher-TF candles via trailing windows
on the 1-minute series (e.g. a 15-minute trend = sign of the 15-bar-spaced close change through
the last completed 15-minute boundary) — developing candles are never used to qualify an earlier
decision. Where a feature is a bar-only approximation of a microstructure quantity, it is named
and documented as such; no Level-2 data is fabricated.

These features back the entry-quality ablation (§5), the multi-timeframe study (§6) and the
supervised opportunity filter (§10). The realized label for each entry is the net return under
the mandatory −0.5% protective exit (from the shared conservative-fill resolver).
"""

from __future__ import annotations

import datetime as _dt
from zoneinfo import ZoneInfo

import numpy as np

from cengine.alpha.portfolio_replay import resolve_exit

FEATURES_VERSION = "alpha-entry-features-1.0.0"
_ET = ZoneInfo("America/New_York")
_STOP = 0.005
_MAX_HOLD = 120

# the feature set (name -> higher-is-"more-of-the-condition"); used by the ablation to scan
# a kept-subset threshold in both directions.
FEATURE_NAMES = (
    "mom5", "mom_accel", "rel_volume", "disp_sma20", "disp_atr", "bb_width",
    "htf_trend_15", "htf_trend_60", "trend_persist", "tod_bucket", "dir_separation",
)


def _rolling_mean(x: np.ndarray, win: int) -> np.ndarray:
    cs = np.cumsum(np.insert(x, 0, 0.0))
    out = np.full(len(x), np.nan)
    if len(x) > win:
        out[win:] = (cs[win:-1] - cs[:-win - 1]) / win
    return out


def _rolling_std(x: np.ndarray, win: int) -> np.ndarray:
    out = np.full(len(x), np.nan)
    if len(x) <= win:
        return out
    m = _rolling_mean(x, win)
    sq = _rolling_mean(x * x, win)
    var = np.clip(sq - m * m, 0.0, None)
    out = np.sqrt(var)
    return out


def compute_features(bars: dict) -> dict:
    """Return a dict of per-bar look-ahead-free feature arrays (NaN during warmup)."""
    c = bars["c"].astype(np.float64)
    h = bars["h"].astype(np.float64)
    low = bars["l"].astype(np.float64)
    v = bars["v"].astype(np.float64)
    ts = bars["ts"]

    # trailing (previous-bar) SMAs / vols — all exclude the current bar via _rolling_mean semantics
    sma20 = _rolling_mean(c, 20)
    sma100 = _rolling_mean(c, 100)
    vol_sma20 = _rolling_mean(v, 20)
    std20 = _rolling_std(c, 20)

    # true range + ATR(14)
    prev_c = np.concatenate(([c[0]], c[:-1]))
    tr = np.maximum(h - low, np.maximum(np.abs(h - prev_c), np.abs(low - prev_c)))
    atr = _rolling_mean(tr, 14)
    atr_frac = np.where(c > 0, np.nan_to_num(atr) / c, 0.0)

    # momentum + acceleration (look-ahead-free: uses current & past closes)
    c_lag5 = np.concatenate((np.full(5, np.nan), c[:-5]))
    c_lag10 = np.concatenate((np.full(10, np.nan), c[:-10]))
    mom5 = np.where(c_lag5 > 0, c / c_lag5 - 1.0, np.nan)
    mom5_prev = np.where(c_lag10 > 0, c_lag5 / c_lag10 - 1.0, np.nan)
    mom_accel = mom5 - mom5_prev

    safe_vol = np.where(vol_sma20 > 0, vol_sma20, np.nan)
    safe_sma = np.where(sma20 > 0, sma20, np.nan)
    safe_atr = np.where(atr > 0, atr, np.nan)
    rel_volume = v / safe_vol
    disp_sma20 = (c - sma20) / safe_sma
    disp_atr = (c - sma20) / safe_atr
    bb_width = 2.0 * std20 / safe_sma

    # higher-timeframe trend via spaced close changes (completed-candle safe)
    c_lag15 = np.concatenate((np.full(15, np.nan), c[:-15]))
    c_lag60 = np.concatenate((np.full(60, np.nan), c[:-60]))
    htf_trend_15 = np.sign(c - c_lag15)
    htf_trend_60 = np.sign(c - c_lag60)
    safe_sma100 = np.where(sma100 > 0, sma100, np.nan)
    trend_persist = (c - sma100) / safe_sma100
    dir_separation = (c - c_lag15) / (safe_atr * 15.0)

    # time-of-day bucket (0=open 09:30-10:30, 1=midday, 2=close 15:00-16:00) in ET
    minutes = np.array([_et_minute(int(t)) for t in ts], dtype=np.float64)
    tod_bucket = np.where(minutes < 60, 0.0, np.where(minutes >= 330, 2.0, 1.0))

    return {
        "mom5": mom5, "mom_accel": mom_accel, "rel_volume": rel_volume,
        "disp_sma20": disp_sma20, "disp_atr": disp_atr, "bb_width": bb_width,
        "htf_trend_15": htf_trend_15, "htf_trend_60": htf_trend_60,
        "trend_persist": trend_persist, "tod_bucket": tod_bucket,
        "dir_separation": dir_separation, "atr_frac": atr_frac,
    }


def _et_minute(ts_ns: int) -> int:
    """Minutes since 09:30 ET for a UTC-ns timestamp (DST-correct)."""
    dt = _dt.datetime.fromtimestamp(ts_ns / 1e9, tz=_ET)
    return (dt.hour - 9) * 60 + (dt.minute - 30)


def build_entry_dataset(bars: dict, entries: list, *, cost_frac: float = 0.0004) -> dict:
    """Assemble (features, label) rows for each entry.

    Returns {"X": {feature: array}, "y": net_returns, "entry_idx": [...], "ts": [...]} — one row
    per entry whose exit resolves. The label is the realized net return under the −0.5% stop.
    """
    feats = compute_features(bars)
    ts = bars["ts"]
    rows_idx = []
    y = []
    for e in entries:
        res = resolve_exit(bars, e, cost_frac=cost_frac, protective_stop=_STOP,
                           emergency_stop=0.03, max_hold=_MAX_HOLD)
        if res is None:
            continue
        rows_idx.append(e)
        y.append(res[3])
    rows_idx_arr = np.array(rows_idx, dtype=np.int64)
    X = {name: feats[name][rows_idx_arr] for name in FEATURE_NAMES}
    return {"X": X, "y": np.array(y, dtype=np.float64), "entry_idx": rows_idx,
            "ts": ts[rows_idx_arr] if len(rows_idx_arr) else np.array([], dtype=np.int64)}
