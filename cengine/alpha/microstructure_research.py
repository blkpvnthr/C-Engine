"""Seller-exhaustion & microstructure research (M8 §9).

Genuine Level-1 microstructure features (quote imbalance, order-flow imbalance, microprice, bid
replenishment, spread normalization, inferred cumulative volume delta) require **trade and quote
ticks**. The historical lakehouse contains only OHLCV **bars** (verified: no market_trades /
market_quotes datasets), so genuine L1 microstructure research is **INSUFFICIENT_DATA on history**
and is deferred to live/paper L1 capture (§23). No Level-2 data is fabricated.

What IS computable from bars is a clearly-labeled **bar-only proxy** for intrabar buying pressure —
the close-location value CLV = ((c−l) − (h−c)) / (h−l) ∈ [−1, 1] (and signed volume = CLV·volume).
This is a weak approximation of order-flow pressure, NOT genuine microstructure. We evaluate it
only as an entry gate (does closing in the upper half of the bar's range improve entry timing?),
on a chronological TRAIN/VALIDATION split, and report it as an approximation.
"""

from __future__ import annotations

import numpy as np

from cengine.alpha.features import build_entry_dataset
from cengine.certification.contracts import DATA_LABEL_HISTORICAL_REPLAY, safe_ticker
from cengine.optimization.exits.experiment_matrix import _entries_and_atr, _load_1m

MICROSTRUCTURE_VERSION = "microstructure-research-1.0.0"
_GENUINE_L1_FEATURES = ("quote_imbalance", "order_flow_imbalance", "microprice",
                        "inferred_cum_volume_delta", "bid_replenishment",
                        "liquidity_resilience", "spread_normalization",
                        "selling_intensity_deterioration")


def _clv(bars: dict, idx: np.ndarray) -> np.ndarray:
    h = bars["h"].astype(np.float64)[idx]
    low = bars["l"].astype(np.float64)[idx]
    c = bars["c"].astype(np.float64)[idx]
    rng = h - low
    safe_rng = np.where(rng > 0, rng, np.nan)
    clv = ((c - low) - (h - c)) / safe_rng
    return np.where(np.isfinite(clv), clv, 0.0)


def run_microstructure_research(symbols: list, *, config=None) -> dict:
    if config is None:
        from cengine.lakehouse.config import LakehouseConfig
        config = LakehouseConfig.from_env()
    # data availability
    from pathlib import Path
    has_ticks = any(
        Path(str(config.dataset_path("bronze", ds))).exists()
        and any(Path(str(config.dataset_path("bronze", ds))).rglob("*.parquet"))
        for ds in ("market_trades", "market_quotes", "trades", "quotes"))

    # bar-only proxy evaluation (clearly labeled approximation). The gate (CLV>0) is a FIXED rule
    # with no fitted threshold, so it is measured on the validation slice vs the val baseline.
    va_clv, va_y = [], []
    per_symbol = {}
    for sym in symbols:
        s = safe_ticker(sym)
        bars = _load_1m(config, s)
        if bars is None:
            per_symbol[s] = {"available": False}
            continue
        entries = _entries_and_atr(bars)[0]
        ds = build_entry_dataset(bars, entries)
        n = len(ds["y"])
        if n < 100:
            per_symbol[s] = {"available": True, "n": n, "note": "too few"}
            continue
        idx = np.array(ds["entry_idx"], dtype=np.int64)
        clv = _clv(bars, idx)
        cut = int(n * 0.70)
        va_clv.append(clv[cut:])
        va_y.append(ds["y"][cut:])
        per_symbol[s] = {"available": True, "n": n}

    proxy = {"status": "insufficient"}
    if va_y:
        vac = np.concatenate(va_clv)
        vay = np.concatenate(va_y)
        base_val = float(vay.mean())
        # gate: keep entries closing in the upper half (CLV>0) — a buying-pressure proxy
        gate = vac > 0
        if gate.sum() >= 100:
            gated_val = float(vay[gate].mean())
            proxy = {"status": "ok", "proxy": "close_location_value>0 (bar-only buying pressure)",
                     "baseline_val_expectancy": round(base_val, 6),
                     "gated_val_expectancy": round(gated_val, 6),
                     "gated_keep_fraction": round(float(gate.mean()), 4),
                     "improves": gated_val > base_val, "positive": gated_val > 0,
                     "caveat": "bar-only approximation of order-flow pressure, NOT genuine L1"}

    return {
        "version": MICROSTRUCTURE_VERSION, "data_label": DATA_LABEL_HISTORICAL_REPLAY,
        "genuine_l1_tick_data_available": has_ticks,
        "genuine_l1_features_status": "INSUFFICIENT_DATA" if not has_ticks else "available",
        "genuine_l1_features_deferred": list(_GENUINE_L1_FEATURES),
        "bar_only_proxy_evaluation": proxy, "per_symbol": per_symbol,
        "conclusion": (
            "Genuine L1 microstructure (quote/order-flow imbalance, microprice, bid replenishment, "
            "spread normalization, seller-exhaustion) is INSUFFICIENT_DATA — the historical "
            "lakehouse holds bars only; it is deferred to live/paper L1 capture. The bar-only "
            "close-location proxy "
            + (("improves but stays non-positive" if proxy.get("improves")
                and not proxy.get("positive")
                else "does not improve") if proxy.get("status") == "ok"
               else "could not be evaluated")
            + " entry timing — a weak approximation, not genuine microstructure evidence."),
        "note": "no Level-2 observations fabricated; bar-only proxies separately identified",
    }
