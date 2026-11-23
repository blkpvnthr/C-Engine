# Microstructure Feature Contract (Milestone 4 addendum)

`MicrostructureFeatures` (`cengine/microstructure.py`, `feature_version =
"microstructure-1.0.0"`) is the versioned L1 snapshot. Fields are grouped so
consumers and the future ML/bandit/RL platform can distinguish what was **observed**
from what was **inferred** and how trustworthy it is.

## Observed (directly from SIP L1)

`bid`, `ask`, `mid`, `spread_bps`, `bid_size`, `ask_size`, `last_trade_price`.

## Inferred (derived; approximate; never exchange truth)

`microprice`, `microprice_displacement_bps`, `quote_imbalance`,
`order_flow_imbalance`, `cumulative_volume_delta`, `trade_arrival_intensity`,
`aggressive_imbalance`, `mean_trade_size`, `p95_trade_size`, `large_trade_fraction`,
`kyle_lambda`, `amihud_illiquidity`, `liquidity_resilience`, `seller_exhaustion`,
`buyer_exhaustion`, `adverse_selection_bps`. Any value that cannot be determined is
`None` — never fabricated.

## Quality

`is_stale`, `quote_age_s`, `trade_age_s`, `crossed_or_locked`, `n_trades_window`,
`confidence` (heuristic [0,1], **not** calibrated), `quality` ∈
{`OK`, `STALE`, `CROSSED`, `INSUFFICIENT_DATA`}.

## Fill-probability contract (§18)

`fill_probability_features() -> dict` returns a flat, point-in-time snapshot
(feature_version + spread, quote imbalance, OFI, microprice displacement, arrival
intensity, aggressive imbalance, Kyle, Amihud, liquidity resilience, adverse
selection, is_stale, confidence) for a **future** fill-probability model. No model is
fit in this milestone — the contract is prepared for the later supervised / contextual-
bandit / constrained offline-RL platform. Decision events carry the snapshot
(`MICROSTRUCTURE_SNAPSHOT`) with stable ids, versioned schema, and point-in-time
timestamps, recoverable for offline learning (Parquet/DuckDB platform deferred).
