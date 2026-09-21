#include <metal_stdlib>
using namespace metal;

// ============================================================================
// imbalance.metal
// ============================================================================
//
// Observation-only Metal backend for market microstructure features.
//
// Designed for:
//   MarketDepthBook
//      -> MarketDepthSoA
//      -> Metal host adapter
//      -> imbalance.metal
//      -> FeatureEngine / StatArb model / Risk analytics
//
// This file NEVER:
//   - inserts/cancels/amends orders,
//   - matches orders,
//   - treats SIP NBBO as fabricated Level-II,
//   - approves risk,
//   - generates broker orders.
//
// A snapshot may represent:
//   1. genuine provider depth, or
//   2. top-of-book-only consolidated quote.
//
// depth_source and valid_bid_levels / valid_ask_levels tell kernels what data
// actually exists. Padded zeros are never interpreted as market liquidity.
// ============================================================================

namespace gme {

// --------------------------------------------------------------------------
// Constants / enums
// --------------------------------------------------------------------------

constant float kImbalanceEpsilon = 1.0e-12f;

enum DepthSource : uint {
    DEPTH_SOURCE_UNKNOWN            = 0u,
    DEPTH_SOURCE_CONSOLIDATED_QUOTE = 1u,
    DEPTH_SOURCE_PROVIDER_DEPTH     = 2u,
    DEPTH_SOURCE_REPLAY             = 3u
};

enum ImbalanceStatus : uint {
    IMBALANCE_OK                = 0u,
    IMBALANCE_EMPTY             = 1u,
    IMBALANCE_INVALID_INPUT     = 2u,
    IMBALANCE_CROSSED_OR_LOCKED = 3u,
    IMBALANCE_ONE_SIDED         = 4u
};

// --------------------------------------------------------------------------
// ABI structures
// --------------------------------------------------------------------------
//
// Host-side mirrors should use fixed-width uint32_t / int64_t fields and
// static_assert sizeof/offsetof before dispatch.
// --------------------------------------------------------------------------

struct DepthBatchParams {
    uint snapshot_count;
    uint depth;
    uint row_stride;
    uint reserved;

    float level_decay;
    float minimum_total_size;
    float trade_flow_scale;
    float pad0;
};

struct DepthSnapshotMeta {
    uint valid_bid_levels;
    uint valid_ask_levels;
    uint depth_source;
    uint reserved0;

    ulong version;
    ulong captured_ns;

    ulong quote_timestamp_ns;
    ulong trade_timestamp_ns;
};

struct TradeFlowInput {
    ulong at_or_above_ask_volume;
    ulong at_or_below_bid_volume;
    ulong inside_spread_volume;
    ulong outside_quote_volume;

    ulong total_volume;
    ulong trade_count;

    long last_trade_price_ticks;
    ulong last_trade_size;
};

struct ImbalanceFeatures {
    float top_level_imbalance;
    float cumulative_imbalance;
    float weighted_imbalance;
    float notional_imbalance;

    float spread_ticks;
    float midpoint_ticks;
    float microprice_ticks;
    float microprice_offset_ticks;

    float bid_depth;
    float ask_depth;
    float bid_weighted_depth;
    float ask_weighted_depth;

    float trade_flow_imbalance;
    float trade_flow_participation;
    float combined_pressure;
    float top_level_size_ratio;

    uint bid_levels;
    uint ask_levels;
    uint depth_source;
    uint status;

    ulong version;
    ulong captured_ns;
};

struct LevelImbalance {
    float imbalance;
    float cumulative_imbalance;
    float weight;
    float reserved;

    ulong bid_size;
    ulong ask_size;

    long bid_price_ticks;
    long ask_price_ticks;
};

struct DepthValidationResult {
    uint valid;
    uint status;
    uint bid_levels;
    uint ask_levels;

    long best_bid_ticks;
    long best_ask_ticks;

    ulong bid_total_size;
    ulong ask_total_size;
};

// --------------------------------------------------------------------------
// Helpers
// --------------------------------------------------------------------------

inline bool finite_value(const float value) {
    return isfinite(value);
}

inline float normalized_difference(
    const float lhs,
    const float rhs
) {
    const float denominator =
        lhs + rhs;

    if (!(denominator > kImbalanceEpsilon)) {
        return 0.0f;
    }

    return (lhs - rhs) /
           denominator;
}

inline float safe_ratio(
    const float numerator,
    const float denominator
) {
    if (!(fabs(denominator) >
          kImbalanceEpsilon)) {
        return 0.0f;
    }

    return numerator /
           denominator;
}

inline float level_weight(
    const uint level,
    const float decay
) {
    if (!(decay > 0.0f) ||
        !(decay <= 1.0f)) {
        return 1.0f /
               float(level + 1u);
    }

    return pow(
        decay,
        float(level)
    );
}

inline bool valid_bid_level(
    const long price,
    const ulong size
) {
    return price > 0 &&
           size > 0u;
}

inline bool valid_ask_level(
    const long price,
    const ulong size
) {
    return price > 0 &&
           size > 0u;
}

// --------------------------------------------------------------------------
// Snapshot validation
// --------------------------------------------------------------------------
//
// One thread per snapshot.
//
// Checks:
//   - declared valid level counts,
//   - positive price/size for declared levels,
//   - descending bid ordering,
//   - ascending ask ordering,
//   - best bid < best ask,
//   - no non-zero levels after declared valid range.
//
// A one-sided book is reported explicitly.
// --------------------------------------------------------------------------

kernel void validate_depth_snapshots(
    device const long* bid_price_ticks       [[buffer(0)]],
    device const ulong* bid_sizes            [[buffer(1)]],
    device const long* ask_price_ticks       [[buffer(2)]],
    device const ulong* ask_sizes            [[buffer(3)]],
    device const DepthSnapshotMeta* meta     [[buffer(4)]],
    constant DepthBatchParams& params        [[buffer(5)]],
    device DepthValidationResult* output     [[buffer(6)]],
    uint snapshot                            [[thread_position_in_grid]]
) {
    if (snapshot >= params.snapshot_count) {
        return;
    }

    DepthValidationResult result = {};
    result.status =
        IMBALANCE_INVALID_INPUT;

    if (params.depth == 0u ||
        params.row_stride < params.depth) {
        output[snapshot] =
            result;
        return;
    }

    const auto info =
        meta[snapshot];

    if (info.valid_bid_levels >
            params.depth ||
        info.valid_ask_levels >
            params.depth) {
        output[snapshot] =
            result;
        return;
    }

    result.bid_levels =
        info.valid_bid_levels;
    result.ask_levels =
        info.valid_ask_levels;

    if (info.valid_bid_levels == 0u &&
        info.valid_ask_levels == 0u) {
        result.status =
            IMBALANCE_EMPTY;
        output[snapshot] =
            result;
        return;
    }

    const uint base =
        snapshot *
        params.row_stride;

    long previous_bid = 0;
    long previous_ask = 0;

    ulong bid_total = 0u;
    ulong ask_total = 0u;

    for (uint level = 0u;
         level < params.depth;
         ++level) {
        const long bid_price =
            bid_price_ticks[base + level];

        const ulong bid_size =
            bid_sizes[base + level];

        if (level <
            info.valid_bid_levels) {
            if (!valid_bid_level(
                    bid_price,
                    bid_size
                )) {
                output[snapshot] =
                    result;
                return;
            }

            if (level > 0u &&
                bid_price >=
                    previous_bid) {
                output[snapshot] =
                    result;
                return;
            }

            previous_bid =
                bid_price;

            bid_total +=
                bid_size;
        } else if (
            bid_price != 0 ||
            bid_size != 0u
        ) {
            output[snapshot] =
                result;
            return;
        }

        const long ask_price =
            ask_price_ticks[base + level];

        const ulong ask_size =
            ask_sizes[base + level];

        if (level <
            info.valid_ask_levels) {
            if (!valid_ask_level(
                    ask_price,
                    ask_size
                )) {
                output[snapshot] =
                    result;
                return;
            }

            if (level > 0u &&
                ask_price <=
                    previous_ask) {
                output[snapshot] =
                    result;
                return;
            }

            previous_ask =
                ask_price;

            ask_total +=
                ask_size;
        } else if (
            ask_price != 0 ||
            ask_size != 0u
        ) {
            output[snapshot] =
                result;
            return;
        }
    }

    result.bid_total_size =
        bid_total;
    result.ask_total_size =
        ask_total;

    if (info.valid_bid_levels > 0u) {
        result.best_bid_ticks =
            bid_price_ticks[base];
    }

    if (info.valid_ask_levels > 0u) {
        result.best_ask_ticks =
            ask_price_ticks[base];
    }

    if (info.valid_bid_levels == 0u ||
        info.valid_ask_levels == 0u) {
        result.status =
            IMBALANCE_ONE_SIDED;
        output[snapshot] =
            result;
        return;
    }

    if (result.best_bid_ticks >=
        result.best_ask_ticks) {
        result.status =
            IMBALANCE_CROSSED_OR_LOCKED;
        output[snapshot] =
            result;
        return;
    }

    result.valid = 1u;
    result.status =
        IMBALANCE_OK;

    output[snapshot] =
        result;
}

// --------------------------------------------------------------------------
// Per-level imbalance
// --------------------------------------------------------------------------
//
// One thread per (snapshot, level).
//
// imbalance = (bid_size - ask_size) / (bid_size + ask_size)
//
// cumulative_imbalance uses levels [0..level].
//
// Missing levels contribute zero. They are not fabricated.
// --------------------------------------------------------------------------

kernel void compute_level_imbalance(
    device const long* bid_price_ticks      [[buffer(0)]],
    device const ulong* bid_sizes           [[buffer(1)]],
    device const long* ask_price_ticks      [[buffer(2)]],
    device const ulong* ask_sizes           [[buffer(3)]],
    device const DepthSnapshotMeta* meta    [[buffer(4)]],
    constant DepthBatchParams& params       [[buffer(5)]],
    device LevelImbalance* output           [[buffer(6)]],
    uint gid                                [[thread_position_in_grid]]
) {
    const uint total =
        params.snapshot_count *
        params.depth;

    if (gid >= total ||
        params.depth == 0u ||
        params.row_stride <
            params.depth) {
        return;
    }

    const uint snapshot =
        gid / params.depth;

    const uint level =
        gid % params.depth;

    const uint base =
        snapshot *
        params.row_stride;

    const auto info =
        meta[snapshot];

    const bool has_bid =
        level <
        info.valid_bid_levels;

    const bool has_ask =
        level <
        info.valid_ask_levels;

    const ulong bid_size =
        has_bid
            ? bid_sizes[base + level]
            : 0u;

    const ulong ask_size =
        has_ask
            ? ask_sizes[base + level]
            : 0u;

    LevelImbalance result = {};

    result.bid_size =
        bid_size;
    result.ask_size =
        ask_size;

    result.bid_price_ticks =
        has_bid
            ? bid_price_ticks[
                base + level
              ]
            : 0;

    result.ask_price_ticks =
        has_ask
            ? ask_price_ticks[
                base + level
              ]
            : 0;

    result.weight =
        level_weight(
            level,
            params.level_decay
        );

    result.imbalance =
        normalized_difference(
            float(bid_size),
            float(ask_size)
        );

    float cumulative_bid = 0.0f;
    float cumulative_ask = 0.0f;

    for (uint cursor = 0u;
         cursor <= level;
         ++cursor) {
        if (cursor <
            info.valid_bid_levels) {
            cumulative_bid +=
                float(
                    bid_sizes[
                        base + cursor
                    ]
                );
        }

        if (cursor <
            info.valid_ask_levels) {
            cumulative_ask +=
                float(
                    ask_sizes[
                        base + cursor
                    ]
                );
        }
    }

    result.cumulative_imbalance =
        normalized_difference(
            cumulative_bid,
            cumulative_ask
        );

    output[gid] =
        result;
}

// --------------------------------------------------------------------------
// Full snapshot imbalance / microstructure feature kernel
// --------------------------------------------------------------------------
//
// One thread per snapshot.
//
// Features:
//   - top-level size imbalance,
//   - cumulative depth imbalance,
//   - exponentially/rank weighted depth imbalance,
//   - price-notional depth imbalance,
//   - spread,
//   - midpoint,
//   - size-weighted microprice,
//   - microprice offset,
//   - quote-relative trade-flow imbalance,
//   - trade-flow participation,
//   - combined depth + trade pressure.
//
// For consolidated-quote-only snapshots:
//   valid_bid_levels/valid_ask_levels should be 1.
//   The result is valid top-of-book information, but NOT claimed to be L2.
// --------------------------------------------------------------------------

kernel void compute_imbalance_features(
    device const long* bid_price_ticks       [[buffer(0)]],
    device const ulong* bid_sizes            [[buffer(1)]],
    device const long* ask_price_ticks       [[buffer(2)]],
    device const ulong* ask_sizes            [[buffer(3)]],
    device const DepthSnapshotMeta* meta     [[buffer(4)]],
    device const TradeFlowInput* trade_flow  [[buffer(5)]],
    constant DepthBatchParams& params        [[buffer(6)]],
    device ImbalanceFeatures* output         [[buffer(7)]],
    uint snapshot                            [[thread_position_in_grid]]
) {
    if (snapshot >= params.snapshot_count) {
        return;
    }

    ImbalanceFeatures result = {};

    result.status =
        IMBALANCE_INVALID_INPUT;

    if (params.depth == 0u ||
        params.row_stride <
            params.depth) {
        output[snapshot] =
            result;
        return;
    }

    const auto info =
        meta[snapshot];

    result.bid_levels =
        info.valid_bid_levels;
    result.ask_levels =
        info.valid_ask_levels;
    result.depth_source =
        info.depth_source;
    result.version =
        info.version;
    result.captured_ns =
        info.captured_ns;

    if (info.valid_bid_levels >
            params.depth ||
        info.valid_ask_levels >
            params.depth) {
        output[snapshot] =
            result;
        return;
    }

    if (info.valid_bid_levels == 0u &&
        info.valid_ask_levels == 0u) {
        result.status =
            IMBALANCE_EMPTY;
        output[snapshot] =
            result;
        return;
    }

    if (info.valid_bid_levels == 0u ||
        info.valid_ask_levels == 0u) {
        result.status =
            IMBALANCE_ONE_SIDED;
        output[snapshot] =
            result;
        return;
    }

    const uint base =
        snapshot *
        params.row_stride;

    const long best_bid =
        bid_price_ticks[base];

    const long best_ask =
        ask_price_ticks[base];

    const ulong best_bid_size =
        bid_sizes[base];

    const ulong best_ask_size =
        ask_sizes[base];

    if (!valid_bid_level(
            best_bid,
            best_bid_size
        ) ||
        !valid_ask_level(
            best_ask,
            best_ask_size
        )) {
        output[snapshot] =
            result;
        return;
    }

    if (best_bid >= best_ask) {
        result.status =
            IMBALANCE_CROSSED_OR_LOCKED;
        output[snapshot] =
            result;
        return;
    }

    float total_bid = 0.0f;
    float total_ask = 0.0f;

    float weighted_bid = 0.0f;
    float weighted_ask = 0.0f;

    float bid_notional = 0.0f;
    float ask_notional = 0.0f;

    long previous_bid = 0;
    long previous_ask = 0;

    for (uint level = 0u;
         level < params.depth;
         ++level) {
        if (level <
            info.valid_bid_levels) {
            const long price =
                bid_price_ticks[
                    base + level
                ];

            const ulong size =
                bid_sizes[
                    base + level
                ];

            if (!valid_bid_level(
                    price,
                    size
                ) ||
                (
                    level > 0u &&
                    price >= previous_bid
                )) {
                output[snapshot] =
                    result;
                return;
            }

            previous_bid =
                price;

            const float fsize =
                float(size);

            const float weight =
                level_weight(
                    level,
                    params.level_decay
                );

            total_bid +=
                fsize;

            weighted_bid +=
                fsize * weight;

            bid_notional +=
                fsize *
                float(price);
        }

        if (level <
            info.valid_ask_levels) {
            const long price =
                ask_price_ticks[
                    base + level
                ];

            const ulong size =
                ask_sizes[
                    base + level
                ];

            if (!valid_ask_level(
                    price,
                    size
                ) ||
                (
                    level > 0u &&
                    price <= previous_ask
                )) {
                output[snapshot] =
                    result;
                return;
            }

            previous_ask =
                price;

            const float fsize =
                float(size);

            const float weight =
                level_weight(
                    level,
                    params.level_decay
                );

            total_ask +=
                fsize;

            weighted_ask +=
                fsize * weight;

            ask_notional +=
                fsize *
                float(price);
        }
    }

    if (total_bid + total_ask <
        params.minimum_total_size) {
        result.status =
            IMBALANCE_EMPTY;
        output[snapshot] =
            result;
        return;
    }

    result.top_level_imbalance =
        normalized_difference(
            float(best_bid_size),
            float(best_ask_size)
        );

    result.cumulative_imbalance =
        normalized_difference(
            total_bid,
            total_ask
        );

    result.weighted_imbalance =
        normalized_difference(
            weighted_bid,
            weighted_ask
        );

    result.notional_imbalance =
        normalized_difference(
            bid_notional,
            ask_notional
        );

    result.bid_depth =
        total_bid;
    result.ask_depth =
        total_ask;

    result.bid_weighted_depth =
        weighted_bid;
    result.ask_weighted_depth =
        weighted_ask;

    result.spread_ticks =
        float(
            best_ask -
            best_bid
        );

    result.midpoint_ticks =
        (
            float(best_bid) +
            float(best_ask)
        ) * 0.5f;

    // Standard top-of-book microprice:
    //
    //     ask * bid_size + bid * ask_size
    //     --------------------------------
    //             bid_size + ask_size
    //
    // Larger bid size pushes microprice toward ask; larger ask size pushes
    // it toward bid.
    const float top_size_sum =
        float(best_bid_size) +
        float(best_ask_size);

    if (top_size_sum >
        kImbalanceEpsilon) {
        result.microprice_ticks =
            (
                float(best_ask) *
                    float(best_bid_size) +
                float(best_bid) *
                    float(best_ask_size)
            ) /
            top_size_sum;

        result.microprice_offset_ticks =
            result.microprice_ticks -
            result.midpoint_ticks;
    }

    result.top_level_size_ratio =
        safe_ratio(
            float(best_bid_size),
            float(best_ask_size)
        );

    const auto flow =
        trade_flow[snapshot];

    const float aggressive_buy =
        float(
            flow.at_or_above_ask_volume
        );

    const float aggressive_sell =
        float(
            flow.at_or_below_bid_volume
        );

    result.trade_flow_imbalance =
        normalized_difference(
            aggressive_buy,
            aggressive_sell
        );

    if (flow.total_volume > 0u) {
        result.trade_flow_participation =
            (
                aggressive_buy +
                aggressive_sell
            ) /
            float(flow.total_volume);
    }

    // Bounded fusion. trade_flow_scale=0 makes this pure weighted depth.
    const float flow_scale =
        max(
            params.trade_flow_scale,
            0.0f
        );

    result.combined_pressure =
        (
            result.weighted_imbalance +
            flow_scale *
                result.trade_flow_imbalance
        ) /
        (
            1.0f +
            flow_scale
        );

    result.status =
        IMBALANCE_OK;

    output[snapshot] =
        result;
}

// --------------------------------------------------------------------------
// Top-level-only imbalance
// --------------------------------------------------------------------------
//
// Lightweight path for SIP/NBBO-only observations.
//
// One thread per snapshot. This kernel intentionally reads ONLY level zero.
// It must not be described as Level-II imbalance.
// --------------------------------------------------------------------------

kernel void compute_top_level_imbalance(
    device const long* bid_price_ticks       [[buffer(0)]],
    device const ulong* bid_sizes            [[buffer(1)]],
    device const long* ask_price_ticks       [[buffer(2)]],
    device const ulong* ask_sizes            [[buffer(3)]],
    device const DepthSnapshotMeta* meta     [[buffer(4)]],
    constant DepthBatchParams& params        [[buffer(5)]],
    device float* output                     [[buffer(6)]],
    uint snapshot                            [[thread_position_in_grid]]
) {
    if (snapshot >= params.snapshot_count) {
        return;
    }

    output[snapshot] = 0.0f;

    if (params.depth == 0u ||
        params.row_stride <
            params.depth) {
        return;
    }

    const auto info =
        meta[snapshot];

    if (info.valid_bid_levels == 0u ||
        info.valid_ask_levels == 0u) {
        return;
    }

    const uint base =
        snapshot *
        params.row_stride;

    const long bid_price =
        bid_price_ticks[base];

    const long ask_price =
        ask_price_ticks[base];

    const ulong bid_size =
        bid_sizes[base];

    const ulong ask_size =
        ask_sizes[base];

    if (!valid_bid_level(
            bid_price,
            bid_size
        ) ||
        !valid_ask_level(
            ask_price,
            ask_size
        ) ||
        bid_price >= ask_price) {
        return;
    }

    output[snapshot] =
        normalized_difference(
            float(bid_size),
            float(ask_size)
        );
}

// --------------------------------------------------------------------------
// Weighted depth imbalance
// --------------------------------------------------------------------------
//
// Lightweight scalar path when the caller does not need the full feature
// struct. One thread per snapshot.
// --------------------------------------------------------------------------

kernel void compute_weighted_depth_imbalance(
    device const ulong* bid_sizes            [[buffer(0)]],
    device const ulong* ask_sizes            [[buffer(1)]],
    device const DepthSnapshotMeta* meta     [[buffer(2)]],
    constant DepthBatchParams& params        [[buffer(3)]],
    device float* output                     [[buffer(4)]],
    uint snapshot                            [[thread_position_in_grid]]
) {
    if (snapshot >= params.snapshot_count) {
        return;
    }

    output[snapshot] = 0.0f;

    if (params.depth == 0u ||
        params.row_stride <
            params.depth) {
        return;
    }

    const auto info =
        meta[snapshot];

    const uint base =
        snapshot *
        params.row_stride;

    float bid = 0.0f;
    float ask = 0.0f;

    for (uint level = 0u;
         level < params.depth;
         ++level) {
        const float weight =
            level_weight(
                level,
                params.level_decay
            );

        if (level <
            info.valid_bid_levels) {
            bid +=
                float(
                    bid_sizes[
                        base + level
                    ]
                ) *
                weight;
        }

        if (level <
            info.valid_ask_levels) {
            ask +=
                float(
                    ask_sizes[
                        base + level
                    ]
                ) *
                weight;
        }
    }

    output[snapshot] =
        normalized_difference(
            bid,
            ask
        );
}

// --------------------------------------------------------------------------
// Depth slope / concentration
// --------------------------------------------------------------------------
//
// Produces:
//   x = bid concentration at top level
//   y = ask concentration at top level
//   z = bid average distance from best, in ticks
//   w = ask average distance from best, in ticks
//
// Useful for distinguishing the same aggregate imbalance arising from very
// different depth shapes.
// --------------------------------------------------------------------------

kernel void compute_depth_shape(
    device const long* bid_price_ticks       [[buffer(0)]],
    device const ulong* bid_sizes            [[buffer(1)]],
    device const long* ask_price_ticks       [[buffer(2)]],
    device const ulong* ask_sizes            [[buffer(3)]],
    device const DepthSnapshotMeta* meta     [[buffer(4)]],
    constant DepthBatchParams& params        [[buffer(5)]],
    device float4* output                    [[buffer(6)]],
    uint snapshot                            [[thread_position_in_grid]]
) {
    if (snapshot >= params.snapshot_count) {
        return;
    }

    output[snapshot] =
        float4(0.0f);

    if (params.depth == 0u ||
        params.row_stride <
            params.depth) {
        return;
    }

    const auto info =
        meta[snapshot];

    if (info.valid_bid_levels == 0u ||
        info.valid_ask_levels == 0u) {
        return;
    }

    const uint base =
        snapshot *
        params.row_stride;

    const long best_bid =
        bid_price_ticks[base];

    const long best_ask =
        ask_price_ticks[base];

    if (best_bid <= 0 ||
        best_ask <= 0 ||
        best_bid >= best_ask) {
        return;
    }

    float bid_total = 0.0f;
    float ask_total = 0.0f;

    float bid_distance_sum = 0.0f;
    float ask_distance_sum = 0.0f;

    for (uint level = 0u;
         level < params.depth;
         ++level) {
        if (level <
            info.valid_bid_levels) {
            const float size =
                float(
                    bid_sizes[
                        base + level
                    ]
                );

            const float distance =
                float(
                    best_bid -
                    bid_price_ticks[
                        base + level
                    ]
                );

            bid_total +=
                size;

            bid_distance_sum +=
                size *
                max(distance, 0.0f);
        }

        if (level <
            info.valid_ask_levels) {
            const float size =
                float(
                    ask_sizes[
                        base + level
                    ]
                );

            const float distance =
                float(
                    ask_price_ticks[
                        base + level
                    ] -
                    best_ask
                );

            ask_total +=
                size;

            ask_distance_sum +=
                size *
                max(distance, 0.0f);
        }
    }

    const float bid_concentration =
        bid_total >
            kImbalanceEpsilon
        ? float(bid_sizes[base]) /
            bid_total
        : 0.0f;

    const float ask_concentration =
        ask_total >
            kImbalanceEpsilon
        ? float(ask_sizes[base]) /
            ask_total
        : 0.0f;

    const float bid_average_distance =
        bid_total >
            kImbalanceEpsilon
        ? bid_distance_sum /
            bid_total
        : 0.0f;

    const float ask_average_distance =
        ask_total >
            kImbalanceEpsilon
        ? ask_distance_sum /
            ask_total
        : 0.0f;

    output[snapshot] =
        float4(
            bid_concentration,
            ask_concentration,
            bid_average_distance,
            ask_average_distance
        );
}

// --------------------------------------------------------------------------
// Cross-symbol pressure spread
// --------------------------------------------------------------------------
//
// Intended for QQQ/SQQQ observation analytics after each symbol's independent
// ImbalanceFeatures has already been computed.
//
// output.x = QQQ pressure
// output.y = SQQQ pressure
// output.z = QQQ pressure - SQQQ pressure
// output.w = QQQ pressure + SQQQ pressure
//
// No directional interpretation is imposed here. The CPU/stat-arb model
// decides how (or whether) this feature is useful.
// --------------------------------------------------------------------------

kernel void compute_pair_pressure_features(
    device const ImbalanceFeatures* qqq      [[buffer(0)]],
    device const ImbalanceFeatures* sqqq     [[buffer(1)]],
    device float4* output                    [[buffer(2)]],
    uint gid                                 [[thread_position_in_grid]]
) {
    const auto q =
        qqq[gid];

    const auto s =
        sqqq[gid];

    if (q.status != IMBALANCE_OK ||
        s.status != IMBALANCE_OK) {
        output[gid] =
            float4(0.0f);
        return;
    }

    const float q_pressure =
        q.combined_pressure;

    const float s_pressure =
        s.combined_pressure;

    output[gid] =
        float4(
            q_pressure,
            s_pressure,
            q_pressure - s_pressure,
            q_pressure + s_pressure
        );
}

} // namespace gme
