#include <metal_stdlib>
using namespace metal;

// ============================================================================
// opportunity_screening.metal
// ----------------------------------------------------------------------------
// GPU individual-ticker entry / technical-exit screening (one thread per symbol).
// Implements the mandatory shared qualification contract (architectural
// correction §11/§12, master spec §48). The host (ScreeningInput/ScreeningOutput
// in opportunity_discovery/screening_types.hpp) and the native CPU oracle
// (cpu_reference.cpp screen_market) mirror this logic byte- and branch-for-branch
// so GPU and CPU agree exactly. Kernels carry NO trading policy beyond this
// contract and never place orders.
//
// Struct layout: all fields are 4-byte float/uint with no padding so the Metal
// and C++ structs are binary-identical.
// ============================================================================

enum ScreenFlag : uint {
    ENTRY_QUALIFIED   = 1u << 0,
    TECHNICAL_EXIT    = 1u << 1,
    BULLISH_DIRECTION = 1u << 2,
    BEARISH_DIRECTION = 1u << 3,
    POSITIVE_MOMENTUM = 1u << 4,
    NEGATIVE_MOMENTUM = 1u << 5,
    ABOVE_VWAP        = 1u << 6,
    BELOW_VWAP        = 1u << 7,
    DATA_INVALID      = 1u << 8
};

struct ScreeningInput {
    float price;
    float previous_close_1m;
    float di_plus_5m;
    float di_minus_5m;
    float session_vwap;
    float sma20_5m;
    float bb_middle_5m;
    uint valid_market;
    uint valid_direction;
    uint valid_1m;
    uint valid_vwap;
    uint valid_sma;
    uint valid_bb;
};

struct ScreeningOutput {
    uint flags;
    float momentum_1m_pct;
    float directional_strength;
    float vwap_distance_pct;
};

kernel void screen_market(
    device const ScreeningInput* inputs [[buffer(0)]],
    device ScreeningOutput* outputs [[buffer(1)]],
    constant uint& symbol_count [[buffer(2)]],
    uint id [[thread_position_in_grid]]) {
    if (id >= symbol_count) {
        return;
    }

    ScreeningInput s = inputs[id];
    ScreeningOutput result = {};

    bool price_valid =
        s.valid_market && isfinite(s.price) && s.price > 0.0f;
    bool direction_valid =
        s.valid_direction && isfinite(s.di_plus_5m) && isfinite(s.di_minus_5m);
    bool momentum_valid =
        price_valid && s.valid_1m && isfinite(s.previous_close_1m) &&
        s.previous_close_1m > 0.0f;
    bool vwap_valid =
        s.valid_vwap && isfinite(s.session_vwap) && s.session_vwap > 0.0f;
    bool sma_valid =
        s.valid_sma && isfinite(s.sma20_5m) && s.sma20_5m > 0.0f;
    bool bb_valid =
        s.valid_bb && isfinite(s.bb_middle_5m) && s.bb_middle_5m > 0.0f;

    bool bullish = direction_valid && s.di_plus_5m > s.di_minus_5m;
    bool bearish = direction_valid && s.di_minus_5m > s.di_plus_5m;

    bool above_vwap = price_valid && vwap_valid && s.price > s.session_vwap;
    bool below_vwap = price_valid && vwap_valid && s.price < s.session_vwap;

    bool above_reference =
        price_valid &&
        (above_vwap || (sma_valid && s.price > s.sma20_5m) ||
         (bb_valid && s.price > s.bb_middle_5m));
    bool below_reference =
        price_valid &&
        (below_vwap || (sma_valid && s.price < s.sma20_5m) ||
         (bb_valid && s.price < s.bb_middle_5m));

    bool positive_1m = momentum_valid && s.price > s.previous_close_1m;
    bool negative_1m = momentum_valid && s.price < s.previous_close_1m;

    if (bullish) result.flags |= BULLISH_DIRECTION;
    if (bearish) result.flags |= BEARISH_DIRECTION;
    if (positive_1m) result.flags |= POSITIVE_MOMENTUM;
    if (negative_1m) result.flags |= NEGATIVE_MOMENTUM;
    if (above_vwap) result.flags |= ABOVE_VWAP;
    if (below_vwap) result.flags |= BELOW_VWAP;

    if (bullish && above_reference && positive_1m) {
        result.flags |= ENTRY_QUALIFIED;
    }
    if (bearish && below_reference && negative_1m) {
        result.flags |= TECHNICAL_EXIT;
    }
    if (!price_valid || !direction_valid || !momentum_valid ||
        !(vwap_valid || sma_valid || bb_valid)) {
        result.flags |= DATA_INVALID;
    }

    result.momentum_1m_pct =
        momentum_valid ? (s.price / s.previous_close_1m - 1.0f) * 100.0f : NAN;
    result.directional_strength =
        direction_valid ? s.di_plus_5m - s.di_minus_5m : NAN;
    result.vwap_distance_pct =
        (price_valid && vwap_valid) ? (s.price / s.session_vwap - 1.0f) * 100.0f
                                    : NAN;

    outputs[id] = result;
}
