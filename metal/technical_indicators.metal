#include <metal_stdlib>
using namespace metal;

// ============================================================================
// technical_indicators.metal
// ----------------------------------------------------------------------------
// Phase B.3 GPU technical-feature computation: one thread per symbol computes,
// over that symbol's rolling bar window, the directional/technical features that
// feed ScreeningInput — Wilder DMI (DI+/DI-/ADX, period dmi_period), SMA
// (sma_period) + Bollinger middle/upper/lower (population stddev), ATR, and
// session VWAP, plus last/previous completed closes. The CPU oracle
// cpu_ref::technical_features mirrors this logic in float32 in the SAME order so
// GPU and CPU agree within a justified tolerance (directive §27/§124).
//
// Input layout: closes/highs/lows/volumes are symbol_count*max_window row-major
// float (row = symbol, bars oldest->newest in [0,count)); counts[symbol] gives
// valid bar depth. Structs are 4-byte-field, no padding, binary-identical to the
// host TechWindowParams / TechFeatures in screening_types.hpp.
// ============================================================================

struct TechWindowParams {
    uint symbol_count;
    uint max_window;
    uint dmi_period;
    uint sma_period;
};

enum TechValid : uint {
    TECH_CLOSE      = 1u << 0,
    TECH_PREV_CLOSE = 1u << 1,
    TECH_SMA        = 1u << 2,
    TECH_BB         = 1u << 3,
    TECH_DMI        = 1u << 4,
    TECH_ADX        = 1u << 5,
    TECH_VWAP       = 1u << 6,
};

struct TechFeatures {
    float last_close;
    float prev_close;
    float sma;
    float bb_middle;
    float bb_upper;
    float bb_lower;
    float di_plus;
    float di_minus;
    float adx;
    float atr;
    float vwap;
    uint valid;
};

kernel void technical_features(
    device const float* closes [[buffer(0)]],
    device const float* highs [[buffer(1)]],
    device const float* lows [[buffer(2)]],
    device const float* volumes [[buffer(3)]],
    device const uint* counts [[buffer(4)]],
    constant TechWindowParams& params [[buffer(5)]],
    device TechFeatures* out [[buffer(6)]],
    uint s [[thread_position_in_grid]]) {
    if (s >= params.symbol_count) {
        return;
    }

    const uint W = params.max_window;
    const uint p = params.dmi_period == 0u ? 14u : params.dmi_period;
    const uint sp = params.sma_period == 0u ? 20u : params.sma_period;
    const uint base = s * W;
    uint n = counts[s];
    if (n > W) n = W;

    TechFeatures f = {};

    if (n >= 1u) {
        f.last_close = closes[base + n - 1u];
        f.valid |= TECH_CLOSE;
    }
    if (n >= 2u) {
        f.prev_close = closes[base + n - 2u];
        f.valid |= TECH_PREV_CLOSE;
    }

    // SMA / Bollinger over the last sp closes (population stddev).
    if (n >= sp) {
        float sum = 0.0f;
        for (uint i = n - sp; i < n; ++i) sum += closes[base + i];
        float sma = sum / (float)sp;
        float var = 0.0f;
        for (uint i = n - sp; i < n; ++i) {
            float d = closes[base + i] - sma;
            var += d * d;
        }
        float sd = sqrt(var / (float)sp);
        f.sma = sma;
        f.bb_middle = sma;
        f.bb_upper = sma + 2.0f * sd;
        f.bb_lower = sma - 2.0f * sd;
        f.valid |= TECH_SMA | TECH_BB;
    }

    // Session VWAP over the window (typical price * volume).
    float vsum = 0.0f;
    float pvsum = 0.0f;
    for (uint i = 0u; i < n; ++i) {
        float tp = (highs[base + i] + lows[base + i] + closes[base + i]) / 3.0f;
        vsum += volumes[base + i];
        pvsum += tp * volumes[base + i];
    }
    if (vsum > 0.0f) {
        f.vwap = pvsum / vsum;
        f.valid |= TECH_VWAP;
    }

    // Wilder DMI / ATR / ADX (sum-form smoothing, matches the CPU oracle).
    if (n >= p + 1u) {
        float fp = (float)p;
        float s_tr = 0.0f, s_dmp = 0.0f, s_dmm = 0.0f;
        for (uint i = 1u; i <= p; ++i) {
            float up = highs[base + i] - highs[base + i - 1u];
            float dn = lows[base + i - 1u] - lows[base + i];
            float dmp = (up > dn && up > 0.0f) ? up : 0.0f;
            float dmm = (dn > up && dn > 0.0f) ? dn : 0.0f;
            float hl = highs[base + i] - lows[base + i];
            float hc = fabs(highs[base + i] - closes[base + i - 1u]);
            float lc = fabs(lows[base + i] - closes[base + i - 1u]);
            float tr = max(hl, max(hc, lc));
            s_tr += tr;
            s_dmp += dmp;
            s_dmm += dmm;
        }
        float di_p = s_tr > 0.0f ? 100.0f * s_dmp / s_tr : 0.0f;
        float di_m = s_tr > 0.0f ? 100.0f * s_dmm / s_tr : 0.0f;
        float dx = (di_p + di_m) > 0.0f
                       ? 100.0f * fabs(di_p - di_m) / (di_p + di_m)
                       : 0.0f;
        float dx_sum = dx;
        uint dx_count = 1u;
        bool adx_started = false;
        float adx = 0.0f;
        if (p == 1u) {
            adx = dx;
            adx_started = true;
        }
        for (uint i = p + 1u; i < n; ++i) {
            float up = highs[base + i] - highs[base + i - 1u];
            float dn = lows[base + i - 1u] - lows[base + i];
            float dmp = (up > dn && up > 0.0f) ? up : 0.0f;
            float dmm = (dn > up && dn > 0.0f) ? dn : 0.0f;
            float hl = highs[base + i] - lows[base + i];
            float hc = fabs(highs[base + i] - closes[base + i - 1u]);
            float lc = fabs(lows[base + i] - closes[base + i - 1u]);
            float tr = max(hl, max(hc, lc));
            s_tr = s_tr - s_tr / fp + tr;
            s_dmp = s_dmp - s_dmp / fp + dmp;
            s_dmm = s_dmm - s_dmm / fp + dmm;
            di_p = s_tr > 0.0f ? 100.0f * s_dmp / s_tr : 0.0f;
            di_m = s_tr > 0.0f ? 100.0f * s_dmm / s_tr : 0.0f;
            dx = (di_p + di_m) > 0.0f
                     ? 100.0f * fabs(di_p - di_m) / (di_p + di_m)
                     : 0.0f;
            if (!adx_started) {
                dx_sum += dx;
                ++dx_count;
                if (dx_count == p) {
                    adx = dx_sum / fp;
                    adx_started = true;
                }
            } else {
                adx = (adx * (fp - 1.0f) + dx) / fp;
            }
        }
        f.di_plus = di_p;
        f.di_minus = di_m;
        f.atr = s_tr / fp;
        f.valid |= TECH_DMI;
        if (adx_started) {
            f.adx = adx;
            f.valid |= TECH_ADX;
        }
    }

    out[s] = f;
}
