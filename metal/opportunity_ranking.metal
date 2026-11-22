#include <metal_stdlib>
using namespace metal;

// ============================================================================
// opportunity_ranking.metal
// ----------------------------------------------------------------------------
// Phase B GPU cross-sectional candidate scoring: one thread per candidate
// computes the composite [0,1] score that previously ran in Python
// market_scan.ranking.score_candidate. Per-candidate raw components (rs,
// order_flow, dir_sign, eligible) are precomputed host-side by reusing the exact
// scanner functions, so this kernel cannot drift from validated ranking
// semantics — it performs the normalization + weighted composite (the
// market-relative part) in parallel. The CPU oracle cpu_ref::rank_scores mirrors
// this float-for-float; sorting / top-K is a deterministic host step.
//
// Structs are 4-byte-field, no padding: binary-identical to RankInput /
// RankParams / RankOutput in screening_types.hpp.
// ============================================================================

struct RankInput {
    float rs;
    float short_mom;
    float vwap_dist;
    float bb_mid_dist;
    float rel_volume;
    float addv;
    float spread_pct;
    float order_flow;
    float dir_sign;
    uint has_rs;
    uint has_spread;
    uint eligible;
};

struct RankParams {
    float w_rel_strength;
    float w_momentum;
    float w_volume_expansion;
    float w_liquidity;
    float w_order_flow;
    float w_regime_align;
    float spy_score;
    float qqq_score;
    float regime_score;
    uint count;
};

struct RankOutput {
    float total_score;
    float rel_strength;
    float momentum;
    float volume_expansion;
    float liquidity;
    float order_flow;
    float regime_align;
    uint eligible;
};

inline float clip01(float x) { return clamp(x, 0.0f, 1.0f); }

kernel void rank_candidates(
    device const RankInput* inputs [[buffer(0)]],
    device RankOutput* out [[buffer(1)]],
    constant RankParams& p [[buffer(2)]],
    uint id [[thread_position_in_grid]]) {
    if (id >= p.count) {
        return;
    }
    RankInput f = inputs[id];
    RankOutput r = {};

    float total_w = p.w_rel_strength + p.w_momentum + p.w_volume_expansion +
                    p.w_liquidity + p.w_order_flow + p.w_regime_align;

    float rel_strength;
    if (f.has_rs) {
        float market_avg = 0.5f * (p.spy_score + p.qqq_score);
        rel_strength = clip01(0.5f + (f.rs - market_avg) / 200.0f);
    } else {
        rel_strength = 0.3f;
    }

    float structure = 0.0f;
    if (f.vwap_dist > 0.0f) structure += 0.5f;
    if (f.bb_mid_dist > 0.0f) structure += 0.5f;
    float momentum = clip01(
        0.5f * (0.5f + 0.5f * clamp(f.short_mom * 20.0f, -1.0f, 1.0f)) +
        0.5f * structure);

    float volume_expansion = clip01((f.rel_volume - 0.5f) / 1.5f);

    float addv_norm = f.addv > 0.0f ? f.addv / (f.addv + 1.0e8f) : 0.0f;
    float spread_norm = f.has_spread
                            ? 1.0f - clip01(f.spread_pct / max(1e-9f, 0.02f))
                            : 0.5f;
    float liquidity = clip01(0.6f * addv_norm + 0.4f * spread_norm);

    float order_flow = clip01(f.order_flow);

    float regime_align = clip01(
        0.5f + 0.5f * clamp(p.regime_score * f.dir_sign / 100.0f, -1.0f, 1.0f));

    float score = p.w_rel_strength * rel_strength + p.w_momentum * momentum +
                  p.w_volume_expansion * volume_expansion +
                  p.w_liquidity * liquidity + p.w_order_flow * order_flow +
                  p.w_regime_align * regime_align;
    score = clip01(total_w > 0.0f ? score / total_w : 0.0f);

    r.total_score = score;
    r.rel_strength = rel_strength;
    r.momentum = momentum;
    r.volume_expansion = volume_expansion;
    r.liquidity = liquidity;
    r.order_flow = order_flow;
    r.regime_align = regime_align;
    r.eligible = f.eligible;
    out[id] = r;
}
