#include <metal_stdlib>
using namespace metal;

// ============================================================================
// covariance.metal
// ============================================================================
//
// Numerical Metal backend for:
//   1. General covariance / EWMA covariance workloads.
//   2. QQQ/SQQQ rolling statistical-arbitrage calculations.
//   3. VIX/VXN read-only volatility-factor transforms.
//
// This file contains NO:
//   - order placement,
//   - pair state machine,
//   - risk approval,
//   - broker connectivity,
//   - execution authority.
//
// CPU/C++ remains the correctness oracle and execution authority.
//
// Memory contract:
//   - Inputs are contiguous float buffers.
//   - Pair windows are row-major:
//         y_returns[window_index * stride + sample]
//         x_returns[window_index * stride + sample]
//   - Factor windows follow the same convention.
//   - Host-side ABI structs must mirror the Metal structs exactly.
// ============================================================================

namespace gme {

// --------------------------------------------------------------------------
// Constants / status
// --------------------------------------------------------------------------

constant float kEpsilon = 1.0e-12f;

enum StatArbStatus : uint {
    STATARB_OK                  = 0u,
    STATARB_INSUFFICIENT        = 1u,
    STATARB_INVALID_INPUT       = 2u,
    STATARB_ZERO_VARIANCE       = 3u,
    STATARB_RESIDUAL_DEGENERATE = 4u
};

// --------------------------------------------------------------------------
// ABI structures
// --------------------------------------------------------------------------

struct CovarianceParams {
    uint observations;
    uint assets;
    uint row_stride;
    uint reserved;
};

struct EwmaParams {
    uint observations;
    uint assets;
    uint row_stride;
    uint reserved;
    float lambda;
    float minimum_variance;
    float pad0;
    float pad1;
};

struct PairWindowParams {
    uint window_count;
    uint samples_per_window;
    uint row_stride;
    uint minimum_samples;

    float minimum_variance;
    float minimum_residual_std;
    float minimum_abs_correlation;
    float reserved;
};

struct PairRegressionResult {
    float alpha;
    float beta;
    float correlation;
    float covariance;

    float variance_y;
    float variance_x;
    float residual;
    float residual_mean;

    float residual_std;
    float zscore;
    float latest_y_return;
    float latest_x_return;

    uint samples;
    uint status;
    uint valid;
    uint reserved;
};

struct VolatilityFactorParams {
    uint window_count;
    uint samples_per_window;
    uint row_stride;
    uint minimum_samples;

    float minimum_std;
    float pad0;
    float pad1;
    float pad2;
};

struct VolatilityFactorResult {
    float vix;
    float vxn;
    float vix_change;
    float vxn_change;

    float vix_zscore;
    float vxn_zscore;
    float vxn_minus_vix;
    float spread_change;

    float spread_mean;
    float spread_std;
    float spread_zscore;
    float reserved0;

    uint samples;
    uint status;
    uint valid;
    uint reserved1;
};

// --------------------------------------------------------------------------
// Helpers
// --------------------------------------------------------------------------

inline bool finite_value(const float value) {
    return isfinite(value);
}

inline float safe_sqrt(const float value) {
    return sqrt(max(value, 0.0f));
}

inline uint matrix_index(
    const uint row,
    const uint column,
    const uint width
) {
    return row * width + column;
}

// --------------------------------------------------------------------------
// General arithmetic mean
// --------------------------------------------------------------------------
//
// One thread per asset. Returns mean across observations.
// Input matrix shape: observations x assets.
// --------------------------------------------------------------------------

kernel void column_means(
    device const float* values          [[buffer(0)]],
    constant CovarianceParams& params   [[buffer(1)]],
    device float* means                 [[buffer(2)]],
    uint asset                          [[thread_position_in_grid]]
) {
    if (asset >= params.assets ||
        params.observations == 0u ||
        params.row_stride < params.assets) {
        return;
    }

    float sum = 0.0f;

    for (uint row = 0u;
         row < params.observations;
         ++row) {
        const float value =
            values[row * params.row_stride + asset];

        if (!finite_value(value)) {
            means[asset] = NAN;
            return;
        }

        sum += value;
    }

    means[asset] =
        sum / float(params.observations);
}

// --------------------------------------------------------------------------
// Sample covariance matrix
// --------------------------------------------------------------------------
//
// One thread per matrix cell.
// Output shape: assets x assets, row-major.
//
// Host should normally dispatch assets * assets threads.
// For deterministic parity this kernel does not use parallel reductions.
// --------------------------------------------------------------------------

kernel void covariance_matrix(
    device const float* values          [[buffer(0)]],
    device const float* means           [[buffer(1)]],
    constant CovarianceParams& params   [[buffer(2)]],
    device float* covariance            [[buffer(3)]],
    uint gid                            [[thread_position_in_grid]]
) {
    const uint cells =
        params.assets * params.assets;

    if (gid >= cells) {
        return;
    }

    const uint i =
        gid / params.assets;
    const uint j =
        gid % params.assets;

    if (params.observations < 2u ||
        params.row_stride < params.assets) {
        covariance[gid] = NAN;
        return;
    }

    const float mean_i = means[i];
    const float mean_j = means[j];

    if (!finite_value(mean_i) ||
        !finite_value(mean_j)) {
        covariance[gid] = NAN;
        return;
    }

    float accumulator = 0.0f;

    for (uint row = 0u;
         row < params.observations;
         ++row) {
        const uint base =
            row * params.row_stride;

        const float xi =
            values[base + i];
        const float xj =
            values[base + j];

        if (!finite_value(xi) ||
            !finite_value(xj)) {
            covariance[gid] = NAN;
            return;
        }

        accumulator +=
            (xi - mean_i) *
            (xj - mean_j);
    }

    covariance[gid] =
        accumulator /
        float(params.observations - 1u);
}

// --------------------------------------------------------------------------
// EWMA covariance matrix
// --------------------------------------------------------------------------
//
// One thread per matrix cell.
//
// Weight convention:
//     newest observation weight = 1
//     previous = lambda
//     previous = lambda^2
//
// Weights are normalized by their sum.
//
// The kernel computes weighted means internally to keep the ABI simple.
// --------------------------------------------------------------------------

kernel void ewma_covariance_matrix(
    device const float* values        [[buffer(0)]],
    constant EwmaParams& params       [[buffer(1)]],
    device float* covariance          [[buffer(2)]],
    uint gid                          [[thread_position_in_grid]]
) {
    const uint cells =
        params.assets * params.assets;

    if (gid >= cells) {
        return;
    }

    const uint i =
        gid / params.assets;
    const uint j =
        gid % params.assets;

    if (params.observations == 0u ||
        params.assets == 0u ||
        params.row_stride < params.assets ||
        !(params.lambda > 0.0f) ||
        !(params.lambda <= 1.0f)) {
        covariance[gid] = NAN;
        return;
    }

    float weight_sum = 0.0f;
    float weighted_i = 0.0f;
    float weighted_j = 0.0f;
    float weight = 1.0f;

    for (uint reverse = 0u;
         reverse < params.observations;
         ++reverse) {
        const uint row =
            params.observations - 1u - reverse;

        const uint base =
            row * params.row_stride;

        const float xi =
            values[base + i];
        const float xj =
            values[base + j];

        if (!finite_value(xi) ||
            !finite_value(xj)) {
            covariance[gid] = NAN;
            return;
        }

        weighted_i +=
            weight * xi;
        weighted_j +=
            weight * xj;
        weight_sum +=
            weight;

        weight *=
            params.lambda;
    }

    if (!(weight_sum > kEpsilon)) {
        covariance[gid] = NAN;
        return;
    }

    const float mean_i =
        weighted_i / weight_sum;
    const float mean_j =
        weighted_j / weight_sum;

    float weighted_covariance = 0.0f;
    weight = 1.0f;

    for (uint reverse = 0u;
         reverse < params.observations;
         ++reverse) {
        const uint row =
            params.observations - 1u - reverse;

        const uint base =
            row * params.row_stride;

        const float xi =
            values[base + i];
        const float xj =
            values[base + j];

        weighted_covariance +=
            weight *
            (xi - mean_i) *
            (xj - mean_j);

        weight *=
            params.lambda;
    }

    float result =
        weighted_covariance /
        weight_sum;

    if (i == j) {
        result =
            max(
                result,
                params.minimum_variance
            );
    }

    covariance[gid] =
        result;
}

// --------------------------------------------------------------------------
// QQQ / SQQQ rolling regression
// --------------------------------------------------------------------------
//
// Model:
//     y = QQQ log return
//     x = SQQQ log return
//
//     y = alpha + beta*x + epsilon
//
// beta is estimated. It is NOT forced to -1/3, -3, or any fixed leverage.
//
// Each GPU thread evaluates one complete rolling window. This is intentionally
// correctness-first. A later optimized implementation may use threadgroup
// reductions after parity tests establish the contract.
//
// Residual statistics are recomputed across the same window using the current
// window alpha/beta, matching the CPU reference design.
// --------------------------------------------------------------------------

kernel void statarb_pair_regression(
    device const float* y_returns          [[buffer(0)]],
    device const float* x_returns          [[buffer(1)]],
    constant PairWindowParams& params      [[buffer(2)]],
    device PairRegressionResult* output    [[buffer(3)]],
    uint window_index                      [[thread_position_in_grid]]
) {
    if (window_index >= params.window_count) {
        return;
    }

    PairRegressionResult result = {};
    result.status = STATARB_INVALID_INPUT;

    const uint n =
        params.samples_per_window;

    if (n < params.minimum_samples ||
        n < 3u ||
        params.row_stride < n) {
        result.status = STATARB_INSUFFICIENT;
        result.samples = n;
        output[window_index] = result;
        return;
    }

    const uint base =
        window_index * params.row_stride;

    float sum_x = 0.0f;
    float sum_y = 0.0f;

    for (uint sample = 0u;
         sample < n;
         ++sample) {
        const float y =
            y_returns[base + sample];
        const float x =
            x_returns[base + sample];

        if (!finite_value(y) ||
            !finite_value(x)) {
            output[window_index] = result;
            return;
        }

        sum_y += y;
        sum_x += x;
    }

    const float inv_n =
        1.0f / float(n);

    const float mean_y =
        sum_y * inv_n;
    const float mean_x =
        sum_x * inv_n;

    float sxx = 0.0f;
    float syy = 0.0f;
    float sxy = 0.0f;

    for (uint sample = 0u;
         sample < n;
         ++sample) {
        const float y =
            y_returns[base + sample];
        const float x =
            x_returns[base + sample];

        const float dx =
            x - mean_x;
        const float dy =
            y - mean_y;

        sxx += dx * dx;
        syy += dy * dy;
        sxy += dx * dy;
    }

    const float minimum_variance =
        max(
            params.minimum_variance,
            kEpsilon
        );

    if (!(sxx > minimum_variance) ||
        !(syy > minimum_variance)) {
        result.status = STATARB_ZERO_VARIANCE;
        result.samples = n;
        output[window_index] = result;
        return;
    }

    const float beta =
        sxy / sxx;

    const float alpha =
        mean_y -
        beta * mean_x;

    const float denominator =
        safe_sqrt(sxx * syy);

    if (!(denominator > kEpsilon)) {
        result.status = STATARB_ZERO_VARIANCE;
        result.samples = n;
        output[window_index] = result;
        return;
    }

    const float correlation =
        sxy / denominator;

    float residual_sum = 0.0f;

    for (uint sample = 0u;
         sample < n;
         ++sample) {
        const float residual =
            y_returns[base + sample] -
            (
                alpha +
                beta *
                x_returns[base + sample]
            );

        residual_sum += residual;
    }

    const float residual_mean =
        residual_sum * inv_n;

    float residual_ss = 0.0f;

    for (uint sample = 0u;
         sample < n;
         ++sample) {
        const float residual =
            y_returns[base + sample] -
            (
                alpha +
                beta *
                x_returns[base + sample]
            );

        const float delta =
            residual -
            residual_mean;

        residual_ss +=
            delta * delta;
    }

    const float residual_std =
        safe_sqrt(
            residual_ss /
            float(n - 1u)
        );

    const float latest_y =
        y_returns[base + n - 1u];

    const float latest_x =
        x_returns[base + n - 1u];

    const float latest_residual =
        latest_y -
        (
            alpha +
            beta * latest_x
        );

    result.alpha =
        alpha;
    result.beta =
        beta;
    result.correlation =
        correlation;
    result.covariance =
        sxy / float(n - 1u);

    result.variance_y =
        syy / float(n - 1u);
    result.variance_x =
        sxx / float(n - 1u);
    result.residual =
        latest_residual;
    result.residual_mean =
        residual_mean;

    result.residual_std =
        residual_std;
    result.latest_y_return =
        latest_y;
    result.latest_x_return =
        latest_x;
    result.samples =
        n;

    const float minimum_residual_std =
        max(
            params.minimum_residual_std,
            kEpsilon
        );

    if (!(residual_std >
          minimum_residual_std)) {
        result.status =
            STATARB_RESIDUAL_DEGENERATE;
        output[window_index] =
            result;
        return;
    }

    result.zscore =
        (
            latest_residual -
            residual_mean
        ) /
        residual_std;

    const bool finite_result =
        finite_value(result.alpha) &&
        finite_value(result.beta) &&
        finite_value(result.correlation) &&
        finite_value(result.zscore);

    const bool stable_relationship =
        fabs(result.correlation) >=
        params.minimum_abs_correlation;

    result.status =
        finite_result
            ? STATARB_OK
            : STATARB_INVALID_INPUT;

    result.valid =
        (
            finite_result &&
            stable_relationship
        )
            ? 1u
            : 0u;

    output[window_index] =
        result;
}

// --------------------------------------------------------------------------
// Residual vector generation
// --------------------------------------------------------------------------
//
// Useful when the host wants the full residual series for diagnostics,
// additional GPU transforms, or parity testing.
//
// One thread per (window, sample).
// --------------------------------------------------------------------------

kernel void statarb_residual_series(
    device const float* y_returns                 [[buffer(0)]],
    device const float* x_returns                 [[buffer(1)]],
    device const PairRegressionResult* regression [[buffer(2)]],
    constant PairWindowParams& params             [[buffer(3)]],
    device float* residuals                       [[buffer(4)]],
    uint gid                                      [[thread_position_in_grid]]
) {
    const uint total =
        params.window_count *
        params.samples_per_window;

    if (gid >= total ||
        params.samples_per_window == 0u) {
        return;
    }

    const uint window =
        gid /
        params.samples_per_window;

    const uint sample =
        gid %
        params.samples_per_window;

    const uint input_index =
        window *
        params.row_stride +
        sample;

    const auto stats =
        regression[window];

    if (stats.status != STATARB_OK) {
        residuals[gid] = NAN;
        return;
    }

    residuals[gid] =
        y_returns[input_index] -
        (
            stats.alpha +
            stats.beta *
            x_returns[input_index]
        );
}

// --------------------------------------------------------------------------
// VIX / VXN factor transform
// --------------------------------------------------------------------------
//
// VIX/VXN are statistical inputs only.
//
// For each rolling factor window this produces:
//   current VIX
//   current VXN
//   one-step changes
//   VIX z-score
//   VXN z-score
//   VXN - VIX spread
//   spread change
//   spread mean/std/z-score
//
// This kernel never maps volatility factors into an order or direction.
// --------------------------------------------------------------------------

kernel void volatility_factor_features(
    device const float* vix_values              [[buffer(0)]],
    device const float* vxn_values              [[buffer(1)]],
    constant VolatilityFactorParams& params     [[buffer(2)]],
    device VolatilityFactorResult* output       [[buffer(3)]],
    uint window_index                           [[thread_position_in_grid]]
) {
    if (window_index >= params.window_count) {
        return;
    }

    VolatilityFactorResult result = {};
    result.status = STATARB_INVALID_INPUT;

    const uint n =
        params.samples_per_window;

    if (n < params.minimum_samples ||
        n < 2u ||
        params.row_stride < n) {
        result.status = STATARB_INSUFFICIENT;
        result.samples = n;
        output[window_index] = result;
        return;
    }

    const uint base =
        window_index * params.row_stride;

    float sum_vix = 0.0f;
    float sum_vxn = 0.0f;
    float sum_spread = 0.0f;

    for (uint sample = 0u;
         sample < n;
         ++sample) {
        const float vix =
            vix_values[base + sample];

        const float vxn =
            vxn_values[base + sample];

        if (!finite_value(vix) ||
            !finite_value(vxn) ||
            !(vix > 0.0f) ||
            !(vxn > 0.0f)) {
            output[window_index] = result;
            return;
        }

        sum_vix += vix;
        sum_vxn += vxn;
        sum_spread +=
            vxn - vix;
    }

    const float inv_n =
        1.0f / float(n);

    const float mean_vix =
        sum_vix * inv_n;

    const float mean_vxn =
        sum_vxn * inv_n;

    const float mean_spread =
        sum_spread * inv_n;

    float ss_vix = 0.0f;
    float ss_vxn = 0.0f;
    float ss_spread = 0.0f;

    for (uint sample = 0u;
         sample < n;
         ++sample) {
        const float vix =
            vix_values[base + sample];

        const float vxn =
            vxn_values[base + sample];

        const float spread =
            vxn - vix;

        const float dvix =
            vix - mean_vix;

        const float dvxn =
            vxn - mean_vxn;

        const float dspread =
            spread - mean_spread;

        ss_vix += dvix * dvix;
        ss_vxn += dvxn * dvxn;
        ss_spread +=
            dspread * dspread;
    }

    const float divisor =
        float(n - 1u);

    const float std_vix =
        safe_sqrt(
            ss_vix / divisor
        );

    const float std_vxn =
        safe_sqrt(
            ss_vxn / divisor
        );

    const float std_spread =
        safe_sqrt(
            ss_spread / divisor
        );

    const uint latest =
        base + n - 1u;

    const uint previous =
        base + n - 2u;

    result.vix =
        vix_values[latest];

    result.vxn =
        vxn_values[latest];

    result.vix_change =
        vix_values[latest] -
        vix_values[previous];

    result.vxn_change =
        vxn_values[latest] -
        vxn_values[previous];

    result.vxn_minus_vix =
        result.vxn -
        result.vix;

    const float previous_spread =
        vxn_values[previous] -
        vix_values[previous];

    result.spread_change =
        result.vxn_minus_vix -
        previous_spread;

    result.spread_mean =
        mean_spread;

    result.spread_std =
        std_spread;

    const float minimum_std =
        max(
            params.minimum_std,
            kEpsilon
        );

    if (std_vix > minimum_std) {
        result.vix_zscore =
            (result.vix - mean_vix) /
            std_vix;
    }

    if (std_vxn > minimum_std) {
        result.vxn_zscore =
            (result.vxn - mean_vxn) /
            std_vxn;
    }

    if (std_spread > minimum_std) {
        result.spread_zscore =
            (
                result.vxn_minus_vix -
                mean_spread
            ) /
            std_spread;
    }

    const bool finite_result =
        finite_value(result.vix) &&
        finite_value(result.vxn) &&
        finite_value(result.vix_change) &&
        finite_value(result.vxn_change) &&
        finite_value(result.vxn_minus_vix) &&
        finite_value(result.spread_change);

    result.samples =
        n;

    result.status =
        finite_result
            ? STATARB_OK
            : STATARB_INVALID_INPUT;

    result.valid =
        finite_result
            ? 1u
            : 0u;

    output[window_index] =
        result;
}

// --------------------------------------------------------------------------
// Pair signed-weight generation
// --------------------------------------------------------------------------
//
// Pure numerical transform only.
//
// long residual:
//     dependent_weight = +1
//     hedge_weight     = -beta
//
// short residual:
//     dependent_weight = -1
//     hedge_weight     = +beta
//
// direction:
//      +1 = long residual
//      -1 = short residual
//       0 = no position
//
// Output is normalized so abs(w_y) + abs(w_x) == 1 when active.
//
// IMPORTANT:
// This is NOT an order-generation kernel. The C++ engine remains responsible
// for translating approved weights into quantities after PairRiskGate.
// --------------------------------------------------------------------------

kernel void statarb_normalized_weights(
    device const PairRegressionResult* regression [[buffer(0)]],
    device const int* direction                   [[buffer(1)]],
    device float2* weights                        [[buffer(2)]],
    uint gid                                      [[thread_position_in_grid]]
) {
    const auto stats =
        regression[gid];

    const int side =
        direction[gid];

    if (stats.valid == 0u ||
        (side != 1 && side != -1)) {
        weights[gid] =
            float2(0.0f);
        return;
    }

    const float dependent =
        side > 0
            ? 1.0f
            : -1.0f;

    const float hedge =
        side > 0
            ? -stats.beta
            : stats.beta;

    const float gross =
        fabs(dependent) +
        fabs(hedge);

    if (!(gross > kEpsilon) ||
        !finite_value(gross)) {
        weights[gid] =
            float2(0.0f);
        return;
    }

    weights[gid] =
        float2(
            dependent / gross,
            hedge / gross
        );
}

} // namespace gme
