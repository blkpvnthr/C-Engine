#include <metal_stdlib>
using namespace metal;

// ============================================================================
// monte-carlo.metal
// ============================================================================
//
// Numerical Monte Carlo backend for the Apple-Silicon market engine.
//
// Intended pipeline:
//
//   covariance.metal
//        |
//        +--> means / covariance / Cholesky factor
//        |
//        v
//   monte-carlo.metal
//        |
//        +--> correlated shocks
//        +--> simulated returns
//        +--> portfolio P&L
//        +--> QQQ/SQQQ residual scenarios
//        +--> path terminal values / drawdowns
//        |
//        v
//   CPU/C++ aggregation + RiskEngine / PairRiskGate
//
// IMPORTANT AUTHORITY BOUNDARY
// ----------------------------
// This file performs numerical simulation only. It NEVER:
//   - approves/rejects an order,
//   - changes pair state,
//   - submits/cancels/amends orders,
//   - connects to a broker,
//   - invents market depth,
//   - chooses live-money risk limits.
//
// Random seeds are supplied by the host. Reproducible research should persist
// seed + simulation parameters + model/policy version.
//
// Layout conventions
// ------------------
// Scenario-major matrix:
//
//   values[scenario * row_stride + asset]
//
// Cholesky matrix is dense row-major:
//
//   L[row * assets + column]
//
// and is assumed lower triangular.
//
// P&L values are expressed in host-selected monetary/notional units.
// ============================================================================

namespace gme {

// --------------------------------------------------------------------------
// Constants / status
// --------------------------------------------------------------------------

constant float kMcEpsilon = 1.0e-12f;
constant float kTwoPi =
    6.28318530717958647692f;

enum MonteCarloStatus : uint {
    MC_OK                 = 0u,
    MC_INVALID_INPUT      = 1u,
    MC_NUMERICAL_FAILURE  = 2u,
    MC_INSUFFICIENT_DATA  = 3u
};

enum ReturnModel : uint {
    RETURN_MODEL_ARITHMETIC = 0u,
    RETURN_MODEL_LOGNORMAL  = 1u
};

// --------------------------------------------------------------------------
// ABI structures
// --------------------------------------------------------------------------

struct MonteCarloParams {
    uint scenarios;
    uint assets;
    uint row_stride;
    uint return_model;

    float horizon_scale;
    float volatility_scale;
    float shock_clip;
    float reserved0;

    ulong base_seed;
    ulong stream_id;
};

struct PortfolioParams {
    uint scenarios;
    uint assets;
    uint row_stride;
    uint reserved0;

    float gross_notional;
    float reserved1;
    float reserved2;
    float reserved3;
};

struct PairMonteCarloParams {
    uint scenarios;
    uint row_stride;
    uint reserved0;
    uint reserved1;

    float alpha;
    float beta;
    float residual_mean;
    float residual_std;

    float dependent_weight;
    float hedge_weight;
    float gross_notional;
    float shock_clip;

    ulong base_seed;
    ulong stream_id;
};

struct PathParams {
    uint scenarios;
    uint steps;
    uint assets;
    uint row_stride;

    uint return_model;
    uint reserved0;
    uint reserved1;
    uint reserved2;

    float dt;
    float volatility_scale;
    float shock_clip;
    float reserved3;

    ulong base_seed;
    ulong stream_id;
};

struct ScenarioSummary {
    float portfolio_return;
    float pnl;
    float gross_shock;
    float max_abs_asset_return;

    uint status;
    uint reserved0;
    uint reserved1;
    uint reserved2;
};

struct PairScenarioResult {
    float qqq_return;
    float sqqq_return;
    float residual;
    float residual_z;

    float pair_return;
    float pnl;
    float gross_abs_return;
    float reserved0;

    uint status;
    uint reserved1;
    uint reserved2;
    uint reserved3;
};

struct PathResult {
    float terminal_return;
    float terminal_pnl;
    float max_drawdown;
    float realized_variance;

    uint status;
    uint reserved0;
    uint reserved1;
    uint reserved2;
};

struct TailMomentPartial {
    float loss_sum;
    float loss_sq_sum;
    float downside_sum;
    float downside_sq_sum;

    float worst_loss;
    float best_pnl;

    uint loss_count;
    uint sample_count;
};

// --------------------------------------------------------------------------
// PRNG helpers
// --------------------------------------------------------------------------
//
// SplitMix64 is used as a deterministic counter/hash mixer. Each output is
// derived from host seed + stream + scenario/asset/sample coordinates.
//
// This avoids mutable global RNG state and makes CPU/GPU replay easier.
//
// This is NOT intended as a cryptographic RNG.
// --------------------------------------------------------------------------

inline ulong splitmix64(ulong x) {
    x += 0x9E3779B97F4A7C15ul;
    x =
        (x ^ (x >> 30u)) *
        0xBF58476D1CE4E5B9ul;
    x =
        (x ^ (x >> 27u)) *
        0x94D049BB133111EBul;
    return x ^ (x >> 31u);
}

inline ulong coordinate_seed(
    const ulong base_seed,
    const ulong stream_id,
    const ulong scenario,
    const ulong dimension,
    const ulong sample
) {
    ulong x = base_seed;

    x ^= splitmix64(
        stream_id +
        0xD1B54A32D192ED03ul
    );

    x ^= splitmix64(
        scenario +
        0x94D049BB133111EBul
    );

    x ^= splitmix64(
        dimension +
        0xBF58476D1CE4E5B9ul
    );

    x ^= splitmix64(
        sample +
        0x9E3779B97F4A7C15ul
    );

    return splitmix64(x);
}

inline float uniform_open01(
    const ulong state
) {
    // Use 24 high-quality bits, avoiding exact 0 and 1.
    const uint bits =
        uint(
            (state >> 40u) &
            0xFFFFFFul
        );

    return (
        float(bits) +
        0.5f
    ) /
    16777216.0f;
}

inline float normal_from_seed(
    const ulong seed
) {
    const float u1 =
        max(
            uniform_open01(
                splitmix64(seed)
            ),
            1.0e-7f
        );

    const float u2 =
        uniform_open01(
            splitmix64(
                seed ^
                0xA0761D6478BD642Ful
            )
        );

    const float radius =
        sqrt(
            -2.0f *
            log(u1)
        );

    const float angle =
        kTwoPi * u2;

    return radius *
           cos(angle);
}

inline float clipped_normal(
    const ulong seed,
    const float clip
) {
    const float z =
        normal_from_seed(seed);

    if (!(clip > 0.0f)) {
        return z;
    }

    return clamp(
        z,
        -clip,
        clip
    );
}

inline bool finite_value(
    const float value
) {
    return isfinite(value);
}

// --------------------------------------------------------------------------
// Independent standard-normal matrix
// --------------------------------------------------------------------------
//
// One thread per (scenario, asset).
//
// Useful when the host wants to inspect/reuse the raw Gaussian draws.
// --------------------------------------------------------------------------

kernel void generate_standard_normals(
    constant MonteCarloParams& params  [[buffer(0)]],
    device float* normals              [[buffer(1)]],
    uint gid                           [[thread_position_in_grid]]
) {
    const uint total =
        params.scenarios *
        params.assets;

    if (gid >= total ||
        params.assets == 0u ||
        params.row_stride <
            params.assets) {
        return;
    }

    const uint scenario =
        gid / params.assets;

    const uint asset =
        gid % params.assets;

    const uint index =
        scenario *
        params.row_stride +
        asset;

    const ulong seed =
        coordinate_seed(
            params.base_seed,
            params.stream_id,
            ulong(scenario),
            ulong(asset),
            0ul
        );

    normals[index] =
        clipped_normal(
            seed,
            params.shock_clip
        );
}

// --------------------------------------------------------------------------
// Correlate standard normals with lower-triangular Cholesky matrix
// --------------------------------------------------------------------------
//
// correlated = L * z
//
// One thread per (scenario, asset).
//
// `cholesky` is assets x assets dense row-major. Entries above the diagonal
// may be zero.
// --------------------------------------------------------------------------

kernel void correlate_normals(
    device const float* normals        [[buffer(0)]],
    device const float* cholesky       [[buffer(1)]],
    constant MonteCarloParams& params  [[buffer(2)]],
    device float* correlated           [[buffer(3)]],
    uint gid                           [[thread_position_in_grid]]
) {
    const uint total =
        params.scenarios *
        params.assets;

    if (gid >= total ||
        params.assets == 0u ||
        params.row_stride <
            params.assets) {
        return;
    }

    const uint scenario =
        gid / params.assets;

    const uint asset =
        gid % params.assets;

    const uint scenario_base =
        scenario *
        params.row_stride;

    const uint matrix_base =
        asset *
        params.assets;

    float value = 0.0f;

    for (uint column = 0u;
         column <= asset;
         ++column) {
        const float l =
            cholesky[
                matrix_base +
                column
            ];

        const float z =
            normals[
                scenario_base +
                column
            ];

        if (!finite_value(l) ||
            !finite_value(z)) {
            correlated[
                scenario_base +
                asset
            ] = NAN;
            return;
        }

        value += l * z;
    }

    correlated[
        scenario_base +
        asset
    ] = value;
}

// --------------------------------------------------------------------------
// Fused correlated return generation
// --------------------------------------------------------------------------
//
// One thread per (scenario, asset), but each thread regenerates the independent
// normals required for its Cholesky row. This avoids an intermediate normal
// buffer at the cost of additional RNG work.
//
// simulated_return = mean * horizon_scale
//                  + correlated_shock * volatility_scale * sqrt(horizon)
//
// The Cholesky matrix should already represent the covariance scale expected
// by the host model.
// --------------------------------------------------------------------------

kernel void generate_correlated_returns(
    device const float* means          [[buffer(0)]],
    device const float* cholesky       [[buffer(1)]],
    constant MonteCarloParams& params  [[buffer(2)]],
    device float* returns              [[buffer(3)]],
    uint gid                           [[thread_position_in_grid]]
) {
    const uint total =
        params.scenarios *
        params.assets;

    if (gid >= total ||
        params.assets == 0u ||
        params.row_stride <
            params.assets ||
        !(params.horizon_scale >= 0.0f) ||
        !(params.volatility_scale >= 0.0f)) {
        return;
    }

    const uint scenario =
        gid / params.assets;

    const uint asset =
        gid % params.assets;

    const uint matrix_base =
        asset *
        params.assets;

    float shock = 0.0f;

    for (uint column = 0u;
         column <= asset;
         ++column) {
        const float l =
            cholesky[
                matrix_base +
                column
            ];

        if (!finite_value(l)) {
            returns[
                scenario *
                params.row_stride +
                asset
            ] = NAN;
            return;
        }

        const ulong seed =
            coordinate_seed(
                params.base_seed,
                params.stream_id,
                ulong(scenario),
                ulong(column),
                0ul
            );

        const float z =
            clipped_normal(
                seed,
                params.shock_clip
            );

        shock += l * z;
    }

    const float horizon =
        params.horizon_scale;

    const float diffusion_scale =
        params.volatility_scale *
        sqrt(max(horizon, 0.0f));

    const float simulated =
        means[asset] *
            horizon +
        shock *
            diffusion_scale;

    returns[
        scenario *
        params.row_stride +
        asset
    ] = simulated;
}

// --------------------------------------------------------------------------
// Portfolio scenario P&L
// --------------------------------------------------------------------------
//
// weights are signed normalized or exposure weights supplied by CPU.
//
// For arithmetic returns:
//     portfolio_return = sum(weight_i * r_i)
//
// For lognormal asset returns, the input buffer should contain LOG returns;
// this kernel converts each asset to simple return exp(r)-1 before weighting.
//
// One thread per scenario.
// --------------------------------------------------------------------------

kernel void portfolio_scenario_pnl(
    device const float* simulated_returns  [[buffer(0)]],
    device const float* weights            [[buffer(1)]],
    constant MonteCarloParams& mc          [[buffer(2)]],
    constant PortfolioParams& portfolio    [[buffer(3)]],
    device ScenarioSummary* output         [[buffer(4)]],
    uint scenario                          [[thread_position_in_grid]]
) {
    if (scenario >= mc.scenarios ||
        scenario >= portfolio.scenarios) {
        return;
    }

    ScenarioSummary result = {};
    result.status =
        MC_INVALID_INPUT;

    if (mc.assets == 0u ||
        portfolio.assets != mc.assets ||
        mc.row_stride < mc.assets ||
        portfolio.row_stride <
            portfolio.assets) {
        output[scenario] =
            result;
        return;
    }

    const uint base =
        scenario *
        mc.row_stride;

    float portfolio_return = 0.0f;
    float gross_shock = 0.0f;
    float max_abs_return = 0.0f;

    for (uint asset = 0u;
         asset < mc.assets;
         ++asset) {
        const float raw =
            simulated_returns[
                base + asset
            ];

        const float weight =
            weights[asset];

        if (!finite_value(raw) ||
            !finite_value(weight)) {
            output[scenario] =
                result;
            return;
        }

        float simple_return = raw;

        if (mc.return_model ==
            RETURN_MODEL_LOGNORMAL) {
            simple_return =
                exp(raw) - 1.0f;
        }

        portfolio_return +=
            weight *
            simple_return;

        gross_shock +=
            fabs(
                weight *
                simple_return
            );

        max_abs_return =
            max(
                max_abs_return,
                fabs(simple_return)
            );
    }

    result.portfolio_return =
        portfolio_return;

    result.pnl =
        portfolio_return *
        portfolio.gross_notional;

    result.gross_shock =
        gross_shock;

    result.max_abs_asset_return =
        max_abs_return;

    result.status =
        MC_OK;

    output[scenario] =
        result;
}

// --------------------------------------------------------------------------
// QQQ/SQQQ residual Monte Carlo
// --------------------------------------------------------------------------
//
// Revised stat-arb model:
//
//     r_QQQ = alpha + beta * r_SQQQ + epsilon
//
// This kernel treats SQQQ return and residual noise as stochastic inputs.
//
// The host supplies:
//   alpha
//   beta
//   residual mean/std
//   signed dependent/hedge weights
//
// It does NOT force beta to -1/3 or -3.
//
// Each scenario draws:
//   z_x   -> SQQQ shock
//   z_eps -> residual shock
//
// For this compact kernel, x_sigma is passed in `x_sigma[0]` and x_mean in
// `x_mean[0]`; they are separate buffers so the ABI can later support
// per-batch parameter vectors without changing the result structure.
// --------------------------------------------------------------------------

kernel void statarb_pair_monte_carlo(
    device const float* x_mean              [[buffer(0)]],
    device const float* x_sigma             [[buffer(1)]],
    constant PairMonteCarloParams& params   [[buffer(2)]],
    device PairScenarioResult* output       [[buffer(3)]],
    uint scenario                           [[thread_position_in_grid]]
) {
    if (scenario >= params.scenarios) {
        return;
    }

    PairScenarioResult result = {};
    result.status =
        MC_INVALID_INPUT;

    const float mean_x =
        x_mean[0];

    const float sigma_x =
        x_sigma[0];

    if (!finite_value(mean_x) ||
        !finite_value(sigma_x) ||
        !(sigma_x >= 0.0f) ||
        !finite_value(params.alpha) ||
        !finite_value(params.beta) ||
        !finite_value(params.residual_mean) ||
        !(params.residual_std >= 0.0f)) {
        output[scenario] =
            result;
        return;
    }

    const ulong x_seed =
        coordinate_seed(
            params.base_seed,
            params.stream_id,
            ulong(scenario),
            0ul,
            0ul
        );

    const ulong residual_seed =
        coordinate_seed(
            params.base_seed,
            params.stream_id,
            ulong(scenario),
            1ul,
            0ul
        );

    const float z_x =
        clipped_normal(
            x_seed,
            params.shock_clip
        );

    const float z_residual =
        clipped_normal(
            residual_seed,
            params.shock_clip
        );

    const float sqqq_return =
        mean_x +
        sigma_x *
        z_x;

    const float residual =
        params.residual_mean +
        params.residual_std *
        z_residual;

    const float qqq_return =
        params.alpha +
        params.beta *
        sqqq_return +
        residual;

    if (!finite_value(qqq_return) ||
        !finite_value(sqqq_return) ||
        !finite_value(residual)) {
        result.status =
            MC_NUMERICAL_FAILURE;
        output[scenario] =
            result;
        return;
    }

    result.qqq_return =
        qqq_return;

    result.sqqq_return =
        sqqq_return;

    result.residual =
        residual;

    if (params.residual_std >
        kMcEpsilon) {
        result.residual_z =
            (
                residual -
                params.residual_mean
            ) /
            params.residual_std;
    }

    result.pair_return =
        params.dependent_weight *
            qqq_return +
        params.hedge_weight *
            sqqq_return;

    result.pnl =
        result.pair_return *
        params.gross_notional;

    result.gross_abs_return =
        fabs(
            params.dependent_weight *
            qqq_return
        ) +
        fabs(
            params.hedge_weight *
            sqqq_return
        );

    result.status =
        MC_OK;

    output[scenario] =
        result;
}

// --------------------------------------------------------------------------
// Multi-step portfolio path simulation
// --------------------------------------------------------------------------
//
// One thread per scenario.
//
// Each step regenerates correlated Gaussian shocks from the Cholesky matrix.
// The kernel tracks:
//   - compounded terminal portfolio return,
//   - terminal P&L,
//   - maximum drawdown,
//   - realized portfolio variance proxy.
//
// weights are signed portfolio weights.
//
// For arithmetic model:
//     step simple return = mu*dt + shock*sqrt(dt)
//
// For lognormal model:
//     asset simple return = exp(log_return)-1
//
// Portfolio wealth compounds from 1.0.
// --------------------------------------------------------------------------

kernel void simulate_portfolio_paths(
    device const float* means          [[buffer(0)]],
    device const float* cholesky       [[buffer(1)]],
    device const float* weights        [[buffer(2)]],
    constant PathParams& params        [[buffer(3)]],
    device PathResult* output          [[buffer(4)]],
    uint scenario                      [[thread_position_in_grid]]
) {
    if (scenario >= params.scenarios) {
        return;
    }

    PathResult result = {};
    result.status =
        MC_INVALID_INPUT;

    if (params.steps == 0u ||
        params.assets == 0u ||
        !(params.dt > 0.0f) ||
        !(params.volatility_scale >= 0.0f)) {
        output[scenario] =
            result;
        return;
    }

    float wealth = 1.0f;
    float peak = 1.0f;
    float max_drawdown = 0.0f;

    float sum_step_return = 0.0f;
    float sum_step_return_sq = 0.0f;

    for (uint step = 0u;
         step < params.steps;
         ++step) {
        float portfolio_step_return =
            0.0f;

        for (uint asset = 0u;
             asset < params.assets;
             ++asset) {
            float correlated_shock =
                0.0f;

            const uint matrix_base =
                asset *
                params.assets;

            for (uint column = 0u;
                 column <= asset;
                 ++column) {
                const float l =
                    cholesky[
                        matrix_base +
                        column
                    ];

                if (!finite_value(l)) {
                    result.status =
                        MC_NUMERICAL_FAILURE;
                    output[scenario] =
                        result;
                    return;
                }

                const ulong seed =
                    coordinate_seed(
                        params.base_seed,
                        params.stream_id,
                        ulong(scenario),
                        ulong(column),
                        ulong(step)
                    );

                const float z =
                    clipped_normal(
                        seed,
                        params.shock_clip
                    );

                correlated_shock +=
                    l * z;
            }

            float modeled_return =
                means[asset] *
                    params.dt +
                correlated_shock *
                    params.volatility_scale *
                    sqrt(params.dt);

            if (params.return_model ==
                RETURN_MODEL_LOGNORMAL) {
                modeled_return =
                    exp(modeled_return) -
                    1.0f;
            }

            if (!finite_value(
                    modeled_return
                )) {
                result.status =
                    MC_NUMERICAL_FAILURE;
                output[scenario] =
                    result;
                return;
            }

            portfolio_step_return +=
                weights[asset] *
                modeled_return;
        }

        // Prevent impossible negative wealth from propagating NaNs in an
        // arithmetic-return research path. The host should separately inspect
        // such extreme scenarios.
        const float wealth_multiplier =
            max(
                1.0f +
                portfolio_step_return,
                0.0f
            );

        wealth *=
            wealth_multiplier;

        peak =
            max(
                peak,
                wealth
            );

        if (peak > kMcEpsilon) {
            const float drawdown =
                (
                    peak -
                    wealth
                ) /
                peak;

            max_drawdown =
                max(
                    max_drawdown,
                    drawdown
                );
        }

        sum_step_return +=
            portfolio_step_return;

        sum_step_return_sq +=
            portfolio_step_return *
            portfolio_step_return;
    }

    const float steps =
        float(params.steps);

    const float mean_step =
        sum_step_return /
        steps;

    const float variance =
        max(
            (
                sum_step_return_sq /
                steps
            ) -
            mean_step *
                mean_step,
            0.0f
        );

    result.terminal_return =
        wealth - 1.0f;

    // Host can scale this return to actual capital. Keeping path output
    // normalized avoids mixing account policy into the GPU kernel.
    result.terminal_pnl =
        result.terminal_return;

    result.max_drawdown =
        max_drawdown;

    result.realized_variance =
        variance;

    result.status =
        MC_OK;

    output[scenario] =
        result;
}

// --------------------------------------------------------------------------
// P&L -> loss transform
// --------------------------------------------------------------------------
//
// Risk statistics often operate on positive loss:
//     loss = max(-pnl, 0)
//
// One thread per scenario.
// --------------------------------------------------------------------------

kernel void pnl_to_loss(
    device const float* pnl       [[buffer(0)]],
    device float* loss            [[buffer(1)]],
    uint gid                      [[thread_position_in_grid]]
) {
    const float value =
        pnl[gid];

    loss[gid] =
        finite_value(value)
            ? max(-value, 0.0f)
            : NAN;
}

// --------------------------------------------------------------------------
// ScenarioSummary -> P&L extraction
// --------------------------------------------------------------------------

kernel void extract_scenario_pnl(
    device const ScenarioSummary* scenarios [[buffer(0)]],
    device float* pnl                       [[buffer(1)]],
    uint gid                                [[thread_position_in_grid]]
) {
    const auto scenario =
        scenarios[gid];

    pnl[gid] =
        scenario.status == MC_OK
            ? scenario.pnl
            : NAN;
}

// --------------------------------------------------------------------------
// PairScenarioResult -> P&L extraction
// --------------------------------------------------------------------------

kernel void extract_pair_pnl(
    device const PairScenarioResult* scenarios [[buffer(0)]],
    device float* pnl                          [[buffer(1)]],
    uint gid                                   [[thread_position_in_grid]]
) {
    const auto scenario =
        scenarios[gid];

    pnl[gid] =
        scenario.status == MC_OK
            ? scenario.pnl
            : NAN;
}

// --------------------------------------------------------------------------
// Tail-moment partial reduction
// --------------------------------------------------------------------------
//
// One thread handles one contiguous chunk.
//
// This intentionally does NOT calculate VaR/CVaR directly. Exact quantiles
// require sorting/selection or a host-side reduction. The host can:
//   1. extract P&L,
//   2. sort/select the desired quantile,
//   3. dispatch a threshold-based CVaR reduction if desired.
//
// The partials here support general diagnostics and CPU final aggregation.
// --------------------------------------------------------------------------

kernel void tail_moment_partials(
    device const float* pnl               [[buffer(0)]],
    constant uint& sample_count           [[buffer(1)]],
    constant uint& samples_per_thread     [[buffer(2)]],
    device TailMomentPartial* output      [[buffer(3)]],
    uint gid                              [[thread_position_in_grid]]
) {
    TailMomentPartial result = {};

    result.worst_loss = 0.0f;
    result.best_pnl =
        -INFINITY;

    const uint begin =
        gid *
        samples_per_thread;

    if (begin >= sample_count) {
        output[gid] =
            result;
        return;
    }

    const uint end =
        min(
            begin +
                samples_per_thread,
            sample_count
        );

    for (uint index = begin;
         index < end;
         ++index) {
        const float value =
            pnl[index];

        if (!finite_value(value)) {
            continue;
        }

        const float loss =
            max(
                -value,
                0.0f
            );

        result.loss_sum +=
            loss;

        result.loss_sq_sum +=
            loss * loss;

        result.best_pnl =
            max(
                result.best_pnl,
                value
            );

        result.worst_loss =
            max(
                result.worst_loss,
                loss
            );

        if (value < 0.0f) {
            const float downside =
                -value;

            result.downside_sum +=
                downside;

            result.downside_sq_sum +=
                downside *
                downside;

            ++result.loss_count;
        }

        ++result.sample_count;
    }

    output[gid] =
        result;
}

// --------------------------------------------------------------------------
// Threshold tail-loss partials
// --------------------------------------------------------------------------
//
// After the CPU/GPU selection stage determines a VaR loss threshold, this
// kernel accumulates losses >= threshold. CPU can sum partials to obtain:
//
//     CVaR = tail_loss_sum / tail_count
//
// This separates quantile selection from tail expectation and keeps policy
// confidence levels outside the Metal execution boundary.
// --------------------------------------------------------------------------

struct TailThresholdPartial {
    float tail_loss_sum;
    float tail_loss_sq_sum;
    float worst_loss;
    float reserved0;

    uint tail_count;
    uint sample_count;
    uint reserved1;
    uint reserved2;
};

kernel void threshold_tail_partials(
    device const float* pnl                   [[buffer(0)]],
    constant uint& sample_count               [[buffer(1)]],
    constant uint& samples_per_thread         [[buffer(2)]],
    constant float& loss_threshold            [[buffer(3)]],
    device TailThresholdPartial* output       [[buffer(4)]],
    uint gid                                  [[thread_position_in_grid]]
) {
    TailThresholdPartial result = {};

    const uint begin =
        gid *
        samples_per_thread;

    if (begin >= sample_count) {
        output[gid] =
            result;
        return;
    }

    const uint end =
        min(
            begin +
                samples_per_thread,
            sample_count
        );

    for (uint index = begin;
         index < end;
         ++index) {
        const float value =
            pnl[index];

        if (!finite_value(value)) {
            continue;
        }

        const float loss =
            max(
                -value,
                0.0f
            );

        result.worst_loss =
            max(
                result.worst_loss,
                loss
            );

        if (loss >=
            loss_threshold) {
            result.tail_loss_sum +=
                loss;

            result.tail_loss_sq_sum +=
                loss * loss;

            ++result.tail_count;
        }

        ++result.sample_count;
    }

    output[gid] =
        result;
}

// --------------------------------------------------------------------------
// Stress-shock application
// --------------------------------------------------------------------------
//
// Applies deterministic host-defined stress vectors to a base return vector.
//
// Layout:
//   stress_shocks[stress_case * stress_stride + asset]
//   output[stress_case * output_stride + asset]
//
// This is useful alongside stochastic Monte Carlo for explicit scenarios such
// as equity selloffs, volatility spikes, or correlation-break assumptions.
// --------------------------------------------------------------------------

struct StressParams {
    uint stress_cases;
    uint assets;
    uint stress_stride;
    uint output_stride;

    float scale;
    float reserved0;
    float reserved1;
    float reserved2;
};

kernel void apply_stress_shocks(
    device const float* base_returns      [[buffer(0)]],
    device const float* stress_shocks     [[buffer(1)]],
    constant StressParams& params         [[buffer(2)]],
    device float* output                  [[buffer(3)]],
    uint gid                              [[thread_position_in_grid]]
) {
    const uint total =
        params.stress_cases *
        params.assets;

    if (gid >= total ||
        params.assets == 0u ||
        params.stress_stride <
            params.assets ||
        params.output_stride <
            params.assets) {
        return;
    }

    const uint stress_case =
        gid / params.assets;

    const uint asset =
        gid % params.assets;

    const float base =
        base_returns[asset];

    const float shock =
        stress_shocks[
            stress_case *
            params.stress_stride +
            asset
        ];

    output[
        stress_case *
        params.output_stride +
        asset
    ] =
        base +
        params.scale *
        shock;
}

} // namespace gme
