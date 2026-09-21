#include <metal_stdlib>
using namespace metal;

// ============================================================================
// reductions.metal
// ============================================================================
//
// Shared numerical reduction backend for the Apple-Silicon market engine.
//
// Intended consumers:
//   covariance.metal
//   imbalance.metal
//   monte-carlo.metal
//   CPU FeatureEngine / RiskEngine / StatArb analytics
//
// Design:
//   - GPU produces deterministic chunk partials.
//   - CPU or a second GPU pass combines partials.
//   - No risk-policy decision is made here.
//   - NaN/Inf handling is explicit.
//   - No atomics are required for the core reduction path.
//
// This is intentionally a "partials" library. It avoids coupling numerical
// aggregation to a specific confidence level, trading policy, or execution
// decision.
//
// Layout:
//   values[row * row_stride + column]
//
// Host ABI mirrors should use:
//   uint32_t -> uint
//   uint64_t -> ulong
//   int64_t  -> long
//   float    -> float
//
// Validate sizeof/offsetof on the Objective-C++ host before dispatch.
// ============================================================================

namespace gme {

// --------------------------------------------------------------------------
// Constants / status
// --------------------------------------------------------------------------

constant float kReductionEpsilon = 1.0e-12f;

enum ReductionStatus : uint {
    REDUCTION_OK            = 0u,
    REDUCTION_EMPTY         = 1u,
    REDUCTION_INVALID_INPUT = 2u,
    REDUCTION_NONFINITE     = 3u
};

// --------------------------------------------------------------------------
// ABI structures
// --------------------------------------------------------------------------

struct ReductionParams {
    uint sample_count;
    uint samples_per_thread;
    uint row_stride;
    uint column;

    uint ignore_nonfinite;
    uint reserved0;
    uint reserved1;
    uint reserved2;
};

struct MatrixReductionParams {
    uint rows;
    uint columns;
    uint row_stride;
    uint rows_per_thread;

    uint ignore_nonfinite;
    uint reserved0;
    uint reserved1;
    uint reserved2;
};

struct PairReductionParams {
    uint sample_count;
    uint samples_per_thread;
    uint ignore_nonfinite;
    uint reserved0;
};

struct SumPartial {
    float sum;
    float sum_abs;
    float sum_sq;
    float reserved0;

    uint count;
    uint nonfinite_count;
    uint status;
    uint reserved1;
};

struct MomentPartial {
    float sum;
    float sum_sq;
    float sum_cube;
    float sum_fourth;

    float minimum;
    float maximum;
    float reserved0;
    float reserved1;

    uint count;
    uint nonfinite_count;
    uint status;
    uint reserved2;
};

struct MinMaxPartial {
    float minimum;
    float maximum;
    float min_abs;
    float max_abs;

    uint count;
    uint nonfinite_count;
    uint status;
    uint reserved0;
};

struct DotPartial {
    float dot;
    float x_sum;
    float y_sum;
    float x_sq_sum;

    float y_sq_sum;
    float abs_product_sum;
    float reserved0;
    float reserved1;

    uint count;
    uint nonfinite_count;
    uint status;
    uint reserved2;
};

struct CovarianceSufficientPartial {
    float sum_x;
    float sum_y;
    float sum_xx;
    float sum_yy;

    float sum_xy;
    float reserved0;
    float reserved1;
    float reserved2;

    uint count;
    uint nonfinite_count;
    uint status;
    uint reserved3;
};

struct PnlPartial {
    float pnl_sum;
    float pnl_sq_sum;
    float positive_sum;
    float negative_sum;

    float worst_pnl;
    float best_pnl;
    float max_loss;
    float reserved0;

    uint positive_count;
    uint negative_count;
    uint zero_count;
    uint nonfinite_count;

    uint count;
    uint status;
    uint reserved1;
    uint reserved2;
};

struct ThresholdTailPartial {
    float tail_loss_sum;
    float tail_loss_sq_sum;
    float worst_loss;
    float threshold;

    uint tail_count;
    uint count;
    uint nonfinite_count;
    uint status;
};

struct ValidationCountPartial {
    uint valid_count;
    uint invalid_count;
    uint stale_count;
    uint empty_count;

    uint crossed_count;
    uint one_sided_count;
    uint nonfinite_count;
    uint total_count;
};

struct HistogramParams {
    uint sample_count;
    uint bins;
    uint samples_per_thread;
    uint ignore_nonfinite;

    float minimum;
    float maximum;
    float reserved0;
    float reserved1;
};

// --------------------------------------------------------------------------
// Helpers
// --------------------------------------------------------------------------

inline bool finite_value(const float value) {
    return isfinite(value);
}

inline uint chunk_begin(
    const uint gid,
    const uint samples_per_thread
) {
    return gid * samples_per_thread;
}

inline uint chunk_end(
    const uint begin,
    const uint samples_per_thread,
    const uint sample_count
) {
    return min(
        begin + samples_per_thread,
        sample_count
    );
}

inline float positive_loss(
    const float pnl
) {
    return max(-pnl, 0.0f);
}

// --------------------------------------------------------------------------
// Scalar sum / absolute sum / square sum partials
// --------------------------------------------------------------------------

kernel void reduce_sum_partials(
    device const float* values          [[buffer(0)]],
    constant ReductionParams& params    [[buffer(1)]],
    device SumPartial* output           [[buffer(2)]],
    uint gid                            [[thread_position_in_grid]]
) {
    SumPartial result = {};
    result.status = REDUCTION_EMPTY;

    if (params.samples_per_thread == 0u) {
        result.status = REDUCTION_INVALID_INPUT;
        output[gid] = result;
        return;
    }

    const uint begin =
        chunk_begin(
            gid,
            params.samples_per_thread
        );

    if (begin >= params.sample_count) {
        output[gid] = result;
        return;
    }

    const uint end =
        chunk_end(
            begin,
            params.samples_per_thread,
            params.sample_count
        );

    for (uint i = begin; i < end; ++i) {
        const float value = values[i];

        if (!finite_value(value)) {
            ++result.nonfinite_count;

            if (params.ignore_nonfinite == 0u) {
                result.status = REDUCTION_NONFINITE;
                output[gid] = result;
                return;
            }

            continue;
        }

        result.sum += value;
        result.sum_abs += fabs(value);
        result.sum_sq += value * value;
        ++result.count;
    }

    result.status =
        result.count > 0u
            ? REDUCTION_OK
            : REDUCTION_EMPTY;

    output[gid] = result;
}

// --------------------------------------------------------------------------
// Higher raw moments + min/max partials
// --------------------------------------------------------------------------
//
// CPU can combine raw sums then derive:
//   mean
//   variance
//   skewness
//   kurtosis
//
// For very large-magnitude datasets, CPU double-precision finalization is
// preferred.
// --------------------------------------------------------------------------

kernel void reduce_moment_partials(
    device const float* values          [[buffer(0)]],
    constant ReductionParams& params    [[buffer(1)]],
    device MomentPartial* output        [[buffer(2)]],
    uint gid                            [[thread_position_in_grid]]
) {
    MomentPartial result = {};
    result.minimum = INFINITY;
    result.maximum = -INFINITY;
    result.status = REDUCTION_EMPTY;

    if (params.samples_per_thread == 0u) {
        result.status = REDUCTION_INVALID_INPUT;
        output[gid] = result;
        return;
    }

    const uint begin =
        chunk_begin(
            gid,
            params.samples_per_thread
        );

    if (begin >= params.sample_count) {
        output[gid] = result;
        return;
    }

    const uint end =
        chunk_end(
            begin,
            params.samples_per_thread,
            params.sample_count
        );

    for (uint i = begin; i < end; ++i) {
        const float value = values[i];

        if (!finite_value(value)) {
            ++result.nonfinite_count;

            if (params.ignore_nonfinite == 0u) {
                result.status = REDUCTION_NONFINITE;
                output[gid] = result;
                return;
            }

            continue;
        }

        const float square = value * value;

        result.sum += value;
        result.sum_sq += square;
        result.sum_cube += square * value;
        result.sum_fourth += square * square;

        result.minimum =
            min(result.minimum, value);

        result.maximum =
            max(result.maximum, value);

        ++result.count;
    }

    result.status =
        result.count > 0u
            ? REDUCTION_OK
            : REDUCTION_EMPTY;

    output[gid] = result;
}

// --------------------------------------------------------------------------
// Min / max partials
// --------------------------------------------------------------------------

kernel void reduce_minmax_partials(
    device const float* values          [[buffer(0)]],
    constant ReductionParams& params    [[buffer(1)]],
    device MinMaxPartial* output        [[buffer(2)]],
    uint gid                            [[thread_position_in_grid]]
) {
    MinMaxPartial result = {};

    result.minimum = INFINITY;
    result.maximum = -INFINITY;
    result.min_abs = INFINITY;
    result.max_abs = 0.0f;
    result.status = REDUCTION_EMPTY;

    if (params.samples_per_thread == 0u) {
        result.status = REDUCTION_INVALID_INPUT;
        output[gid] = result;
        return;
    }

    const uint begin =
        chunk_begin(
            gid,
            params.samples_per_thread
        );

    if (begin >= params.sample_count) {
        output[gid] = result;
        return;
    }

    const uint end =
        chunk_end(
            begin,
            params.samples_per_thread,
            params.sample_count
        );

    for (uint i = begin; i < end; ++i) {
        const float value = values[i];

        if (!finite_value(value)) {
            ++result.nonfinite_count;

            if (params.ignore_nonfinite == 0u) {
                result.status = REDUCTION_NONFINITE;
                output[gid] = result;
                return;
            }

            continue;
        }

        const float absolute =
            fabs(value);

        result.minimum =
            min(result.minimum, value);

        result.maximum =
            max(result.maximum, value);

        result.min_abs =
            min(result.min_abs, absolute);

        result.max_abs =
            max(result.max_abs, absolute);

        ++result.count;
    }

    result.status =
        result.count > 0u
            ? REDUCTION_OK
            : REDUCTION_EMPTY;

    output[gid] = result;
}

// --------------------------------------------------------------------------
// Dot-product sufficient statistics
// --------------------------------------------------------------------------
//
// Produces enough partial information for:
//   dot(x,y)
//   cosine similarity
//   means
//   variances
//   covariance
//   correlation
// --------------------------------------------------------------------------

kernel void reduce_dot_partials(
    device const float* x               [[buffer(0)]],
    device const float* y               [[buffer(1)]],
    constant PairReductionParams& params [[buffer(2)]],
    device DotPartial* output           [[buffer(3)]],
    uint gid                            [[thread_position_in_grid]]
) {
    DotPartial result = {};
    result.status = REDUCTION_EMPTY;

    if (params.samples_per_thread == 0u) {
        result.status = REDUCTION_INVALID_INPUT;
        output[gid] = result;
        return;
    }

    const uint begin =
        chunk_begin(
            gid,
            params.samples_per_thread
        );

    if (begin >= params.sample_count) {
        output[gid] = result;
        return;
    }

    const uint end =
        chunk_end(
            begin,
            params.samples_per_thread,
            params.sample_count
        );

    for (uint i = begin; i < end; ++i) {
        const float xv = x[i];
        const float yv = y[i];

        if (!finite_value(xv) ||
            !finite_value(yv)) {
            ++result.nonfinite_count;

            if (params.ignore_nonfinite == 0u) {
                result.status = REDUCTION_NONFINITE;
                output[gid] = result;
                return;
            }

            continue;
        }

        const float product =
            xv * yv;

        result.dot += product;
        result.x_sum += xv;
        result.y_sum += yv;
        result.x_sq_sum += xv * xv;
        result.y_sq_sum += yv * yv;
        result.abs_product_sum +=
            fabs(product);

        ++result.count;
    }

    result.status =
        result.count > 0u
            ? REDUCTION_OK
            : REDUCTION_EMPTY;

    output[gid] = result;
}

// --------------------------------------------------------------------------
// Covariance sufficient-statistic partials
// --------------------------------------------------------------------------
//
// CPU finalization:
//
//   mean_x = sum_x / n
//   mean_y = sum_y / n
//
//   cov_num = sum_xy - sum_x*sum_y/n
//   var_x   = sum_xx - sum_x*sum_x/n
//   var_y   = sum_yy - sum_y*sum_y/n
//
// sample covariance = cov_num / (n-1)
//
// This is useful for validating covariance.metal and rolling pair models.
// --------------------------------------------------------------------------

kernel void reduce_covariance_sufficient_partials(
    device const float* x                [[buffer(0)]],
    device const float* y                [[buffer(1)]],
    constant PairReductionParams& params [[buffer(2)]],
    device CovarianceSufficientPartial* output
                                             [[buffer(3)]],
    uint gid                              [[thread_position_in_grid]]
) {
    CovarianceSufficientPartial result = {};
    result.status = REDUCTION_EMPTY;

    if (params.samples_per_thread == 0u) {
        result.status = REDUCTION_INVALID_INPUT;
        output[gid] = result;
        return;
    }

    const uint begin =
        chunk_begin(
            gid,
            params.samples_per_thread
        );

    if (begin >= params.sample_count) {
        output[gid] = result;
        return;
    }

    const uint end =
        chunk_end(
            begin,
            params.samples_per_thread,
            params.sample_count
        );

    for (uint i = begin; i < end; ++i) {
        const float xv = x[i];
        const float yv = y[i];

        if (!finite_value(xv) ||
            !finite_value(yv)) {
            ++result.nonfinite_count;

            if (params.ignore_nonfinite == 0u) {
                result.status = REDUCTION_NONFINITE;
                output[gid] = result;
                return;
            }

            continue;
        }

        result.sum_x += xv;
        result.sum_y += yv;
        result.sum_xx += xv * xv;
        result.sum_yy += yv * yv;
        result.sum_xy += xv * yv;

        ++result.count;
    }

    result.status =
        result.count > 0u
            ? REDUCTION_OK
            : REDUCTION_EMPTY;

    output[gid] = result;
}

// --------------------------------------------------------------------------
// Matrix column partials
// --------------------------------------------------------------------------
//
// One thread handles one (chunk, column).
//
// gid mapping:
//   column = gid % columns
//   chunk  = gid / columns
//
// Useful for batched feature means/variances without transposing SoA/AoS
// buffers.
// --------------------------------------------------------------------------

kernel void reduce_matrix_column_partials(
    device const float* values              [[buffer(0)]],
    constant MatrixReductionParams& params  [[buffer(1)]],
    device SumPartial* output               [[buffer(2)]],
    uint gid                                [[thread_position_in_grid]]
) {
    if (params.columns == 0u ||
        params.rows_per_thread == 0u ||
        params.row_stride < params.columns) {
        return;
    }

    const uint column =
        gid % params.columns;

    const uint chunk =
        gid / params.columns;

    const uint begin =
        chunk *
        params.rows_per_thread;

    SumPartial result = {};
    result.status = REDUCTION_EMPTY;

    if (begin >= params.rows) {
        output[gid] = result;
        return;
    }

    const uint end =
        min(
            begin +
                params.rows_per_thread,
            params.rows
        );

    for (uint row = begin;
         row < end;
         ++row) {
        const float value =
            values[
                row *
                params.row_stride +
                column
            ];

        if (!finite_value(value)) {
            ++result.nonfinite_count;

            if (params.ignore_nonfinite == 0u) {
                result.status = REDUCTION_NONFINITE;
                output[gid] = result;
                return;
            }

            continue;
        }

        result.sum += value;
        result.sum_abs += fabs(value);
        result.sum_sq += value * value;
        ++result.count;
    }

    result.status =
        result.count > 0u
            ? REDUCTION_OK
            : REDUCTION_EMPTY;

    output[gid] = result;
}

// --------------------------------------------------------------------------
// P&L distribution partials
// --------------------------------------------------------------------------
//
// Shared by Monte Carlo portfolio and pair simulations.
// --------------------------------------------------------------------------

kernel void reduce_pnl_partials(
    device const float* pnl             [[buffer(0)]],
    constant ReductionParams& params    [[buffer(1)]],
    device PnlPartial* output           [[buffer(2)]],
    uint gid                            [[thread_position_in_grid]]
) {
    PnlPartial result = {};

    result.worst_pnl = INFINITY;
    result.best_pnl = -INFINITY;
    result.status = REDUCTION_EMPTY;

    if (params.samples_per_thread == 0u) {
        result.status = REDUCTION_INVALID_INPUT;
        output[gid] = result;
        return;
    }

    const uint begin =
        chunk_begin(
            gid,
            params.samples_per_thread
        );

    if (begin >= params.sample_count) {
        output[gid] = result;
        return;
    }

    const uint end =
        chunk_end(
            begin,
            params.samples_per_thread,
            params.sample_count
        );

    for (uint i = begin; i < end; ++i) {
        const float value = pnl[i];

        if (!finite_value(value)) {
            ++result.nonfinite_count;

            if (params.ignore_nonfinite == 0u) {
                result.status = REDUCTION_NONFINITE;
                output[gid] = result;
                return;
            }

            continue;
        }

        result.pnl_sum += value;
        result.pnl_sq_sum +=
            value * value;

        result.worst_pnl =
            min(
                result.worst_pnl,
                value
            );

        result.best_pnl =
            max(
                result.best_pnl,
                value
            );

        result.max_loss =
            max(
                result.max_loss,
                positive_loss(value)
            );

        if (value > 0.0f) {
            result.positive_sum += value;
            ++result.positive_count;
        } else if (value < 0.0f) {
            result.negative_sum += value;
            ++result.negative_count;
        } else {
            ++result.zero_count;
        }

        ++result.count;
    }

    result.status =
        result.count > 0u
            ? REDUCTION_OK
            : REDUCTION_EMPTY;

    output[gid] = result;
}

// --------------------------------------------------------------------------
// Threshold tail-loss partials
// --------------------------------------------------------------------------
//
// The host chooses the threshold after quantile selection.
//
// loss = max(-pnl, 0)
//
// Tail membership:
//     loss >= threshold
//
// CPU combines partial sums to compute CVaR / expected shortfall.
// --------------------------------------------------------------------------

kernel void reduce_threshold_tail_partials(
    device const float* pnl               [[buffer(0)]],
    constant ReductionParams& params      [[buffer(1)]],
    constant float& loss_threshold        [[buffer(2)]],
    device ThresholdTailPartial* output   [[buffer(3)]],
    uint gid                              [[thread_position_in_grid]]
) {
    ThresholdTailPartial result = {};
    result.threshold = loss_threshold;
    result.status = REDUCTION_EMPTY;

    if (params.samples_per_thread == 0u ||
        !finite_value(loss_threshold) ||
        loss_threshold < 0.0f) {
        result.status = REDUCTION_INVALID_INPUT;
        output[gid] = result;
        return;
    }

    const uint begin =
        chunk_begin(
            gid,
            params.samples_per_thread
        );

    if (begin >= params.sample_count) {
        output[gid] = result;
        return;
    }

    const uint end =
        chunk_end(
            begin,
            params.samples_per_thread,
            params.sample_count
        );

    for (uint i = begin; i < end; ++i) {
        const float value = pnl[i];

        if (!finite_value(value)) {
            ++result.nonfinite_count;

            if (params.ignore_nonfinite == 0u) {
                result.status = REDUCTION_NONFINITE;
                output[gid] = result;
                return;
            }

            continue;
        }

        const float loss =
            positive_loss(value);

        result.worst_loss =
            max(
                result.worst_loss,
                loss
            );

        if (loss >= loss_threshold) {
            result.tail_loss_sum += loss;
            result.tail_loss_sq_sum +=
                loss * loss;

            ++result.tail_count;
        }

        ++result.count;
    }

    result.status =
        result.count > 0u
            ? REDUCTION_OK
            : REDUCTION_EMPTY;

    output[gid] = result;
}

// --------------------------------------------------------------------------
// Boolean / status validation counters
// --------------------------------------------------------------------------
//
// Generic status convention expected from host adapter:
//
//   0 = valid
//   1 = invalid
//   2 = stale
//   3 = empty
//   4 = crossed/locked
//   5 = one-sided
//   6 = non-finite/numerical failure
//
// This does not require the source kernels to share enum values; the host can
// normalize their statuses before dispatch.
// --------------------------------------------------------------------------

kernel void reduce_validation_counts(
    device const uint* normalized_status    [[buffer(0)]],
    constant uint& sample_count             [[buffer(1)]],
    constant uint& samples_per_thread       [[buffer(2)]],
    device ValidationCountPartial* output   [[buffer(3)]],
    uint gid                                [[thread_position_in_grid]]
) {
    ValidationCountPartial result = {};

    if (samples_per_thread == 0u) {
        output[gid] = result;
        return;
    }

    const uint begin =
        chunk_begin(
            gid,
            samples_per_thread
        );

    if (begin >= sample_count) {
        output[gid] = result;
        return;
    }

    const uint end =
        chunk_end(
            begin,
            samples_per_thread,
            sample_count
        );

    for (uint i = begin; i < end; ++i) {
        switch (normalized_status[i]) {
            case 0u:
                ++result.valid_count;
                break;

            case 1u:
                ++result.invalid_count;
                break;

            case 2u:
                ++result.stale_count;
                break;

            case 3u:
                ++result.empty_count;
                break;

            case 4u:
                ++result.crossed_count;
                break;

            case 5u:
                ++result.one_sided_count;
                break;

            case 6u:
                ++result.nonfinite_count;
                break;

            default:
                ++result.invalid_count;
                break;
        }

        ++result.total_count;
    }

    output[gid] = result;
}

// --------------------------------------------------------------------------
// Histogram partials
// --------------------------------------------------------------------------
//
// Output layout:
//     partial_histograms[gid * bins + bin]
//
// Each thread owns its histogram slice, so no atomics are needed.
// CPU or another pass combines slices.
//
// Values == maximum are placed in the final bin.
// Values outside [minimum, maximum] are clamped to edge bins.
// --------------------------------------------------------------------------

kernel void reduce_histogram_partials(
    device const float* values          [[buffer(0)]],
    constant HistogramParams& params    [[buffer(1)]],
    device uint* partial_histograms     [[buffer(2)]],
    uint gid                            [[thread_position_in_grid]]
) {
    if (params.bins == 0u ||
        params.samples_per_thread == 0u ||
        !(params.maximum >
          params.minimum)) {
        return;
    }

    const uint histogram_base =
        gid *
        params.bins;

    for (uint bin = 0u;
         bin < params.bins;
         ++bin) {
        partial_histograms[
            histogram_base + bin
        ] = 0u;
    }

    const uint begin =
        chunk_begin(
            gid,
            params.samples_per_thread
        );

    if (begin >= params.sample_count) {
        return;
    }

    const uint end =
        chunk_end(
            begin,
            params.samples_per_thread,
            params.sample_count
        );

    const float range =
        params.maximum -
        params.minimum;

    for (uint i = begin; i < end; ++i) {
        const float value = values[i];

        if (!finite_value(value)) {
            if (params.ignore_nonfinite != 0u) {
                continue;
            }

            // No special NaN bin is reserved in this ABI.
            continue;
        }

        float normalized =
            (
                value -
                params.minimum
            ) /
            range;

        normalized =
            clamp(
                normalized,
                0.0f,
                1.0f
            );

        uint bin =
            uint(
                floor(
                    normalized *
                    float(params.bins)
                )
            );

        if (bin >= params.bins) {
            bin =
                params.bins - 1u;
        }

        ++partial_histograms[
            histogram_base + bin
        ];
    }
}

// --------------------------------------------------------------------------
// Combine SumPartial records
// --------------------------------------------------------------------------
//
// Enables hierarchical GPU reductions:
//
// raw values
//    -> reduce_sum_partials
//    -> combine_sum_partials
//    -> repeat until small enough for CPU finalization.
//
// One output thread combines `partials_per_thread` source records.
// --------------------------------------------------------------------------

kernel void combine_sum_partials(
    device const SumPartial* input          [[buffer(0)]],
    constant uint& partial_count            [[buffer(1)]],
    constant uint& partials_per_thread      [[buffer(2)]],
    device SumPartial* output               [[buffer(3)]],
    uint gid                                [[thread_position_in_grid]]
) {
    SumPartial result = {};
    result.status = REDUCTION_EMPTY;

    if (partials_per_thread == 0u) {
        result.status = REDUCTION_INVALID_INPUT;
        output[gid] = result;
        return;
    }

    const uint begin =
        gid *
        partials_per_thread;

    if (begin >= partial_count) {
        output[gid] = result;
        return;
    }

    const uint end =
        min(
            begin +
                partials_per_thread,
            partial_count
        );

    for (uint i = begin; i < end; ++i) {
        const auto partial =
            input[i];

        result.sum +=
            partial.sum;

        result.sum_abs +=
            partial.sum_abs;

        result.sum_sq +=
            partial.sum_sq;

        result.count +=
            partial.count;

        result.nonfinite_count +=
            partial.nonfinite_count;

        if (partial.status ==
            REDUCTION_NONFINITE) {
            result.status =
                REDUCTION_NONFINITE;
        }
    }

    if (result.status !=
        REDUCTION_NONFINITE) {
        result.status =
            result.count > 0u
                ? REDUCTION_OK
                : REDUCTION_EMPTY;
    }

    output[gid] = result;
}

// --------------------------------------------------------------------------
// Combine covariance sufficient-statistic partials
// --------------------------------------------------------------------------

kernel void combine_covariance_partials(
    device const CovarianceSufficientPartial* input
                                              [[buffer(0)]],
    constant uint& partial_count              [[buffer(1)]],
    constant uint& partials_per_thread        [[buffer(2)]],
    device CovarianceSufficientPartial* output
                                              [[buffer(3)]],
    uint gid                                  [[thread_position_in_grid]]
) {
    CovarianceSufficientPartial result = {};
    result.status = REDUCTION_EMPTY;

    if (partials_per_thread == 0u) {
        result.status = REDUCTION_INVALID_INPUT;
        output[gid] = result;
        return;
    }

    const uint begin =
        gid *
        partials_per_thread;

    if (begin >= partial_count) {
        output[gid] = result;
        return;
    }

    const uint end =
        min(
            begin +
                partials_per_thread,
            partial_count
        );

    for (uint i = begin; i < end; ++i) {
        const auto partial =
            input[i];

        result.sum_x +=
            partial.sum_x;
        result.sum_y +=
            partial.sum_y;
        result.sum_xx +=
            partial.sum_xx;
        result.sum_yy +=
            partial.sum_yy;
        result.sum_xy +=
            partial.sum_xy;

        result.count +=
            partial.count;

        result.nonfinite_count +=
            partial.nonfinite_count;

        if (partial.status ==
            REDUCTION_NONFINITE) {
            result.status =
                REDUCTION_NONFINITE;
        }
    }

    if (result.status !=
        REDUCTION_NONFINITE) {
        result.status =
            result.count > 0u
                ? REDUCTION_OK
                : REDUCTION_EMPTY;
    }

    output[gid] = result;
}

// --------------------------------------------------------------------------
// Combine P&L partials
// --------------------------------------------------------------------------

kernel void combine_pnl_partials(
    device const PnlPartial* input       [[buffer(0)]],
    constant uint& partial_count         [[buffer(1)]],
    constant uint& partials_per_thread   [[buffer(2)]],
    device PnlPartial* output            [[buffer(3)]],
    uint gid                             [[thread_position_in_grid]]
) {
    PnlPartial result = {};

    result.worst_pnl = INFINITY;
    result.best_pnl = -INFINITY;
    result.status = REDUCTION_EMPTY;

    if (partials_per_thread == 0u) {
        result.status = REDUCTION_INVALID_INPUT;
        output[gid] = result;
        return;
    }

    const uint begin =
        gid *
        partials_per_thread;

    if (begin >= partial_count) {
        output[gid] = result;
        return;
    }

    const uint end =
        min(
            begin +
                partials_per_thread,
            partial_count
        );

    for (uint i = begin; i < end; ++i) {
        const auto partial =
            input[i];

        result.pnl_sum +=
            partial.pnl_sum;

        result.pnl_sq_sum +=
            partial.pnl_sq_sum;

        result.positive_sum +=
            partial.positive_sum;

        result.negative_sum +=
            partial.negative_sum;

        if (partial.count > 0u) {
            result.worst_pnl =
                min(
                    result.worst_pnl,
                    partial.worst_pnl
                );

            result.best_pnl =
                max(
                    result.best_pnl,
                    partial.best_pnl
                );
        }

        result.max_loss =
            max(
                result.max_loss,
                partial.max_loss
            );

        result.positive_count +=
            partial.positive_count;

        result.negative_count +=
            partial.negative_count;

        result.zero_count +=
            partial.zero_count;

        result.nonfinite_count +=
            partial.nonfinite_count;

        result.count +=
            partial.count;

        if (partial.status ==
            REDUCTION_NONFINITE) {
            result.status =
                REDUCTION_NONFINITE;
        }
    }

    if (result.status !=
        REDUCTION_NONFINITE) {
        result.status =
            result.count > 0u
                ? REDUCTION_OK
                : REDUCTION_EMPTY;
    }

    output[gid] = result;
}

// --------------------------------------------------------------------------
// Combine threshold-tail partials
// --------------------------------------------------------------------------

kernel void combine_threshold_tail_partials(
    device const ThresholdTailPartial* input
                                           [[buffer(0)]],
    constant uint& partial_count           [[buffer(1)]],
    constant uint& partials_per_thread     [[buffer(2)]],
    device ThresholdTailPartial* output    [[buffer(3)]],
    uint gid                               [[thread_position_in_grid]]
) {
    ThresholdTailPartial result = {};
    result.status = REDUCTION_EMPTY;

    if (partials_per_thread == 0u) {
        result.status = REDUCTION_INVALID_INPUT;
        output[gid] = result;
        return;
    }

    const uint begin =
        gid *
        partials_per_thread;

    if (begin >= partial_count) {
        output[gid] = result;
        return;
    }

    const uint end =
        min(
            begin +
                partials_per_thread,
            partial_count
        );

    bool threshold_set = false;

    for (uint i = begin; i < end; ++i) {
        const auto partial =
            input[i];

        if (!threshold_set &&
            partial.count > 0u) {
            result.threshold =
                partial.threshold;
            threshold_set = true;
        }

        result.tail_loss_sum +=
            partial.tail_loss_sum;

        result.tail_loss_sq_sum +=
            partial.tail_loss_sq_sum;

        result.worst_loss =
            max(
                result.worst_loss,
                partial.worst_loss
            );

        result.tail_count +=
            partial.tail_count;

        result.count +=
            partial.count;

        result.nonfinite_count +=
            partial.nonfinite_count;

        if (partial.status ==
            REDUCTION_NONFINITE) {
            result.status =
                REDUCTION_NONFINITE;
        }
    }

    if (result.status !=
        REDUCTION_NONFINITE) {
        result.status =
            result.count > 0u
                ? REDUCTION_OK
                : REDUCTION_EMPTY;
    }

    output[gid] = result;
}

// --------------------------------------------------------------------------
// Normalize values using host-finalized mean/std
// --------------------------------------------------------------------------
//
// This is technically a transform rather than a reduction, but it belongs
// here because it consumes reduction results and is useful for:
//   - residual z-scores,
//   - VIX/VXN z-scores,
//   - feature standardization,
//   - CPU/GPU parity tests.
// --------------------------------------------------------------------------

struct NormalizeParams {
    uint sample_count;
    uint reserved0;
    uint reserved1;
    uint reserved2;

    float mean;
    float standard_deviation;
    float clip_abs_z;
    float reserved3;
};

kernel void normalize_zscores(
    device const float* values          [[buffer(0)]],
    constant NormalizeParams& params    [[buffer(1)]],
    device float* output                [[buffer(2)]],
    uint gid                            [[thread_position_in_grid]]
) {
    if (gid >= params.sample_count) {
        return;
    }

    const float value =
        values[gid];

    if (!finite_value(value) ||
        !(params.standard_deviation >
          kReductionEpsilon)) {
        output[gid] = NAN;
        return;
    }

    float z =
        (
            value -
            params.mean
        ) /
        params.standard_deviation;

    if (params.clip_abs_z > 0.0f) {
        z =
            clamp(
                z,
                -params.clip_abs_z,
                params.clip_abs_z
            );
    }

    output[gid] = z;
}

} // namespace gme
