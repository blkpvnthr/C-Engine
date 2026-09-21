#include <metal_stdlib>
using namespace metal;

// Prices remain signed integer ticks. Window lengths and all thresholds are
// supplied by the caller; kernels contain no trading policy.
kernel void rolling_sma_ticks(
    device const long* prices [[buffer(0)]],
    device long* output [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    constant uint& window [[buffer(3)]],
    uint i [[thread_position_in_grid]]) {
    if (i >= count || window == 0 || i + 1 < window) return;
    long sum = 0;
    for (uint j = i + 1 - window; j <= i; ++j) sum += prices[j];
    output[i] = sum / long(window);
}

kernel void exponential_moving_average_ticks(
    device const long* prices [[buffer(0)]],
    device long* output [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    constant uint& alpha_numerator [[buffer(3)]],
    constant uint& alpha_denominator [[buffer(4)]],
    uint i [[thread_position_in_grid]]) {
    if (i >= count || alpha_denominator == 0 || alpha_numerator > alpha_denominator) return;
    if (i == 0) { output[0] = prices[0]; return; }
    // This recurrence must be dispatched serially or in ordered passes by the host.
    output[i] = (long(alpha_numerator) * prices[i] +
                 long(alpha_denominator - alpha_numerator) * output[i - 1]) /
                long(alpha_denominator);
}

kernel void average_price_oscillator_ticks(
    device const long* fast_ema [[buffer(0)]],
    device const long* slow_ema [[buffer(1)]],
    device long* output [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint i [[thread_position_in_grid]]) {
    if (i < count) output[i] = fast_ema[i] - slow_ema[i];
}

kernel void true_range_ticks(
    device const long* highs [[buffer(0)]],
    device const long* lows [[buffer(1)]],
    device const long* previous_closes [[buffer(2)]],
    device long* output [[buffer(3)]],
    constant uint& count [[buffer(4)]],
    uint i [[thread_position_in_grid]]) {
    if (i >= count) return;
    long a = highs[i] - lows[i];
    long b = abs(highs[i] - previous_closes[i]);
    long c = abs(lows[i] - previous_closes[i]);
    output[i] = max(a, max(b, c));
}
