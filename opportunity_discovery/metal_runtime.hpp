// ============================================================================
// metal_runtime.hpp
// ----------------------------------------------------------------------------
// The single native Objective-C++ Metal runtime (pure-C++ PIMPL interface;
// Obj-C++ impl in metal_runtime.mm). It owns the one MTLDevice, one
// MTLCommandQueue, the loaded .metallib, a cache of MTLComputePipelineState
// objects, and a persistent BufferPool. This replaces the per-call,
// no-pipeline-cache PyObjC dispatch path for operational work (directive §4).
//
// There must be exactly one operational Metal runtime; callers share an
// instance. PyObjC is retained only for temporary regression parity and is not
// used operationally once this runtime is validated.
//
// Design notes:
//  * create() never throws for "Metal unavailable"; it returns a runtime whose
//    available() is false and whose reason() explains why, so the application
//    degrades to the CPU fallback instead of crashing (directive §4/§13).
//  * The high-level kernel methods (sma/ema/apo/true_range + covariance) mirror
//    the existing cengine.metal_indicators / metal_covariance ABIs exactly so
//    GPU output can be compared for strict parity against the CPU oracle.
//  * Pipelines are built once and cached by resolved kernel name; the resolver
//    is namespace-aware (bare name, else "gme::<name>") to match the shaders.
// ============================================================================
#pragma once

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "opportunity_discovery/metal_buffers.hpp"
#include "opportunity_discovery/screening_types.hpp"

namespace trading::metal {

// Raised by the high-level kernel methods when the runtime is unavailable or an
// argument is invalid. Generic dispatch() reports failure via its bool return.
class MetalError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

class MetalRuntime {
public:
    // Build the runtime from a compiled .metallib path. Always returns a
    // non-null object; check available(). An empty path triggers the same
    // library discovery order the Python backend uses only if the caller does
    // not supply one (callers normally pass the authoritative, build-recorded
    // path to avoid stale artifacts — directive §2).
    [[nodiscard]] static std::unique_ptr<MetalRuntime> create(
        const std::string& metallib_path);

    ~MetalRuntime();
    MetalRuntime(const MetalRuntime&) = delete;
    MetalRuntime& operator=(const MetalRuntime&) = delete;

    [[nodiscard]] bool available() const noexcept;
    [[nodiscard]] const std::string& reason() const noexcept;
    [[nodiscard]] std::string device_name() const;
    [[nodiscard]] const std::string& library_path() const noexcept;

    // Entry-point names present in the loaded library (as reported by the
    // device), and a namespace-aware membership test.
    [[nodiscard]] std::vector<std::string> function_names() const;
    [[nodiscard]] bool has_kernel(const std::string& name) const;

    // Build and cache an MTLComputePipelineState for `kernel` (namespace-aware)
    // without executing it. Returns true on success. Used by the registry to
    // verify pipeline-callability from live evidence.
    [[nodiscard]] bool ensure_pipeline(const std::string& kernel);

    // --- low-level generic 1-D dispatch ------------------------------------
    // Binds buffers at indices 0..n-1, dispatches `grid_size` threads with an
    // auto threadgroup width, waits for completion, and updates telemetry
    // (pipeline-cache hit/miss, submitted/completed/failures, GPU duration).
    // Returns false on any failure (and increments dispatch_failures).
    bool dispatch(const std::string& kernel,
                  const std::vector<GpuBuffer>& buffers,
                  std::uint64_t grid_size);

    // --- high-level operational kernels (parity-testable) ------------------
    // Integer-tick indicators (mirror indicators.metal exactly).
    [[nodiscard]] std::vector<std::int64_t> sma(
        const std::vector<std::int64_t>& prices, std::uint32_t window);
    [[nodiscard]] std::vector<std::int64_t> ema(
        const std::vector<std::int64_t>& prices,
        std::uint32_t alpha_numerator, std::uint32_t alpha_denominator);
    [[nodiscard]] std::vector<std::int64_t> apo(
        const std::vector<std::int64_t>& fast_ema,
        const std::vector<std::int64_t>& slow_ema);
    [[nodiscard]] std::vector<std::int64_t> true_range(
        const std::vector<std::int64_t>& highs,
        const std::vector<std::int64_t>& lows,
        const std::vector<std::int64_t>& previous_closes);

    // Covariance (gme:: namespace; values is obs*assets row-major float32).
    // Returns assets means / assets*assets covariance flattened row-major.
    [[nodiscard]] std::vector<float> column_means(
        const std::vector<float>& values, std::uint32_t observations,
        std::uint32_t assets);
    [[nodiscard]] std::vector<float> covariance_matrix(
        const std::vector<float>& values, std::uint32_t observations,
        std::uint32_t assets);
    [[nodiscard]] std::vector<float> ewma_covariance_matrix(
        const std::vector<float>& values, std::uint32_t observations,
        std::uint32_t assets, float lambda, float minimum_variance);

    // Opportunity screening (one GPU thread per symbol). Evaluates the mandatory
    // shared entry contract on device via metal/opportunity_screening.metal and
    // returns one ScreeningOutput per input, bit-for-bit matching the CPU oracle
    // cpu_ref::screen_market (directive §28/§32/§37).
    [[nodiscard]] std::vector<ScreeningOutput> screen(
        const std::vector<ScreeningInput>& inputs);

    // Batch technical-feature computation (one GPU thread per symbol) over
    // row-major (symbol_count * max_window) bar windows. Dispatches
    // technical_features on device; returns one TechFeatures per symbol,
    // matching cpu_ref::technical_features within tolerance (directive §27).
    [[nodiscard]] std::vector<TechFeatures> technical_features(
        const std::vector<float>& closes, const std::vector<float>& highs,
        const std::vector<float>& lows, const std::vector<float>& volumes,
        const std::vector<std::uint32_t>& counts, std::uint32_t symbol_count,
        std::uint32_t max_window, std::uint32_t dmi_period = 14,
        std::uint32_t sma_period = 20);

    // --- introspection / telemetry -----------------------------------------
    [[nodiscard]] RuntimeTelemetry telemetry() const;
    [[nodiscard]] std::size_t cached_pipeline_count() const;
    [[nodiscard]] BufferPool& buffers();

private:
    MetalRuntime();
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace trading::metal
