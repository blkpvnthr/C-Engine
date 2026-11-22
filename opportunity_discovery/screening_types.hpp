// ============================================================================
// screening_types.hpp
// ----------------------------------------------------------------------------
// Pure-C++ POD contracts shared across the native opportunity-discovery layer.
//
// This header is deliberately free of Objective-C / Metal headers so it can be
// included by both plain C++ translation units and the Objective-C++ (.mm)
// Metal host. The .mm files own the MTLDevice / command queues / pipelines;
// everything here is host-side, framework-agnostic data.
//
// Phase A scope: the native Metal runtime, its telemetry, and the kernel
// registry. Phase B extends this header with the per-symbol screening inputs,
// discovery records, and membership snapshots.
// ============================================================================
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace trading::metal {

// Which backend actually produced a result. Telemetry and discovery records
// must never conflate a measured GPU execution with a CPU fallback.
enum class ComputeBackend : std::uint8_t {
    Metal = 0,
    Cpu = 1,
};

[[nodiscard]] inline const char* to_string(ComputeBackend b) noexcept {
    return b == ComputeBackend::Metal ? "metal" : "cpu";
}

// Integration lifecycle for a single Metal kernel. Mirrors the states the
// architectural-correction directive (§10) requires us to distinguish so that
// "it compiled" is never confused with "it ran on hardware and matched the CPU
// oracle" or "a real consumer uses it".
enum class KernelStatus : std::uint8_t {
    Discovered = 0,            // present in a .metal source file
    Compiled = 1,              // present in the loaded .metallib
    Registered = 2,            // described in this registry
    PipelineCached = 3,        // an MTLComputePipelineState was built + cached
    HostCallable = 4,          // a native host interface can dispatch it
    HardwareExecuted = 5,      // dispatched on a real device at least once
    ConsumerConnected = 6,     // a legitimate operational/research consumer uses it
    NumericallyValidated = 7,  // GPU output matched the CPU reference oracle
    Benchmarked = 8,           // GPU vs CPU timing measured on device
    Blocked = 9,               // cannot be wired yet (documented reason)
};

[[nodiscard]] const char* to_string(KernelStatus s) noexcept;

// Whether a kernel is wired for live operational use, kept for research only,
// or blocked outright (e.g. needs L2 depth that IEX does not provide).
enum class KernelMode : std::uint8_t {
    Operational = 0,
    Research = 1,
    Blocked = 2,
};

[[nodiscard]] const char* to_string(KernelMode m) noexcept;

// One row of the native kernel inventory. Field names intentionally match the
// directive's required inventory schema (§10/§14) so the emitted JSON is a
// faithful, auditable mirror of the registry.
struct KernelInfo {
    std::string shader_file;        // e.g. "indicators.metal"
    std::string kernel_name;        // bare name, e.g. "rolling_sma_ticks"
    std::string resolved_name;      // name used to look it up (may add "gme::")
    std::string host_interface;     // dotted native/py host, or "" if none
    std::string buffer_contract;    // short human description of the ABI
    std::string consumer;           // legitimate downstream consumer, or ""
    std::string required_data;      // e.g. "l1", "l2_depth", "returns_matrix"
    std::string blocker;            // reason if blocked, else ""
    KernelMode mode{KernelMode::Blocked};
    bool pipeline_cached{false};
    bool hardware_executed{false};
    bool numerically_validated{false};
    bool benchmarked{false};
};

// Monotonic runtime counters. Everything the live dashboard and the
// live_gpu_execution_trace evidence need in order to prove the GPU is actually
// doing operational work (directive §11/§20). All counts are cumulative.
struct RuntimeTelemetry {
    std::uint64_t dispatches_submitted{0};
    std::uint64_t dispatches_completed{0};
    std::uint64_t dispatch_failures{0};
    std::uint64_t pipeline_cache_hits{0};
    std::uint64_t pipeline_cache_misses{0};
    std::uint64_t buffers_allocated{0};
    std::uint64_t buffers_reused{0};
    std::uint64_t bytes_uploaded{0};
    std::uint64_t bytes_downloaded{0};
    std::uint64_t cpu_fallbacks{0};
    std::uint64_t last_dispatch_duration_ns{0};
    std::uint64_t total_dispatch_duration_ns{0};

    [[nodiscard]] double pipeline_cache_hit_rate() const noexcept {
        const std::uint64_t total = pipeline_cache_hits + pipeline_cache_misses;
        return total == 0 ? 0.0
                          : static_cast<double>(pipeline_cache_hits) /
                                static_cast<double>(total);
    }
    [[nodiscard]] double mean_dispatch_ms() const noexcept {
        return dispatches_completed == 0
                   ? 0.0
                   : (static_cast<double>(total_dispatch_duration_ns) /
                      static_cast<double>(dispatches_completed)) /
                         1.0e6;
    }
};

}  // namespace trading::metal
