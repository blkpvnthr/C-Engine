// ============================================================================
// kernel_registry.hpp
// ----------------------------------------------------------------------------
// Native authoritative inventory of every Metal kernel in the loaded library.
// Seeded with static metadata (shader, name, mode, consumer, blocker) and
// reconciled against the live device (which names actually compiled, which
// pipelines build) so DISCOVERED/COMPILED/PIPELINE_CACHED reflect measured
// evidence rather than assertions (directive §14).
//
// Baseline the correction fixes: 45 existing kernels, 7 previously wired.
// Newly added discovery kernels (Phase B+) register here too and are tracked
// separately via their shader_file.
// ============================================================================
#pragma once

#include <string>
#include <vector>

#include "opportunity_discovery/screening_types.hpp"

namespace trading::metal {

class MetalRuntime;  // fwd

class KernelRegistry {
public:
    // Construct with the static 45-kernel baseline inventory.
    KernelRegistry();

    // All registered kernels (static inventory + any added).
    [[nodiscard]] const std::vector<KernelInfo>& kernels() const noexcept;

    // Add or update a kernel entry (used when new discovery shaders land).
    void upsert(const KernelInfo& info);

    // Reconcile against a live runtime: verify each kernel name is present in
    // the loaded library and that an MTLComputePipelineState can be built
    // (sets pipeline_cached). Does NOT execute kernels.
    void reconcile_with_device(MetalRuntime& runtime);

    // Mark a kernel's lifecycle flags after host wiring / hardware runs /
    // parity checks. No-op if the kernel is unknown.
    void mark_hardware_executed(const std::string& kernel_name);
    void mark_numerically_validated(const std::string& kernel_name);
    void mark_benchmarked(const std::string& kernel_name);
    // Set once a real operational consumer actually dispatches the kernel on the
    // live path (NOT merely because an intended consumer is named).
    void mark_operationally_consumed(const std::string& kernel_name);

    // Coverage summary, computed from live flags.
    struct Coverage {
        std::size_t discovered{0};         // total registered
        std::size_t compiled_callable{0};  // pipeline_cached true
        std::size_t host_callable{0};      // has a host_interface
        std::size_t consumer_connected{0}; // operationally_consumed (real wiring)
        std::size_t intended_consumer{0};  // has a named (intended) consumer
        std::size_t hardware_executed{0};
        std::size_t numerically_validated{0};
        std::size_t benchmarked{0};
        std::size_t blocked{0};            // mode == Blocked
        double pipeline_coverage_pct{0.0};
        double operational_coverage_pct{0.0};  // 100 * consumer_connected / discovered
    };
    [[nodiscard]] Coverage coverage() const;

    // Serialize the inventory + coverage as a JSON string (for reports/*.json).
    [[nodiscard]] std::string to_json() const;

private:
    std::vector<KernelInfo> kernels_;
    [[nodiscard]] KernelInfo* find(const std::string& kernel_name);
};

}  // namespace trading::metal
