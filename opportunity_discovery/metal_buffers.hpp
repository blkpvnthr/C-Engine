// ============================================================================
// metal_buffers.hpp
// ----------------------------------------------------------------------------
// Persistent, reusable Metal buffer pool (pure-C++ interface; Obj-C++ impl in
// metal_buffers.mm). Buffers use Metal shared storage so that, on Apple Silicon
// unified memory, the same allocation is visible to both the host and the GPU
// without an explicit copy. The pool recycles allocations by size class so the
// hot path does not allocate a fresh input/output buffer per market observation
// (directive §4/§19).
//
// Ownership: a GpuBuffer handle returned by acquire()/upload_* is owned by the
// pool. Return it with release() to make it reusable. The pool retains the
// underlying MTLBuffer until reset()/destruction.
// ============================================================================
#pragma once

#include <cstddef>
#include <cstdint>

#include "opportunity_discovery/screening_types.hpp"

namespace trading::metal {

// Opaque handle to a pooled Metal buffer (shared storage). `contents` is the
// CPU-visible pointer valid for the lifetime of the buffer in the pool.
struct GpuBuffer {
    void* handle{nullptr};        // retained id<MTLBuffer>, bridged to void*
    void* contents{nullptr};      // [buffer contents] — CPU/GPU shared pointer
    std::size_t size_bytes{0};    // usable size requested by the caller
    std::size_t capacity_bytes{0};// physical allocation (>= size_bytes)

    [[nodiscard]] bool valid() const noexcept { return handle != nullptr; }
};

class BufferPool {
public:
    // `device` is an id<MTLDevice> bridged to void* (the runtime owns the device
    // and hands it in). `telemetry` is optional and, if provided, is updated
    // with allocation / reuse / byte counts.
    explicit BufferPool(void* device, RuntimeTelemetry* telemetry = nullptr);
    ~BufferPool();

    BufferPool(const BufferPool&) = delete;
    BufferPool& operator=(const BufferPool&) = delete;

    // Acquire a buffer of at least `bytes`, reusing a pooled allocation when a
    // compatible one is free. Contents are NOT zeroed unless `zero` is set.
    [[nodiscard]] GpuBuffer acquire(std::size_t bytes, bool zero = false);

    // Return a buffer to the pool for reuse. Safe to call with an invalid handle.
    void release(const GpuBuffer& buffer);

    // Typed convenience uploaders (copy host data into a shared buffer).
    [[nodiscard]] GpuBuffer upload_i64(const std::int64_t* data, std::size_t count);
    [[nodiscard]] GpuBuffer upload_f32(const float* data, std::size_t count);
    [[nodiscard]] GpuBuffer upload_bytes(const void* data, std::size_t bytes);
    // Scalar uint32 (Metal `constant uint&`).
    [[nodiscard]] GpuBuffer upload_u32(std::uint32_t value);

    // Typed readers (copy shared-buffer contents back to host memory).
    void read_i64(const GpuBuffer& buffer, std::int64_t* out, std::size_t count) const;
    void read_f32(const GpuBuffer& buffer, float* out, std::size_t count) const;

    // Number of buffers currently held free for reuse.
    [[nodiscard]] std::size_t free_count() const noexcept;
    // Number of buffers currently handed out (acquired, not released).
    [[nodiscard]] std::size_t live_count() const noexcept;
    // Drop all pooled (free) buffers; live buffers are unaffected.
    void trim();

private:
    struct Impl;
    Impl* impl_;
};

}  // namespace trading::metal
