// ============================================================================
// metal_buffers.mm — persistent Metal buffer pool (Obj-C++; see metal_buffers.hpp)
//
// Compiled WITHOUT ARC (manual retain/release) so Objective-C buffer objects can
// live inside C++ containers with deterministic lifetimes. `newBufferWithLength:`
// follows Cocoa "new" ownership (returns +1), so we own it and release on trim/
// destruction. Buffers use shared storage: on Apple Silicon unified memory the
// same allocation is host- and GPU-visible with no copy.
// ============================================================================
#import <Metal/Metal.h>

#include <algorithm>
#include <cstring>
#include <vector>

#include "opportunity_discovery/metal_buffers.hpp"

namespace trading::metal {

struct BufferPool::Impl {
    id<MTLDevice> device = nil;
    RuntimeTelemetry* tele = nullptr;
    std::vector<GpuBuffer> free_buffers;  // available for reuse
    std::size_t live = 0;                 // handed out, not yet released
};

BufferPool::BufferPool(void* device, RuntimeTelemetry* telemetry)
    : impl_(new Impl) {
    impl_->device = (id<MTLDevice>)device;
    [impl_->device retain];
    impl_->tele = telemetry;
}

BufferPool::~BufferPool() {
    for (auto& b : impl_->free_buffers) {
        if (b.handle) [(id<MTLBuffer>)b.handle release];
    }
    if (impl_->device) [impl_->device release];
    delete impl_;
}

GpuBuffer BufferPool::acquire(std::size_t bytes, bool zero) {
    if (bytes == 0) bytes = 1;

    // Best-fit reuse: smallest free buffer whose capacity can hold `bytes`.
    int best = -1;
    for (int i = 0; i < static_cast<int>(impl_->free_buffers.size()); ++i) {
        if (impl_->free_buffers[i].capacity_bytes >= bytes) {
            if (best < 0 || impl_->free_buffers[i].capacity_bytes <
                                impl_->free_buffers[best].capacity_bytes) {
                best = i;
            }
        }
    }

    GpuBuffer gb;
    if (best >= 0) {
        gb = impl_->free_buffers[static_cast<std::size_t>(best)];
        impl_->free_buffers.erase(impl_->free_buffers.begin() + best);
        gb.size_bytes = bytes;
        if (impl_->tele) ++impl_->tele->buffers_reused;
    } else {
        id<MTLBuffer> buf =
            [impl_->device newBufferWithLength:bytes
                                      options:MTLResourceStorageModeShared];
        if (!buf) return GpuBuffer{};  // invalid
        gb.handle = (void*)buf;
        gb.contents = [buf contents];
        gb.size_bytes = bytes;
        gb.capacity_bytes = static_cast<std::size_t>([buf length]);
        if (impl_->tele) ++impl_->tele->buffers_allocated;
    }

    if (zero && gb.contents) std::memset(gb.contents, 0, gb.size_bytes);
    ++impl_->live;
    return gb;
}

void BufferPool::release(const GpuBuffer& buffer) {
    if (!buffer.handle) return;
    impl_->free_buffers.push_back(buffer);
    if (impl_->live) --impl_->live;
}

GpuBuffer BufferPool::upload_bytes(const void* data, std::size_t bytes) {
    GpuBuffer gb = acquire(bytes, false);
    if (gb.contents && data) std::memcpy(gb.contents, data, bytes);
    if (impl_->tele) impl_->tele->bytes_uploaded += bytes;
    return gb;
}

GpuBuffer BufferPool::upload_i64(const std::int64_t* data, std::size_t count) {
    return upload_bytes(data, count * sizeof(std::int64_t));
}

GpuBuffer BufferPool::upload_f32(const float* data, std::size_t count) {
    return upload_bytes(data, count * sizeof(float));
}

GpuBuffer BufferPool::upload_u32(std::uint32_t value) {
    return upload_bytes(&value, sizeof(std::uint32_t));
}

void BufferPool::read_i64(const GpuBuffer& buffer, std::int64_t* out,
                          std::size_t count) const {
    if (!buffer.contents || !out) return;
    std::memcpy(out, buffer.contents, count * sizeof(std::int64_t));
    if (impl_->tele) impl_->tele->bytes_downloaded += count * sizeof(std::int64_t);
}

void BufferPool::read_f32(const GpuBuffer& buffer, float* out,
                          std::size_t count) const {
    if (!buffer.contents || !out) return;
    std::memcpy(out, buffer.contents, count * sizeof(float));
    if (impl_->tele) impl_->tele->bytes_downloaded += count * sizeof(float);
}

std::size_t BufferPool::free_count() const noexcept {
    return impl_->free_buffers.size();
}

std::size_t BufferPool::live_count() const noexcept { return impl_->live; }

void BufferPool::trim() {
    for (auto& b : impl_->free_buffers) {
        if (b.handle) [(id<MTLBuffer>)b.handle release];
    }
    impl_->free_buffers.clear();
}

}  // namespace trading::metal
