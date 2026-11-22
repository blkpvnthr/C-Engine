// ============================================================================
// metal_runtime.mm — the single native Objective-C++ Metal runtime.
// See metal_runtime.hpp. Compiled WITHOUT ARC (manual retain/release).
//
// Key behaviours (directive §4):
//  * One MTLDevice, one MTLCommandQueue, one loaded library, a pipeline cache.
//  * Namespace-aware kernel resolution (bare name, else "gme::<name>").
//  * Async submission: a real addCompletedHandler updates telemetry (GPU time,
//    completions, failures); a dispatch_semaphore makes the call deterministic
//    for parity tests while still exercising the completion-handler path.
//  * create() never throws on "Metal unavailable" — it returns a runtime with
//    available()==false and a reason(), so callers fall back to the CPU oracle.
// ============================================================================
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <cstring>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "opportunity_discovery/metal_runtime.hpp"

namespace trading::metal {

struct MetalRuntime::Impl {
    id<MTLDevice> device = nil;
    id<MTLCommandQueue> queue = nil;
    id<MTLLibrary> library = nil;
    std::unordered_map<std::string, id<MTLComputePipelineState>> pipelines;
    std::unordered_map<std::string, bool> fn_present;  // bare names in library
    std::string lib_path;
    std::string reason;
    bool available = false;
    RuntimeTelemetry tele;
    std::unique_ptr<BufferPool> pool;
    std::mutex mtx;  // guards pipelines + telemetry

    // Namespace-aware resolution: return the library function name to use, or ""
    // if neither the bare name nor the gme:: form exists.
    std::string resolve(const std::string& name) const {
        if (fn_present.count(name)) return name;
        const std::string ns = "gme::" + name;
        if (fn_present.count(ns)) return ns;
        return "";
    }
};

MetalRuntime::MetalRuntime() : impl_(std::make_unique<Impl>()) {}
MetalRuntime::~MetalRuntime() {
    if (!impl_) return;
    for (auto& kv : impl_->pipelines) {
        if (kv.second) [kv.second release];
    }
    impl_->pool.reset();
    if (impl_->library) [impl_->library release];
    if (impl_->queue) [impl_->queue release];
    if (impl_->device) [impl_->device release];
}

std::unique_ptr<MetalRuntime> MetalRuntime::create(
    const std::string& metallib_path) {
    std::unique_ptr<MetalRuntime> rt(new MetalRuntime());
    Impl& s = *rt->impl_;
    s.lib_path = metallib_path;

    @autoreleasepool {
        s.device = MTLCreateSystemDefaultDevice();
        if (!s.device) {
            s.reason = "no Metal device available";
            return rt;
        }
        [s.device retain];

        s.queue = [s.device newCommandQueue];  // owned (+1)
        if (!s.queue) {
            s.reason = "failed to create command queue";
            return rt;
        }

        if (metallib_path.empty()) {
            s.reason = "no metallib path supplied";
            return rt;
        }
        NSString* path = [NSString stringWithUTF8String:metallib_path.c_str()];
        NSURL* url = [NSURL fileURLWithPath:path];
        NSError* err = nil;
        s.library = [s.device newLibraryWithURL:url error:&err];  // owned (+1)
        if (!s.library) {
            s.reason = std::string("cannot load metallib: ") +
                       (err ? err.localizedDescription.UTF8String : "unknown");
            return rt;
        }

        for (NSString* fn in [s.library functionNames]) {
            s.fn_present[fn.UTF8String] = true;
        }
        s.pool = std::make_unique<BufferPool>((void*)s.device, &s.tele);
        s.available = true;
    }
    return rt;
}

bool MetalRuntime::available() const noexcept { return impl_->available; }
const std::string& MetalRuntime::reason() const noexcept { return impl_->reason; }
const std::string& MetalRuntime::library_path() const noexcept {
    return impl_->lib_path;
}

std::string MetalRuntime::device_name() const {
    if (!impl_->device) return "";
    return std::string([impl_->device name].UTF8String);
}

std::vector<std::string> MetalRuntime::function_names() const {
    std::vector<std::string> out;
    out.reserve(impl_->fn_present.size());
    for (const auto& kv : impl_->fn_present) out.push_back(kv.first);
    return out;
}

bool MetalRuntime::has_kernel(const std::string& name) const {
    return !impl_->resolve(name).empty();
}

bool MetalRuntime::ensure_pipeline(const std::string& kernel) {
    if (!impl_->available) return false;
    const std::string resolved = impl_->resolve(kernel);
    if (resolved.empty()) return false;
    std::lock_guard<std::mutex> lk(impl_->mtx);
    auto it = impl_->pipelines.find(resolved);
    if (it != impl_->pipelines.end()) return it->second != nil;

    @autoreleasepool {
        id<MTLFunction> fn =
            [impl_->library newFunctionWithName:@(resolved.c_str())];  // owned
        if (!fn) return false;
        NSError* err = nil;
        id<MTLComputePipelineState> pso =
            [impl_->device newComputePipelineStateWithFunction:fn
                                                         error:&err];  // owned
        [fn release];
        if (!pso) return false;
        impl_->pipelines[resolved] = pso;  // keep ownership in the cache
        return true;
    }
}

bool MetalRuntime::dispatch(const std::string& kernel,
                            const std::vector<GpuBuffer>& buffers,
                            std::uint64_t grid_size) {
    if (!impl_->available || grid_size == 0) {
        std::lock_guard<std::mutex> lk(impl_->mtx);
        ++impl_->tele.dispatch_failures;
        return false;
    }

    const std::string resolved = impl_->resolve(kernel);
    if (resolved.empty()) {
        std::lock_guard<std::mutex> lk(impl_->mtx);
        ++impl_->tele.dispatch_failures;
        return false;
    }

    id<MTLComputePipelineState> pso = nil;
    {
        // Cache lookup / build + hit-miss + submitted counters under the lock.
        std::lock_guard<std::mutex> lk(impl_->mtx);
        auto it = impl_->pipelines.find(resolved);
        if (it != impl_->pipelines.end()) {
            pso = it->second;
            ++impl_->tele.pipeline_cache_hits;
        } else {
            @autoreleasepool {
                id<MTLFunction> fn =
                    [impl_->library newFunctionWithName:@(resolved.c_str())];
                if (fn) {
                    NSError* err = nil;
                    pso = [impl_->device newComputePipelineStateWithFunction:fn
                                                                      error:&err];
                    [fn release];
                }
            }
            if (!pso) {
                ++impl_->tele.dispatch_failures;
                return false;
            }
            impl_->pipelines[resolved] = pso;
            ++impl_->tele.pipeline_cache_misses;
        }
        ++impl_->tele.dispatches_submitted;
    }

    __block bool ok = false;
    @autoreleasepool {
        id<MTLCommandBuffer> cmd = [impl_->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pso];
        for (NSUInteger i = 0; i < buffers.size(); ++i) {
            [enc setBuffer:(id<MTLBuffer>)buffers[i].handle offset:0 atIndex:i];
        }
        NSUInteger max_w = [pso maxTotalThreadsPerThreadgroup];
        NSUInteger w = grid_size < max_w ? (NSUInteger)grid_size : max_w;
        if (w == 0) w = 1;
        MTLSize grid = MTLSizeMake((NSUInteger)grid_size, 1, 1);
        MTLSize tg = MTLSizeMake(w, 1, 1);
        [enc dispatchThreads:grid threadsPerThreadgroup:tg];
        [enc endEncoding];

        // Real completion handler: records GPU timing + completion/failure on a
        // background queue. The semaphore makes the call deterministic.
        Impl* s = impl_.get();
        dispatch_semaphore_t sem = dispatch_semaphore_create(0);
        [cmd addCompletedHandler:^(id<MTLCommandBuffer> c) {
            std::lock_guard<std::mutex> lk(s->mtx);
            if (c.status == MTLCommandBufferStatusError) {
                ++s->tele.dispatch_failures;
            } else {
                ++s->tele.dispatches_completed;
                const double secs = c.GPUEndTime - c.GPUStartTime;
                const std::uint64_t ns =
                    secs > 0 ? static_cast<std::uint64_t>(secs * 1.0e9) : 0;
                s->tele.last_dispatch_duration_ns = ns;
                s->tele.total_dispatch_duration_ns += ns;
                ok = true;
            }
            dispatch_semaphore_signal(sem);
        }];
        [cmd commit];
        dispatch_semaphore_wait(sem, DISPATCH_TIME_FOREVER);
        dispatch_release(sem);
        if (cmd.error != nil) ok = false;
    }
    return ok;
}

// --------------------------------------------------------------------------
// High-level operational kernels (mirror the existing Python host ABIs).
// --------------------------------------------------------------------------
std::vector<std::int64_t> MetalRuntime::sma(
    const std::vector<std::int64_t>& prices, std::uint32_t window) {
    if (!available()) throw MetalError("Metal runtime unavailable");
    if (prices.empty()) throw MetalError("prices cannot be empty");
    if (window == 0) throw MetalError("window must be positive");
    const std::size_t n = prices.size();
    auto& pool = *impl_->pool;
    GpuBuffer pb = pool.upload_i64(prices.data(), n);
    GpuBuffer ob = pool.acquire(n * sizeof(std::int64_t), true);
    GpuBuffer cb = pool.upload_u32(static_cast<std::uint32_t>(n));
    GpuBuffer wb = pool.upload_u32(window);
    const bool ok = dispatch("rolling_sma_ticks", {pb, ob, cb, wb}, n);
    std::vector<std::int64_t> out(n, 0);
    if (ok) pool.read_i64(ob, out.data(), n);
    pool.release(pb); pool.release(ob); pool.release(cb); pool.release(wb);
    if (!ok) throw MetalError("rolling_sma_ticks dispatch failed");
    return out;
}

std::vector<std::int64_t> MetalRuntime::ema(
    const std::vector<std::int64_t>& prices, std::uint32_t alpha_numerator,
    std::uint32_t alpha_denominator) {
    if (!available()) throw MetalError("Metal runtime unavailable");
    if (prices.empty()) throw MetalError("prices cannot be empty");
    if (alpha_denominator == 0) throw MetalError("alpha_denominator must be positive");
    if (alpha_numerator > alpha_denominator)
        throw MetalError("alpha_numerator must be <= alpha_denominator");
    const std::size_t n = prices.size();
    auto& pool = *impl_->pool;
    GpuBuffer pb = pool.upload_i64(prices.data(), n);
    GpuBuffer ob = pool.acquire(n * sizeof(std::int64_t), true);
    GpuBuffer cb = pool.upload_u32(static_cast<std::uint32_t>(n));
    GpuBuffer nb = pool.upload_u32(alpha_numerator);
    GpuBuffer db = pool.upload_u32(alpha_denominator);
    // EMA is a sequential recurrence: a single GPU thread walks the series.
    const bool ok = dispatch("exponential_moving_average_ticks",
                             {pb, ob, cb, nb, db}, 1);
    std::vector<std::int64_t> out(n, 0);
    if (ok) pool.read_i64(ob, out.data(), n);
    pool.release(pb); pool.release(ob); pool.release(cb);
    pool.release(nb); pool.release(db);
    if (!ok) throw MetalError("exponential_moving_average_ticks dispatch failed");
    return out;
}

std::vector<std::int64_t> MetalRuntime::apo(
    const std::vector<std::int64_t>& fast_ema,
    const std::vector<std::int64_t>& slow_ema) {
    if (!available()) throw MetalError("Metal runtime unavailable");
    if (fast_ema.empty() || fast_ema.size() != slow_ema.size())
        throw MetalError("fast_ema and slow_ema must have equal non-zero lengths");
    const std::size_t n = fast_ema.size();
    auto& pool = *impl_->pool;
    GpuBuffer fb = pool.upload_i64(fast_ema.data(), n);
    GpuBuffer sb = pool.upload_i64(slow_ema.data(), n);
    GpuBuffer ob = pool.acquire(n * sizeof(std::int64_t), true);
    GpuBuffer cb = pool.upload_u32(static_cast<std::uint32_t>(n));
    const bool ok = dispatch("average_price_oscillator_ticks", {fb, sb, ob, cb}, n);
    std::vector<std::int64_t> out(n, 0);
    if (ok) pool.read_i64(ob, out.data(), n);
    pool.release(fb); pool.release(sb); pool.release(ob); pool.release(cb);
    if (!ok) throw MetalError("average_price_oscillator_ticks dispatch failed");
    return out;
}

std::vector<std::int64_t> MetalRuntime::true_range(
    const std::vector<std::int64_t>& highs,
    const std::vector<std::int64_t>& lows,
    const std::vector<std::int64_t>& previous_closes) {
    if (!available()) throw MetalError("Metal runtime unavailable");
    const std::size_t n = highs.size();
    if (n == 0 || lows.size() != n || previous_closes.size() != n)
        throw MetalError("highs, lows, previous_closes must have equal non-zero lengths");
    auto& pool = *impl_->pool;
    GpuBuffer hb = pool.upload_i64(highs.data(), n);
    GpuBuffer lb = pool.upload_i64(lows.data(), n);
    GpuBuffer cbuf = pool.upload_i64(previous_closes.data(), n);
    GpuBuffer ob = pool.acquire(n * sizeof(std::int64_t), true);
    GpuBuffer nb = pool.upload_u32(static_cast<std::uint32_t>(n));
    const bool ok = dispatch("true_range_ticks", {hb, lb, cbuf, ob, nb}, n);
    std::vector<std::int64_t> out(n, 0);
    if (ok) pool.read_i64(ob, out.data(), n);
    pool.release(hb); pool.release(lb); pool.release(cbuf);
    pool.release(ob); pool.release(nb);
    if (!ok) throw MetalError("true_range_ticks dispatch failed");
    return out;
}

namespace {
// CovarianceParams {observations, assets, row_stride, reserved} as 4x uint32.
struct CovParams {
    std::uint32_t observations, assets, row_stride, reserved;
};
// EwmaParams {4x uint32, lambda, minimum_variance, pad0, pad1}.
struct EwmaParams {
    std::uint32_t observations, assets, row_stride, reserved;
    float lambda, minimum_variance, pad0, pad1;
};
}  // namespace

std::vector<float> MetalRuntime::column_means(const std::vector<float>& values,
                                              std::uint32_t observations,
                                              std::uint32_t assets) {
    if (!available()) throw MetalError("Metal runtime unavailable");
    if (observations == 0 || assets == 0)
        throw MetalError("observations and assets must be positive");
    if (values.size() < static_cast<std::size_t>(observations) * assets)
        throw MetalError("values smaller than observations*assets");
    auto& pool = *impl_->pool;
    GpuBuffer vb = pool.upload_f32(values.data(),
                                   static_cast<std::size_t>(observations) * assets);
    CovParams p{observations, assets, assets, 0};
    GpuBuffer prm = pool.upload_bytes(&p, sizeof(p));
    GpuBuffer mb = pool.acquire(assets * sizeof(float), true);
    const bool ok = dispatch("column_means", {vb, prm, mb}, assets);
    std::vector<float> out(assets, 0.0f);
    if (ok) pool.read_f32(mb, out.data(), assets);
    pool.release(vb); pool.release(prm); pool.release(mb);
    if (!ok) throw MetalError("column_means dispatch failed");
    return out;
}

std::vector<float> MetalRuntime::covariance_matrix(const std::vector<float>& values,
                                                   std::uint32_t observations,
                                                   std::uint32_t assets) {
    if (!available()) throw MetalError("Metal runtime unavailable");
    if (observations == 0 || assets == 0)
        throw MetalError("observations and assets must be positive");
    if (values.size() < static_cast<std::size_t>(observations) * assets)
        throw MetalError("values smaller than observations*assets");
    auto& pool = *impl_->pool;
    GpuBuffer vb = pool.upload_f32(values.data(),
                                   static_cast<std::size_t>(observations) * assets);
    CovParams p{observations, assets, assets, 0};
    GpuBuffer prm = pool.upload_bytes(&p, sizeof(p));
    GpuBuffer mb = pool.acquire(assets * sizeof(float), true);
    GpuBuffer cov = pool.acquire(static_cast<std::size_t>(assets) * assets * sizeof(float), true);
    const bool ok1 = dispatch("column_means", {vb, prm, mb}, assets);
    const bool ok2 =
        ok1 && dispatch("covariance_matrix", {vb, mb, prm, cov},
                        static_cast<std::uint64_t>(assets) * assets);
    std::vector<float> out(static_cast<std::size_t>(assets) * assets, 0.0f);
    if (ok2) pool.read_f32(cov, out.data(), static_cast<std::size_t>(assets) * assets);
    pool.release(vb); pool.release(prm); pool.release(mb); pool.release(cov);
    if (!ok2) throw MetalError("covariance_matrix dispatch failed");
    return out;
}

std::vector<float> MetalRuntime::ewma_covariance_matrix(
    const std::vector<float>& values, std::uint32_t observations,
    std::uint32_t assets, float lambda, float minimum_variance) {
    if (!available()) throw MetalError("Metal runtime unavailable");
    if (observations == 0 || assets == 0)
        throw MetalError("observations and assets must be positive");
    if (!(lambda > 0.0f && lambda <= 1.0f))
        throw MetalError("lambda must be in (0, 1]");
    if (values.size() < static_cast<std::size_t>(observations) * assets)
        throw MetalError("values smaller than observations*assets");
    auto& pool = *impl_->pool;
    GpuBuffer vb = pool.upload_f32(values.data(),
                                   static_cast<std::size_t>(observations) * assets);
    EwmaParams p{observations, assets, assets, 0, lambda, minimum_variance, 0.0f, 0.0f};
    GpuBuffer prm = pool.upload_bytes(&p, sizeof(p));
    GpuBuffer cov = pool.acquire(static_cast<std::size_t>(assets) * assets * sizeof(float), true);
    const bool ok = dispatch("ewma_covariance_matrix", {vb, prm, cov},
                             static_cast<std::uint64_t>(assets) * assets);
    std::vector<float> out(static_cast<std::size_t>(assets) * assets, 0.0f);
    if (ok) pool.read_f32(cov, out.data(), static_cast<std::size_t>(assets) * assets);
    pool.release(vb); pool.release(prm); pool.release(cov);
    if (!ok) throw MetalError("ewma_covariance_matrix dispatch failed");
    return out;
}

std::vector<ScreeningOutput> MetalRuntime::screen(
    const std::vector<ScreeningInput>& inputs) {
    if (!available()) throw MetalError("Metal runtime unavailable");
    if (inputs.empty()) throw MetalError("inputs cannot be empty");
    const std::size_t n = inputs.size();
    auto& pool = *impl_->pool;
    // Shared-storage buffers: inputs[0], outputs[1], symbol_count[2] — mirrors
    // the screen_market signature. SoA-of-structs upload is a single memcpy.
    GpuBuffer ib = pool.upload_bytes(inputs.data(), n * sizeof(ScreeningInput));
    GpuBuffer ob = pool.acquire(n * sizeof(ScreeningOutput), true);
    GpuBuffer cb = pool.upload_u32(static_cast<std::uint32_t>(n));
    const bool ok = dispatch("screen_market", {ib, ob, cb}, n);
    std::vector<ScreeningOutput> out(n);
    if (ok && ob.contents) {
        std::memcpy(out.data(), ob.contents, n * sizeof(ScreeningOutput));
    }
    pool.release(ib);
    pool.release(ob);
    pool.release(cb);
    if (!ok) throw MetalError("screen_market dispatch failed");
    return out;
}

RuntimeTelemetry MetalRuntime::telemetry() const {
    std::lock_guard<std::mutex> lk(impl_->mtx);
    return impl_->tele;
}

std::size_t MetalRuntime::cached_pipeline_count() const {
    std::lock_guard<std::mutex> lk(impl_->mtx);
    return impl_->pipelines.size();
}

BufferPool& MetalRuntime::buffers() {
    if (!impl_->pool) throw MetalError("buffer pool unavailable (Metal down)");
    return *impl_->pool;
}

}  // namespace trading::metal
