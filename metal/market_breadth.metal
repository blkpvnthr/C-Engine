#include <metal_stdlib>
using namespace metal;

// ============================================================================
// market_breadth.metal
// ----------------------------------------------------------------------------
// Phase B.4 GPU reductions over per-symbol screen flags (the ScreeningOutput.
// flags bitmasks from screen_market). One thread per symbol:
//   * market_breadth        — atomically tallies breadth counters.
//   * compact_entry_candidates — atomically compacts ENTRY_QUALIFIED symbol
//     indices into a dense output array, with explicit overflow detection.
// CPU oracles (cpu_ref::market_breadth / compact_entry_candidates) mirror both.
// Counts are order-independent (exact GPU/CPU parity); the compacted id ORDER is
// non-deterministic across GPU threads, so callers compare the id set + count.
//
// Flag bits mirror ScreenFlag in screening_types.hpp / opportunity_screening.metal.
// ============================================================================

enum ScreenFlag : uint {
    ENTRY_QUALIFIED   = 1u << 0,
    TECHNICAL_EXIT    = 1u << 1,
    BULLISH_DIRECTION = 1u << 2,
    BEARISH_DIRECTION = 1u << 3,
    POSITIVE_MOMENTUM = 1u << 4,
    NEGATIVE_MOMENTUM = 1u << 5,
    ABOVE_VWAP        = 1u << 6,
    BELOW_VWAP        = 1u << 7,
    DATA_INVALID      = 1u << 8
};

// Atomic-counter layout; binary-identical to host BreadthCounters (11x uint32).
struct BreadthCountersAtomic {
    atomic_uint total;
    atomic_uint valid;
    atomic_uint entry_qualified;
    atomic_uint technical_exit;
    atomic_uint bullish;
    atomic_uint bearish;
    atomic_uint positive_momentum;
    atomic_uint negative_momentum;
    atomic_uint above_vwap;
    atomic_uint below_vwap;
    atomic_uint data_invalid;
};

inline void bump(device atomic_uint* c) {
    atomic_fetch_add_explicit(c, 1u, memory_order_relaxed);
}

kernel void market_breadth(
    device const uint* flags [[buffer(0)]],
    device BreadthCountersAtomic* out [[buffer(1)]],
    constant uint& symbol_count [[buffer(2)]],
    uint id [[thread_position_in_grid]]) {
    if (id >= symbol_count) {
        return;
    }
    uint f = flags[id];
    bump(&out->total);
    if (!(f & DATA_INVALID)) bump(&out->valid); else bump(&out->data_invalid);
    if (f & ENTRY_QUALIFIED) bump(&out->entry_qualified);
    if (f & TECHNICAL_EXIT) bump(&out->technical_exit);
    if (f & BULLISH_DIRECTION) bump(&out->bullish);
    if (f & BEARISH_DIRECTION) bump(&out->bearish);
    if (f & POSITIVE_MOMENTUM) bump(&out->positive_momentum);
    if (f & NEGATIVE_MOMENTUM) bump(&out->negative_momentum);
    if (f & ABOVE_VWAP) bump(&out->above_vwap);
    if (f & BELOW_VWAP) bump(&out->below_vwap);
}

// Compact the indices of ENTRY_QUALIFIED symbols into out_ids[0..count). The
// running counter (buffer 2, a single atomic_uint) ends at the TOTAL number of
// qualified symbols; if that exceeds `capacity`, overflow occurred and only the
// first `capacity` ids were written (callers must detect counter > capacity).
kernel void compact_entry_candidates(
    device const uint* flags [[buffer(0)]],
    device uint* out_ids [[buffer(1)]],
    device atomic_uint* counter [[buffer(2)]],
    constant uint& symbol_count [[buffer(3)]],
    constant uint& capacity [[buffer(4)]],
    uint id [[thread_position_in_grid]]) {
    if (id >= symbol_count) {
        return;
    }
    if (flags[id] & ENTRY_QUALIFIED) {
        uint slot = atomic_fetch_add_explicit(counter, 1u, memory_order_relaxed);
        if (slot < capacity) {
            out_ids[slot] = id;
        }
    }
}
