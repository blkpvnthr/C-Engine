"""Phase A — native Objective-C++ Metal runtime (architectural correction).

Validates the native ``trading::metal::MetalRuntime`` exposed via
``_cengine_native``: on-device execution of the 7 previously-wired kernels,
strict CPU parity, pipeline caching, persistent-buffer reuse, telemetry, the
native kernel registry, graceful unavailability, and — critically — numerical
parity against the legacy PyObjC path it replaces (the regression gate for the
migration). These require genuine Apple-Silicon hardware; they skip elsewhere.
"""

from __future__ import annotations

import platform
import random

import _cengine_native as native
import pytest

from cengine import metal_backend as mb
from cengine import metal_cpu_ref as ref

pytestmark = pytest.mark.skipif(
    platform.system() != "Darwin"
    or platform.machine() != "arm64"
    or not getattr(native, "metal_available", False),
    reason="native Metal runtime requires macOS on Apple Silicon with Metal built in",
)

PRICES = [100, 101, 102, 100, 99, 101, 103, 105, 104, 106, 108, 107, 109, 111, 110]


@pytest.fixture(scope="module")
def metallib() -> str:
    lib = mb.discover_metallib()
    assert lib is not None, "no cengine.metallib discovered"
    return str(lib)


@pytest.fixture(scope="module")
def runtime(metallib: str) -> "native.MetalRuntime":
    rt = native.MetalRuntime(metallib)
    assert rt.available, f"runtime unavailable: {rt.reason}"
    return rt


# --------------------------------------------------------------------------- init
def test_runtime_initializes_on_device(runtime):
    assert runtime.device_name  # non-empty device name
    assert runtime.reason == ""
    names = runtime.function_names()
    # 45 preserved baseline kernels + the Phase B kernels newly added by the
    # native discovery layer: screen_market (B.1), technical_features (B.3),
    # market_breadth + compact_entry_candidates (B.4).
    assert len(names) == 49, f"expected 49 entry points, got {len(names)}"
    for k in ("screen_market", "technical_features", "market_breadth",
              "compact_entry_candidates"):
        assert k in names


def test_namespace_aware_resolution(runtime):
    # Global indicator kernels resolve by bare name.
    assert runtime.has_kernel("rolling_sma_ticks")
    # gme:: kernels resolve via the namespace fallback.
    assert runtime.has_kernel("covariance_matrix")
    assert runtime.has_kernel("ewma_covariance_matrix")
    assert not runtime.has_kernel("this_kernel_does_not_exist")


def test_unavailable_runtime_reports_reason():
    rt = native.MetalRuntime("/nonexistent/path/cengine.metallib")
    assert rt.available is False
    assert "metallib" in rt.reason.lower() or "cannot load" in rt.reason.lower()


# ------------------------------------------------------------------- indicator parity
def test_sma_exact_cpu_parity(runtime):
    assert runtime.sma(PRICES, 5) == ref.sma_ticks(PRICES, 5)


def test_ema_exact_cpu_parity(runtime):
    assert runtime.ema(PRICES, 2, 10) == ref.ema_ticks(PRICES, 2, 10)


def test_apo_exact_cpu_parity(runtime):
    fast = runtime.ema(PRICES, 3, 10)
    slow = runtime.ema(PRICES, 1, 10)
    assert runtime.apo(fast, slow) == ref.apo_ticks(fast, slow)


def test_true_range_exact_cpu_parity(runtime):
    highs = [p + 2 for p in PRICES]
    lows = [p - 2 for p in PRICES]
    prev = [PRICES[0]] + PRICES[:-1]
    assert runtime.true_range(highs, lows, prev) == ref.true_range_ticks(highs, lows, prev)


def test_indicator_invalid_args_raise(runtime):
    with pytest.raises(native.MetalError):
        runtime.sma([], 5)
    with pytest.raises(native.MetalError):
        runtime.sma(PRICES, 0)
    with pytest.raises(native.MetalError):
        runtime.ema(PRICES, 11, 10)  # numerator > denominator


# ------------------------------------------------------------------ covariance parity
def _random_matrix(obs: int, assets: int, seed: int = 7):
    random.seed(seed)
    return [[random.gauss(0.0, 1.0) for _ in range(assets)] for _ in range(obs)]


def test_covariance_cpu_parity(runtime):
    obs, assets = 24, 5
    mat = _random_matrix(obs, assets)
    flat = [x for row in mat for x in row]
    gpu = runtime.covariance_matrix(flat, obs, assets)
    cpu = [x for row in ref.covariance_matrix(mat) for x in row]
    assert max(abs(a - b) for a, b in zip(gpu, cpu, strict=True)) < 1e-4


def test_ewma_covariance_cpu_parity(runtime):
    obs, assets = 24, 5
    mat = _random_matrix(obs, assets)
    flat = [x for row in mat for x in row]
    gpu = runtime.ewma_covariance_matrix(flat, obs, assets, 0.94, 0.0)
    cpu = [x for row in ref.ewma_covariance_matrix(mat, 0.94) for x in row]
    assert max(abs(a - b) for a, b in zip(gpu, cpu, strict=True)) < 1e-4


# ---------------------------------------------------------- pipeline cache + telemetry
def test_pipeline_cache_and_buffer_reuse(runtime):
    before = runtime.telemetry()
    for _ in range(12):
        runtime.sma(PRICES, 5)
    after = runtime.telemetry()
    # Repeated dispatch of the same kernel must hit the cache, not rebuild.
    assert after.pipeline_cache_hits > before.pipeline_cache_hits
    assert after.dispatches_completed > before.dispatches_completed
    assert after.dispatch_failures == before.dispatch_failures
    # The persistent pool must recycle buffers rather than allocate per call.
    assert after.buffers_reused > before.buffers_reused


def test_telemetry_counts_completions(runtime):
    t = runtime.telemetry()
    assert t.dispatches_completed > 0
    assert t.dispatches_submitted >= t.dispatches_completed
    assert 0.0 <= t.pipeline_cache_hit_rate <= 1.0


# -------------------------------------------------------------------- kernel registry
def test_kernel_registry_reconciles_against_device(runtime):
    reg = native.KernelRegistry()
    reg.reconcile_with_device(runtime)
    cov = reg.coverage()
    # 45 baseline + Phase B screen_market + technical_features (directive §13/§80).
    assert cov.discovered == 49
    assert cov.compiled_callable == 49  # every kernel builds a pipeline on device
    assert cov.host_callable == 11  # 7 baseline + screen/tech/breadth/compaction
    # HONESTY GATE: a hardware-validated kernel is NOT operationally integrated.
    # Nothing is wired to a live consumer yet, so consumer_connected must be 0
    # even though several kernels name an intended consumer (directive §80/§86).
    assert cov.consumer_connected == 0
    assert cov.intended_consumer >= 8
    # JSON serialization is well-formed and reflects the live counts.
    import json

    doc = json.loads(reg.to_json())
    assert doc["discovered"] == 49
    assert len(doc["kernels"]) == 49
    assert doc["consumer_connected"] == 0


def test_operational_consumption_is_explicit(runtime):
    """mark_operationally_consumed is the ONLY way consumer_connected rises."""
    reg = native.KernelRegistry()
    reg.reconcile_with_device(runtime)
    assert reg.coverage().consumer_connected == 0
    reg.mark_operationally_consumed("screen_market")
    assert reg.coverage().consumer_connected == 1
    sm = next(k for k in reg.kernels() if k.kernel_name == "screen_market")
    assert sm.operationally_consumed is True


# ------------------------------------------------- regression parity vs legacy PyObjC
def test_native_matches_legacy_pyobjc_indicators(metallib):
    """The native runtime must agree with the PyObjC path it replaces."""
    legacy = mb.MetalBackend.create(require=False)
    if not legacy.available:
        pytest.skip("legacy PyObjC backend unavailable")
    from cengine.metal_indicators import MetalIndicators
    from cengine.metal_runtime import MetalRuntime as PyMetalRuntime

    py_rt = PyMetalRuntime(metallib)
    py_ind = MetalIndicators(py_rt)
    native_rt = native.MetalRuntime(metallib)
    assert native_rt.sma(PRICES, 5) == py_ind.sma(PRICES, 5)
    assert native_rt.true_range(
        [p + 2 for p in PRICES], [p - 2 for p in PRICES], [PRICES[0]] + PRICES[:-1]
    ) == py_ind.true_range(
        [p + 2 for p in PRICES], [p - 2 for p in PRICES], [PRICES[0]] + PRICES[:-1]
    )


def test_native_matches_legacy_pyobjc_covariance(metallib):
    legacy = mb.MetalBackend.create(require=False)
    if not legacy.available:
        pytest.skip("legacy PyObjC backend unavailable")
    from cengine.metal_covariance import MetalCovariance

    py_cov = MetalCovariance(legacy)
    obs, assets = 20, 4
    mat = _random_matrix(obs, assets, seed=11)
    flat = [x for row in mat for x in row]
    native_rt = native.MetalRuntime(metallib)
    gpu_native = native_rt.covariance_matrix(flat, obs, assets)
    gpu_legacy = [x for row in py_cov.covariance_matrix(mat) for x in row]
    assert max(abs(a - b) for a, b in zip(gpu_native, gpu_legacy, strict=True)) < 1e-4


# -------------------------------------------- Phase B opportunity screening kernel
def _screen_inputs():
    """A mix of entry-qualified, technical-exit, borderline and invalid symbols."""
    SI = native.ScreeningInput
    return [
        # Clean entry: DI+ > DI-, above VWAP, price > prev 1m close.
        SI(price=101.0, previous_close_1m=100.0, di_plus_5m=30.0, di_minus_5m=20.0,
           session_vwap=99.0, sma20_5m=98.0, bb_middle_5m=97.0,
           valid_market=1, valid_direction=1, valid_1m=1, valid_vwap=1,
           valid_sma=1, valid_bb=1),
        # Clean technical exit: DI- > DI+, below VWAP, price < prev 1m close.
        SI(price=97.0, previous_close_1m=100.0, di_plus_5m=18.0, di_minus_5m=29.0,
           session_vwap=99.0, sma20_5m=100.0, bb_middle_5m=101.0,
           valid_market=1, valid_direction=1, valid_1m=1, valid_vwap=1,
           valid_sma=1, valid_bb=1),
        # Bullish direction but below all references -> NOT entry qualified.
        SI(price=95.0, previous_close_1m=96.0, di_plus_5m=30.0, di_minus_5m=20.0,
           session_vwap=99.0, sma20_5m=98.0, bb_middle_5m=97.0,
           valid_market=1, valid_direction=1, valid_1m=1, valid_vwap=1,
           valid_sma=1, valid_bb=1),
        # Missing all reference features -> entry can never qualify; DATA_INVALID.
        SI(price=101.0, previous_close_1m=100.0, di_plus_5m=30.0, di_minus_5m=20.0,
           valid_market=1, valid_direction=1, valid_1m=1,
           valid_vwap=0, valid_sma=0, valid_bb=0),
        # No valid market observation -> DATA_INVALID, nothing qualifies.
        SI(price=101.0, previous_close_1m=100.0, di_plus_5m=30.0, di_minus_5m=20.0,
           session_vwap=99.0, valid_market=0, valid_direction=1, valid_1m=1,
           valid_vwap=1, valid_sma=0, valid_bb=0),
    ]


def test_screen_market_gpu_cpu_exact_parity(runtime):
    """GPU screen_market must match the CPU oracle bit-for-bit on every flag."""
    inputs = _screen_inputs()
    gpu = runtime.screen(inputs)
    cpu = native.screen_market_cpu(inputs)
    assert len(gpu) == len(cpu) == len(inputs)
    for g, c in zip(gpu, cpu, strict=True):
        assert g.flags == c.flags
        # NaN-safe float comparison (invalid refs produce NaN identically).
        for attr in ("momentum_1m_pct", "directional_strength", "vwap_distance_pct"):
            gv, cv = getattr(g, attr), getattr(c, attr)
            if gv != gv or cv != cv:  # NaN
                assert gv != gv and cv != cv
            else:
                assert abs(gv - cv) < 1e-4


def test_screen_market_entry_and_exit_semantics(runtime):
    out = runtime.screen(_screen_inputs())
    EQ = native.SCREEN_ENTRY_QUALIFIED
    TX = native.SCREEN_TECHNICAL_EXIT
    DI = native.SCREEN_DATA_INVALID
    # row 0: entry qualified, not an exit, data valid
    assert out[0].flags & EQ and not (out[0].flags & TX) and not (out[0].flags & DI)
    assert abs(out[0].momentum_1m_pct - 1.0) < 1e-4  # (101/100 - 1) * 100
    # row 1: technical exit, not an entry
    assert out[1].flags & TX and not (out[1].flags & EQ)
    # row 2: bullish but below references -> no entry
    assert not (out[2].flags & EQ)
    # rows 3 & 4: no usable reference / no market -> DATA_INVALID, no entry
    assert out[3].flags & DI and not (out[3].flags & EQ)
    assert out[4].flags & DI and not (out[4].flags & EQ)


def test_screen_market_rejects_empty(runtime):
    with pytest.raises(native.MetalError):
        runtime.screen([])


# ------------------------------------------- Phase B.3 technical feature kernel
def _bar_windows(num_symbols=3, window=40):
    """Deterministic trending/declining/partial bar windows (row-major)."""
    closes, highs, lows, vols, counts = [], [], [], [], []
    for s in range(num_symbols):
        price = 100.0 + s * 50.0
        drift = (0.004, -0.003, 0.001)[s % 3]
        n = window if s < 2 else 25  # third symbol is partial (no ADX yet)
        # tiny deterministic wiggle keyed by (s, i), no RNG
        for i in range(window):
            if i < n:
                price *= 1.0 + drift + 0.0008 * ((i * 7 + s * 3) % 5 - 2)
                closes.append(price)
                highs.append(price * 1.0015)
                lows.append(price * 0.9985)
                vols.append(1000.0 + ((i * 13 + s) % 7) * 50.0)
            else:
                closes.append(0.0)
                highs.append(0.0)
                lows.append(0.0)
                vols.append(0.0)
        counts.append(n)
    return closes, highs, lows, vols, counts, num_symbols, window


def test_technical_features_gpu_cpu_parity(runtime):
    c, h, lo, v, cnt, S, W = _bar_windows()
    gpu = runtime.technical_features(c, h, lo, v, cnt, S, W, 14, 20)
    cpu = native.technical_features_cpu(c, h, lo, v, cnt, S, W, 14, 20)
    assert len(gpu) == len(cpu) == S
    attrs = ("last_close", "prev_close", "sma", "bb_middle", "bb_upper",
             "bb_lower", "di_plus", "di_minus", "adx", "atr", "vwap")
    for g, cc in zip(gpu, cpu, strict=True):
        assert g.valid == cc.valid  # identical validity masks
        for a in attrs:
            # derived multi-step float32: justified tolerance (directive §124)
            assert abs(getattr(g, a) - getattr(cc, a)) < 1e-2


def test_technical_features_directional_semantics(runtime):
    c, h, lo, v, cnt, S, W = _bar_windows()
    out = runtime.technical_features(c, h, lo, v, cnt, S, W, 14, 20)
    # sym0 uptrend -> DI+ dominant; sym1 downtrend -> DI- dominant.
    assert out[0].di_plus > out[0].di_minus
    assert out[1].di_minus > out[1].di_plus
    # Bollinger middle equals SMA (same measurement, reported once; §29).
    for f in out:
        if f.valid & native.TECH_SMA:
            assert abs(f.bb_middle - f.sma) < 1e-4
            assert f.bb_upper >= f.bb_middle >= f.bb_lower


def test_technical_features_valid_mask_respects_bar_depth(runtime):
    # 10 bars: enough for DI (>=15? no) — choose depths that toggle each bit.
    c, h, lo, v, cnt, S, W = _bar_windows(num_symbols=3, window=40)
    out = runtime.technical_features(c, h, lo, v, cnt, S, W, 14, 20)
    # Symbol 2 has only 25 bars: DI valid (>=15) but ADX not (needs >=29).
    assert out[2].valid & native.TECH_DMI
    assert not (out[2].valid & native.TECH_ADX)
    # Symbols 0/1 have 40 bars: ADX valid.
    assert out[0].valid & native.TECH_ADX
    assert out[1].valid & native.TECH_ADX


def test_technical_features_rejects_bad_dims(runtime):
    with pytest.raises(native.MetalError):
        runtime.technical_features([], [], [], [], [], 0, 0)


# ------------------------------------------- Phase B.4 breadth + compaction kernels
def _flag_sample():
    EQ = native.SCREEN_ENTRY_QUALIFIED
    TX = native.SCREEN_TECHNICAL_EXIT
    BULL = native.SCREEN_BULLISH_DIRECTION
    BEAR = native.SCREEN_BEARISH_DIRECTION
    AV = native.SCREEN_ABOVE_VWAP
    BV = native.SCREEN_BELOW_VWAP
    PM = native.SCREEN_POSITIVE_MOMENTUM
    NM = native.SCREEN_NEGATIVE_MOMENTUM
    DI = native.SCREEN_DATA_INVALID
    return [EQ | BULL | AV | PM, TX | BEAR | BV | NM, BULL | AV, DI,
            EQ | BULL | AV | PM, DI, TX | BEAR | BV | NM, BULL | AV | PM]


def test_market_breadth_gpu_cpu_parity(runtime):
    flags = _flag_sample()
    g = runtime.market_breadth(flags)
    c = native.market_breadth_cpu(flags)
    for a in ("total", "valid", "entry_qualified", "technical_exit", "bullish",
              "bearish", "positive_momentum", "negative_momentum", "above_vwap",
              "below_vwap", "data_invalid"):
        assert getattr(g, a) == getattr(c, a), a
    assert g.total == len(flags)
    assert g.valid + g.data_invalid == g.total
    assert g.entry_qualified == 2 and g.technical_exit == 2 and g.data_invalid == 2


def test_compact_entry_candidates_parity_and_overflow(runtime):
    flags = _flag_sample()
    ids, total = runtime.compact_entry_candidates(flags, 10)
    cids, ctotal = native.compact_entry_candidates_cpu(flags, 10)
    assert ids == cids == [0, 4]     # sorted indices of ENTRY_QUALIFIED
    assert total == ctotal == 2
    # Overflow: capacity 1 keeps only 1 id but reports the true total (=2).
    ov_ids, ov_total = runtime.compact_entry_candidates(flags, 1)
    assert len(ov_ids) == 1 and ov_total == 2  # total > capacity signals overflow


def test_breadth_rejects_empty(runtime):
    with pytest.raises(native.MetalError):
        runtime.market_breadth([])
    with pytest.raises(native.MetalError):
        runtime.compact_entry_candidates([], 10)
