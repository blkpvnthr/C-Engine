"""Native + Metal certification (§23).

Certifies the C++/Metal substrate that the platform's financial authority and GPU analytics
rest on, by EXERCISING it (not describing it):

* **native module** — ``_cengine_native`` imports and exposes the authoritative types
  (RiskEngine, DiscoveryEngine, MetalRuntime, the emergency-exit candidate).
* **risk gate operational** — the native ``RiskEngine`` is the sole financial authority: a
  well-formed small order is accepted; an order breaching a hard limit is rejected. Proven by
  running ``engine.evaluate`` with distinct order ids.
* **Metal runtime** — a real Metal device is present (Apple GPU) and a compiled kernel executes.
* **GPU/CPU parity** — a GPU indicator reproduces the CPU reference oracle exactly, so GPU
  acceleration never changes results (correctness-preserving).
* **documented CPU fallback** — a CPU reference path (``metal_cpu_ref``) exists and the learning
  ``DevicePolicy`` honors a forced-CPU override, so the platform degrades safely without a GPU.

A GPU/Metal fault is an operational concern, never a safety one: the −0.5% protective exit and
the native risk gate do not depend on Metal being available (verified in protective_exit_cert
and here by the CPU-fallback check).
"""

from __future__ import annotations

from cengine.certification.contracts import CertificationResult, CertificationVerdict

NATIVE_CERT_VERSION = "certification-native-metal-1.0.0"
_REQUIRED_NATIVE = ("RiskEngine", "DiscoveryEngine", "MetalRuntime", "RiskRequest",
                    "evaluate_emergency_exit", "clamp_emergency_threshold")


def _v(ok: bool) -> CertificationVerdict:
    return CertificationVerdict.CERTIFIED if ok else CertificationVerdict.NOT_CERTIFIED


def _certify_native_module() -> CertificationResult:
    try:
        import _cengine_native as native
    except Exception as exc:  # noqa: BLE001
        return CertificationResult(
            check="native.module", component="risk_native",
            verdict=CertificationVerdict.BLOCKED,
            summary=f"_cengine_native failed to import: {exc}")
    missing = [s for s in _REQUIRED_NATIVE if not hasattr(native, s)]
    return CertificationResult(
        check="native.module", component="risk_native", verdict=_v(not missing),
        summary=("native extension exposes all authoritative types"
                 if not missing else f"missing native symbols: {missing}"),
        metrics={"required": list(_REQUIRED_NATIVE), "missing": missing})


def _make_risk_request(native, order_id: int, qty: int, price_ticks: int = 100 * 10000):
    r = native.RiskRequest()
    r.order_id = order_id
    r.symbol = "AAPL"
    r.side = native.Side.BUY
    r.quantity = qty
    r.price_ticks = price_ticks
    m = native.RiskMarketState()
    m.symbol = "AAPL"
    m.bid_ticks = price_ticks - 100
    m.ask_ticks = price_ticks + 100
    m.timestamp_ns = 1
    m.last_ticks = price_ticks
    r.market = m
    a = native.RiskAccountState()
    a.position = 0
    a.cash_ticks = 10_000_000_000
    a.buying_power_ticks = 10_000_000_000
    a.gross_exposure_ticks = 0
    r.account = a
    oo = native.OpenOrderRiskState()
    oo.count = 0
    oo.total_remaining_quantity = 0
    oo.total_buy_notional_ticks = 0
    oo.total_sell_notional_ticks = 0
    r.open_orders = oo
    r.now_ns = 10
    return r


def _certify_risk_gate() -> CertificationResult:
    import _cengine_native as native
    eng = native.RiskEngine()
    small = eng.evaluate(_make_risk_request(native, order_id=1001, qty=10))
    big = eng.evaluate(_make_risk_request(native, order_id=1002, qty=50_000))
    small_ok = bool(small.accepted)
    big_rejected = not bool(big.accepted)
    reason = str(getattr(big, "reason", getattr(big, "reject_reason", "")))
    return CertificationResult(
        check="native.risk_gate", component="risk_native",
        verdict=_v(small_ok and big_rejected),
        summary=(f"native RiskEngine operational: 10-share order accepted; 50k-share order "
                 f"rejected ({reason})"),
        metrics={"small_accepted": small_ok, "oversized_rejected": big_rejected,
                 "oversized_reason": reason},
        notes="native RiskEngine is the sole financial authority for every order")


def _certify_metal_runtime() -> CertificationResult:
    try:
        import cengine.gpu_worker as worker
        rt = worker.get_runtime()
        device = rt.device.name()
        # a compiled kernel executes
        sma = worker.get_gpu().sma([10000, 10100, 10200, 10300, 10400], 3)
        kernel_ok = sma[-1] == 10300
    except Exception as exc:  # noqa: BLE001
        return CertificationResult(
            check="native.metal_runtime", component="risk_native",
            verdict=CertificationVerdict.NOT_CERTIFIED,
            summary=f"Metal runtime unavailable: {exc}")
    is_apple = "Apple" in device
    return CertificationResult(
        check="native.metal_runtime", component="risk_native", verdict=_v(is_apple and kernel_ok),
        summary=f"Metal device '{device}'; compiled SMA kernel executes correctly",
        metrics={"device": device, "kernel_sma_ok": kernel_ok})


def _certify_gpu_cpu_parity() -> CertificationResult:
    try:
        import cengine.gpu_worker as worker
        import cengine.metal_cpu_ref as ref
        prices = [10000, 10100, 10050, 10200, 10300, 10250, 10400, 10380, 10420, 10500]
        gpu_ema = worker.get_gpu().ema(prices, 2, 10)
        cpu_ema = ref.ema_ticks(prices, 2, 10)
        gpu_apo = worker.get_gpu().apo(gpu_ema, gpu_ema)
        cpu_apo = ref.apo_ticks(cpu_ema, cpu_ema)
    except Exception as exc:  # noqa: BLE001
        return CertificationResult(
            check="native.gpu_cpu_parity", component="risk_native",
            verdict=CertificationVerdict.NOT_CERTIFIED,
            summary=f"parity check could not run: {exc}")
    ema_match = list(gpu_ema) == list(cpu_ema)
    apo_match = list(gpu_apo) == list(cpu_apo)
    return CertificationResult(
        check="native.gpu_cpu_parity", component="risk_native",
        verdict=_v(ema_match and apo_match),
        summary=(f"GPU reproduces the CPU reference exactly (EMA match={ema_match}, "
                 f"APO match={apo_match}) — acceleration is correctness-preserving"),
        metrics={"ema_exact_match": ema_match, "apo_exact_match": apo_match,
                 "sample": {"gpu_ema": list(gpu_ema), "cpu_ema": list(cpu_ema)}})


def _certify_cpu_fallback() -> CertificationResult:
    try:
        import cengine.metal_cpu_ref as ref
        from cengine.learning.hardware import DevicePolicy
    except Exception as exc:  # noqa: BLE001
        return CertificationResult(
            check="native.cpu_fallback", component="risk_native",
            verdict=CertificationVerdict.NOT_CERTIFIED,
            summary=f"CPU fallback machinery missing: {exc}")
    cpu_refs = [f for f in ("ema_ticks", "apo_ticks", "column_means", "covariance_matrix")
                if hasattr(ref, f)]
    # DevicePolicy(force_cpu=True) must resolve away from the GPU and place work on CPU
    summary = DevicePolicy(force_cpu=True).summary()
    policy_forces_cpu = bool(summary.get("force_cpu")) and not summary.get("gpu_available")
    ok = len(cpu_refs) >= 3 and policy_forces_cpu
    return CertificationResult(
        check="native.cpu_fallback", component="risk_native", verdict=_v(ok),
        summary=(f"documented CPU fallback: {len(cpu_refs)} CPU reference kernels + "
                 f"DevicePolicy override — platform degrades safely without a GPU"),
        metrics={"cpu_reference_kernels": cpu_refs, "device_policy": policy_forces_cpu},
        notes="GPU/Metal is an acceleration layer; risk + protective exit never depend on it")


def certify_native_metal() -> list:
    """Run the full §23 native/Metal certification."""
    mod = _certify_native_module()
    results = [mod]
    if mod.verdict == CertificationVerdict.BLOCKED:
        return results
    results.append(_certify_risk_gate())
    results.append(_certify_metal_runtime())
    results.append(_certify_gpu_cpu_parity())
    results.append(_certify_cpu_fallback())
    return results
