"""Failure-injection certification (§24).

Injects faults and verifies the platform ISOLATES them without ever compromising financial
protection or trading authority. Each check exercises the REAL code paths:

* **learning/analytics worker fault** — the :class:`RuntimeSupervisor` isolates a non-essential
  worker failure; essential (P0 protection / P1 execution) workers are never auto-killed and the
  report confirms protection is unaffected.
* **GPU/Metal unavailable** — a forced-CPU device policy resolves without raising, so a GPU fault
  degrades to CPU rather than halting.
* **corrupt / empty lakehouse** — the data-integrity and strategy certifications degrade to
  INSUFFICIENT_DATA rather than crashing when the lakehouse is missing.
* **broker / risk rejection isolated** — the protective-exit coordinator's one-pass scan never
  raises even when the order path throws for one symbol (per-symbol isolation).
* **single-instance guard** — a second trading instance cannot acquire the runtime lock while a
  live one holds it (no competing financial authority).
* **durable checkpoint** — an atomic checkpoint write survives a round-trip (crash recovery).

The through-line: faults in learning, discovery, GPU or data NEVER cascade into the protection
or execution path. This is the operational-robustness complement to the economic certification.
"""

from __future__ import annotations

import asyncio
import os
import tempfile

from cengine.certification.contracts import CertificationResult, CertificationVerdict

FAILURE_INJECTION_VERSION = "certification-failure-injection-1.0.0"


def _v(ok: bool) -> CertificationVerdict:
    return CertificationVerdict.CERTIFIED if ok else CertificationVerdict.NOT_CERTIFIED


def _certify_worker_isolation() -> CertificationResult:
    from cengine.runtime.supervisor import Priority, RuntimeSupervisor
    sup = RuntimeSupervisor()
    sup.register("protection", Priority.P0_PROTECTION, essential=True)
    sup.register("execution", Priority.P1_MARKET_EXEC, essential=True)
    sup.register("ml_training", Priority.P6_ML_TRAINING, essential=False)
    # inject a learning fault
    rep = sup.record_failure("ml_training", "mlx kernel panic")
    # injecting on an essential worker must NOT report it as isolated-killed
    ess = sup.record_failure("protection", "transient")
    isolated_learning = rep.get("isolated") is True
    protection_untouched = (rep.get("financial_protection_affected") is False
                            and ess.get("isolated") is False)
    health = sup.health()
    independent = health.get("protection_independent_of_learning") is True
    ok = isolated_learning and protection_untouched and independent
    return CertificationResult(
        check="failure_injection.worker_isolation", component="continuous_runtime",
        verdict=_v(ok),
        summary=("learning fault isolated; essential protection/execution workers never "
                 "auto-killed; protection independent of learning"),
        metrics={"learning_isolated": isolated_learning,
                 "protection_untouched": protection_untouched,
                 "protection_independent": independent})


def _certify_gpu_unavailable_degrades() -> CertificationResult:
    # force CPU and confirm the device policy resolves without raising
    from cengine.learning.hardware import DevicePolicy
    # force CPU via the real DevicePolicy API and confirm it resolves away from the GPU
    summary = DevicePolicy(force_cpu=True).summary()
    policy_cpu = bool(summary.get("force_cpu")) and not summary.get("gpu_available")
    # and confirm the CPU reference kernels still compute without a GPU
    import cengine.metal_cpu_ref as ref
    cpu_ok = ref.ema_ticks([10000, 10100, 10200], 2, 10) is not None
    ok = cpu_ok and policy_cpu
    return CertificationResult(
        check="failure_injection.gpu_unavailable", component="risk_native", verdict=_v(ok),
        summary="forced-CPU policy resolves off-GPU and CPU kernels compute (safe degradation)",
        metrics={"cpu_compute_ok": cpu_ok, "device_policy_forces_cpu": policy_cpu})


def _certify_empty_lakehouse_degrades() -> CertificationResult:
    from cengine.certification import data_integrity as di
    from cengine.certification.datasource import BarInventory
    orig = di.bar_inventory
    di.bar_inventory = lambda: BarInventory(available=False, glob="/nonexistent/*.parquet")
    try:
        results = di.run_data_integrity_certification()
        degraded = (len(results) == 1
                    and results[0].verdict == CertificationVerdict.INSUFFICIENT_DATA)
    finally:
        di.bar_inventory = orig
    return CertificationResult(
        check="failure_injection.empty_lakehouse", component="lakehouse", verdict=_v(degraded),
        summary="missing lakehouse degrades certification to INSUFFICIENT_DATA (no crash)",
        metrics={"degraded_gracefully": degraded})


def _certify_protective_exit_isolation() -> CertificationResult:
    """The protective-exit scan must never raise even when the order path throws per-symbol."""
    from protective_exit import ProtectiveExitCoordinator

    class _Pos:
        def __init__(self, symbol, qty, cost_ticks):
            self.symbol = symbol
            self.quantity = qty
            self.cost_ticks = cost_ticks

    class _Positions:
        def positions(self):
            return [_Pos("AAPL", 10, 10 * 100 * 10000)]  # avg 100, breaching below

    class _Prices:
        def price_ticks(self, sym):
            return 99.0 * 10000      # −1.0% → breaches −0.5%

    class _ExplodingOrders:
        async def submit(self, intent):
            raise RuntimeError("injected broker/venue failure")

    class _State:
        def can_submit(self, sid, sym):
            return True

    class _Reservations:
        def projected_position(self, sym, qty):
            return qty

    coord = ProtectiveExitCoordinator(
        order_manager=_ExplodingOrders(), positions=_Positions(), price_source=_Prices(),
        reservations=_Reservations(), strategy_state=_State(), threshold=-0.005,
        market_open_fn=lambda: True)
    try:
        submitted = asyncio.run(coord.scan_and_exit())
        did_not_raise = True
    except Exception:  # noqa: BLE001
        did_not_raise = False
        submitted = None
    # the injected failure must be swallowed per-symbol (no submission recorded, no raise)
    ok = did_not_raise and submitted == []
    return CertificationResult(
        check="failure_injection.protective_exit_isolation", component="protective_exit",
        verdict=_v(ok),
        summary=("protective-exit scan isolates a per-symbol broker/venue failure and never "
                 "raises (the monitor must never die)"),
        metrics={"did_not_raise": did_not_raise, "submitted": submitted})


def _certify_single_instance_guard() -> CertificationResult:
    from cengine.runtime.supervisor import SingleInstanceError, SingleInstanceLock
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "trade.lock")
        a = SingleInstanceLock(path)
        a.acquire()
        b = SingleInstanceLock(path)
        blocked = False
        try:
            b.acquire()
        except SingleInstanceError:
            blocked = True
        a.release()
        # after release a fresh instance can acquire
        c = SingleInstanceLock(path)
        reacquired = c.acquire()
        c.release()
    ok = blocked and reacquired
    return CertificationResult(
        check="failure_injection.single_instance", component="continuous_runtime", verdict=_v(ok),
        summary="second trade instance blocked while lock held; lock reusable after release",
        metrics={"second_blocked": blocked, "reacquired_after_release": reacquired})


def _certify_durable_checkpoint() -> CertificationResult:
    from cengine.runtime.supervisor import read_checkpoint, write_checkpoint
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "state", "checkpoint.json")
        state = {"last_state": "regular_session", "positions": 3, "ns": 123456789}
        write_checkpoint(path, state)
        back = read_checkpoint(path)
        missing = read_checkpoint(os.path.join(d, "none.json"))
    ok = back == state and missing is None
    return CertificationResult(
        check="failure_injection.durable_checkpoint", component="continuous_runtime",
        verdict=_v(ok),
        summary="atomic checkpoint survives a write/read round-trip; missing file -> None",
        metrics={"roundtrip_ok": back == state, "missing_returns_none": missing is None})


def certify_failure_injection() -> list:
    """Run the full §24 failure-injection certification."""
    return [
        _certify_worker_isolation(),
        _certify_gpu_unavailable_degrades(),
        _certify_empty_lakehouse_degrades(),
        _certify_protective_exit_isolation(),
        _certify_single_instance_guard(),
        _certify_durable_checkpoint(),
    ]
