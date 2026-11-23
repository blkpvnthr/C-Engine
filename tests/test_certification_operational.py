"""Tests for the operational-robustness certification: native/Metal (§23) + failure injection (§24).

These exercise the real native extension, Metal runtime, supervisor, protective-exit coordinator,
single-instance lock and checkpoint — so they are genuine, not mocked. They run fast.
"""

from __future__ import annotations

from cengine.certification.contracts import CertificationVerdict
from cengine.certification.failure_injection import certify_failure_injection
from cengine.certification.native_cert import certify_native_metal


def test_native_metal_all_certified():
    results = certify_native_metal()
    by = {r.check: r for r in results}
    assert by["native.module"].verdict == CertificationVerdict.CERTIFIED
    assert by["native.risk_gate"].verdict == CertificationVerdict.CERTIFIED
    # the risk gate must reject the oversized order for a genuine limit reason
    assert "LIMIT" in by["native.risk_gate"].metrics["oversized_reason"].upper()
    assert by["native.metal_runtime"].verdict == CertificationVerdict.CERTIFIED
    assert by["native.gpu_cpu_parity"].verdict == CertificationVerdict.CERTIFIED
    assert by["native.gpu_cpu_parity"].metrics["ema_exact_match"] is True
    assert by["native.cpu_fallback"].verdict == CertificationVerdict.CERTIFIED


def test_failure_injection_all_certified():
    results = certify_failure_injection()
    checks = {r.check: r.verdict for r in results}
    for key in ("worker_isolation", "gpu_unavailable", "empty_lakehouse",
                "protective_exit_isolation", "single_instance", "durable_checkpoint"):
        assert checks[f"failure_injection.{key}"] == CertificationVerdict.CERTIFIED, key


def test_protective_exit_isolation_swallows_broker_failure():
    results = {r.check: r for r in certify_failure_injection()}
    r = results["failure_injection.protective_exit_isolation"]
    assert r.metrics["did_not_raise"] is True
    assert r.metrics["submitted"] == []   # injected failure isolated, no submission, no crash


def test_worker_isolation_keeps_protection_independent():
    results = {r.check: r for r in certify_failure_injection()}
    r = results["failure_injection.worker_isolation"]
    assert r.metrics["learning_isolated"] is True
    assert r.metrics["protection_untouched"] is True
    assert r.metrics["protection_independent"] is True
