// ============================================================================
// discovery_bindings.cpp
// ----------------------------------------------------------------------------
// pybind11 surface for the native Objective-C++ Metal runtime + kernel registry.
// This is a plain C++ translation unit: the PIMPL headers keep all Objective-C /
// Metal types out of the binding, so only metal_runtime.mm / metal_buffers.mm
// pull in the frameworks. The GIL is released around on-device dispatch so the
// Python supervisor is never blocked by GPU work (directive §14 boundary rules).
//
// On non-Apple builds (no Metal), register_discovery() is a documented no-op so
// the single _cengine_native module still links everywhere.
// ============================================================================
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

#if defined(CENGINE_WITH_METAL)

#include <memory>
#include <string>
#include <vector>

#include "opportunity_discovery/kernel_registry.hpp"
#include "opportunity_discovery/metal_runtime.hpp"
#include "opportunity_discovery/screening_types.hpp"

using namespace trading::metal;

void register_discovery(py::module_& m) {
    py::register_exception<MetalError>(m, "MetalError");

    py::class_<RuntimeTelemetry>(m, "RuntimeTelemetry")
        .def_readonly("dispatches_submitted", &RuntimeTelemetry::dispatches_submitted)
        .def_readonly("dispatches_completed", &RuntimeTelemetry::dispatches_completed)
        .def_readonly("dispatch_failures", &RuntimeTelemetry::dispatch_failures)
        .def_readonly("pipeline_cache_hits", &RuntimeTelemetry::pipeline_cache_hits)
        .def_readonly("pipeline_cache_misses", &RuntimeTelemetry::pipeline_cache_misses)
        .def_readonly("buffers_allocated", &RuntimeTelemetry::buffers_allocated)
        .def_readonly("buffers_reused", &RuntimeTelemetry::buffers_reused)
        .def_readonly("bytes_uploaded", &RuntimeTelemetry::bytes_uploaded)
        .def_readonly("bytes_downloaded", &RuntimeTelemetry::bytes_downloaded)
        .def_readonly("cpu_fallbacks", &RuntimeTelemetry::cpu_fallbacks)
        .def_readonly("last_dispatch_duration_ns", &RuntimeTelemetry::last_dispatch_duration_ns)
        .def_readonly("total_dispatch_duration_ns", &RuntimeTelemetry::total_dispatch_duration_ns)
        .def_property_readonly("pipeline_cache_hit_rate", &RuntimeTelemetry::pipeline_cache_hit_rate)
        .def_property_readonly("mean_dispatch_ms", &RuntimeTelemetry::mean_dispatch_ms);

    py::class_<MetalRuntime>(m, "MetalRuntime")
        .def(py::init([](const std::string& metallib_path) {
                 return MetalRuntime::create(metallib_path);
             }),
             py::arg("metallib_path"))
        .def_property_readonly("available", &MetalRuntime::available)
        .def_property_readonly("reason", &MetalRuntime::reason)
        .def_property_readonly("device_name", &MetalRuntime::device_name)
        .def_property_readonly("library_path", &MetalRuntime::library_path)
        .def_property_readonly("cached_pipeline_count", &MetalRuntime::cached_pipeline_count)
        .def("function_names", &MetalRuntime::function_names)
        .def("has_kernel", &MetalRuntime::has_kernel, py::arg("name"))
        .def("ensure_pipeline", &MetalRuntime::ensure_pipeline, py::arg("kernel"),
             py::call_guard<py::gil_scoped_release>())
        .def("telemetry", &MetalRuntime::telemetry)
        .def("sma", &MetalRuntime::sma, py::arg("prices"), py::arg("window"),
             py::call_guard<py::gil_scoped_release>())
        .def("ema", &MetalRuntime::ema, py::arg("prices"),
             py::arg("alpha_numerator"), py::arg("alpha_denominator"),
             py::call_guard<py::gil_scoped_release>())
        .def("apo", &MetalRuntime::apo, py::arg("fast_ema"), py::arg("slow_ema"),
             py::call_guard<py::gil_scoped_release>())
        .def("true_range", &MetalRuntime::true_range, py::arg("highs"),
             py::arg("lows"), py::arg("previous_closes"),
             py::call_guard<py::gil_scoped_release>())
        .def("column_means", &MetalRuntime::column_means, py::arg("values"),
             py::arg("observations"), py::arg("assets"),
             py::call_guard<py::gil_scoped_release>())
        .def("covariance_matrix", &MetalRuntime::covariance_matrix,
             py::arg("values"), py::arg("observations"), py::arg("assets"),
             py::call_guard<py::gil_scoped_release>())
        .def("ewma_covariance_matrix", &MetalRuntime::ewma_covariance_matrix,
             py::arg("values"), py::arg("observations"), py::arg("assets"),
             py::arg("lambda_"), py::arg("minimum_variance") = 0.0f,
             py::call_guard<py::gil_scoped_release>());

    py::class_<KernelInfo>(m, "KernelInfo")
        .def_readonly("shader_file", &KernelInfo::shader_file)
        .def_readonly("kernel_name", &KernelInfo::kernel_name)
        .def_readonly("resolved_name", &KernelInfo::resolved_name)
        .def_readonly("host_interface", &KernelInfo::host_interface)
        .def_readonly("buffer_contract", &KernelInfo::buffer_contract)
        .def_readonly("consumer", &KernelInfo::consumer)
        .def_readonly("required_data", &KernelInfo::required_data)
        .def_readonly("blocker", &KernelInfo::blocker)
        .def_readonly("pipeline_cached", &KernelInfo::pipeline_cached)
        .def_readonly("hardware_executed", &KernelInfo::hardware_executed)
        .def_readonly("numerically_validated", &KernelInfo::numerically_validated)
        .def_readonly("benchmarked", &KernelInfo::benchmarked);

    py::class_<KernelRegistry::Coverage>(m, "KernelCoverage")
        .def_readonly("discovered", &KernelRegistry::Coverage::discovered)
        .def_readonly("compiled_callable", &KernelRegistry::Coverage::compiled_callable)
        .def_readonly("host_callable", &KernelRegistry::Coverage::host_callable)
        .def_readonly("consumer_connected", &KernelRegistry::Coverage::consumer_connected)
        .def_readonly("hardware_executed", &KernelRegistry::Coverage::hardware_executed)
        .def_readonly("numerically_validated", &KernelRegistry::Coverage::numerically_validated)
        .def_readonly("benchmarked", &KernelRegistry::Coverage::benchmarked)
        .def_readonly("blocked", &KernelRegistry::Coverage::blocked)
        .def_readonly("pipeline_coverage_pct", &KernelRegistry::Coverage::pipeline_coverage_pct)
        .def_readonly("operational_coverage_pct", &KernelRegistry::Coverage::operational_coverage_pct);

    py::class_<KernelRegistry>(m, "KernelRegistry")
        .def(py::init<>())
        .def("kernels", &KernelRegistry::kernels,
             py::return_value_policy::reference_internal)
        .def("reconcile_with_device", &KernelRegistry::reconcile_with_device,
             py::arg("runtime"), py::call_guard<py::gil_scoped_release>())
        .def("mark_hardware_executed", &KernelRegistry::mark_hardware_executed)
        .def("mark_numerically_validated", &KernelRegistry::mark_numerically_validated)
        .def("mark_benchmarked", &KernelRegistry::mark_benchmarked)
        .def("coverage", &KernelRegistry::coverage)
        .def("to_json", &KernelRegistry::to_json);

    m.attr("metal_available") = true;
}

#else  // !CENGINE_WITH_METAL

void register_discovery(py::module_& m) {
    // Non-Apple / Metal-disabled build: expose the flag so callers can detect
    // the native runtime is absent and fall back to the CPU path.
    m.attr("metal_available") = false;
}

#endif
