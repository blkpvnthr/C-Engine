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

#include "opportunity_discovery/cpu_reference.hpp"
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

    // --- opportunity screening contract (Phase B) --------------------------
    // Bitmask flag values (mirror ScreenFlag in screening_types.hpp /
    // opportunity_screening.metal). Exposed as plain ints so Python can test
    // `output.flags & native.SCREEN_ENTRY_QUALIFIED` ergonomically.
    m.attr("SCREEN_ENTRY_QUALIFIED") = static_cast<std::uint32_t>(kEntryQualified);
    m.attr("SCREEN_TECHNICAL_EXIT") = static_cast<std::uint32_t>(kTechnicalExit);
    m.attr("SCREEN_BULLISH_DIRECTION") = static_cast<std::uint32_t>(kBullishDirection);
    m.attr("SCREEN_BEARISH_DIRECTION") = static_cast<std::uint32_t>(kBearishDirection);
    m.attr("SCREEN_POSITIVE_MOMENTUM") = static_cast<std::uint32_t>(kPositiveMomentum);
    m.attr("SCREEN_NEGATIVE_MOMENTUM") = static_cast<std::uint32_t>(kNegativeMomentum);
    m.attr("SCREEN_ABOVE_VWAP") = static_cast<std::uint32_t>(kAboveVwap);
    m.attr("SCREEN_BELOW_VWAP") = static_cast<std::uint32_t>(kBelowVwap);
    m.attr("SCREEN_DATA_INVALID") = static_cast<std::uint32_t>(kDataInvalid);

    py::class_<ScreeningInput>(m, "ScreeningInput")
        .def(py::init<>())
        .def(py::init([](float price, float previous_close_1m, float di_plus_5m,
                         float di_minus_5m, float session_vwap, float sma20_5m,
                         float bb_middle_5m, std::uint32_t valid_market,
                         std::uint32_t valid_direction, std::uint32_t valid_1m,
                         std::uint32_t valid_vwap, std::uint32_t valid_sma,
                         std::uint32_t valid_bb) {
                 ScreeningInput s;
                 s.price = price;
                 s.previous_close_1m = previous_close_1m;
                 s.di_plus_5m = di_plus_5m;
                 s.di_minus_5m = di_minus_5m;
                 s.session_vwap = session_vwap;
                 s.sma20_5m = sma20_5m;
                 s.bb_middle_5m = bb_middle_5m;
                 s.valid_market = valid_market;
                 s.valid_direction = valid_direction;
                 s.valid_1m = valid_1m;
                 s.valid_vwap = valid_vwap;
                 s.valid_sma = valid_sma;
                 s.valid_bb = valid_bb;
                 return s;
             }),
             py::arg("price") = 0.0f, py::arg("previous_close_1m") = 0.0f,
             py::arg("di_plus_5m") = 0.0f, py::arg("di_minus_5m") = 0.0f,
             py::arg("session_vwap") = 0.0f, py::arg("sma20_5m") = 0.0f,
             py::arg("bb_middle_5m") = 0.0f, py::arg("valid_market") = 0u,
             py::arg("valid_direction") = 0u, py::arg("valid_1m") = 0u,
             py::arg("valid_vwap") = 0u, py::arg("valid_sma") = 0u,
             py::arg("valid_bb") = 0u)
        .def_readwrite("price", &ScreeningInput::price)
        .def_readwrite("previous_close_1m", &ScreeningInput::previous_close_1m)
        .def_readwrite("di_plus_5m", &ScreeningInput::di_plus_5m)
        .def_readwrite("di_minus_5m", &ScreeningInput::di_minus_5m)
        .def_readwrite("session_vwap", &ScreeningInput::session_vwap)
        .def_readwrite("sma20_5m", &ScreeningInput::sma20_5m)
        .def_readwrite("bb_middle_5m", &ScreeningInput::bb_middle_5m)
        .def_readwrite("valid_market", &ScreeningInput::valid_market)
        .def_readwrite("valid_direction", &ScreeningInput::valid_direction)
        .def_readwrite("valid_1m", &ScreeningInput::valid_1m)
        .def_readwrite("valid_vwap", &ScreeningInput::valid_vwap)
        .def_readwrite("valid_sma", &ScreeningInput::valid_sma)
        .def_readwrite("valid_bb", &ScreeningInput::valid_bb);

    py::class_<ScreeningOutput>(m, "ScreeningOutput")
        .def_readonly("flags", &ScreeningOutput::flags)
        .def_readonly("momentum_1m_pct", &ScreeningOutput::momentum_1m_pct)
        .def_readonly("directional_strength", &ScreeningOutput::directional_strength)
        .def_readonly("vwap_distance_pct", &ScreeningOutput::vwap_distance_pct);

    // Native CPU oracle / fallback for the screening contract. Parity tests
    // compare this against MetalRuntime.screen on real hardware.
    m.def("screen_market_cpu", &cpu_ref::screen_market, py::arg("inputs"));

    // --- technical features (Phase B.3) ------------------------------------
    m.attr("TECH_CLOSE") = static_cast<std::uint32_t>(kTechClose);
    m.attr("TECH_PREV_CLOSE") = static_cast<std::uint32_t>(kTechPrevClose);
    m.attr("TECH_SMA") = static_cast<std::uint32_t>(kTechSma);
    m.attr("TECH_BB") = static_cast<std::uint32_t>(kTechBb);
    m.attr("TECH_DMI") = static_cast<std::uint32_t>(kTechDmi);
    m.attr("TECH_ADX") = static_cast<std::uint32_t>(kTechAdx);
    m.attr("TECH_VWAP") = static_cast<std::uint32_t>(kTechVwap);

    py::class_<TechFeatures>(m, "TechFeatures")
        .def_readonly("last_close", &TechFeatures::last_close)
        .def_readonly("prev_close", &TechFeatures::prev_close)
        .def_readonly("sma", &TechFeatures::sma)
        .def_readonly("bb_middle", &TechFeatures::bb_middle)
        .def_readonly("bb_upper", &TechFeatures::bb_upper)
        .def_readonly("bb_lower", &TechFeatures::bb_lower)
        .def_readonly("di_plus", &TechFeatures::di_plus)
        .def_readonly("di_minus", &TechFeatures::di_minus)
        .def_readonly("adx", &TechFeatures::adx)
        .def_readonly("atr", &TechFeatures::atr)
        .def_readonly("vwap", &TechFeatures::vwap)
        .def_readonly("valid", &TechFeatures::valid);

    // CPU oracle / fallback for the technical-feature pipeline.
    m.def("technical_features_cpu",
          [](const std::vector<float>& closes, const std::vector<float>& highs,
             const std::vector<float>& lows, const std::vector<float>& volumes,
             const std::vector<std::uint32_t>& counts, std::uint32_t symbol_count,
             std::uint32_t max_window, std::uint32_t dmi_period,
             std::uint32_t sma_period) {
              TechWindowParams p{symbol_count, max_window, dmi_period, sma_period};
              return cpu_ref::technical_features(closes, highs, lows, volumes,
                                                 counts, p);
          },
          py::arg("closes"), py::arg("highs"), py::arg("lows"), py::arg("volumes"),
          py::arg("counts"), py::arg("symbol_count"), py::arg("max_window"),
          py::arg("dmi_period") = 14, py::arg("sma_period") = 20);

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
             py::call_guard<py::gil_scoped_release>())
        .def("screen", &MetalRuntime::screen, py::arg("inputs"),
             py::call_guard<py::gil_scoped_release>())
        .def("technical_features", &MetalRuntime::technical_features,
             py::arg("closes"), py::arg("highs"), py::arg("lows"),
             py::arg("volumes"), py::arg("counts"), py::arg("symbol_count"),
             py::arg("max_window"), py::arg("dmi_period") = 14,
             py::arg("sma_period") = 20, py::call_guard<py::gil_scoped_release>());

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
        .def_readonly("benchmarked", &KernelInfo::benchmarked)
        .def_readonly("operationally_consumed", &KernelInfo::operationally_consumed);

    py::class_<KernelRegistry::Coverage>(m, "KernelCoverage")
        .def_readonly("discovered", &KernelRegistry::Coverage::discovered)
        .def_readonly("compiled_callable", &KernelRegistry::Coverage::compiled_callable)
        .def_readonly("host_callable", &KernelRegistry::Coverage::host_callable)
        .def_readonly("consumer_connected", &KernelRegistry::Coverage::consumer_connected)
        .def_readonly("intended_consumer", &KernelRegistry::Coverage::intended_consumer)
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
        .def("mark_operationally_consumed", &KernelRegistry::mark_operationally_consumed)
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
