// ============================================================================
// kernel_registry.cpp — native 45-kernel inventory (see kernel_registry.hpp).
// ============================================================================
#include "opportunity_discovery/kernel_registry.hpp"

#include <sstream>

#include "opportunity_discovery/metal_runtime.hpp"

namespace trading::metal {

// ---- screening_types.hpp to_string definitions -----------------------------
const char* to_string(KernelStatus s) noexcept {
    switch (s) {
        case KernelStatus::Discovered: return "DISCOVERED";
        case KernelStatus::Compiled: return "COMPILED";
        case KernelStatus::Registered: return "REGISTERED";
        case KernelStatus::PipelineCached: return "PIPELINE_CACHED";
        case KernelStatus::HostCallable: return "HOST_CALLABLE";
        case KernelStatus::HardwareExecuted: return "HARDWARE_EXECUTED";
        case KernelStatus::ConsumerConnected: return "CONSUMER_CONNECTED";
        case KernelStatus::NumericallyValidated: return "NUMERICALLY_VALIDATED";
        case KernelStatus::Benchmarked: return "BENCHMARKED";
        case KernelStatus::Blocked: return "BLOCKED";
    }
    return "UNKNOWN";
}

const char* to_string(KernelMode m) noexcept {
    switch (m) {
        case KernelMode::Operational: return "operational";
        case KernelMode::Research: return "research";
        case KernelMode::Blocked: return "blocked";
    }
    return "unknown";
}

namespace {

KernelInfo op(const char* shader, const char* name, const char* host,
              const char* contract, const char* consumer, const char* data) {
    KernelInfo k;
    k.shader_file = shader;
    k.kernel_name = name;
    k.host_interface = host;
    k.buffer_contract = contract;
    k.consumer = consumer;
    k.required_data = data;
    k.mode = KernelMode::Operational;
    return k;
}

KernelInfo blocked(const char* shader, const char* name, const char* consumer,
                   const char* data, const char* reason,
                   KernelMode mode = KernelMode::Blocked) {
    KernelInfo k;
    k.shader_file = shader;
    k.kernel_name = name;
    k.consumer = consumer;
    k.required_data = data;
    k.blocker = reason;
    k.mode = mode;
    return k;
}

}  // namespace

KernelRegistry::KernelRegistry() {
    // --- indicators.metal (4) : the operational integer-tick indicators ------
    kernels_.push_back(op("indicators.metal", "rolling_sma_ticks",
                          "MetalRuntime.sma",
                          "prices[i64@0] out[i64@1] count[u32@2] window[u32@3]",
                          "technical_features,strategies,scanner", "ticks_1d"));
    kernels_.push_back(op("indicators.metal", "exponential_moving_average_ticks",
                          "MetalRuntime.ema",
                          "prices[i64@0] out[i64@1] count[u32@2] num[u32@3] den[u32@4]",
                          "technical_features,strategies", "ticks_1d"));
    kernels_.push_back(op("indicators.metal", "average_price_oscillator_ticks",
                          "MetalRuntime.apo",
                          "fast[i64@0] slow[i64@1] out[i64@2] count[u32@3]",
                          "technical_features,apo_strategy", "ticks_1d"));
    kernels_.push_back(op("indicators.metal", "true_range_ticks",
                          "MetalRuntime.true_range",
                          "highs[i64@0] lows[i64@1] prev[i64@2] out[i64@3] count[u32@4]",
                          "technical_features,atr,di_bb", "ohlc_ticks"));

    // --- opportunity_screening.metal (1) : Phase B native discovery kernel -----
    // Newly added (directive §13/§80): one GPU thread per symbol evaluates the
    // mandatory shared entry contract on device. Operational; CPU oracle is
    // cpu_ref::screen_market.
    kernels_.push_back(op("opportunity_screening.metal", "screen_market",
                          "MetalRuntime.screen",
                          "inputs[ScreeningInput@0] outputs[ScreeningOutput@1] count[u32@2]",
                          "opportunity_discovery,scanner,strategies", "l1_quote"));

    // --- covariance.metal (7) : 3 wired, 4 research/blocked -------------------
    kernels_.push_back(op("covariance.metal", "column_means",
                          "MetalRuntime.column_means",
                          "values[f32@0] params[CovarianceParams@1] means[f32@2]",
                          "portfolio_optimizer", "returns_matrix"));
    kernels_.push_back(op("covariance.metal", "covariance_matrix",
                          "MetalRuntime.covariance_matrix",
                          "values[f32@0] means[f32@1] params[CovarianceParams@2] cov[f32@3]",
                          "portfolio_optimizer,risk_manager", "returns_matrix"));
    kernels_.push_back(op("covariance.metal", "ewma_covariance_matrix",
                          "MetalRuntime.ewma_covariance_matrix",
                          "values[f32@0] params[EwmaParams@1] cov[f32@2]",
                          "portfolio_optimizer,regime", "returns_matrix"));
    kernels_.push_back(blocked("covariance.metal", "statarb_pair_regression",
                               "statarb_research", "pair_returns",
                               "structured StatArb ABI; research-only, not yet wired",
                               KernelMode::Research));
    kernels_.push_back(blocked("covariance.metal", "statarb_residual_series",
                               "statarb_research", "pair_returns",
                               "structured StatArb ABI; research-only, not yet wired",
                               KernelMode::Research));
    kernels_.push_back(blocked("covariance.metal", "volatility_factor_features",
                               "regime_research", "factor_windows",
                               "structured factor ABI; not yet wired",
                               KernelMode::Research));
    kernels_.push_back(blocked("covariance.metal", "statarb_normalized_weights",
                               "statarb_research", "pair_returns",
                               "long-only engine; statarb weights research-only",
                               KernelMode::Research));

    // --- monte-carlo.metal (12) : research / multi-stage pipelines ------------
    const char* mc = "monte-carlo.metal";
    kernels_.push_back(blocked(mc, "generate_standard_normals", "risk_scenarios",
                               "rng_seeds", "MonteCarloParams ulong ABI; wiring pending",
                               KernelMode::Research));
    kernels_.push_back(blocked(mc, "correlate_normals", "risk_scenarios", "cholesky",
                               "structured Cholesky ABI; pipeline not yet wired",
                               KernelMode::Research));
    kernels_.push_back(blocked(mc, "generate_correlated_returns", "risk_scenarios",
                               "cholesky", "depends on correlate_normals; not wired",
                               KernelMode::Research));
    kernels_.push_back(blocked(mc, "portfolio_scenario_pnl", "portfolio_risk",
                               "scenarios", "multi-stage MC pipeline; not wired",
                               KernelMode::Research));
    kernels_.push_back(blocked(mc, "statarb_pair_monte_carlo", "statarb_research",
                               "pair_returns", "research-only; not wired",
                               KernelMode::Research));
    kernels_.push_back(blocked(mc, "simulate_portfolio_paths", "research", "scenarios",
                               "multi-stage MC pipeline; not wired", KernelMode::Research));
    kernels_.push_back(blocked(mc, "pnl_to_loss", "risk_scenarios", "scenarios",
                               "MC aggregation stage; not wired", KernelMode::Research));
    kernels_.push_back(blocked(mc, "extract_scenario_pnl", "risk_scenarios", "scenarios",
                               "MC aggregation stage; not wired", KernelMode::Research));
    kernels_.push_back(blocked(mc, "extract_pair_pnl", "statarb_research", "scenarios",
                               "MC aggregation stage; not wired", KernelMode::Research));
    kernels_.push_back(blocked(mc, "tail_moment_partials", "risk_scenarios", "scenarios",
                               "feeds reduce_* combine stage; not wired", KernelMode::Research));
    kernels_.push_back(blocked(mc, "threshold_tail_partials", "risk_scenarios", "scenarios",
                               "feeds reduce_threshold_tail; not wired", KernelMode::Research));
    kernels_.push_back(blocked(mc, "apply_stress_shocks", "stress_testing", "scenarios",
                               "structured stress ABI; not wired", KernelMode::Research));

    // --- imbalance.metal (7) : L2-depth dependent; IEX provides L1 only --------
    const char* im = "imbalance.metal";
    kernels_.push_back(blocked(im, "validate_depth_snapshots", "", "l2_depth",
                               "requires genuine L2 depth; IEX provides L1 only"));
    kernels_.push_back(blocked(im, "compute_level_imbalance", "", "l2_depth",
                               "requires genuine L2 depth; IEX provides L1 only"));
    kernels_.push_back(blocked(im, "compute_imbalance_features", "", "l2_depth",
                               "requires genuine L2 depth; IEX provides L1 only"));
    kernels_.push_back(blocked(im, "compute_top_level_imbalance", "order_flow,scanner",
                               "l1_quote",
                               "L1-compatible; structured depth ABI wiring pending",
                               KernelMode::Research));
    kernels_.push_back(blocked(im, "compute_weighted_depth_imbalance", "", "l2_depth",
                               "requires genuine L2 depth; IEX provides L1 only"));
    kernels_.push_back(blocked(im, "compute_depth_shape", "", "l2_depth",
                               "requires genuine L2 depth; IEX provides L1 only"));
    kernels_.push_back(blocked(im, "compute_pair_pressure_features", "", "l2_depth",
                               "requires genuine L2 depth; IEX provides L1 only"));

    // --- reductions.metal (15) : two-stage partial/combine pipelines ----------
    const char* re = "reductions.metal";
    const char* two = "two-stage partial/combine pipeline wiring pending";
    kernels_.push_back(blocked(re, "reduce_sum_partials", "aggregation,portfolio_stats",
                               "scalars", two, KernelMode::Research));
    kernels_.push_back(blocked(re, "reduce_moment_partials", "statistics,volatility",
                               "scalars", two, KernelMode::Research));
    kernels_.push_back(blocked(re, "reduce_minmax_partials", "statistics", "scalars",
                               two, KernelMode::Research));
    kernels_.push_back(blocked(re, "reduce_dot_partials", "", "vectors",
                               "combine stage wiring pending", KernelMode::Research));
    kernels_.push_back(blocked(re, "reduce_covariance_sufficient_partials",
                               "portfolio_optimizer", "returns_matrix",
                               "pairs with combine_covariance_partials; pending",
                               KernelMode::Research));
    kernels_.push_back(blocked(re, "reduce_matrix_column_partials", "", "matrix",
                               "combine stage wiring pending", KernelMode::Research));
    kernels_.push_back(blocked(re, "reduce_pnl_partials", "portfolio_stats", "scenarios",
                               "pairs with combine_pnl_partials; pending", KernelMode::Research));
    kernels_.push_back(blocked(re, "reduce_threshold_tail_partials", "risk_scenarios",
                               "scenarios", "pairs with combine_threshold_tail; pending",
                               KernelMode::Research));
    kernels_.push_back(blocked(re, "reduce_validation_counts", "", "scalars",
                               "combine stage wiring pending", KernelMode::Research));
    kernels_.push_back(blocked(re, "reduce_histogram_partials", "", "scalars",
                               "combine stage wiring pending", KernelMode::Research));
    kernels_.push_back(blocked(re, "combine_sum_partials", "aggregation", "scalars",
                               "second stage of reduce_sum; pending", KernelMode::Research));
    kernels_.push_back(blocked(re, "combine_covariance_partials", "portfolio_optimizer",
                               "returns_matrix", "second stage; pending", KernelMode::Research));
    kernels_.push_back(blocked(re, "combine_pnl_partials", "", "scenarios",
                               "second stage; pending", KernelMode::Research));
    kernels_.push_back(blocked(re, "combine_threshold_tail_partials", "", "scenarios",
                               "second stage; pending", KernelMode::Research));
    kernels_.push_back(blocked(re, "normalize_zscores", "", "vectors",
                               "combine stage wiring pending", KernelMode::Research));

    // Set resolved_name default to bare name (reconcile adjusts covariance/gme).
    for (auto& k : kernels_) k.resolved_name = k.kernel_name;
}

const std::vector<KernelInfo>& KernelRegistry::kernels() const noexcept {
    return kernels_;
}

KernelInfo* KernelRegistry::find(const std::string& name) {
    for (auto& k : kernels_) {
        if (k.kernel_name == name) return &k;
    }
    return nullptr;
}

void KernelRegistry::upsert(const KernelInfo& info) {
    if (KernelInfo* existing = find(info.kernel_name)) {
        *existing = info;
    } else {
        kernels_.push_back(info);
    }
}

void KernelRegistry::reconcile_with_device(MetalRuntime& runtime) {
    if (!runtime.available()) return;
    for (auto& k : kernels_) {
        // Namespace-aware presence + pipeline build (does not execute).
        if (runtime.has_kernel(k.kernel_name)) {
            k.resolved_name =
                runtime.has_kernel(k.kernel_name) ? k.kernel_name : k.resolved_name;
            k.pipeline_cached = runtime.ensure_pipeline(k.kernel_name);
        } else {
            k.pipeline_cached = false;
        }
    }
}

void KernelRegistry::mark_hardware_executed(const std::string& name) {
    if (KernelInfo* k = find(name)) k->hardware_executed = true;
}
void KernelRegistry::mark_numerically_validated(const std::string& name) {
    if (KernelInfo* k = find(name)) k->numerically_validated = true;
}
void KernelRegistry::mark_benchmarked(const std::string& name) {
    if (KernelInfo* k = find(name)) k->benchmarked = true;
}
void KernelRegistry::mark_operationally_consumed(const std::string& name) {
    if (KernelInfo* k = find(name)) k->operationally_consumed = true;
}

KernelRegistry::Coverage KernelRegistry::coverage() const {
    Coverage c;
    c.discovered = kernels_.size();
    for (const auto& k : kernels_) {
        if (k.pipeline_cached) ++c.compiled_callable;
        if (!k.host_interface.empty()) ++c.host_callable;
        // consumer_connected counts REAL operational wiring, not intent. A named
        // `consumer` only increments intended_consumer (directive §80/§86).
        if (k.operationally_consumed) ++c.consumer_connected;
        if (!k.consumer.empty()) ++c.intended_consumer;
        if (k.hardware_executed) ++c.hardware_executed;
        if (k.numerically_validated) ++c.numerically_validated;
        if (k.benchmarked) ++c.benchmarked;
        if (k.mode == KernelMode::Blocked) ++c.blocked;
    }
    if (c.discovered > 0) {
        c.pipeline_coverage_pct =
            100.0 * static_cast<double>(c.compiled_callable) / c.discovered;
        c.operational_coverage_pct =
            100.0 * static_cast<double>(c.consumer_connected) / c.discovered;
    }
    return c;
}

namespace {
std::string esc(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 2);
    for (char ch : s) {
        if (ch == '"' || ch == '\\') out.push_back('\\');
        out.push_back(ch);
    }
    return out;
}
}  // namespace

std::string KernelRegistry::to_json() const {
    const Coverage c = coverage();
    std::ostringstream o;
    o << "{\n  \"discovered\": " << c.discovered
      << ",\n  \"compiled_callable\": " << c.compiled_callable
      << ",\n  \"host_callable\": " << c.host_callable
      << ",\n  \"consumer_connected\": " << c.consumer_connected
      << ",\n  \"intended_consumer\": " << c.intended_consumer
      << ",\n  \"hardware_executed\": " << c.hardware_executed
      << ",\n  \"numerically_validated\": " << c.numerically_validated
      << ",\n  \"benchmarked\": " << c.benchmarked
      << ",\n  \"blocked\": " << c.blocked
      << ",\n  \"pipeline_coverage_pct\": " << c.pipeline_coverage_pct
      << ",\n  \"operational_coverage_pct\": " << c.operational_coverage_pct
      << ",\n  \"kernels\": [\n";
    for (std::size_t i = 0; i < kernels_.size(); ++i) {
        const auto& k = kernels_[i];
        o << "    {\"shader_file\": \"" << esc(k.shader_file) << "\""
          << ", \"kernel_name\": \"" << esc(k.kernel_name) << "\""
          << ", \"resolved_name\": \"" << esc(k.resolved_name) << "\""
          << ", \"host_interface\": \"" << esc(k.host_interface) << "\""
          << ", \"buffer_contract\": \"" << esc(k.buffer_contract) << "\""
          << ", \"consumer\": \"" << esc(k.consumer) << "\""
          << ", \"required_data\": \"" << esc(k.required_data) << "\""
          << ", \"mode\": \"" << to_string(k.mode) << "\""
          << ", \"pipeline_cached\": " << (k.pipeline_cached ? "true" : "false")
          << ", \"hardware_executed\": " << (k.hardware_executed ? "true" : "false")
          << ", \"numerically_validated\": "
          << (k.numerically_validated ? "true" : "false")
          << ", \"benchmarked\": " << (k.benchmarked ? "true" : "false")
          << ", \"operationally_consumed\": "
          << (k.operationally_consumed ? "true" : "false")
          << ", \"blocker\": \"" << esc(k.blocker) << "\"}";
        o << (i + 1 < kernels_.size() ? ",\n" : "\n");
    }
    o << "  ]\n}\n";
    return o.str();
}

}  // namespace trading::metal
