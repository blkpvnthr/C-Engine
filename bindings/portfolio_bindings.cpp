// ============================================================================
// portfolio_bindings.cpp — pybind11 surface for the native portfolio allocator
// ============================================================================
//
// Exposes the native deterministic risk-based allocation + portfolio-risk math
// (risk/portfolio_allocator.{hpp,cpp}) as _cengine_native.* so the Python
// portfolio pipeline can consume native optimization (§3/§25). No order path.
// ============================================================================

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "risk/portfolio_allocator.hpp"

namespace py = pybind11;

void register_portfolio(py::module_& m) {
    using trading::portfolio::AllocationResult;

    py::class_<AllocationResult>(m, "PortfolioAllocation")
        .def_readonly("weights", &AllocationResult::weights)
        .def_readonly("risk_contributions", &AllocationResult::risk_contributions)
        .def_readonly("portfolio_variance", &AllocationResult::portfolio_variance)
        .def_readonly("portfolio_volatility", &AllocationResult::portfolio_volatility)
        .def_readonly("gross_weight", &AllocationResult::gross_weight)
        .def_readonly("cash_weight", &AllocationResult::cash_weight)
        .def_readonly("per_asset_cap_binding", &AllocationResult::per_asset_cap_binding)
        .def_readonly("gross_cap_binding", &AllocationResult::gross_cap_binding)
        .def_readonly("valid", &AllocationResult::valid);

    m.def("portfolio_variance", &trading::portfolio::portfolio_variance,
          py::arg("weights"), py::arg("covariance"), py::arg("assets"));
    m.def("portfolio_marginal_variance", &trading::portfolio::marginal_variance,
          py::arg("weights"), py::arg("covariance"), py::arg("assets"));
    m.def("portfolio_risk_contributions", &trading::portfolio::risk_contributions,
          py::arg("weights"), py::arg("covariance"), py::arg("assets"));
    m.def("inverse_volatility_weights", &trading::portfolio::inverse_volatility_weights,
          py::arg("variances"));
    m.def("project_long_only_capped", &trading::portfolio::project_long_only_capped,
          py::arg("raw_weights"), py::arg("per_asset_cap"), py::arg("gross_cap"));
    m.def("allocate_risk_based", &trading::portfolio::allocate_risk_based,
          py::arg("variances"), py::arg("covariance"), py::arg("assets"),
          py::arg("per_asset_cap"), py::arg("gross_cap"));
}
