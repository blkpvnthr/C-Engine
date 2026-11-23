// Candidate native emergency protective-exit (Milestone 6B, §14.7) — INACTIVE.
//
// A pure, stateless, crash-independent emergency liquidation trigger implemented in native
// C++ so it can remain operational when Python/MLX/GPU/model-loading fail. It is a CANDIDATE
// only: it is NOT wired into RiskEngine::evaluate or any live path, and it must stay inactive
// in production until independent Milestone 7 certification authorizes it.
//
// Semantics: a signed unrealized-return fraction (e.g. -0.03 == -3%). `triggered` is true when
// the loss is at or beyond the emergency threshold. No ML/RL recommendation may override an
// activated trigger, and the optimizer must never raise (loosen) the emergency threshold. The
// threshold is a LIQUIDATION LEVEL, not a guaranteed realized maximum loss (gaps/halts/partial
// fills can produce worse fills). A tighter active stop, if present, always takes precedence.

#ifndef CENGINE_RISK_PROTECTIVE_EXIT_CANDIDATE_HPP
#define CENGINE_RISK_PROTECTIVE_EXIT_CANDIDATE_HPP

namespace trading {
namespace risk {

// Default candidate emergency threshold (signed fraction). Research value, not production.
inline constexpr double kCandidateEmergencyThreshold = -0.03;

struct EmergencyExitDecision {
    bool triggered{false};
    double unrealized_return{0.0};
    double threshold{kCandidateEmergencyThreshold};
    // This candidate never authorizes bypassing a tighter active stop.
    bool overrides_tighter_stop{false};
};

// Pure evaluation. triggered when unrealized_return <= emergency_threshold. Deterministic and
// side-effect-free; safe to call from any context including degraded/fallback paths.
inline EmergencyExitDecision evaluate_emergency_exit(
        double unrealized_return,
        double emergency_threshold = kCandidateEmergencyThreshold) noexcept {
    EmergencyExitDecision d;
    d.unrealized_return = unrealized_return;
    d.threshold = emergency_threshold;
    d.triggered = unrealized_return <= emergency_threshold;
    d.overrides_tighter_stop = false;  // never — a tighter active stop wins
    return d;
}

// The optimizer may never loosen the emergency threshold. This guard returns the SAFER (tighter,
// i.e. closer to zero loss / larger magnitude negative is looser) of the two: a proposed
// threshold is only accepted if it is at least as protective as the current one.
inline double clamp_emergency_threshold(double proposed, double current) noexcept {
    // more-negative == wider == looser. Never allow a looser (more-negative) threshold.
    return proposed < current ? current : proposed;
}

}  // namespace risk
}  // namespace trading

#endif  // CENGINE_RISK_PROTECTIVE_EXIT_CANDIDATE_HPP
