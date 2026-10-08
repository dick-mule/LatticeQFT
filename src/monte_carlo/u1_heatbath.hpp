#pragma once

/**
 * @file u1_heatbath.hpp
 * @brief Heat-bath update for compact U(1) gauge theory.
 *
 * The single-link conditional distribution at fixed staple A is the
 * von Mises distribution:
 *
 *     P(θ | A) ∝ exp(β · Re[e^{iθ} A]) = exp(β |A| cos(θ + φ_A))
 *
 * where φ_A = arg(A). Equivalently P(θ | A) = vM(μ = −φ_A, κ = β |A|).
 *
 * Sampling: Best & Fisher (1979) — rejection sampling on a wrapped-Cauchy
 * envelope. Acceptance probability stays above 0.9 across all κ ≥ 0, so
 * the inner loop runs ~1.05 iterations on average. Output is exact —
 * unlike Metropolis there is no acceptance step, and the new link is an
 * iid draw from the conditional, so autocorrelation per sweep is much
 * shorter than Metropolis especially at strong coupling.
 *
 * The κ → 0 limit (strong coupling, vanishing staple magnitude) is the
 * uniform distribution on [−π, π); we special-case it because the
 * Best-Fisher constants involve 1/κ.
 *
 * Both `vonMisesSample` and `heatBathSweep` are dimension-agnostic — the
 * sweep iterates the same color partition `U1Model<Dim>` exposes for
 * Metropolis, so the GPU port at Phase 4 will run heat-bath color-by-color
 * the same way it would run Metropolis.
 */

#include "../fields/link_field.hpp"
#include "../models/u1.hpp"
#include "../rng/rng.hpp"

#include <cmath>

namespace lqft::u1
{

/// Best-Fisher 1979 sampler for vM(0, κ). Returns z ∈ [−π, π).
inline double bestFisherSample(double kappa, Rng& rng)
{
    constexpr double kTwoPi = 6.28318530717958647692;
    constexpr double kPi    = 3.14159265358979323846;

    if (kappa < 1e-12)
    {
        // κ → 0: uniform on the circle.
        return (rng.uniform() - 0.5) * kTwoPi;
    }

    const double a = 1.0 + std::sqrt(1.0 + 4.0 * kappa * kappa);
    const double b = (a - std::sqrt(2.0 * a)) / (2.0 * kappa);
    const double r = (1.0 + b * b) / (2.0 * b);

    while (true)
    {
        const double u1 = rng.uniform();
        const double u2 = rng.uniform();
        const double z  = std::cos(kPi * u1);
        const double f  = (1.0 + r * z) / (r + z);
        const double c  = kappa * (r - f);

        // Quick-accept then log-criterion fallback.
        if (c * (2.0 - c) > u2)
        {
            const double u3   = rng.uniform();
            const double sign = (u3 < 0.5) ? -1.0 : +1.0;
            return sign * std::acos(f);
        }
        // log accept-criterion handles the tail.
        if (std::log(c / u2) + 1.0 - c >= 0.0)
        {
            const double u3   = rng.uniform();
            const double sign = (u3 < 0.5) ? -1.0 : +1.0;
            return sign * std::acos(f);
        }
    }
}

/// One heat-bath update at link (site, μ): sample θ from the von Mises
/// conditional given the staple. No accept/reject — always replaces.
template<int Dim>
inline double heatBathUpdateOneLink(
    const U1Model<Dim>& model,
    const LinkField<double, Dim>& field,
    int site, int mu, Rng& rng)
{
    const auto A     = model.staple(field, site, mu);
    const double mag = std::sqrt(A.re * A.re + A.im * A.im);
    const double kappa = model.beta() * mag;
    const double phi_A = std::atan2(A.im, A.re); // ∈ (−π, π]
    const double z     = bestFisherSample(kappa, rng);
    // Conditional has mean direction μ = −φ_A.
    return -phi_A + z;
}

/// Full heat-bath sweep over every link, color by color, matching the
/// Metropolis color iteration so the algorithms can be swapped freely.
template<int Dim>
void heatBathSweep(
    const U1Model<Dim>& model,
    LinkField<double, Dim>& field,
    Rng& rng)
{
    const int n_colors = model.numColors();
    for (int c = 0; c < n_colors; ++c)
    {
        const auto& units = model.unitsOfColor(c);
        for (int unit : units)
        {
            const int site = U1Model<Dim>::siteOfUnit(unit);
            const int mu   = U1Model<Dim>::muOfUnit(unit);
            field(site, mu) = heatBathUpdateOneLink(model, field, site, mu, rng);
        }
    }
}

/// Repeat `n` sweeps. Convenience.
template<int Dim>
void heatBathSweepN(
    const U1Model<Dim>& model,
    LinkField<double, Dim>& field,
    Rng& rng,
    int n)
{
    for (int i = 0; i < n; ++i) heatBathSweep(model, field, rng);
}

} // namespace lqft::u1
