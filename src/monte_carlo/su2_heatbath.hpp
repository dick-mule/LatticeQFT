#pragma once

/**
 * @file su2_heatbath.hpp
 * @brief Kennedy–Pendleton heat-bath for SU(2) pure-gauge updates.
 *
 * The single-link conditional at fixed staple `A_μ(y)` is
 *
 *     P(U | A) ∝ exp((β/2) Re tr(U · A))
 *             = exp(β k · w_0(U V))
 *
 * where `k = |A| = √(s_A² + |v_A|²)`, `V = A / k ∈ SU(2)`, and `w_0` is the
 * scalar (quaternion) component of `W = U · V`. The conditional on `W`
 * factorizes:
 *
 *     P(W) ∝ exp(β k w_0) · √(1 − w_0²) · dΩ_{n̂},
 *
 * the second factor being the Haar measure restricted to the level set
 * `w_0 = const`, and `n̂` uniform on S² in the SU(2) parameterization
 * `W = (w_0, √(1 − w_0²) n̂)`.
 *
 * Kennedy and Pendleton (Phys. Lett. B 156, 393, 1985) gave a rejection
 * algorithm for sampling `w_0` directly via the parameterization
 * `w_0 = 1 − 2 λ²`:
 *
 *     loop {
 *         X₁, X₂, X₃, X₄ ∼ Uniform(0, 1)
 *         λ² = -[ ln X₁ + cos²(2π X₂) · ln X₃ ] / (2 β k)
 *         if X₄² ≤ 1 − λ² : break
 *     }
 *
 * The acceptance is ≥ 0.95 across all β k, so the inner loop is < 2
 * iterations on average — dramatically better than Creutz's earlier method
 * which scales as 1/√(βk) at weak coupling.
 *
 * After sampling `w_0`, draw `n̂` uniformly on S² and form `W`. The new
 * link is `U_new = W · V†`.
 *
 * The sweep iterates the same color partition (2·Dim sublattices) as the
 * Metropolis path; same generic `MetropolisSweep<T>` interface so users
 * can swap between updates freely.
 */

#include "../fields/link_field.hpp"
#include "../math/su2.hpp"
#include "../models/su2.hpp"
#include "../rng/rng.hpp"

#include <cmath>

namespace lqft::su2_model
{

/// Sample `w_0` for the SU(2) heat-bath conditional ∝ exp(a · w_0) √(1−w_0²).
inline double kennedyPendletonW0(double a, Rng& rng)
{
    constexpr double kTwoPi = 6.28318530717958647692;
    if (a < 1e-12)
    {
        // Degenerate staple — uniform on the angle.
        return 1.0 - 2.0 * rng.uniform();
    }
    const double inv2a = 0.5 / a;
    for (int trial = 0; trial < 200; ++trial)
    {
        const double X1 = std::max(rng.uniform(), 1e-15);
        const double X2 = rng.uniform();
        const double X3 = std::max(rng.uniform(), 1e-15);
        const double X4 = rng.uniform();
        const double cos2 = std::cos(kTwoPi * X2);
        const double lam2 = -(std::log(X1) + cos2 * cos2 * std::log(X3)) * inv2a;
        if (X4 * X4 <= 1.0 - lam2)
            return 1.0 - 2.0 * lam2;
    }
    // Pathological — fall back to uniform.
    return 1.0 - 2.0 * rng.uniform();
}

/// Sample a unit 3-vector uniformly on S² (used for the SU(2) heat-bath
/// vector part after `w_0` has been chosen).
inline std::array<double, 3> uniformS2(Rng& rng)
{
    constexpr double kTwoPi = 6.28318530717958647692;
    const double cos_th = 2.0 * rng.uniform() - 1.0;
    const double sin_th = std::sqrt(std::max(0.0, 1.0 - cos_th * cos_th));
    const double phi    = kTwoPi * rng.uniform();
    return { sin_th * std::cos(phi),
             sin_th * std::sin(phi),
             cos_th };
}

/// One heat-bath update at link (site, μ): sample U_new directly from
/// the von-Mises-on-SU(2) conditional. 100% acceptance by construction.
template<int Dim>
inline su2::Element heatBathUpdateOneLink(
    const SU2Model<Dim>& model,
    const LinkField<su2::Element, Dim>& field,
    int site, int mu, Rng& rng)
{
    const su2::Element A = model.staple(field, site, mu);
    const double k = su2::norm(A);
    if (k < 1e-12)
    {
        // No information from the staple → uniform SU(2) draw.
        return SU2Model<Dim>::randomUniformSU2(rng);
    }
    const double a = model.beta() * k;
    const double w0 = kennedyPendletonW0(a, rng);

    const auto n  = uniformS2(rng);
    const double r = std::sqrt(std::max(0.0, 1.0 - w0 * w0));
    const su2::Element W = { w0, { r * n[0], r * n[1], r * n[2] } };

    // U_new = W · V†   with V = A / k.
    const su2::Element V_dag = su2::scale(1.0 / k, su2::dagger(A));
    return su2::multiply(W, V_dag);
}

/// Full heat-bath sweep — color-by-color iteration matching Metropolis.
template<int Dim>
void heatBathSweep(
    const SU2Model<Dim>& model,
    LinkField<su2::Element, Dim>& field,
    Rng& rng)
{
    const int n_colors = model.numColors();
    for (int c = 0; c < n_colors; ++c)
    {
        const auto& units = model.unitsOfColor(c);
        for (int unit : units)
        {
            const int site = SU2Model<Dim>::siteOfUnit(unit);
            const int mu   = SU2Model<Dim>::muOfUnit(unit);
            field(site, mu) = heatBathUpdateOneLink(model, field, site, mu, rng);
        }
    }
}

template<int Dim>
void heatBathSweepN(
    const SU2Model<Dim>& model,
    LinkField<su2::Element, Dim>& field,
    Rng& rng, int n)
{
    for (int i = 0; i < n; ++i) heatBathSweep(model, field, rng);
}

} // namespace lqft::su2_model
