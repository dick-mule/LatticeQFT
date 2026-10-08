#pragma once

/**
 * @file su3_heatbath.hpp
 * @brief Cabibbo–Marinari heat-bath for SU(3) pure-gauge updates.
 *
 * SU(3) has no closed-form heat-bath, but Cabibbo and Marinari (Phys. Lett. B 119,
 * 387, 1982) showed that updating the link through a sequence of SU(2) subgroups
 * is enough: each subgroup update is an exact heat-bath in that subgroup, the
 * three subgroups (0,1), (0,2), (1,2) generate SU(3), and the composition leaves
 * the Wilson measure invariant.
 *
 * The single-link weight at fixed staple sum `A` is `exp((β/3) Re tr(U A))`. Write
 * the update as `U → α U` with `α` an SU(2) element embedded in rows/columns
 * `(i, j)`. Then, with `W = U A`,
 *
 *     Re tr(α U A) = Re tr(α_{2×2} W_{(ij)}) + (terms independent of α),
 *
 * where `W_{(ij)}` is the 2×2 block of `W`. Any 2×2 complex matrix splits into a
 * part proportional to an SU(2) element (the quaternions, `[[a, b], [−b̄, ā]]`) and
 * a part whose trace against every SU(2) element is purely imaginary, so only the
 * quaternion projection
 *
 *     a = ½ (W_ii + W̄_jj),   b = ½ (W_ij − W̄_ji),   k = √(|a|² + |b|²),   V = [[a, b], [−b̄, ā]] / k
 *
 * matters and the conditional for `α` is the SU(2) heat-bath weight
 * `exp((β/3) k Re tr(α V)) = exp((2β/3) k · w₀(α V))`. That is the Kennedy–Pendleton
 * problem of `su2_heatbath.hpp` with `a = (2β/3) k`: sample `X = (w₀, √(1−w₀²) n̂)`,
 * set `α = X V†`, embed, multiply. The three subgroups are visited in turn, with
 * `W` recomputed after each, and the link is re-projected to SU(3) at the end
 * against round-off drift.
 */

#include "../fields/link_field.hpp"
#include "../math/su2.hpp"
#include "../math/su3.hpp"
#include "../models/su3.hpp"
#include "../rng/rng.hpp"
#include "su2_heatbath.hpp"

#include <cmath>
#include <complex>

namespace lqft::su3_model
{

/// Quaternion projection of the (i, j) block of a 3×3 complex matrix: the unique
/// `k · V` with `V ∈ SU(2)` whose real trace against every SU(2) element matches
/// the block's. Returned in the `s + i v·σ` convention of `su2::Element`
/// (matching `su3::embedSU2`).
inline su2::Element quaternionBlock(const su3::Element& W, int i, int j)
{
    const std::complex<double> a = 0.5 * (W.m[i][i] + std::conj(W.m[j][j]));
    const std::complex<double> b = 0.5 * (W.m[i][j] - std::conj(W.m[j][i]));
    // embedSU2: block = [[s + i v2, v1 + i v0], [−v1 + i v0, s − i v2]]
    return su2::Element{ a.real(), { b.imag(), b.real(), a.imag() } };
}

/// One Cabibbo–Marinari heat-bath update at link (site, μ).
template<int Dim>
inline su3::Element heatBathUpdateOneLink(
    const SU3Model<Dim>& model,
    const LinkField<su3::Element, Dim>& field,
    int site, int mu, Rng& rng)
{
    const su3::Element A = model.staple(field, site, mu);
    su3::Element U = field(site, mu);
    constexpr int pairs[3][2] = { {0, 1}, {0, 2}, {1, 2} };
    for (const auto& p : pairs)
    {
        const int i = p[0], j = p[1];
        const su3::Element W = su3::multiply(U, A);
        const su2::Element R = quaternionBlock(W, i, j);
        const double k = su2::norm(R);
        su2::Element alpha;
        if (k < 1e-12)
        {
            alpha = su2_model::SU2Model<Dim>::randomUniformSU2(rng);
        }
        else
        {
            const double a  = (2.0 * model.beta() / 3.0) * k;
            const double w0 = su2_model::kennedyPendletonW0(a, rng);
            const auto   n  = su2_model::uniformS2(rng);
            const double r  = std::sqrt(std::max(0.0, 1.0 - w0 * w0));
            const su2::Element X = { w0, { r * n[0], r * n[1], r * n[2] } };
            alpha = su2::multiply(X, su2::scale(1.0 / k, su2::dagger(R)));
        }
        U = su3::multiply(su3::embedSU2(i, j, alpha.s, alpha.v[0], alpha.v[1], alpha.v[2]), U);
    }
    return su3::project_to_su3(U);
}

/// Full heat-bath sweep over the same colour partition as the Metropolis path.
template<int Dim>
void heatBathSweep(const SU3Model<Dim>& model,
                   LinkField<su3::Element, Dim>& field,
                   Rng& rng)
{
    const int n_colors = model.numColors();
    for (int c = 0; c < n_colors; ++c)
        for (int unit : model.unitsOfColor(c))
        {
            const int site = SU3Model<Dim>::siteOfUnit(unit);
            const int mu   = SU3Model<Dim>::muOfUnit(unit);
            field(site, mu) = heatBathUpdateOneLink(model, field, site, mu, rng);
        }
}

template<int Dim>
void heatBathSweepN(const SU3Model<Dim>& model,
                    LinkField<su3::Element, Dim>& field,
                    Rng& rng, int n)
{
    for (int i = 0; i < n; ++i) heatBathSweep(model, field, rng);
}

/// Translation-averaged Wilson loop W(R, T) in plane (μ, ν), (1/3) Re tr.
template<int Dim>
double averageWilsonLoop(const Lattice<Dim>& lat,
                         const LinkField<su3::Element, Dim>& field,
                         int mu, int nu, int R, int T)
{
    double sum = 0.0;
    const int V = lat.volume();
    for (int s = 0; s < V; ++s) sum += wilsonLoopAt(lat, field, s, mu, nu, R, T);
    return sum / static_cast<double>(V);
}

} // namespace lqft::su3_model
