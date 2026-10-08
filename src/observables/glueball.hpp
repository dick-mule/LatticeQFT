#pragma once

/**
 * @file glueball.hpp
 * @brief Zero-momentum scalar (0⁺⁺) glueball operator and its time-slice correlator
 *        for SU(2) pure gauge theory.
 *
 * The simplest operator with vacuum quantum numbers on a time slice `τ` is the sum of all
 * spatial plaquettes on that slice,
 *
 *     O(τ) = Σ_{x ∈ slice τ} Σ_{i<j spatial} ½ Re tr U_ij(x),
 *
 * which projects onto zero momentum and onto the A₁⁺⁺ representation of the cubic group,
 * the lattice version of J^PC = 0⁺⁺. Its connected correlator
 *
 *     C(t) = (1/N_t) Σ_τ ⟨O(τ) O(τ + t)⟩ − ⟨O⟩²
 *
 * decays as `Σ_n |⟨0|O|n⟩|² e^{−m_n t}` (plus the periodic image), so the effective mass
 * `m_eff(t) = ln C(t)/C(t+1)` plateaus at the lightest glueball. The overlap of a thin
 * plaquette on the glueball is poor (the state is ~1 fm across); the driver therefore
 * builds `O` from APE-smeared **spatial** links (`smearing/ape.hpp`, temporal links left
 * alone so the transfer matrix is untouched), which is the standard cure.
 *
 * The vacuum subtraction makes `C(t)` a difference of two large numbers; the driver keeps
 * per-sample `O(τ)` data and uses a blocked jackknife for the errors on `C(t)` and `m_eff`.
 */

#include "../fields/link_field.hpp"
#include "../lattice/lattice.hpp"
#include "../math/su2.hpp"

#include <cmath>
#include <vector>

namespace lqft::obs
{

/// `O(τ)` for every time slice τ along `t_axis`: the sum of ½ Re tr over the spatial
/// plaquettes (planes not containing `t_axis`) at the sites of that slice.
template<int Dim>
std::vector<double> plaquetteSlices(const Lattice<Dim>& lattice,
                                    const LinkField<su2::Element, Dim>& field,
                                    int t_axis)
{
    const int V  = lattice.volume();
    const int Nt = lattice.coords(V - 1)[static_cast<std::size_t>(t_axis)] + 1;
    std::vector<double> O(static_cast<std::size_t>(Nt), 0.0);
    for (int s = 0; s < V; ++s)
    {
        const int tau = lattice.coords(s)[static_cast<std::size_t>(t_axis)];
        double acc = 0.0;
        for (int mu = 0; mu < Dim; ++mu)
        {
            if (mu == t_axis) continue;
            for (int nu = mu + 1; nu < Dim; ++nu)
            {
                if (nu == t_axis) continue;
                const int s_mu = lattice.forward(s, mu);
                const int s_nu = lattice.forward(s, nu);
                su2::Element P = su2::multiply(field(s, mu), field(s_mu, nu));
                P = su2::multiply(P, su2::dagger(field(s_nu, mu)));
                P = su2::multiply(P, su2::dagger(field(s, nu)));
                acc += P.s;   // ½ tr
            }
        }
        O[static_cast<std::size_t>(tau)] += acc;
    }
    return O;
}

/// Time-averaged slice products for one configuration: `c(t) = (1/N_t) Σ_τ O(τ) O(τ+t)`,
/// periodic in `t`. Together with the slice mean these are the per-sample inputs to the
/// vacuum-subtracted correlator.
inline std::vector<double> sliceProducts(const std::vector<double>& O)
{
    const int Nt = static_cast<int>(O.size());
    std::vector<double> c(static_cast<std::size_t>(Nt), 0.0);
    for (int t = 0; t < Nt; ++t)
    {
        double acc = 0.0;
        for (int tau = 0; tau < Nt; ++tau)
            acc += O[static_cast<std::size_t>(tau)] * O[static_cast<std::size_t>((tau + t) % Nt)];
        c[static_cast<std::size_t>(t)] = acc / static_cast<double>(Nt);
    }
    return c;
}

/// Vacuum-subtracted correlator from per-sample products and means:
/// `C(t) = ⟨c(t)⟩ − ⟨ō⟩²`.
inline std::vector<double> connectedCorrelator(const std::vector<std::vector<double>>& c,
                                               const std::vector<double>& obar)
{
    const std::size_t n = c.size();
    if (n == 0) return {};
    const std::size_t Nt = c[0].size();
    std::vector<double> C(Nt, 0.0);
    double om = 0.0;
    for (std::size_t s = 0; s < n; ++s)
    {
        om += obar[s];
        for (std::size_t t = 0; t < Nt; ++t) C[t] += c[s][t];
    }
    om /= static_cast<double>(n);
    for (std::size_t t = 0; t < Nt; ++t) C[t] = C[t] / static_cast<double>(n) - om * om;
    return C;
}

/// Effective mass from the periodic cosh form (bisection), NaN when undefined.
inline double effectiveMassPeriodic(double Ct, double Ct1, int t, int Nt)
{
    if (!(Ct > 0.0) || !(Ct1 > 0.0)) return std::nan("");
    const double target = Ct / Ct1;
    const double u = t - 0.5 * Nt, v = t + 1 - 0.5 * Nt;
    auto f = [&](double M) { return std::cosh(M * u) / std::cosh(M * v); };
    double lo = 1e-6, hi = 10.0;
    if ((f(lo) - target) * (f(hi) - target) > 0.0) return std::nan("");
    for (int it = 0; it < 200; ++it)
    {
        const double mid = 0.5 * (lo + hi);
        if ((f(mid) - target) * (f(lo) - target) <= 0.0) hi = mid; else lo = mid;
    }
    return 0.5 * (lo + hi);
}

} // namespace lqft::obs
