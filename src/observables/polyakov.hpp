#pragma once

/**
 * @file polyakov.hpp
 * @brief Polyakov loops and the deconfinement order parameter for SU(2) on an
 *        asymmetric (finite-temperature) lattice.
 *
 * On a lattice with temporal extent N_t along `t_axis`, the Polyakov loop at a spatial
 * site x is the gauge-invariant trace of the product of temporal links around the
 * periodic time direction,
 *
 *     P(x) = ½ tr ∏_{t=0}^{N_t−1} U_t(x, t)        (real for SU(2))
 *
 * `⟨P⟩` is the order parameter of the deconfinement transition: it transforms under
 * the Z₂ centre of SU(2) as P → −P, so in the confined phase `⟨P⟩ = 0` and `e^{−N_t a F_q}`
 * = 0 (infinite free energy of an isolated static charge), while in the deconfined
 * phase the centre symmetry breaks and `⟨|P̄|⟩ ≠ 0`. On a finite lattice the symmetric
 * phase tunnels between the two Z₂ sectors, so the volume-averaged loop P̄ enters through
 * its modulus:
 *
 *     ⟨|P̄|⟩,     χ_P = V_s (⟨P̄²⟩ − ⟨|P̄|⟩²)
 *
 * whose peak locates β_c (N_t). Known 4D SU(2) values: β_c ≈ 1.88 at N_t = 2,
 * β_c ≈ 2.30 at N_t = 4 (Engels et al.; Fingberg, Heller & Karsch 1993).
 */

#include "../fields/link_field.hpp"
#include "../lattice/lattice.hpp"
#include "../math/su2.hpp"
#include "../math/su3.hpp"

#include <complex>

namespace lqft::obs
{

/// ½ tr of the Polyakov loop through the spatial site `site0` (any site on the line; the
/// trace is cyclic, so the starting t does not matter) along `t_axis`, with temporal
/// extent `Nt`.
template<int Dim>
double polyakovLoopAt(const Lattice<Dim>& lattice,
                      const LinkField<su2::Element, Dim>& field,
                      int site0, int t_axis, int Nt)
{
    su2::Element P = su2::Element::identity();
    int s = site0;
    for (int t = 0; t < Nt; ++t)
    {
        P = su2::multiply(P, field(s, t_axis));
        s = lattice.forward(s, t_axis);
    }
    return P.s;   // ½ tr for SU(2)
}

/// Volume-averaged Polyakov loop P̄ (one number per configuration) over all spatial sites
/// (the sites with t = 0 along `t_axis`).
template<int Dim>
double averagePolyakovLoop(const Lattice<Dim>& lattice,
                           const LinkField<su2::Element, Dim>& field,
                           int t_axis, int Nt)
{
    double sum = 0.0;
    long   n   = 0;
    const int V = lattice.volume();
    for (int s = 0; s < V; ++s)
    {
        if (lattice.coords(s)[static_cast<std::size_t>(t_axis)] != 0) continue;
        sum += polyakovLoopAt(lattice, field, s, t_axis, Nt);
        ++n;
    }
    return (n > 0) ? sum / static_cast<double>(n) : 0.0;
}

// ---------------------------------------------------------------------------
// SU(3): the loop is complex, P = ⅓ tr ∏ U_t, and the centre is Z₃ (P → e^{2πi/3} P),
// so the modulus is again the finite-volume order parameter. The SU(3) transition is
// first order; known β_c(N_t = 4) ≈ 5.69 for the Wilson action.
// ---------------------------------------------------------------------------

template<int Dim>
std::complex<double> polyakovLoopAt(const Lattice<Dim>& lattice,
                                    const LinkField<su3::Element, Dim>& field,
                                    int site0, int t_axis, int Nt)
{
    su3::Element P = su3::Element::identity();
    int s = site0;
    for (int t = 0; t < Nt; ++t)
    {
        P = su3::multiply(P, field(s, t_axis));
        s = lattice.forward(s, t_axis);
    }
    return su3::trace(P) / 3.0;
}

template<int Dim>
std::complex<double> averagePolyakovLoop(const Lattice<Dim>& lattice,
                                         const LinkField<su3::Element, Dim>& field,
                                         int t_axis, int Nt)
{
    std::complex<double> sum{0.0, 0.0};
    long n = 0;
    const int V = lattice.volume();
    for (int s = 0; s < V; ++s)
    {
        if (lattice.coords(s)[static_cast<std::size_t>(t_axis)] != 0) continue;
        sum += polyakovLoopAt(lattice, field, s, t_axis, Nt);
        ++n;
    }
    return (n > 0) ? sum / static_cast<double>(n) : std::complex<double>{0.0, 0.0};
}

} // namespace lqft::obs
