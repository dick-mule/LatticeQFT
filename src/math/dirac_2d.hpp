#pragma once

/**
 * @file dirac_2d.hpp
 * @brief Wilson-Dirac operator on a 2D lattice with a U(1) gauge background.
 *
 * Two-component Dirac spinor ψ(x) at each site. γ-matrix convention
 * (Euclidean, Hermitian):
 *
 *     γ⁰ = σ_x = [[0, 1], [1, 0]]
 *     γ¹ = σ_y = [[0, -i], [i, 0]]
 *     γ⁵ = σ_z = [[1, 0], [0, -1]]
 *
 * All three γ's are Hermitian; γ⁵ anticommutes with γ⁰ and γ¹.
 *
 * Wilson-Dirac operator:
 *
 *     (D ψ)(x) = (m + 2) ψ(x)
 *              − ½ Σ_μ [ (1 − γ_μ) U_μ(x)   ψ(x+ê_μ)
 *                      + (1 + γ_μ) U_μ†(x−ê_μ) ψ(x−ê_μ) ].
 *
 * The (1 ± γ_μ) spinor projectors are rank-1 in 2D: each one collapses to
 *
 *     (1 − γ⁰) ψ = (ψ_0 − ψ_1) · (+1, −1)
 *     (1 + γ⁰) ψ = (ψ_0 + ψ_1) · (+1, +1)
 *     (1 − γ¹) ψ = (ψ_0 + i ψ_1) · (+1, −i)
 *     (1 + γ¹) ψ = (ψ_0 − i ψ_1) · (+1, +i)
 *
 * so each hopping term is one complex link-multiply followed by writing the
 * result into two spinor components with a sign / ±i factor. Same projector
 * trick is what makes 4D Wilson-Dirac efficient.
 *
 * γ⁵-Hermiticity:  γ⁵ D γ⁵ = D†.  In practice D† equals D with the
 * substitution γ_μ → −γ_μ in every hopping term, which swaps the projectors
 * (1 − γ_μ) ↔ (1 + γ_μ). We use this to implement applyDagger without a
 * second copy of the operator.
 */

#include "../fields/link_field.hpp"
#include "../fields/spinor_field.hpp"
#include "../lattice/lattice.hpp"

#include <complex>

namespace lqft::dirac
{

/// Wilson-Dirac operator on a 2D U(1) gauge background.
class WilsonDirac2D
{
public:
    using Complex     = std::complex<double>;
    using GaugeField  = LinkField<double, 2>;
    using Spinor      = SpinorField<2, 2>;

    WilsonDirac2D(const Lattice<2>& lattice, double mass)
        : m_lattice(lattice), m_mass(mass) {}

    double mass() const { return m_mass; }
    void   setMass(double m) { m_mass = m; }
    const Lattice<2>& lattice() const { return m_lattice; }

    // ------------------------------------------------------------------------
    // y = D[U] x.
    // Sign convention: forward hop uses (1 − γ_μ), backward uses (1 + γ_μ).
    // ------------------------------------------------------------------------

    void apply(const GaugeField& U, const Spinor& x, Spinor& y) const
    {
        applyImpl(U, x, y, /*dagger*/ false);
    }

    /// y = D†[U] x. Via γ⁵-Hermiticity this is the same code path as `apply`
    /// with (1 ± γ_μ) projectors swapped.
    void applyDagger(const GaugeField& U, const Spinor& x, Spinor& y) const
    {
        applyImpl(U, x, y, /*dagger*/ true);
    }

    /// y = D†[U] D[U] x. Requires one workspace spinor `tmp`.
    /// This is the positive-semidefinite Hermitian operator CG inverts.
    void applyDagD(const GaugeField& U,
                   const Spinor& x,
                   Spinor& y,
                   Spinor& tmp) const
    {
        apply       (U, x,   tmp);
        applyDagger (U, tmp, y);
    }

private:
    // Internal: one implementation does both apply and applyDagger.
    // dagger = true swaps the two projectors that get used at forward and
    // backward hops — equivalent to γ_μ → −γ_μ everywhere.
    void applyImpl(const GaugeField& U,
                   const Spinor& x,
                   Spinor& y,
                   bool dagger) const
    {
        const Complex I_unit(0.0, 1.0);
        const double  m_plus_2 = m_mass + 2.0;
        const int     V = m_lattice.volume();

        // The (1 ± γ_μ) action on a 2-spinor collapses to a scalar "combined"
        // and a write-out pattern. We compute combined once per neighbor visit.

        for (int s = 0; s < V; ++s)
        {
            // Diagonal: (m + 2) ψ(x)
            Complex y0 = m_plus_2 * x(s, 0);
            Complex y1 = m_plus_2 * x(s, 1);

            // --------------- μ = 0 ---------------
            {
                const int s_p   = m_lattice.forward (s, 0);
                const int s_m   = m_lattice.backward(s, 0);
                const Complex U_fwd  = std::polar(1.0,  U(s,   0));
                const Complex U_bwd  = std::polar(1.0, -U(s_m, 0));

                // Forward hop at +μ.
                // Standard: (1 − γ⁰)/2 → combined = ψ_0(x+μ) − ψ_1(x+μ),
                //           y0 −= ½ U·combined,  y1 −= ½ U·(−combined)
                // Dagger:   (1 + γ⁰)/2 → combined = ψ_0(x+μ) + ψ_1(x+μ),
                //           y0 −= ½ U·combined,  y1 −= ½ U·(+combined)
                const Complex p_fwd = dagger
                    ?  U_fwd * (x(s_p, 0) + x(s_p, 1))
                    :  U_fwd * (x(s_p, 0) - x(s_p, 1));
                y0 -= 0.5 * p_fwd;
                y1 -= 0.5 * (dagger ? +p_fwd : -p_fwd);

                // Backward hop at −μ.
                const Complex p_bwd = dagger
                    ?  U_bwd * (x(s_m, 0) - x(s_m, 1))
                    :  U_bwd * (x(s_m, 0) + x(s_m, 1));
                y0 -= 0.5 * p_bwd;
                y1 -= 0.5 * (dagger ? -p_bwd : +p_bwd);
            }

            // --------------- μ = 1 ---------------
            {
                const int s_p   = m_lattice.forward (s, 1);
                const int s_m   = m_lattice.backward(s, 1);
                const Complex U_fwd  = std::polar(1.0,  U(s,   1));
                const Complex U_bwd  = std::polar(1.0, -U(s_m, 1));

                // Standard: (1 − γ¹)/2 → combined = ψ_0 + i ψ_1, write (+1, −i)·combined.
                // Dagger:   (1 + γ¹)/2 → combined = ψ_0 − i ψ_1, write (+1, +i)·combined.
                const Complex p_fwd = dagger
                    ?  U_fwd * (x(s_p, 0) - I_unit * x(s_p, 1))
                    :  U_fwd * (x(s_p, 0) + I_unit * x(s_p, 1));
                y0 -= 0.5 * p_fwd;
                y1 -= 0.5 * (dagger ? +I_unit * p_fwd : -I_unit * p_fwd);

                // Backward hop at −μ: standard (1 + γ¹)/2, dagger (1 − γ¹)/2.
                const Complex p_bwd = dagger
                    ?  U_bwd * (x(s_m, 0) + I_unit * x(s_m, 1))
                    :  U_bwd * (x(s_m, 0) - I_unit * x(s_m, 1));
                y0 -= 0.5 * p_bwd;
                y1 -= 0.5 * (dagger ? -I_unit * p_bwd : +I_unit * p_bwd);
            }

            y(s, 0) = y0;
            y(s, 1) = y1;
        }
    }

    const Lattice<2>& m_lattice;
    double            m_mass;
};

} // namespace lqft::dirac
