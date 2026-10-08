#pragma once

/**
 * @file stout.hpp
 * @brief Stout (analytic / differentiable) link smearing for SU(2).
 *
 * Morningstar & Peardon, Phys. Rev. D 69 (2004) 054501. The "modern" cousin
 * of APE smearing:
 *
 *     C_μ(x)  = Σ_{ν ⊥ μ ∈ smear_dir} [staples around link (x, μ)]
 *     Ω_μ(x) = C_μ(x) · U_μ†(x)
 *     Q_μ(x) = (i/2) [Ω† − Ω] − (i / 2N_c) · tr[Ω† − Ω] · I
 *     U_μ^{(new)}(x) = exp(i · ρ · Q_μ(x)) · U_μ(x)
 *
 * The sign on (Ω† − Ω) matters: with this convention Q is the gradient of
 * +Re tr(C · U†) on the SU(N) manifold, so positive ρ pushes U toward the
 * staple-aligned configuration (smoothing). The opposite sign would heat
 * the field.
 *
 * Differences vs APE:
 *   - **Analytic** (no Π_{SU(N)} projection): every stout step is a smooth
 *     map U → U, so the chain is differentiable. That's the reason it's the
 *     smearing of choice for HMC fermion actions; here we use it just for
 *     measurement, but the smoothness still gives marginally better
 *     signal-to-noise per iteration than APE's project-back-to-the-group.
 *   - The single parameter ρ plays a role analogous to APE's α / n_terms.
 *     Common range: ρ ∈ [0.05, 0.15].
 *
 * **SU(2)-specific simplifications:**
 *   - Ω.s + Ω.s = 2 · Ω.s (real), so Ω − Ω† has Ω.s component = 0;
 *     `tr(Ω − Ω†) = 0` automatically, so we drop the trace-subtraction term.
 *   - Q is then anti-Hermitian-times-i = pure −Ω.v·σ (Hermitian, traceless).
 *   - exp(i ρ Q) = exp(−i ρ Ω.v · σ) closes in the standard SU(2)
 *     (s, v) = (cos|w|, sin|w| ŵ) form with w = −ρ Ω.v. One sin / cos
 *     per link — same cost as APE's renormalization sqrt.
 */

#include "../fields/link_field.hpp"
#include "../lattice/lattice.hpp"
#include "../math/su2.hpp"

#include <array>
#include <cmath>

namespace lqft::smearing
{

template<int Dim>
void stoutSmearStep(
    const Lattice<Dim>&                  lattice,
    const LinkField<su2::Element, Dim>&  U_in,
    LinkField<su2::Element, Dim>&        U_out,
    double                               rho,
    const std::array<bool, Dim>&         smear_dir)
{
    const int V = lattice.volume();
    for (int s = 0; s < V; ++s)
    {
        for (int mu = 0; mu < Dim; ++mu)
        {
            if (!smear_dir[mu])
            {
                U_out(s, mu) = U_in(s, mu);
                continue;
            }

            // Staple sum (same construction as APE).
            su2::Element staple = su2::Element::zero();
            const int s_pmu = lattice.forward(s, mu);
            for (int nu = 0; nu < Dim; ++nu)
            {
                if (nu == mu || !smear_dir[nu]) continue;
                const int s_pnu     = lattice.forward(s, nu);
                const int s_mnu     = lattice.backward(s, nu);
                const int s_pmu_mnu = lattice.backward(s_pmu, nu);

                su2::Element fwd = su2::multiply(
                    U_in(s_pmu, nu), su2::dagger(U_in(s_pnu, mu)));
                fwd = su2::multiply(fwd, su2::dagger(U_in(s, nu)));
                staple = su2::add(staple, fwd);

                su2::Element bwd = su2::multiply(
                    su2::dagger(U_in(s_pmu_mnu, nu)),
                    su2::dagger(U_in(s_mnu, mu)));
                bwd = su2::multiply(bwd, U_in(s_mnu, nu));
                staple = su2::add(staple, bwd);
            }

            // Ω = C_MP · U†, where Morningstar-Peardon's "fat staple" C_MP
            // is the OPEN path going *from* x *to* x+μ. Our `staple` sum is
            // the closed-completing path going x+μ → x (so U_μ · staple is a
            // plaquette), i.e., C_MP = dagger(staple). Same matrix-trace
            // identities, but Q lands in the actual cooling-direction Lie
            // algebra rather than its negative.
            const su2::Element U_dag = su2::dagger(U_in(s, mu));
            const su2::Element C_mp  = su2::dagger(staple);
            const su2::Element Omega = su2::multiply(C_mp, U_dag);

            // Q = (i/2)(Ω† − Ω) − (i / 2N_c) tr[Ω† − Ω] I.  For SU(2) the
            // trace term vanishes (Ω.s − Ω†.s = 0), and (i/2)(Ω† − Ω) reduces
            // to the matrix +Ω.v · σ — a Hermitian traceless 2×2.
            //
            // U_new = exp(i ρ Q) U_old = exp(+i ρ Ω.v · σ) U_old.
            //
            // exp(i w·σ) = cos|w| I + i (sin|w| / |w|) w·σ, so with
            // w = +ρ Ω.v we get the SU(2) element
            //
            //     R = (cos(ρ|Ω.v|),  +sinc(ρ|Ω.v|) · ρ Ω.v).
            const double v_norm
                = std::sqrt(Omega.v[0] * Omega.v[0]
                          + Omega.v[1] * Omega.v[1]
                          + Omega.v[2] * Omega.v[2]);
            su2::Element R;
            if (v_norm < 1e-14)
            {
                R = su2::Element::identity();
            }
            else
            {
                const double theta = rho * v_norm;
                const double c     = std::cos(theta);
                const double sinc  = std::sin(theta) / v_norm;
                R = su2::Element{
                    c,
                    { +sinc * Omega.v[0], +sinc * Omega.v[1], +sinc * Omega.v[2] }
                };
            }
            U_out(s, mu) = su2::multiply(R, U_in(s, mu));
        }
    }
}

template<int Dim>
void stoutSmearN(
    const Lattice<Dim>&                  lattice,
    LinkField<su2::Element, Dim>&        U,
    LinkField<su2::Element, Dim>&        scratch,
    double                               rho,
    int                                  n_steps,
    const std::array<bool, Dim>&         smear_dir)
{
    for (int k = 0; k < n_steps; ++k)
    {
        stoutSmearStep<Dim>(lattice, U, scratch, rho, smear_dir);
        std::swap(U, scratch);
    }
}

} // namespace lqft::smearing
