#pragma once

/**
 * @file ape.hpp
 * @brief APE-style link smearing for SU(2) gauge fields.
 *
 * The Albanese–Petronzio "fuzzed link" prescription (Phys. Lett. B 192, 163,
 * 1987): replace each link with a weighted sum of itself plus the staples
 * around it, then re-project onto SU(2). One smearing step:
 *
 *     C_μ(x) = U_μ(x) + α · Σ_{ν ⊥ μ, ν ∈ spatial} [forward + backward staples]
 *     U_μ^{(new)}(x) = Π_{SU(2)} C_μ(x) = C_μ(x) / ‖C_μ(x)‖
 *
 * For SU(2), `Π_{SU(2)}` is exact normalization (since SU(2) ≅ unit
 * quaternions). For SU(N > 2) it would be polar decomposition or maximum-Re-tr
 * projection — that's the only piece that doesn't carry over to SU(3).
 *
 * Iterating the smearing step ~5–15 times dramatically improves the
 * signal-to-noise of large Wilson loops by filtering out short-distance
 * vacuum fluctuations while leaving long-distance physics (string tension,
 * flux-tube width) intact. This is the standard procedure for static-charge
 * potential and flux-tube measurements (Bali–Schilling–Schlichter).
 *
 * In the flux-tube measurement we want to keep the temporal links untouched
 * so the transfer-matrix structure of the static-charge propagator is
 * preserved. `smear_dir[μ] = true` flags which link directions get smeared;
 * staples are formed using only other `smear_dir`-true directions.
 */

#include "../fields/link_field.hpp"
#include "../lattice/lattice.hpp"
#include "../math/su2.hpp"

#include <array>

namespace lqft::smearing
{

/// One APE smearing step. `U_in` is read-only; `U_out` receives the smeared
/// gauge field. Pass `U_out` distinct from `U_in` (no aliasing).
template<int Dim>
void apeSmearStep(
    const Lattice<Dim>&                       lattice,
    const LinkField<su2::Element, Dim>&       U_in,
    LinkField<su2::Element, Dim>&             U_out,
    double                                    alpha,
    const std::array<bool, Dim>&              smear_dir)
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

            // Sum staples in every perpendicular `smear_dir`-true direction.
            // `s_pmu` depends only on mu — hoist it out of the nu loop.
            su2::Element staple = su2::Element::zero();
            int n_terms = 0;
            const int s_pmu = lattice.forward(s, mu);
            for (int nu = 0; nu < Dim; ++nu)
            {
                if (nu == mu || !smear_dir[nu]) continue;

                const int s_pnu     = lattice.forward(s, nu);
                const int s_mnu     = lattice.backward(s, nu);
                const int s_pmu_mnu = lattice.backward(s_pmu, nu);

                // Forward staple:  U_ν(s+μ) · U_μ†(s+ν) · U_ν†(s)
                su2::Element fwd = su2::multiply(
                    U_in(s_pmu, nu), su2::dagger(U_in(s_pnu, mu)));
                fwd = su2::multiply(fwd, su2::dagger(U_in(s, nu)));
                staple = su2::add(staple, fwd);

                // Backward staple:  U_ν†(s+μ−ν) · U_μ†(s−ν) · U_ν(s−ν)
                su2::Element bwd = su2::multiply(
                    su2::dagger(U_in(s_pmu_mnu, nu)),
                    su2::dagger(U_in(s_mnu, mu)));
                bwd = su2::multiply(bwd, U_in(s_mnu, nu));
                staple = su2::add(staple, bwd);

                n_terms += 2;
            }

            // The cooling fixed point for the link is A†/‖A‖ (the link that
            // maximizes Re tr(U·A) with A the action staple). APE smears
            // toward that point, so we interpolate U with A† — *not* A —
            // and then renormalize.
            //
            //     C = U + (α/n_terms) · Σ staples†
            //
            // The (α/n_terms) factor keeps the natural range of α
            // independent of how many perpendicular directions contribute.
            su2::Element C = U_in(s, mu);
            if (n_terms > 0)
                C = su2::add(C, su2::scale(alpha / n_terms, su2::dagger(staple)));

            // Project to SU(2): exact normalization for unit quaternions.
            U_out(s, mu) = su2::project_to_su2(C);
        }
    }
}

/// Repeat `n` APE smearing steps. Ping-pong between two link buffers; the
/// final result lands in `U`.
template<int Dim>
void apeSmearN(
    const Lattice<Dim>&                  lattice,
    LinkField<su2::Element, Dim>&        U,         // in/out
    LinkField<su2::Element, Dim>&        scratch,   // workspace
    double                               alpha,
    int                                  n_steps,
    const std::array<bool, Dim>&         smear_dir)
{
    for (int k = 0; k < n_steps; ++k)
    {
        apeSmearStep<Dim>(lattice, U, scratch, alpha, smear_dir);
        std::swap(U, scratch); // logically copy back; cheap since each is a vector wrapper
    }
}

} // namespace lqft::smearing
