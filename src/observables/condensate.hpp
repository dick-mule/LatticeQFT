#pragma once

/**
 * @file condensate.hpp
 * @brief Stochastic estimator for the chiral condensate ⟨ψ̄ψ⟩.
 *
 * The fermion bilinear expectation in the path-integral formulation is
 *
 *     ⟨ψ̄ψ⟩ = (1/V) · tr_{(site, spinor)} D⁻¹[U]
 *
 * (sign conventions aside — see docs/math/dirac_wilson_2d.md). We have a
 * Dirac operator with `apply / applyDagger / applyDagD` and a CG solver
 * for D†D; the missing piece is `D⁻¹` itself. Two algorithm choices:
 *
 *   1. Direct trace: invert D against `2·V` unit sources e_{(x, a)} and
 *      accumulate the (x, a) component of each solution. Exact but costs
 *      `2·V` CG solves per measurement.
 *
 *   2. Stochastic trace: pick a random noise vector η with
 *      `⟨η_i ⊗ η_j⟩ = δ_{ij}`, solve D x = η, accumulate ⟨η, x⟩.
 *      Costs one CG solve per source draw. Variance drops as 1/N_src,
 *      and we can afford many sources cheaply.
 *
 * We implement (2) with **real Z₂ noise** (each component independently
 * ±1), the standard variance-reducing choice for tr-D⁻¹ estimators. Z₂
 * has the property `η_i² ≡ 1`, which means diagonal entries of D⁻¹ are
 * captured exactly by any single source — only the off-diagonals add
 * variance.
 *
 * The D⁻¹ action is implemented via the identity D⁻¹ = D† (D†D)⁻¹: one
 * CG solve, then one applyDagger.
 */

#include "../fields/link_field.hpp"
#include "../fields/spinor_field.hpp"
#include "../lattice/lattice.hpp"
#include "../rng/rng.hpp"
#include "../solvers/conjugate_gradient.hpp"

#include <complex>

namespace lqft::obs
{

/// Fill `eta` with iid real Z₂ noise (each component ±1 + 0i).
inline void z2_noise(SpinorField<2, 2>& eta, Rng& rng)
{
    using Complex = std::complex<double>;
    const int N = eta.totalSize();
    auto* p = eta.data();
    for (int i = 0; i < N; ++i)
        p[i] = Complex{ rng.uniform() < 0.5 ? -1.0 : +1.0, 0.0 };
}

struct CondensateResult
{
    double mean;             // (1/V) · stochastic estimate of tr D⁻¹
    double stochastic_error; // naïve standard error across the N_src draws
    int    avg_cg_iters;     // average CG iterations per source (informational)
    bool   all_converged;
};

/// Stochastic estimator of (1/V) · tr_{site,spinor} D⁻¹[U] using N_src
/// random Z₂ sources. Each source needs one CG solve on D†D.
///
/// Templated on the Dirac operator type so SU(N) / 4D variants will reuse
/// this same estimator without changes — only the operator's apply* contract
/// matters.
template<typename DiracOp>
CondensateResult stochasticCondensate(
    const DiracOp& D,
    const LinkField<double, 2>& U,
    const Lattice<2>& lattice,
    Rng& rng,
    int    n_sources,
    double cg_tol      = 1e-9,
    int    cg_max_iters = 2000)
{
    using Complex = std::complex<double>;

    SpinorField<2, 2> eta(lattice), y(lattice), x(lattice);
    SpinorField<2, 2> r(lattice), p(lattice), Ap(lattice), tmp(lattice);

    auto apply_DagD = [&](const SpinorField<2, 2>& in, SpinorField<2, 2>& out)
    {
        D.applyDagD(U, in, out, tmp);
    };

    double sum    = 0.0;
    double sumsq  = 0.0;
    long   iters  = 0;
    bool   all_ok = true;
    const double inv_V = 1.0 / static_cast<double>(lattice.volume());

    for (int k = 0; k < n_sources; ++k)
    {
        z2_noise(eta, rng);
        y.zero();
        auto res = solvers::conjugateGradient<2, 2>(
            apply_DagD, eta, y, cg_tol, cg_max_iters, r, p, Ap);
        if (!res.converged) all_ok = false;
        iters += res.iterations;

        // x = D⁻¹ η = D† (D†D)⁻¹ η.
        D.applyDagger(U, y, x);

        // Per-source contribution: (1/V) ⟨η, x⟩, take the real part —
        // imaginary part averages to zero by the reality of tr D⁻¹.
        const Complex inner = inner_product(eta, x);
        const double  est_k = inner.real() * inv_V;
        sum   += est_k;
        sumsq += est_k * est_k;
    }

    const double N    = static_cast<double>(n_sources);
    const double mean = sum / N;
    const double var  = (n_sources < 2) ? 0.0 : (sumsq / N - mean * mean);
    const double se   = (n_sources < 2) ? 0.0 : std::sqrt(var / N);
    const int    avg_iters = (n_sources > 0)
        ? static_cast<int>(iters / static_cast<long>(n_sources)) : 0;

    return CondensateResult{ mean, se, avg_iters, all_ok };
}

// ----------------------------------------------------------------------------
// Exact free-field 2D condensate (U ≡ 0). Closed form: the free Wilson-Dirac
// operator is diagonal in momentum space, and tr D(p)⁻¹ = 2 M(p) / (M² + s²)
// with M = m + 2 − Σ cos p_μ and s² = Σ sin² p_μ. The condensate is the
// momentum average.
// ----------------------------------------------------------------------------

inline double freeFieldCondensate2D(const Lattice<2>& lattice, double mass)
{
    constexpr double kTwoPi = 6.28318530717958647692;
    const int L0 = lattice.extent(0);
    const int L1 = lattice.extent(1);
    double sum = 0.0;
    for (int k0 = 0; k0 < L0; ++k0)
        for (int k1 = 0; k1 < L1; ++k1)
        {
            const double p0 = kTwoPi * static_cast<double>(k0) / static_cast<double>(L0);
            const double p1 = kTwoPi * static_cast<double>(k1) / static_cast<double>(L1);
            const double M  = mass + 2.0 - std::cos(p0) - std::cos(p1);
            const double s0 = std::sin(p0);
            const double s1 = std::sin(p1);
            const double s2 = s0 * s0 + s1 * s1;
            sum += 2.0 * M / (M * M + s2);
        }
    return sum / static_cast<double>(L0 * L1);
}

} // namespace lqft::obs
