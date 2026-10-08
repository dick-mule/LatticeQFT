#pragma once

/**
 * @file schwinger_hmc.hpp
 * @brief Pseudofermion provider for the 2D Schwinger model with HMC.
 *
 * Adds the dynamical-fermion contribution to the gauge HMC by implementing
 * the same `ForceProvider` interface that `hmc::GaugeHMC` already consumes
 * (`totalAction`, `computeAllGaugeForces`, `lattice()` — plus `FieldT`).
 *
 * Effective action:
 *
 *     S[θ] = S_gauge[θ] + log |det D[θ]|²
 *
 * The fermion determinant is sampled via a pseudofermion field φ (complex,
 * two spinor components per site) drawn fresh at the start of every
 * trajectory:
 *
 *     χ ∼ N(0, ½) complex Gaussian
 *     φ = D† χ                                ⇒ P(φ | θ) ∝ exp(−φ†(D†D)⁻¹ φ)
 *
 * Then `S_pf = φ† (D†D)⁻¹ φ` is integrated by Hamiltonian dynamics. Evaluating
 * S_pf requires one CG solve; evaluating the force requires one CG solve plus
 * one apply, and the force at every link is collected in a single sweep.
 *
 * The link force derivation: ∂D/∂θ_μ(x) is sparse (only the (y = x, x+μ)
 * and (y = x+μ, x) matrix elements depend on θ_μ(x)), so
 *
 *   F_pf(x, μ) = -∂S_pf/∂θ_μ(x)
 *              = 2 Re[ψ† D† (∂D/∂θ_μ(x)) ψ]
 *              = Im[U_μ(x)  · A_μ] − Im[U_μ†(x) · B_μ],
 *
 * where ψ = (D†D)⁻¹ φ, η = D ψ, and
 *
 *   μ = 0:  A = (η_0*−η_1*)(x)   · (ψ_0−ψ_1)(x+μ)
 *           B = (η_0*+η_1*)(x+μ) · (ψ_0+ψ_1)(x)
 *
 *   μ = 1:  A = (η_0*−i η_1*)(x)   · (ψ_0+i ψ_1)(x+μ)
 *           B = (η_0*+i η_1*)(x+μ) · (ψ_0−i ψ_1)(x).
 *
 * This is the projector trick applied to ∂D/∂θ — same algebra that
 * collapsed the original `apply`, reused.
 *
 * Driver pattern:
 *
 *     SchwingerProvider provider(gauge, dirac, lattice);
 *     hmc::GaugeHMC<SchwingerProvider> hmc(provider, dt, n_steps);
 *     for trajectory in range(N):
 *         provider.refreshPseudofermion(U, rng);   // re-sample φ
 *         hmc.trajectory(U, rng);                   // leapfrog + accept/reject
 */

#include "../fields/link_field.hpp"
#include "../fields/spinor_field.hpp"
#include "../lattice/lattice.hpp"
#include "../math/dirac_2d.hpp"
#include "../models/u1.hpp"
#include "../rng/rng.hpp"
#include "../solvers/conjugate_gradient.hpp"

#include <cmath>
#include <complex>

namespace lqft::schwinger
{

class SchwingerProvider
{
public:
    using FieldT     = LinkField<double, 2>;
    using SpinorT    = SpinorField<2, 2>;
    using GaugeModel = u1::U1Model<2>;
    using Complex    = std::complex<double>;

    SchwingerProvider(const GaugeModel& gauge,
                      const dirac::WilsonDirac2D& dirac,
                      const Lattice<2>& lattice,
                      double cg_tol      = 1e-10,
                      int    cg_max_iters = 5000)
        : m_gauge(gauge)
        , m_dirac(dirac)
        , m_lattice(lattice)
        , m_cg_tol(cg_tol)
        , m_cg_max_iters(cg_max_iters)
        , m_phi(lattice)
        , m_psi(lattice)
        , m_eta(lattice)
        , m_tmp(lattice)
        , m_r(lattice), m_p(lattice), m_Ap(lattice)
    {}

    const Lattice<2>& lattice() const { return m_lattice; }

    /// Sample χ ∼ complex N(0, ½) per component and set φ = D† χ.
    /// This makes the marginal of φ exactly exp(−φ†(D†D)⁻¹ φ).
    void refreshPseudofermion(const FieldT& U, Rng& rng)
    {
        const int N = m_psi.totalSize();
        const double sigma = 1.0 / std::sqrt(2.0);
        for (int i = 0; i < N; ++i)
            m_psi.data()[i] = Complex{ sigma * rng.normal(), sigma * rng.normal() };
        m_dirac.applyDagger(U, m_psi, m_phi);
    }

    /// S_gauge + S_pf. S_pf = Re ⟨φ, (D†D)⁻¹ φ⟩ — one CG solve.
    double totalAction(const FieldT& U) const
    {
        const double S_gauge = m_gauge.totalAction(U);

        m_psi.zero();
        auto apply_DagD = [&](const SpinorT& in, SpinorT& out)
        {
            m_dirac.applyDagD(U, in, out, m_tmp);
        };
        solvers::conjugateGradient<2, 2>(
            apply_DagD, m_phi, m_psi,
            m_cg_tol, m_cg_max_iters, m_r, m_p, m_Ap);
        const double S_pf = inner_product(m_phi, m_psi).real();
        return S_gauge + S_pf;
    }

    /// Gauge force + fermion force on every link, written into `force`.
    void computeAllGaugeForces(const FieldT& U, FieldT& force) const
    {
        // 1. Gauge sector.
        m_gauge.computeAllGaugeForces(U, force);

        // 2. Solve ψ = (D†D)⁻¹ φ, then η = D ψ. These are the two spinors
        //    that show up in the fermion-force formula.
        m_psi.zero();
        auto apply_DagD = [&](const SpinorT& in, SpinorT& out)
        {
            m_dirac.applyDagD(U, in, out, m_tmp);
        };
        solvers::conjugateGradient<2, 2>(
            apply_DagD, m_phi, m_psi,
            m_cg_tol, m_cg_max_iters, m_r, m_p, m_Ap);
        m_dirac.apply(U, m_psi, m_eta);

        // 3. Walk every link and add F_pf(x, μ) = Im[U A] − Im[U† B].
        const Complex I_unit(0.0, 1.0);
        const int     V = m_lattice.volume();
        for (int s = 0; s < V; ++s)
        {
            const int s_p0 = m_lattice.forward(s, 0);
            const int s_p1 = m_lattice.forward(s, 1);
            const Complex U_0  = std::polar(1.0,  U(s, 0));
            const Complex U_1  = std::polar(1.0,  U(s, 1));
            const Complex U_0d = std::conj(U_0);
            const Complex U_1d = std::conj(U_1);

            // μ = 0
            const Complex A0 = (std::conj(m_eta(s,   0)) - std::conj(m_eta(s,   1)))
                             * (m_psi(s_p0, 0) - m_psi(s_p0, 1));
            const Complex B0 = (std::conj(m_eta(s_p0, 0)) + std::conj(m_eta(s_p0, 1)))
                             * (m_psi(s,   0) + m_psi(s,   1));
            force(s, 0) += (U_0 * A0).imag() - (U_0d * B0).imag();

            // μ = 1
            const Complex A1 = (std::conj(m_eta(s,   0)) - I_unit * std::conj(m_eta(s,   1)))
                             * (m_psi(s_p1, 0) + I_unit * m_psi(s_p1, 1));
            const Complex B1 = (std::conj(m_eta(s_p1, 0)) + I_unit * std::conj(m_eta(s_p1, 1)))
                             * (m_psi(s,   0) - I_unit * m_psi(s,   1));
            force(s, 1) += (U_1 * A1).imag() - (U_1d * B1).imag();
        }
    }

    /// Read-only access to the current pseudofermion field (tests).
    const SpinorT& phi() const { return m_phi; }

    /// Direct evaluation of just S_pf (without the gauge piece). Used by
    /// the finite-difference check for the fermion force.
    double pseudofermionAction(const FieldT& U) const
    {
        m_psi.zero();
        auto apply_DagD = [&](const SpinorT& in, SpinorT& out)
        {
            m_dirac.applyDagD(U, in, out, m_tmp);
        };
        solvers::conjugateGradient<2, 2>(
            apply_DagD, m_phi, m_psi,
            m_cg_tol, m_cg_max_iters, m_r, m_p, m_Ap);
        return inner_product(m_phi, m_psi).real();
    }

private:
    const GaugeModel&            m_gauge;
    const dirac::WilsonDirac2D&  m_dirac;
    const Lattice<2>&            m_lattice;
    double                       m_cg_tol;
    int                          m_cg_max_iters;

    SpinorT                      m_phi;   // owned: pseudofermion (refresh changes)
    mutable SpinorT              m_psi;   // workspace
    mutable SpinorT              m_eta;
    mutable SpinorT              m_tmp;
    mutable SpinorT              m_r, m_p, m_Ap;
};

} // namespace lqft::schwinger
