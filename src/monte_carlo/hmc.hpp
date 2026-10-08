#pragma once

/**
 * @file hmc.hpp
 * @brief Hybrid (Hamiltonian) Monte Carlo with leapfrog evolution.
 *
 * Augments the gauge field θ_μ(x) ∈ ℝ with conjugate momenta P_μ(x) ∈ ℝ
 * (mass = 1 implicit) and samples from the extended distribution
 *
 *     ρ(θ, P) ∝ exp(−H(θ, P)),    H = ½ Σ P² + S[θ].
 *
 * One trajectory:
 *   1. Refresh momenta:        P_μ(x) ← N(0, 1) iid.
 *   2. Evolve (θ, P) for `n_steps` leapfrog steps of size `dt`. The
 *      leapfrog integrator is symplectic and time-reversible, so detailed
 *      balance follows from a Metropolis accept/reject on the final ΔH.
 *   3. Accept (θ', P') with probability min(1, exp(−ΔH)). If rejected,
 *      restore the pre-trajectory θ.
 *
 * Leapfrog (kick-drift form, n_steps full drifts, n_steps + 1 force evals):
 *
 *     P ← P + (dt/2) F[θ]
 *     repeat n_steps − 1 times:
 *         θ ← θ + dt · P
 *         P ← P + dt · F[θ]
 *     θ ← θ + dt · P
 *     P ← P + (dt/2) F[θ]
 *
 * ΔH per trajectory scales as O(dt²) for fixed total time τ = n_steps · dt.
 * The acceptance rate at small dt is ≈ exp(-σ_dH²/2) with σ_dH = O(dt²) too,
 * so a sensible tuning point is dt such that the per-trajectory ΔH std-dev
 * is around 0.5.
 *
 * Template parameter `ForceProvider` is any model that exposes:
 *
 *     using FieldT = ...;
 *     double totalAction(const FieldT&) const;
 *     void   computeAllGaugeForces(const FieldT&, FieldT&) const;
 *     const Lattice<Dim>& lattice() const;
 *
 * `u1::U1Model<Dim>` implements this interface; a combined gauge + fermion
 * ForceProvider for full dynamical Schwinger will plug in unchanged at
 * Round 3b.
 */

#include "../rng/rng.hpp"

#include <cmath>
#include <cstdint>

namespace lqft::hmc
{

template<typename ForceProvider>
class GaugeHMC
{
public:
    using FieldT = typename ForceProvider::FieldT; // a LinkField<double, Dim>

    GaugeHMC(const ForceProvider& provider,
             double dt,
             int    n_steps)
        : m_provider(provider)
        , m_dt(dt)
        , m_n_steps(n_steps)
        , m_force (provider.lattice())
    {}

    double dt()     const { return m_dt; }
    int    nSteps() const { return m_n_steps; }
    void   setDt(double dt) { m_dt = dt; }
    void   setNSteps(int n) { m_n_steps = n; }

    struct Result
    {
        double dH;
        bool   accepted;
    };

    /// One full trajectory: refresh P, leapfrog, accept/reject.
    Result trajectory(FieldT& theta, Rng& rng)
    {
        FieldT P(m_provider.lattice());
        sampleMomenta(P, rng);

        FieldT theta_orig = theta;
        const double H_orig = hamiltonian(theta, P);

        leapfrog(theta, P);

        const double H_new = hamiltonian(theta, P);
        const double dH    = H_new - H_orig;

        bool accepted = (dH <= 0.0) || (rng.uniform() < std::exp(-dH));
        if (!accepted) theta = theta_orig;
        ++m_attempted_total;
        if (accepted) ++m_accepted_total;
        return Result{ dH, accepted };
    }

    /// Pure leapfrog evolution with the configured (dt, n_steps).
    /// Exposed for the reversibility / energy-conservation tests.
    void leapfrog(FieldT& theta, FieldT& P)
    {
        const int N_dof = theta.volume() * FieldT::dimension();

        // Initial half-kick.
        m_provider.computeAllGaugeForces(theta, m_force);
        for (int i = 0; i < N_dof; ++i) P.data()[i] += 0.5 * m_dt * m_force.data()[i];

        for (int step = 0; step < m_n_steps - 1; ++step)
        {
            for (int i = 0; i < N_dof; ++i) theta.data()[i] += m_dt * P.data()[i];
            m_provider.computeAllGaugeForces(theta, m_force);
            for (int i = 0; i < N_dof; ++i) P.data()[i] += m_dt * m_force.data()[i];
        }

        // Final drift + half-kick.
        for (int i = 0; i < N_dof; ++i) theta.data()[i] += m_dt * P.data()[i];
        m_provider.computeAllGaugeForces(theta, m_force);
        for (int i = 0; i < N_dof; ++i) P.data()[i] += 0.5 * m_dt * m_force.data()[i];
    }

    double kineticEnergy(const FieldT& P) const
    {
        const int N = P.volume() * FieldT::dimension();
        double sum = 0.0;
        for (int i = 0; i < N; ++i) sum += P.data()[i] * P.data()[i];
        return 0.5 * sum;
    }

    double hamiltonian(const FieldT& theta, const FieldT& P) const
    {
        return kineticEnergy(P) + m_provider.totalAction(theta);
    }

    /// Each P component independently N(0, 1). Standard Gaussian momenta.
    void sampleMomenta(FieldT& P, Rng& rng) const
    {
        const int N = P.volume() * FieldT::dimension();
        for (int i = 0; i < N; ++i) P.data()[i] = rng.normal();
    }

    double cumulativeAcceptance() const
    {
        return m_attempted_total == 0
            ? 0.0
            : static_cast<double>(m_accepted_total) / static_cast<double>(m_attempted_total);
    }

    void resetCounters() { m_accepted_total = m_attempted_total = 0; }

private:
    const ForceProvider& m_provider;
    double               m_dt;
    int                  m_n_steps;
    FieldT               m_force;             // reused workspace
    std::uint64_t        m_accepted_total  = 0;
    std::uint64_t        m_attempted_total = 0;
};

} // namespace lqft::hmc
