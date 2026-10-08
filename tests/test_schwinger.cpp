/**
 * @file test_schwinger.cpp
 * @brief Tests of the 2D Schwinger model with dynamical fermions via HMC.
 *
 *   1. Fermion force = -∂S_pf/∂θ at one link — verified by finite difference
 *      of `pseudofermionAction()` (which is the CG solve). Tight CG tol
 *      keeps the FD signal above CG noise.
 *   2. Dynamical HMC runs cleanly: trajectories accept at reasonable rate
 *      and the plaquette settles into a value comparable to quenched at
 *      moderate (β, m).
 *   3. Pseudofermion refresh produces the right marginal: ⟨φ†(D†D)⁻¹ φ⟩ over
 *      many refreshes ≈ dim(spinor space) = 2 · V (the trace identity).
 */

#include "fields/link_field.hpp"
#include "fields/spinor_field.hpp"
#include "lattice/lattice.hpp"
#include "math/dirac_2d.hpp"
#include "models/u1.hpp"
#include "monte_carlo/hmc.hpp"
#include "monte_carlo/schwinger_hmc.hpp"
#include "observables/observables.hpp"
#include "rng/rng.hpp"

#include <gtest/gtest.h>
#include <cmath>

using namespace lqft;

TEST(SchwingerForce, FermionForceMatchesFiniteDifference)
{
    Lattice<2> lattice = Lattice<2>::cube(6);
    LinkField<double, 2> U(lattice);
    Rng rng(42u);
    u1::U1Model<2>::hot(U, rng);

    u1::U1Model<2>           gauge(lattice, /*β*/2.0);
    dirac::WilsonDirac2D     dirac(lattice, /*mass*/0.5);
    schwinger::SchwingerProvider provider(gauge, dirac, lattice,
                                          /*cg_tol*/1e-13, /*cg_max*/8000);
    provider.refreshPseudofermion(U, rng);

    // Compute analytic full force (gauge + fermion).
    LinkField<double, 2> F_full (lattice);
    LinkField<double, 2> F_gauge(lattice);
    provider.computeAllGaugeForces(U, F_full);
    gauge   .computeAllGaugeForces(U, F_gauge);

    constexpr double eps = 1e-5;
    for (int site : {0, 7, 14, 21})
        for (int mu = 0; mu < 2; ++mu)
        {
            const double F_pf_analytic = F_full(site, mu) - F_gauge(site, mu);

            const double original = U(site, mu);
            U(site, mu) = original + eps;
            const double S_plus  = provider.pseudofermionAction(U);
            U(site, mu) = original - eps;
            const double S_minus = provider.pseudofermionAction(U);
            U(site, mu) = original;
            const double F_pf_fd = -(S_plus - S_minus) / (2.0 * eps);

            // FD tolerance: O(eps²) ≈ 1e-10 plus CG residual ≈ 1e-6.
            EXPECT_NEAR(F_pf_analytic, F_pf_fd, 5e-4)
                << "site=" << site << " μ=" << mu
                << "  analytic=" << F_pf_analytic
                << "  fd="       << F_pf_fd;
        }
}

TEST(SchwingerHMC, RefreshGivesCorrectPseudofermionAction)
{
    // ⟨φ†(D†D)⁻¹ φ⟩ over fresh draws = ⟨χ† χ⟩ (since φ = D† χ implies
    // φ†(D†D)⁻¹ φ = χ† χ exactly). For χ_i ∼ complex N(0, ½) each with
    // ⟨|χ_i|²⟩ = 1, the average over draws is the total DOF count
    // dim = Nc · V = 2 · 36 = 72 on an L = 6 lattice.
    Lattice<2> lattice = Lattice<2>::cube(6);
    LinkField<double, 2> U(lattice);
    Rng rng(1u);
    u1::U1Model<2>::hot(U, rng);

    u1::U1Model<2>           gauge(lattice, /*β*/1.8);
    dirac::WilsonDirac2D     dirac(lattice, /*mass*/0.5);
    schwinger::SchwingerProvider provider(gauge, dirac, lattice);

    const int   N_draws = 60;
    const int   expected_dim = 2 * lattice.volume();
    double sum = 0.0;
    for (int k = 0; k < N_draws; ++k)
    {
        provider.refreshPseudofermion(U, rng);
        sum += provider.pseudofermionAction(U);
    }
    const double mean = sum / N_draws;
    const double rel  = std::abs(mean - expected_dim) / static_cast<double>(expected_dim);
    EXPECT_LT(rel, 0.10)
        << "mean=" << mean << " expected=" << expected_dim << " rel=" << rel;
}

TEST(SchwingerHMC, DynamicalTrajectoriesAcceptAndProducePlaquette)
{
    constexpr int    L     = 6;
    constexpr double beta  = 2.0;
    constexpr double mass  = 0.5;
    Lattice<2> lattice = Lattice<2>::cube(L);
    LinkField<double, 2> U(lattice);
    Rng rng(2024u);
    u1::U1Model<2>::hot(U, rng);

    u1::U1Model<2>           gauge(lattice, beta);
    dirac::WilsonDirac2D     dirac(lattice, mass);
    schwinger::SchwingerProvider provider(gauge, dirac, lattice,
                                          /*cg_tol*/1e-10, /*cg_max*/3000);
    hmc::GaugeHMC<schwinger::SchwingerProvider> hmc(provider,
                                                    /*dt*/0.04,
                                                    /*n_steps*/20);

    // Thermalize.
    for (int s = 0; s < 30; ++s)
    {
        provider.refreshPseudofermion(U, rng);
        hmc.trajectory(U, rng);
    }
    hmc.resetCounters();

    obs::Mean plaq;
    for (int s = 0; s < 60; ++s)
    {
        provider.refreshPseudofermion(U, rng);
        hmc.trajectory(U, rng);
        plaq.add(u1::averagePlaquette(lattice, U));
    }

    EXPECT_GT(hmc.cumulativeAcceptance(), 0.5)
        << "acceptance=" << hmc.cumulativeAcceptance();
    // Plaquette is bounded in [-1, 1] for any β; for β = 2 with light
    // dynamical fermions we expect 0.5–0.85 territory.
    EXPECT_GT(plaq.mean(), 0.45);
    EXPECT_LT(plaq.mean(), 0.95);
}
