/**
 * @file test_hmc.cpp
 * @brief Tests of the gauge-sector HMC infrastructure.
 *
 *   1. gauge force = -∂S/∂θ at a single link, verified by finite difference
 *      against the existing `totalAction`.
 *   2. Leapfrog reversibility: forward N steps, flip P, forward N steps
 *      returns to the original (θ, -P) to round-off.
 *   3. Energy conservation: ΔH per trajectory scales as dt² at fixed total
 *      time τ = N · dt.
 *   4. Gauge-only HMC reproduces the U(1) plaquette expectation of link
 *      Metropolis at the same β.
 */

#include "fields/link_field.hpp"
#include "lattice/lattice.hpp"
#include "models/u1.hpp"
#include "monte_carlo/hmc.hpp"
#include "monte_carlo/metropolis.hpp"
#include "monte_carlo/tuning.hpp"
#include "observables/observables.hpp"
#include "rng/rng.hpp"

#include <gtest/gtest.h>
#include <cmath>

using namespace lqft;

TEST(GaugeForce, MatchesFiniteDifferenceOfTotalAction)
{
    // Verify gaugeForce(θ, site, μ) ≈ -[S(θ + ε) - S(θ - ε)] / (2ε) on a
    // random hot configuration. Stress every plaquette plane and a
    // handful of sites.
    Lattice<2> lattice = Lattice<2>::cube(6);
    LinkField<double, 2> U(lattice);
    Rng rng(1u);
    u1::U1Model<2>::hot(U, rng);

    u1::U1Model<2> model(lattice, /*β*/2.3);
    constexpr double eps = 1e-6;

    for (int site : {0, 7, 15, 22, 31})
        for (int mu = 0; mu < 2; ++mu)
        {
            const double F_analytic = model.gaugeForce(U, site, mu);

            const double original = U(site, mu);
            U(site, mu) = original + eps;
            const double S_plus  = model.totalAction(U);
            U(site, mu) = original - eps;
            const double S_minus = model.totalAction(U);
            U(site, mu) = original;
            const double F_fd = -(S_plus - S_minus) / (2.0 * eps);

            EXPECT_NEAR(F_analytic, F_fd, 1e-5)
                << "site " << site << " μ=" << mu;
        }
}

TEST(GaugeHMC, LeapfrogIsTimeReversible)
{
    // Evolve forward, flip momenta, evolve forward again. End state
    // should match (θ_0, -P_0) up to round-off.
    Lattice<2> lattice = Lattice<2>::cube(6);
    LinkField<double, 2> theta(lattice), theta0(lattice);
    LinkField<double, 2> P(lattice),     P0(lattice);
    Rng rng(7u);
    u1::U1Model<2>::hot(theta, rng);

    u1::U1Model<2> model(lattice, /*β*/1.8);
    hmc::GaugeHMC hmc(model, /*dt*/0.1, /*n_steps*/12);
    hmc.sampleMomenta(P, rng);

    // Snapshot.
    theta0 = theta;
    P0     = P;

    hmc.leapfrog(theta, P);
    // Flip momenta and evolve again; net result is the original state.
    const int N = P.volume() * 2;
    for (int i = 0; i < N; ++i) P.data()[i] = -P.data()[i];
    hmc.leapfrog(theta, P);
    for (int i = 0; i < N; ++i) P.data()[i] = -P.data()[i];

    double dtheta2 = 0.0, dP2 = 0.0;
    for (int i = 0; i < N; ++i)
    {
        const double dth = theta.data()[i] - theta0.data()[i];
        const double dp  = P    .data()[i] - P0    .data()[i];
        dtheta2 += dth * dth;
        dP2     += dp  * dp;
    }
    EXPECT_LT(std::sqrt(dtheta2), 1e-9);
    EXPECT_LT(std::sqrt(dP2),     1e-9);
}

TEST(GaugeHMC, EnergyChangeScalesAsDtSquared)
{
    // For fixed total trajectory time τ = n_steps · dt, leapfrog is a
    // second-order symplectic integrator: ΔH per trajectory scales as
    // dt². Compare ΔH at dt = 0.04 vs dt = 0.08; ratio should be ≈ 4
    // (within a factor of a few from stochastic initial conditions).
    Lattice<2> lattice = Lattice<2>::cube(8);
    LinkField<double, 2> theta(lattice);
    LinkField<double, 2> P(lattice);
    Rng rng(31337u);
    u1::U1Model<2>::hot(theta, rng);

    u1::U1Model<2> model(lattice, /*β*/2.0);

    auto dH_at = [&](double dt, int n)
    {
        LinkField<double, 2> t = theta;
        LinkField<double, 2> p(lattice);
        hmc::GaugeHMC hmc(model, dt, n);
        hmc.sampleMomenta(p, rng);
        const double H0 = hmc.hamiltonian(t, p);
        hmc.leapfrog(t, p);
        const double H1 = hmc.hamiltonian(t, p);
        return std::abs(H1 - H0);
    };

    // τ = 0.4 fixed; vary dt.
    const double dH_small = dH_at(0.04, 10);
    const double dH_large = dH_at(0.08, 5);
    // Expect dH_large ≈ 4 · dH_small. Allow loose factor for stochastic
    // momentum draws giving slightly different trajectories.
    EXPECT_GT(dH_large, 2.0 * dH_small);
    EXPECT_LT(dH_large, 8.0 * dH_small);
}

TEST(GaugeHMC, ReproducesMetropolisPlaquetteAtFixedBeta)
{
    // Long-run plaquette expectation from gauge-only HMC should agree
    // with link Metropolis at the same β within MC error.
    constexpr int    L    = 12;
    constexpr double beta = 1.5;
    Lattice<2> lattice = Lattice<2>::cube(L);

    // ---- Metropolis baseline ----
    double plaq_metro;
    {
        LinkField<double, 2> U(lattice);
        Rng rng(42u);
        u1::U1Model<2>::hot(U, rng);
        u1::U1Model<2> model(lattice, beta);
        mc::autoTuneStepSize(model, U, rng, /*batch*/80, /*max*/12, /*target*/0.5, /*tol*/0.05);
        mc::MetropolisSweep<u1::U1Model<2>> sweeper(model);
        sweeper.sweepN(U, rng, 600);
        obs::Mean p;
        for (int s = 0; s < 1500; ++s)
        {
            sweeper.sweep(U, rng);
            if ((s % 3) == 0) p.add(u1::averagePlaquette(lattice, U));
        }
        plaq_metro = p.mean();
    }

    // ---- HMC ----
    double plaq_hmc;
    {
        LinkField<double, 2> U(lattice);
        Rng rng(123u);
        u1::U1Model<2>::hot(U, rng);
        u1::U1Model<2> model(lattice, beta);
        hmc::GaugeHMC hmc(model, /*dt*/0.05, /*n_steps*/20);
        // Thermalize.
        for (int s = 0; s < 200; ++s) hmc.trajectory(U, rng);
        hmc.resetCounters();
        obs::Mean p;
        for (int s = 0; s < 800; ++s)
        {
            hmc.trajectory(U, rng);
            if ((s % 2) == 0) p.add(u1::averagePlaquette(lattice, U));
        }
        plaq_hmc = p.mean();
        // Acceptance at these (dt, n_steps) should be comfortably high.
        EXPECT_GT(hmc.cumulativeAcceptance(), 0.70)
            << "acc=" << hmc.cumulativeAcceptance();
    }

    EXPECT_NEAR(plaq_hmc, plaq_metro, 0.02)
        << "metro=" << plaq_metro << "  hmc=" << plaq_hmc;
}
