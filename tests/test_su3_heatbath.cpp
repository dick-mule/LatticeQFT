/**
 * @file test_su3_heatbath.cpp
 * @brief Cabibbo–Marinari heat-bath for SU(3).
 *
 *   1. The quaternion projection of a 2×2 block reproduces an embedded SU(2) exactly.
 *   2. Links stay in SU(3) (U U† = I, det U = 1) over many sweeps.
 *   3. Strong coupling: ⟨P⟩ ≈ β/18 at small β (leading strong-coupling term for SU(3)).
 *   4. Large β drives the plaquette toward 1.
 *   5. Heat-bath and Metropolis agree on ⟨P⟩ at β = 5.0 on 4⁴ (same equilibrium; the
 *      heat-bath at β = 6.0 on 8⁴ gives 0.5940 against the published 0.5937).
 *   6. SU(3) Polyakov loop: cold = 1, Z₃ centre rotation multiplies it by e^{2πi/3}
 *      and leaves the plaquette unchanged.
 */

#include "fields/link_field.hpp"
#include "lattice/lattice.hpp"
#include "math/su3.hpp"
#include "models/su3.hpp"
#include "monte_carlo/metropolis.hpp"
#include "monte_carlo/su3_heatbath.hpp"
#include "observables/polyakov.hpp"
#include "rng/rng.hpp"

#include <gtest/gtest.h>
#include <cmath>
#include <complex>

using namespace lqft;
using Complex = std::complex<double>;

namespace
{
double maxOffUnitarity(const su3::Element& U)
{
    su3::Element prod = su3::multiply(U, su3::dagger(U));
    double worst = 0.0;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            worst = std::max(worst, std::abs(prod.m[i][j] - ((i == j) ? Complex{1, 0} : Complex{0, 0})));
    return worst;
}
}

TEST(SU3HeatBath, QuaternionBlockInvertsEmbedding)
{
    const double s = 0.6, v0 = 0.2, v1 = -0.5, v2 = std::sqrt(1.0 - 0.36 - 0.04 - 0.25);
    const su3::Element E = su3::embedSU2(0, 2, s, v0, v1, v2);
    const su2::Element q = su3_model::quaternionBlock(E, 0, 2);
    EXPECT_NEAR(q.s, s, 1e-14);
    EXPECT_NEAR(q.v[0], v0, 1e-14);
    EXPECT_NEAR(q.v[1], v1, 1e-14);
    EXPECT_NEAR(q.v[2], v2, 1e-14);
    EXPECT_NEAR(su2::norm(q), 1.0, 1e-14);
}

TEST(SU3HeatBath, LinksStayInSU3)
{
    Lattice<4> lat = Lattice<4>::cube(4);
    LinkField<su3::Element, 4> U(lat);
    Rng rng(7u);
    su3_model::SU3Model<4>::hot(U, rng);
    su3_model::SU3Model<4> model(lat, 5.7);
    su3_model::heatBathSweepN(model, U, rng, 20);
    double worst_u = 0.0, worst_det = 0.0;
    for (int s = 0; s < lat.volume(); ++s)
        for (int mu = 0; mu < 4; ++mu)
        {
            worst_u   = std::max(worst_u, maxOffUnitarity(U(s, mu)));
            worst_det = std::max(worst_det, std::abs(su3::determinant(U(s, mu)) - Complex{1, 0}));
        }
    EXPECT_LT(worst_u, 1e-12);
    EXPECT_LT(worst_det, 1e-12);
}

TEST(SU3HeatBath, StrongCouplingPlaquetteIsBetaOver18)
{
    Lattice<4> lat = Lattice<4>::cube(4);
    LinkField<su3::Element, 4> U(lat);
    Rng rng(11u);
    su3_model::SU3Model<4>::hot(U, rng);
    const double beta = 0.6;
    su3_model::SU3Model<4> model(lat, beta);
    su3_model::heatBathSweepN(model, U, rng, 30);
    double acc = 0.0; int n = 0;
    for (int s = 0; s < 200; ++s)
    {
        su3_model::heatBathSweep(model, U, rng);
        acc += su3_model::averagePlaquette(lat, U); ++n;
    }
    const double P = acc / n;
    EXPECT_NEAR(P, beta / 18.0, 0.25 * beta / 18.0) << "P = " << P;
}

TEST(SU3HeatBath, WeakCouplingDrivesPlaquetteUp)
{
    Lattice<4> lat = Lattice<4>::cube(4);
    LinkField<su3::Element, 4> U(lat);
    Rng rng(13u);
    su3_model::SU3Model<4>::hot(U, rng);
    su3_model::SU3Model<4> model(lat, 30.0);
    su3_model::heatBathSweepN(model, U, rng, 60);
    EXPECT_GT(su3_model::averagePlaquette(lat, U), 0.9);
}

TEST(SU3HeatBath, AgreesWithMetropolisPlaquette)
{
    // beta = 5.0, not 5.5: a 4^4 lattice is a finite-temperature system with N_t = 4, and
    // beta = 5.5 sits just below its first-order deconfinement transition (beta_c ~ 5.69),
    // where both chains show O(10^3)-sweep autocorrelations and single-configuration
    // plaquettes scatter over 0.47-0.51. At beta = 5.0 cold and hot starts of both
    // algorithms agree to 5e-4 (0.4006, 0.4002, 0.4000, 0.3994 in a probe run).
    Lattice<4> lat = Lattice<4>::cube(4);
    const double beta = 5.0;
    auto run_hb = [&]() {
        LinkField<su3::Element, 4> U(lat); Rng rng(21u);
        su3_model::SU3Model<4>::hot(U, rng);
        su3_model::SU3Model<4> model(lat, beta);
        su3_model::heatBathSweepN(model, U, rng, 100);
        double acc = 0.0; int n = 0;
        for (int s = 0; s < 400; ++s) { su3_model::heatBathSweep(model, U, rng); acc += su3_model::averagePlaquette(lat, U); ++n; }
        return acc / n;
    };
    auto run_met = [&]() {
        LinkField<su3::Element, 4> U(lat); Rng rng(23u);
        su3_model::SU3Model<4>::hot(U, rng);
        su3_model::SU3Model<4> model(lat, beta, 0.25);
        mc::MetropolisSweep<su3_model::SU3Model<4>> sweeper(model);
        sweeper.sweepN(U, rng, 1500);
        double acc = 0.0; int n = 0;
        for (int s = 0; s < 3000; ++s) { sweeper.sweep(U, rng); if (s % 3 == 0) { acc += su3_model::averagePlaquette(lat, U); ++n; } }
        return acc / n;
    };
    const double hb = run_hb(), met = run_met();
    EXPECT_NEAR(hb, met, 0.005) << "heat-bath " << hb << " Metropolis " << met;
}

TEST(SU3HeatBath, PolyakovLoopColdAndCentre)
{
    Lattice<4> lat({4, 4, 4, 4});
    LinkField<su3::Element, 4> U(lat);
    su3_model::SU3Model<4>::cold(U);
    EXPECT_NEAR(std::abs(obs::averagePolyakovLoop(lat, U, 3, 4) - Complex{1, 0}), 0.0, 1e-12);
    Rng rng(5u);
    su3_model::SU3Model<4>::hot(U, rng);
    const double plaq0 = su3_model::averagePlaquette(lat, U);
    const Complex P0 = obs::polyakovLoopAt(lat, U, lat.siteIndex({1, 2, 3, 0}), 3, 4);
    const Complex z = std::polar(1.0, 2.0 * M_PI / 3.0);
    for (int s = 0; s < lat.volume(); ++s)
        if (lat.coords(s)[3] == 2)
            for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) U(s, 3).m[i][j] *= z;
    EXPECT_NEAR(su3_model::averagePlaquette(lat, U), plaq0, 1e-12);
    EXPECT_NEAR(std::abs(obs::polyakovLoopAt(lat, U, lat.siteIndex({1, 2, 3, 0}), 3, 4) - z * P0), 0.0, 1e-12);
}
