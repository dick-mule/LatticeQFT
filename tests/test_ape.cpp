/**
 * @file test_ape.cpp
 * @brief APE-smearing invariants.
 *
 *   1. Smearing preserves SU(2) at every link.
 *   2. Smearing the cold (identity) configuration is a fixed point.
 *   3. Smearing a hot configuration strictly increases the mean plaquette
 *      (it smooths short-distance fluctuations).
 *   4. Direction mask: links flagged `false` are bit-identical after smearing.
 */

#include "fields/link_field.hpp"
#include "lattice/lattice.hpp"
#include "math/su2.hpp"
#include "models/su2.hpp"
#include "monte_carlo/su2_heatbath.hpp"
#include "rng/rng.hpp"
#include "smearing/ape.hpp"

#include <gtest/gtest.h>
#include <array>
#include <cmath>

using namespace lqft;

namespace
{

double meanPlaquette3D(const Lattice<3>& L,
                      const LinkField<su2::Element, 3>& U)
{
    double sum = 0.0;
    long long count = 0;
    const int V = L.volume();
    for (int s = 0; s < V; ++s)
        for (int mu = 0; mu < 3; ++mu)
            for (int nu = mu + 1; nu < 3; ++nu)
            {
                const int s_mu = L.forward(s, mu);
                const int s_nu = L.forward(s, nu);
                su2::Element P = su2::multiply(U(s, mu), U(s_mu, nu));
                P = su2::multiply(P, su2::dagger(U(s_nu, mu)));
                P = su2::multiply(P, su2::dagger(U(s, nu)));
                sum += P.s;
                ++count;
            }
    return sum / static_cast<double>(count);
}

double maxSU2Deviation(const LinkField<su2::Element, 3>& U)
{
    double worst = 0.0;
    const int V = U.volume();
    for (int s = 0; s < V; ++s)
        for (int mu = 0; mu < 3; ++mu)
        {
            const auto& e = U(s, mu);
            const double n2 = e.s * e.s + e.v[0] * e.v[0]
                            + e.v[1] * e.v[1] + e.v[2] * e.v[2];
            worst = std::max(worst, std::abs(n2 - 1.0));
        }
    return worst;
}

} // namespace

TEST(APE, PreservesSU2)
{
    Lattice<3> L = Lattice<3>::cube(6);
    LinkField<su2::Element, 3> U(L), scratch(L);
    Rng rng(123u);
    su2_model::SU2Model<3>::hot(U, rng);

    std::array<bool, 3> spatial = { true, true, false };
    smearing::apeSmearN<3>(L, U, scratch, /*α*/0.6, /*iters*/8, spatial);

    EXPECT_LT(maxSU2Deviation(U), 1e-12);
}

TEST(APE, ColdIsFixedPoint)
{
    Lattice<3> L = Lattice<3>::cube(5);
    LinkField<su2::Element, 3> U(L), scratch(L);
    su2_model::SU2Model<3>::cold(U);

    std::array<bool, 3> spatial = { true, true, false };
    smearing::apeSmearN<3>(L, U, scratch, /*α*/0.9, /*iters*/10, spatial);

    // Every link should still be the identity (or numerically indistinguishable).
    EXPECT_LT(maxSU2Deviation(U), 1e-14);
    const double plaq = meanPlaquette3D(L, U);
    EXPECT_NEAR(plaq, 1.0, 1e-12);
}

TEST(APE, SmoothsThermalizedConfiguration)
{
    // APE is a smoothing flow that suppresses high-action fluctuations on top
    // of an *already-smooth* background. On a totally hot configuration the
    // staples themselves are random and APE has nothing to align to. We
    // therefore thermalize at a fairly cold β so the field has real
    // structure, then check that APE drives the plaquette closer to 1.
    Lattice<3> L = Lattice<3>::cube(8);
    LinkField<su2::Element, 3> U(L), scratch(L);
    Rng rng(777u);
    su2_model::SU2Model<3>::hot(U, rng);

    su2_model::SU2Model<3> model(L, /*β*/5.0, /*step*/0.4);
    su2_model::heatBathSweepN(model, U, rng, /*sweeps*/40);

    const double plaq_before = meanPlaquette3D(L, U);
    std::array<bool, 3> spatial = { true, true, true };
    smearing::apeSmearN<3>(L, U, scratch, /*α*/0.6, /*iters*/6, spatial);
    const double plaq_after = meanPlaquette3D(L, U);

    EXPECT_GT(plaq_before, 0.5) << "thermalization didn't take";
    EXPECT_GT(plaq_after, plaq_before)
        << "before=" << plaq_before << "  after=" << plaq_after;
}

TEST(APE, RespectsDirectionMask)
{
    Lattice<3> L = Lattice<3>::cube(6);
    LinkField<su2::Element, 3> U(L), U_copy(L), scratch(L);
    Rng rng(55u);
    su2_model::SU2Model<3>::hot(U, rng);
    U_copy = U;

    // Mask: smear only μ = 0; μ = 1, 2 must be bit-identical after smearing.
    std::array<bool, 3> only_x = { true, false, false };
    smearing::apeSmearN<3>(L, U, scratch, /*α*/0.5, /*iters*/3, only_x);

    const int V = L.volume();
    for (int s = 0; s < V; ++s)
        for (int mu : { 1, 2 })
        {
            const auto& a = U(s, mu);
            const auto& b = U_copy(s, mu);
            EXPECT_EQ(a.s, b.s);
            EXPECT_EQ(a.v[0], b.v[0]);
            EXPECT_EQ(a.v[1], b.v[1]);
            EXPECT_EQ(a.v[2], b.v[2]);
        }
}
