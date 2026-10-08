/**
 * @file test_stout.cpp
 * @brief Stout-smearing invariants — analytic SU(2) variant.
 *
 *   1. Output is still in SU(2) at every link.
 *   2. Cold (identity) configuration is a fixed point.
 *   3. Thermalized configuration smooths (mean plaquette grows).
 *   4. Direction mask: flagged-false links bit-identical.
 *   5. ρ = 0 is the identity transformation (no-op).
 */

#include "fields/link_field.hpp"
#include "lattice/lattice.hpp"
#include "math/su2.hpp"
#include "models/su2.hpp"
#include "monte_carlo/su2_heatbath.hpp"
#include "rng/rng.hpp"
#include "smearing/stout.hpp"

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

TEST(Stout, PreservesSU2)
{
    Lattice<3> L = Lattice<3>::cube(6);
    LinkField<su2::Element, 3> U(L), scratch(L);
    Rng rng(321u);
    su2_model::SU2Model<3>::hot(U, rng);

    std::array<bool, 3> spatial = { true, true, false };
    smearing::stoutSmearN<3>(L, U, scratch, /*ρ*/0.10, /*iters*/12, spatial);
    EXPECT_LT(maxSU2Deviation(U), 1e-12);
}

TEST(Stout, ColdIsFixedPoint)
{
    Lattice<3> L = Lattice<3>::cube(5);
    LinkField<su2::Element, 3> U(L), scratch(L);
    su2_model::SU2Model<3>::cold(U);

    std::array<bool, 3> spatial = { true, true, false };
    smearing::stoutSmearN<3>(L, U, scratch, /*ρ*/0.15, /*iters*/8, spatial);

    EXPECT_LT(maxSU2Deviation(U), 1e-14);
    EXPECT_NEAR(meanPlaquette3D(L, U), 1.0, 1e-12);
}

TEST(Stout, SmoothsThermalizedConfiguration)
{
    Lattice<3> L = Lattice<3>::cube(8);
    LinkField<su2::Element, 3> U(L), scratch(L);
    Rng rng(101u);
    su2_model::SU2Model<3>::hot(U, rng);

    su2_model::SU2Model<3> model(L, /*β*/5.0, /*step*/0.4);
    su2_model::heatBathSweepN(model, U, rng, /*sweeps*/40);

    const double plaq_before = meanPlaquette3D(L, U);
    std::array<bool, 3> spatial = { true, true, true };
    smearing::stoutSmearN<3>(L, U, scratch, /*ρ*/0.12, /*iters*/6, spatial);
    const double plaq_after = meanPlaquette3D(L, U);

    EXPECT_GT(plaq_before, 0.5) << "thermalization didn't take";
    EXPECT_GT(plaq_after, plaq_before)
        << "before=" << plaq_before << "  after=" << plaq_after;
}

TEST(Stout, RespectsDirectionMask)
{
    Lattice<3> L = Lattice<3>::cube(6);
    LinkField<su2::Element, 3> U(L), U_copy(L), scratch(L);
    Rng rng(55u);
    su2_model::SU2Model<3>::hot(U, rng);
    U_copy = U;

    std::array<bool, 3> only_x = { true, false, false };
    smearing::stoutSmearN<3>(L, U, scratch, /*ρ*/0.1, /*iters*/3, only_x);

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

TEST(Stout, ZeroRhoIsIdentity)
{
    Lattice<3> L = Lattice<3>::cube(5);
    LinkField<su2::Element, 3> U(L), U_copy(L), scratch(L);
    Rng rng(7u);
    su2_model::SU2Model<3>::hot(U, rng);
    U_copy = U;

    std::array<bool, 3> all_spatial = { true, true, true };
    smearing::stoutSmearN<3>(L, U, scratch, /*ρ*/0.0, /*iters*/5, all_spatial);

    const int V = L.volume();
    for (int s = 0; s < V; ++s)
        for (int mu = 0; mu < 3; ++mu)
        {
            const auto& a = U(s, mu);
            const auto& b = U_copy(s, mu);
            EXPECT_NEAR(a.s, b.s, 1e-14);
            for (int k = 0; k < 3; ++k)
                EXPECT_NEAR(a.v[k], b.v[k], 1e-14);
        }
}
