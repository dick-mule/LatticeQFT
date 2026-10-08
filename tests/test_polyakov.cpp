/**
 * @file test_polyakov.cpp
 * @brief Polyakov loops on an asymmetric SU(2) lattice.
 *
 *   1. Cold configuration: P = 1 at every spatial site, P̄ = 1.
 *   2. Cyclicity: the loop started from any t on the same line has the same trace.
 *   3. Z₂ centre transformation: flipping the sign of every temporal link on one time
 *      slice flips every Polyakov loop (P → −P) while every plaquette is unchanged —
 *      the lattice statement that ⟨P⟩ is a centre-symmetry order parameter.
 *   4. Hot configuration on 8³×2: |P̄| is small (confined-like, strong coupling) while the
 *      per-site loops are O(1).
 */

#include "fields/link_field.hpp"
#include "lattice/lattice.hpp"
#include "math/su2.hpp"
#include "models/su2.hpp"
#include "observables/polyakov.hpp"
#include "rng/rng.hpp"

#include <gtest/gtest.h>

#include <cmath>

using namespace lqft;

TEST(Polyakov, ColdIsOne)
{
    Lattice<4> lat({4, 4, 4, 2});
    LinkField<su2::Element, 4> U(lat);
    su2_model::SU2Model<4>::cold(U);
    EXPECT_NEAR(obs::averagePolyakovLoop(lat, U, 3, 2), 1.0, 1e-12);
}

TEST(Polyakov, CyclicAlongTheLine)
{
    Lattice<3> lat({6, 6, 4});
    LinkField<su2::Element, 3> U(lat);
    Rng rng(11u);
    su2_model::SU2Model<3>::hot(U, rng);
    const int s0 = lat.siteIndex({2, 3, 0});
    const double P0 = obs::polyakovLoopAt(lat, U, s0, 2, 4);
    int s = s0;
    for (int t = 1; t < 4; ++t)
    {
        s = lat.forward(s, 2);
        EXPECT_NEAR(obs::polyakovLoopAt(lat, U, s, 2, 4), P0, 1e-12) << "start t = " << t;
    }
}

TEST(Polyakov, CentreFlipNegatesLoopsAndKeepsPlaquettes)
{
    Lattice<4> lat({4, 4, 4, 4});
    LinkField<su2::Element, 4> U(lat);
    Rng rng(5u);
    su2_model::SU2Model<4>::hot(U, rng);
    const double plaq0 = su2_model::averagePlaquette(lat, U);
    std::vector<double> P0;
    for (int s = 0; s < lat.volume(); ++s)
        if (lat.coords(s)[3] == 0) P0.push_back(obs::polyakovLoopAt(lat, U, s, 3, 4));
    // Centre transformation: U_t(x, t = 1) -> -U_t(x, t = 1) for all x.
    for (int s = 0; s < lat.volume(); ++s)
        if (lat.coords(s)[3] == 1) U(s, 3) = su2::scale(-1.0, U(s, 3));
    EXPECT_NEAR(su2_model::averagePlaquette(lat, U), plaq0, 1e-12);
    std::size_t k = 0;
    for (int s = 0; s < lat.volume(); ++s)
        if (lat.coords(s)[3] == 0) { EXPECT_NEAR(obs::polyakovLoopAt(lat, U, s, 3, 4), -P0[k], 1e-12); ++k; }
}

TEST(Polyakov, HotIsSmallOnAverage)
{
    Lattice<4> lat({8, 8, 8, 2});
    LinkField<su2::Element, 4> U(lat);
    Rng rng(21u);
    su2_model::SU2Model<4>::hot(U, rng);
    EXPECT_LT(std::abs(obs::averagePolyakovLoop(lat, U, 3, 2)), 0.1);
}
