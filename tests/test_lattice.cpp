/**
 * @file test_lattice.cpp
 * @brief Sanity checks on the Dim-templated lattice infrastructure.
 *
 * These tests don't depend on any model — they check that the topology
 * itself (indexing, neighbors, parity) is correct in 2D, 3D, and 4D so
 * lifting Phase-0 algorithms to higher Dim does not silently break.
 */

#include "lattice/lattice.hpp"
#include <gtest/gtest.h>

using namespace lqft;

// -----------------------------------------------------------------------------
// Volume and indexing round-trip
// -----------------------------------------------------------------------------

TEST(Lattice2D, VolumeIsProductOfShape)
{
    Lattice<2> L({8, 4});
    EXPECT_EQ(L.volume(), 8 * 4);
    EXPECT_EQ(L.extent(0), 8);
    EXPECT_EQ(L.extent(1), 4);
}

TEST(Lattice4D, VolumeIsProductOfShape)
{
    Lattice<4> L({4, 4, 4, 4});
    EXPECT_EQ(L.volume(), 256);
}

TEST(Lattice2D, IndexRoundTrip)
{
    Lattice<2> L = Lattice<2>::cube(6);
    for (int s = 0; s < L.volume(); ++s)
        EXPECT_EQ(L.siteIndex(L.coords(s)), s);
}

TEST(Lattice3D, IndexRoundTrip)
{
    Lattice<3> L = Lattice<3>::cube(5);
    for (int s = 0; s < L.volume(); ++s)
        EXPECT_EQ(L.siteIndex(L.coords(s)), s);
}

// -----------------------------------------------------------------------------
// Neighbors: forward then backward should return us to the start, periodic
// boundaries should wrap, and stepping Lmu times in one direction should
// return us to the starting site.
// -----------------------------------------------------------------------------

TEST(Lattice2D, NeighborsAreInvolutions)
{
    Lattice<2> L = Lattice<2>::cube(7);
    for (int s = 0; s < L.volume(); ++s)
        for (int mu = 0; mu < 2; ++mu)
        {
            EXPECT_EQ(L.backward(L.forward(s, mu), mu), s);
            EXPECT_EQ(L.forward(L.backward(s, mu), mu), s);
        }
}

TEST(Lattice3D, FullLoopPeriodic)
{
    Lattice<3> L = Lattice<3>::cube(5);
    for (int s = 0; s < L.volume(); ++s)
        for (int mu = 0; mu < 3; ++mu)
        {
            int t = s;
            for (int i = 0; i < L.extent(mu); ++i) t = L.forward(t, mu);
            EXPECT_EQ(t, s) << "site " << s << " mu " << mu;
        }
}

// -----------------------------------------------------------------------------
// Parity: even/odd sublattices together cover the lattice exactly once,
// and every neighbor of an even site is odd (this is the property the
// checkerboard MC depends on).
// -----------------------------------------------------------------------------

TEST(Lattice2D, ParityPartitionIsExact)
{
    Lattice<2> L = Lattice<2>::cube(8);
    const auto& even = L.sitesOfParity(0);
    const auto& odd  = L.sitesOfParity(1);
    EXPECT_EQ(even.size() + odd.size(), static_cast<std::size_t>(L.volume()));
    EXPECT_EQ(even.size(), odd.size()); // L even ⇒ equal partition
}

TEST(Lattice3D, NeighborsOfEvenAreOdd)
{
    Lattice<3> L = Lattice<3>::cube(6);
    for (int s : L.sitesOfParity(0))
        for (int mu = 0; mu < 3; ++mu)
        {
            EXPECT_EQ(L.parityOf(L.forward(s, mu)), 1);
            EXPECT_EQ(L.parityOf(L.backward(s, mu)), 1);
        }
}

TEST(Lattice4D, NeighborsOfOddAreEven)
{
    Lattice<4> L = Lattice<4>::cube(4);
    for (int s : L.sitesOfParity(1))
        for (int mu = 0; mu < 4; ++mu)
        {
            EXPECT_EQ(L.parityOf(L.forward(s, mu)), 0);
            EXPECT_EQ(L.parityOf(L.backward(s, mu)), 0);
        }
}
