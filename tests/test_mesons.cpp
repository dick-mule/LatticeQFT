/**
 * @file test_mesons.cpp
 * @brief The dense propagator and the pseudoscalar correlators on known backgrounds.
 *
 *   1. D · D⁻¹ = I to round-off on a hot U(1) background (the inverse is exact).
 *   2. γ⁵-Hermiticity of the propagator: S(y, x) = γ⁵ S(x, y)† γ⁵.
 *   3. Free field (U = 0): the connected pseudoscalar correlator is positive, symmetric
 *      under t → L_t − t, and its effective mass at the plateau is twice the free Wilson
 *      pole mass, 2 ln(1 + m): two fermions at rest.
 *   4. Free field: the disconnected piece vanishes (tr γ⁵ S(x, x) = 0 by parity at U = 0).
 */

#include "fields/link_field.hpp"
#include "lattice/lattice.hpp"
#include "math/dirac_2d.hpp"
#include "models/u1.hpp"
#include "observables/meson_correlators.hpp"
#include "rng/rng.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <complex>

using namespace lqft;
using Complex = std::complex<double>;

TEST(DensePropagator, InverseIsExactOnHotBackground)
{
    Lattice<2> lat = Lattice<2>::cube(6);
    LinkField<double, 2> U(lat);
    Rng rng(3u);
    u1::U1Model<2>::hot(U, rng);
    dirac::WilsonDirac2D D(lat, 0.3);
    const auto M = obs::denseDirac(D, U, lat);
    const auto S = obs::densePropagator(D, U, lat);
    const int n = M.n;
    double worst = 0.0;
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
        {
            Complex acc{0, 0};
            for (int k = 0; k < n; ++k) acc += M(i, k) * S(k, j);
            worst = std::max(worst, std::abs(acc - ((i == j) ? Complex{1, 0} : Complex{0, 0})));
        }
    EXPECT_LT(worst, 1e-10);
}

TEST(DensePropagator, Gamma5Hermiticity)
{
    Lattice<2> lat = Lattice<2>::cube(6);
    LinkField<double, 2> U(lat);
    Rng rng(5u);
    u1::U1Model<2>::hot(U, rng);
    dirac::WilsonDirac2D D(lat, 0.2);
    const auto S = obs::densePropagator(D, U, lat);
    const int V = lat.volume();
    const double g5[2] = {1.0, -1.0};
    double worst = 0.0;
    for (int x = 0; x < V; ++x)
        for (int y = 0; y < V; ++y)
            for (int a = 0; a < 2; ++a)
                for (int b = 0; b < 2; ++b)
                {
                    // S_ab(y, x) = γ⁵_a conj(S_ba(x, y)) γ⁵_b
                    const Complex lhs = S(2 * y + a, 2 * x + b);
                    const Complex rhs = g5[a] * std::conj(S(2 * x + b, 2 * y + a)) * g5[b];
                    worst = std::max(worst, std::abs(lhs - rhs));
                }
    EXPECT_LT(worst, 1e-10);
}

TEST(MesonCorrelators, FreeFieldPionIsTheTwoFermionThreshold)
{
    // At U = 0 the "pion" correlator is a two-free-fermion state. Its threshold is twice the
    // Wilson pole mass, 2 ln(1 + m), and in two dimensions the two-particle phase space gives
    // a t^{-1/2} prefactor, so the effective mass approaches the threshold from ABOVE as
    // 2 ln(1 + m) + O(1/t): monotone decreasing, never below threshold, within 10% by t = 10
    // on L = 24 (measured: 0.480 vs 0.446 at m = 0.25).
    constexpr int L = 24;
    constexpr double m = 0.25;
    Lattice<2> lat = Lattice<2>::cube(L);
    LinkField<double, 2> U(lat);
    for (int s = 0; s < lat.volume(); ++s) for (int mu = 0; mu < 2; ++mu) U(s, mu) = 0.0;
    dirac::WilsonDirac2D D(lat, m);
    const auto S = obs::densePropagator(D, U, lat);
    const auto C = obs::mesonCorrelators(S, lat);
    for (int t = 0; t < L; ++t)
    {
        EXPECT_GT(C.conn[static_cast<std::size_t>(t)], 0.0);
        EXPECT_NEAR(C.conn[static_cast<std::size_t>(t)], C.conn[static_cast<std::size_t>((L - t) % L)], 1e-10 * C.conn[0]);
        EXPECT_NEAR(C.disc[static_cast<std::size_t>(t)], 0.0, 1e-12);
    }
    const double threshold = 2.0 * std::log(1.0 + m);
    double prev = 1e9;
    for (int t = 3; t <= 10; ++t)
    {
        const double Meff = obs::effectiveMassCosh(C.conn[static_cast<std::size_t>(t)], C.conn[static_cast<std::size_t>(t + 1)], t, L);
        ASSERT_FALSE(std::isnan(Meff));
        EXPECT_GT(Meff, threshold) << "t = " << t;
        EXPECT_LT(Meff, prev)      << "t = " << t;
        prev = Meff;
    }
    EXPECT_LT(prev, threshold * 1.10) << "Meff(10) = " << prev << " threshold " << threshold;
}
