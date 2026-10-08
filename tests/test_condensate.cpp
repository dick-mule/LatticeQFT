/**
 * @file test_condensate.cpp
 * @brief Tests of the stochastic chiral condensate estimator.
 *
 *   1. Free-field benchmark (U ≡ 0): the stochastic estimator matches the
 *      closed-form lattice-momentum-sum value.
 *   2. Heavy-mass limit: at large m the operator is m+2 on the diagonal
 *      plus small hopping, so ⟨ψ̄ψ⟩ → 2/(m + 2) per site. Z₂ noise is
 *      *exact* on diagonal contributions, so 1 source suffices.
 *   3. Mass monotonicity: heavier fermion ⇒ smaller condensate.
 */

#include "fields/link_field.hpp"
#include "fields/spinor_field.hpp"
#include "lattice/lattice.hpp"
#include "math/dirac_2d.hpp"
#include "observables/condensate.hpp"
#include "rng/rng.hpp"

#include <gtest/gtest.h>
#include <cmath>

using namespace lqft;

TEST(Condensate, FreeFieldMatchesAnalyticAtModerateMass)
{
    constexpr int    L     = 8;
    constexpr double mass  = 0.5;
    Lattice<2> lattice = Lattice<2>::cube(L);
    LinkField<double, 2> U(lattice); // U ≡ 1 (all θ = 0)

    dirac::WilsonDirac2D D(lattice, mass);
    Rng rng(2024u);

    const auto r = obs::stochasticCondensate(
        D, U, lattice, rng,
        /*n_sources*/200, /*tol*/1e-10, /*cg_max*/2000);

    const double analytic = obs::freeFieldCondensate2D(lattice, mass);
    const double rel = std::abs(r.mean - analytic) / std::abs(analytic);

    EXPECT_TRUE(r.all_converged);
    // 200 Z₂ sources on V = 64 — stochastic noise is ~few %, give 4 % bound.
    EXPECT_LT(rel, 0.04)
        << "estimator=" << r.mean
        << "  analytic=" << analytic
        << "  rel="      << rel
        << "  se="       << r.stochastic_error
        << "  avg_iters="<< r.avg_cg_iters;
}

TEST(Condensate, HeavyMassLimitMatchesAnalytic)
{
    // m = 20 ⇒ the diagonal (m + 2) dominates and the leading-order condensate
    // is ≈ 2/(m + 2). Z₂ noise η² ≡ 1 component-wise gives the diagonal entry
    // exactly per source, but the O(1/(m+2)²) off-diagonal Wilson hops still
    // contribute stochastic variance — a few · 10⁻³ at this mass.
    constexpr int    L     = 4;
    constexpr double mass  = 20.0;
    Lattice<2> lattice = Lattice<2>::cube(L);
    LinkField<double, 2> U(lattice);

    dirac::WilsonDirac2D D(lattice, mass);
    Rng rng(7u);

    const auto r = obs::stochasticCondensate(
        D, U, lattice, rng, /*n_sources*/40);

    const double analytic      = obs::freeFieldCondensate2D(lattice, mass);
    const double leading_order = 2.0 / (mass + 2.0);

    EXPECT_TRUE(r.all_converged);
    EXPECT_NEAR(r.mean, analytic, 5e-3) << "se=" << r.stochastic_error;
    // The 2/(m+2) leading-order should match the exact lattice sum at this
    // (m, L); together they pin down our normalization.
    EXPECT_NEAR(analytic, leading_order, 5e-4);
}

TEST(Condensate, IsMonotoneInMass)
{
    Lattice<2> lattice = Lattice<2>::cube(8);
    LinkField<double, 2> U(lattice);
    Rng rng(42u);

    auto cond_at = [&](double m) -> double
    {
        dirac::WilsonDirac2D D(lattice, m);
        Rng local(rng.raw()); // independent stream per mass
        return obs::stochasticCondensate(
            D, U, lattice, local, /*n_sources*/40, /*tol*/1e-9).mean;
    };

    const double c1 = cond_at(0.4);
    const double c2 = cond_at(0.8);
    const double c3 = cond_at(1.5);

    // Free-field condensate at U=0 is positive and decreases monotonically
    // with mass (the diagonal ~ 1/(m+2) dominates).
    EXPECT_GT(c1, 0.0);
    EXPECT_GT(c2, 0.0);
    EXPECT_GT(c3, 0.0);
    EXPECT_GT(c1, c2);
    EXPECT_GT(c2, c3);
}
