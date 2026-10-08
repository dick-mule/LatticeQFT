/**
 * @file test_su2.cpp
 * @brief Tests of the SU(2) algebra, the Wilson-action model, and the
 *        Kennedy-Pendleton heat-bath.
 *
 *   1. Algebra:
 *      - Identity is neutral under multiplication.
 *      - U U† = I to round-off.
 *      - Multiplication preserves unit norm (the canonical SU(2) closure).
 *      - randomUniformSU2 produces unit elements.
 *      - randomNearIdentity is close to identity at small sigma.
 *
 *   2. SU(2) model:
 *      - Cold start has S = 0.
 *      - Δ-action matches totalAction difference at machine precision.
 *      - Action is invariant under random gauge transformations
 *            U_μ(x) → Ω(x) U_μ(x) Ω†(x + ê_μ).
 *      - Color partition has 2·Dim sublattices covering V·Dim units.
 *
 *   3. Heat-bath:
 *      - Strong-coupling plaquette is approximately zero (random gauge field).
 *      - Weak-coupling plaquette → 1.
 *      - Acceptance is 100% (no Metropolis reject step needed by construction).
 *
 *   4. Dim-lift:
 *      - All exact-arithmetic tests pass at Dim = 3 and Dim = 4 too.
 */

#include "fields/link_field.hpp"
#include "lattice/lattice.hpp"
#include "math/su2.hpp"
#include "models/su2.hpp"
#include "monte_carlo/metropolis.hpp"
#include "monte_carlo/su2_heatbath.hpp"
#include "observables/observables.hpp"
#include "rng/rng.hpp"

#include <gtest/gtest.h>
#include <cmath>

using namespace lqft;

// =============================================================================
// 1. Algebra
// =============================================================================

TEST(SU2Math, IdentityIsNeutralForMultiplication)
{
    Rng rng(1u);
    const auto U = su2_model::SU2Model<2>::randomUniformSU2(rng);
    const auto I = su2::Element::identity();
    const auto U1 = su2::multiply(U, I);
    const auto U2 = su2::multiply(I, U);
    EXPECT_NEAR(U1.s, U.s, 1e-12);
    EXPECT_NEAR(U2.s, U.s, 1e-12);
    for (int i = 0; i < 3; ++i)
    {
        EXPECT_NEAR(U1.v[i], U.v[i], 1e-12);
        EXPECT_NEAR(U2.v[i], U.v[i], 1e-12);
    }
}

TEST(SU2Math, UTimesUDaggerIsIdentity)
{
    Rng rng(2u);
    for (int i = 0; i < 32; ++i)
    {
        const auto U = su2_model::SU2Model<2>::randomUniformSU2(rng);
        const auto P = su2::multiply(U, su2::dagger(U));
        EXPECT_NEAR(P.s, 1.0, 1e-12);
        EXPECT_NEAR(P.v[0], 0.0, 1e-12);
        EXPECT_NEAR(P.v[1], 0.0, 1e-12);
        EXPECT_NEAR(P.v[2], 0.0, 1e-12);
    }
}

TEST(SU2Math, MultiplicationPreservesUnitNorm)
{
    Rng rng(3u);
    for (int i = 0; i < 32; ++i)
    {
        const auto A = su2_model::SU2Model<2>::randomUniformSU2(rng);
        const auto B = su2_model::SU2Model<2>::randomUniformSU2(rng);
        const auto C = su2::multiply(A, B);
        EXPECT_NEAR(su2::norm_squared(C), 1.0, 1e-12);
    }
}

TEST(SU2Math, RandomNearIdentityIsCloseToIdentityForSmallSigma)
{
    Rng rng(4u);
    for (int i = 0; i < 16; ++i)
    {
        const auto U = su2_model::SU2Model<2>::randomNearIdentity(0.05, rng);
        EXPECT_NEAR(su2::norm_squared(U), 1.0, 1e-12);
        EXPECT_GT(U.s, 0.99); // close to 1
    }
}

// =============================================================================
// 2. Model (Dim = 2)
// =============================================================================

TEST(SU2_2D, ColdStartHasZeroAction)
{
    Lattice<2> L = Lattice<2>::cube(6);
    LinkField<su2::Element, 2> field(L);
    su2_model::SU2Model<2>::cold(field);

    su2_model::SU2Model<2> model(L, /*β*/2.0);
    EXPECT_DOUBLE_EQ(model.totalAction(field), 0.0);
    EXPECT_DOUBLE_EQ(su2_model::averagePlaquette(L, field), 1.0);
}

TEST(SU2_2D, DeltaSMatchesActionDifference)
{
    Lattice<2> L = Lattice<2>::cube(6);
    LinkField<su2::Element, 2> field(L);
    Rng rng(7u);
    su2_model::SU2Model<2>::hot(field, rng);

    su2_model::SU2Model<2> model(L, /*β*/2.2);

    for (int unit : {0, 5, 13, 25, 47})
    {
        const int site = su2_model::SU2Model<2>::siteOfUnit(unit);
        const int mu   = su2_model::SU2Model<2>::muOfUnit(unit);
        const auto U_old = field(site, mu);

        // Use the model's own proposal kernel so the test reflects the
        // production path.
        const auto U_new = model.propose(field, unit, rng);

        const double S_before = model.totalAction(field);
        const double dS_pred  = model.delta_action(field, unit, U_new);
        model.apply(field, unit, U_new);
        const double S_after  = model.totalAction(field);
        EXPECT_NEAR(S_after - S_before, dS_pred, 1e-9) << "unit " << unit;

        // Restore for next iteration.
        model.apply(field, unit, U_old);
    }
}

TEST(SU2_2D, ActionIsGaugeInvariant)
{
    // Random Ω(x) ∈ SU(2). Action invariant under U_μ(x) → Ω(x) U_μ(x) Ω†(x+ê_μ).
    Lattice<2> L = Lattice<2>::cube(8);
    LinkField<su2::Element, 2> field(L);
    Rng rng(9u);
    su2_model::SU2Model<2>::hot(field, rng);

    su2_model::SU2Model<2> model(L, /*β*/2.0);
    const double S_before = model.totalAction(field);

    // Per-site gauge transformation matrices.
    std::vector<su2::Element> omega(static_cast<std::size_t>(L.volume()));
    for (int s = 0; s < L.volume(); ++s)
        omega[s] = su2_model::SU2Model<2>::randomUniformSU2(rng);

    // Apply: U_μ(x) ← Ω(x) U_μ(x) Ω†(x + ê_μ).
    for (int s = 0; s < L.volume(); ++s)
        for (int mu = 0; mu < 2; ++mu)
        {
            const int s_next = L.forward(s, mu);
            field(s, mu) = su2::multiply(omega[s],
                          su2::multiply(field(s, mu), su2::dagger(omega[s_next])));
        }

    const double S_after = model.totalAction(field);
    EXPECT_NEAR(S_before, S_after, 1e-8);
}

TEST(SU2_2D, ColorPartitionCovers)
{
    Lattice<2> L = Lattice<2>::cube(8);
    su2_model::SU2Model<2> model(L, 1.0);
    ASSERT_EQ(model.numColors(), 4);
    long total = 0;
    for (int c = 0; c < model.numColors(); ++c)
        total += static_cast<long>(model.unitsOfColor(c).size());
    EXPECT_EQ(total, static_cast<long>(L.volume()) * 2);
}

// =============================================================================
// 3. Heat-bath
// =============================================================================

TEST(SU2HeatBath, StrongAndWeakCouplingPlaquetteBoundsAreSane)
{
    auto plaq_at = [](double beta, std::uint64_t seed) {
        Lattice<2> L = Lattice<2>::cube(10);
        LinkField<su2::Element, 2> field(L);
        Rng rng(seed);
        su2_model::SU2Model<2>::hot(field, rng);
        su2_model::SU2Model<2> model(L, beta);
        // Thermalize then measure.
        su2_model::heatBathSweepN(model, field, rng, 200);
        obs::Mean p;
        for (int s = 0; s < 600; ++s)
        {
            su2_model::heatBathSweep(model, field, rng);
            if ((s % 2) == 0) p.add(su2_model::averagePlaquette(L, field));
        }
        return p.mean();
    };

    // SU(2) leading-order weak-coupling: ⟨½ tr U⟩ ≈ 1 − (N²−1)/(2N · β) = 1 − 3/(4β).
    // At β = 10 this predicts ≈ 0.925, so > 0.80 is comfortable.
    // Strong-coupling: ⟨½ tr U⟩ ≈ β/4 at small β; at β = 0.4 → ~0.1, so < 0.30 is loose.
    const double p_strong = plaq_at(/*β*/0.4,  11u);
    const double p_weak   = plaq_at(/*β*/10.0, 12u);
    EXPECT_LT(p_strong, 0.30);
    EXPECT_GT(p_weak,   0.80);
    EXPECT_LT(p_strong, p_weak);
}

TEST(SU2HeatBath, ProducesValidSU2)
{
    // Sanity check: after many heat-bath sweeps, every link is still a
    // unit quaternion to round-off.
    Lattice<2> L = Lattice<2>::cube(8);
    LinkField<su2::Element, 2> field(L);
    Rng rng(13u);
    su2_model::SU2Model<2>::hot(field, rng);
    su2_model::SU2Model<2> model(L, 2.5);
    su2_model::heatBathSweepN(model, field, rng, 50);

    for (int s = 0; s < L.volume(); ++s)
        for (int mu = 0; mu < 2; ++mu)
            EXPECT_NEAR(su2::norm_squared(field(s, mu)), 1.0, 1e-10)
                << "site " << s << " μ " << mu;
}

// =============================================================================
// 4. Dim-lift: instantiate the model at Dim = 3 and Dim = 4.
// =============================================================================

TEST(SU2_3D, ColdStartHasZeroAction)
{
    Lattice<3> L = Lattice<3>::cube(4);
    LinkField<su2::Element, 3> field(L);
    su2_model::SU2Model<3>::cold(field);
    su2_model::SU2Model<3> model(L, 2.0);
    EXPECT_DOUBLE_EQ(model.totalAction(field), 0.0);
    EXPECT_DOUBLE_EQ(su2_model::averagePlaquette(L, field), 1.0);
}

TEST(SU2_3D, ColorPartitionIsSixColors)
{
    Lattice<3> L = Lattice<3>::cube(4);
    su2_model::SU2Model<3> model(L, 1.0);
    EXPECT_EQ(model.numColors(), 6);
    long total = 0;
    for (int c = 0; c < model.numColors(); ++c)
        total += static_cast<long>(model.unitsOfColor(c).size());
    EXPECT_EQ(total, static_cast<long>(L.volume()) * 3);
}

TEST(SU2_3D, DeltaSMatchesActionDifference)
{
    Lattice<3> L = Lattice<3>::cube(5);
    LinkField<su2::Element, 3> field(L);
    Rng rng(31u);
    su2_model::SU2Model<3>::hot(field, rng);
    su2_model::SU2Model<3> model(L, 1.8);
    for (int unit : {0, 14, 51, 117, 244})
    {
        const auto U_old = field(su2_model::SU2Model<3>::siteOfUnit(unit),
                                 su2_model::SU2Model<3>::muOfUnit(unit));
        const auto U_new = model.propose(field, unit, rng);
        const double S_before = model.totalAction(field);
        const double dS_pred  = model.delta_action(field, unit, U_new);
        model.apply(field, unit, U_new);
        const double S_after  = model.totalAction(field);
        EXPECT_NEAR(S_after - S_before, dS_pred, 1e-8) << "unit " << unit;
        model.apply(field, unit, U_old);
    }
}

TEST(SU2_4D, ColorPartitionAndColdAction)
{
    Lattice<4> L = Lattice<4>::cube(4);
    LinkField<su2::Element, 4> field(L);
    su2_model::SU2Model<4>::cold(field);
    su2_model::SU2Model<4> model(L, 2.0);
    EXPECT_DOUBLE_EQ(model.totalAction(field), 0.0);
    EXPECT_EQ(model.numColors(), 8);
}

// =============================================================================
// 5. Wilson loops + Creutz ratio (Round 3b additions).
// =============================================================================

TEST(SU2WilsonLoop, W11EqualsAveragePlaquette)
{
    // ⟨W(1,1)⟩ should be identical to the average plaquette by construction
    // for any single (μ, ν) plane.
    Lattice<2> L = Lattice<2>::cube(8);
    LinkField<su2::Element, 2> field(L);
    Rng rng(11u);
    su2_model::SU2Model<2>::hot(field, rng);

    const double W11 = su2_model::averageWilsonLoop(L, field, 0, 1, 1, 1);
    const double avg = su2_model::averagePlaquette(L, field);
    EXPECT_NEAR(W11, avg, 1e-12);
}

TEST(SU2WilsonLoop, GaugeInvariantUnderRandomTransformation)
{
    // Closed Wilson loops are gauge invariant by the cyclic-trace argument.
    Lattice<3> L = Lattice<3>::cube(6);
    LinkField<su2::Element, 3> field(L);
    Rng rng(22u);
    su2_model::SU2Model<3>::hot(field, rng);

    const double W21_before = su2_model::averageWilsonLoop(L, field, 0, 1, 2, 1);
    const double W22_before = su2_model::averageWilsonLoop(L, field, 0, 1, 2, 2);

    // Apply U_μ(x) → Ω(x) U_μ(x) Ω†(x + ê_μ).
    std::vector<su2::Element> omega(static_cast<std::size_t>(L.volume()));
    for (int s = 0; s < L.volume(); ++s)
        omega[s] = su2_model::SU2Model<3>::randomUniformSU2(rng);
    for (int s = 0; s < L.volume(); ++s)
        for (int mu = 0; mu < 3; ++mu)
        {
            const int s_next = L.forward(s, mu);
            field(s, mu) = su2::multiply(
                omega[s],
                su2::multiply(field(s, mu), su2::dagger(omega[s_next])));
        }

    const double W21_after = su2_model::averageWilsonLoop(L, field, 0, 1, 2, 1);
    const double W22_after = su2_model::averageWilsonLoop(L, field, 0, 1, 2, 2);
    EXPECT_NEAR(W21_before, W21_after, 1e-9);
    EXPECT_NEAR(W22_before, W22_after, 1e-9);
}

TEST(SU2WilsonLoop, CreutzRatio3DIsPositive)
{
    // 3D SU(2) is confining for all β (no deconfinement transition in 3D).
    // The Creutz ratio χ(2, 2) should be positive at any β; the magnitude
    // shrinks at weak coupling but the sign is the confinement signature.
    constexpr int    L    = 10;
    constexpr double beta = 2.0;
    Lattice<3> lat = Lattice<3>::cube(L);
    LinkField<su2::Element, 3> field(lat);
    Rng rng(33u);
    su2_model::SU2Model<3>::hot(field, rng);

    su2_model::SU2Model<3> model(lat, beta);
    su2_model::heatBathSweepN(model, field, rng, /*therm*/300);

    obs::Mean W11, W21, W12, W22;
    for (int s = 0; s < 400; ++s)
    {
        su2_model::heatBathSweep(model, field, rng);
        if ((s % 2) == 0)
        {
            // Average over the three plaquette planes for better statistics.
            double w11 = 0, w12 = 0, w21 = 0, w22 = 0;
            int np = 0;
            for (int mu = 0; mu < 3; ++mu)
                for (int nu = mu + 1; nu < 3; ++nu)
                {
                    w11 += su2_model::averageWilsonLoop(lat, field, mu, nu, 1, 1);
                    w12 += su2_model::averageWilsonLoop(lat, field, mu, nu, 1, 2);
                    w21 += su2_model::averageWilsonLoop(lat, field, mu, nu, 2, 1);
                    w22 += su2_model::averageWilsonLoop(lat, field, mu, nu, 2, 2);
                    ++np;
                }
            W11.add(w11 / np); W12.add(w12 / np);
            W21.add(w21 / np); W22.add(w22 / np);
        }
    }
    const double chi22 = su2_model::creutzRatio(
        W22.mean(), W11.mean(), W12.mean(), W21.mean());
    EXPECT_GT(W22.mean(), 0.0);
    EXPECT_LT(W22.mean(), W11.mean()); // smaller loop has larger expectation
    EXPECT_GT(chi22, 0.0)
        << "χ(2,2)=" << chi22 << " W11=" << W11.mean()
        << " W12=" << W12.mean() << " W21=" << W21.mean() << " W22=" << W22.mean();
}
