/**
 * @file test_su3.cpp
 * @brief SU(3) group math + lattice gauge-action correctness checks.
 *
 *   1. Identity and zero literals match.
 *   2. multiply(I, A) == A and multiply(A, I) == A.
 *   3. multiply is associative.
 *   4. dagger(A·B) == dagger(B) · dagger(A).
 *   5. Random SU(3) element has det = 1 and U U† = I.
 *   6. project_to_su3 fixes drifted matrices: U U† = I, det = 1.
 *   7. embedSU2 produces unit-determinant 3×3.
 *   8. Cold field → average plaquette = 1.
 *   9. Hot field → average plaquette ≈ 0 (Z[U]-uniform sampling).
 *  10. Metropolis at large β (cold limit) drives ⟨P⟩ → 1.
 *  11. Metropolis at small β (random limit) drives ⟨P⟩ → 0.
 *  12. Single-link ΔS matches direct re-computation S[U_new] − S[U_old].
 */

#include "fields/link_field.hpp"
#include "lattice/lattice.hpp"
#include "math/su3.hpp"
#include "models/su3.hpp"
#include "monte_carlo/metropolis.hpp"
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
        {
            const Complex target(i == j ? 1.0 : 0.0, 0.0);
            worst = std::max(worst, std::abs(prod.m[i][j] - target));
        }
    return worst;
}

} // namespace

TEST(SU3Math, IdentityAndZero)
{
    const su3::Element I = su3::Element::identity();
    const su3::Element Z = su3::Element::zero();
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
        {
            EXPECT_EQ(I.m[i][j], Complex(i == j ? 1.0 : 0.0, 0.0));
            EXPECT_EQ(Z.m[i][j], Complex(0.0, 0.0));
        }
}

TEST(SU3Math, MultiplyByIdentity)
{
    Rng rng(11u);
    const su3::Element A = su3_model::SU3Model<3>::randomSU3(rng);
    const su3::Element I = su3::Element::identity();
    const su3::Element L = su3::multiply(I, A);
    const su3::Element R = su3::multiply(A, I);
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
        {
            EXPECT_NEAR(L.m[i][j].real(), A.m[i][j].real(), 1e-12);
            EXPECT_NEAR(L.m[i][j].imag(), A.m[i][j].imag(), 1e-12);
            EXPECT_NEAR(R.m[i][j].real(), A.m[i][j].real(), 1e-12);
            EXPECT_NEAR(R.m[i][j].imag(), A.m[i][j].imag(), 1e-12);
        }
}

TEST(SU3Math, MultiplyAssociative)
{
    Rng rng(42u);
    const su3::Element A = su3_model::SU3Model<3>::randomSU3(rng);
    const su3::Element B = su3_model::SU3Model<3>::randomSU3(rng);
    const su3::Element C = su3_model::SU3Model<3>::randomSU3(rng);
    const su3::Element L = su3::multiply(su3::multiply(A, B), C);
    const su3::Element R = su3::multiply(A, su3::multiply(B, C));
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
        {
            EXPECT_NEAR(L.m[i][j].real(), R.m[i][j].real(), 1e-10);
            EXPECT_NEAR(L.m[i][j].imag(), R.m[i][j].imag(), 1e-10);
        }
}

TEST(SU3Math, DaggerOfProduct)
{
    Rng rng(7u);
    const su3::Element A = su3_model::SU3Model<3>::randomSU3(rng);
    const su3::Element B = su3_model::SU3Model<3>::randomSU3(rng);
    const su3::Element L = su3::dagger(su3::multiply(A, B));
    const su3::Element R = su3::multiply(su3::dagger(B), su3::dagger(A));
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
        {
            EXPECT_NEAR(L.m[i][j].real(), R.m[i][j].real(), 1e-12);
            EXPECT_NEAR(L.m[i][j].imag(), R.m[i][j].imag(), 1e-12);
        }
}

TEST(SU3Math, RandomSU3IsUnitary)
{
    Rng rng(100u);
    for (int trial = 0; trial < 16; ++trial)
    {
        const su3::Element U = su3_model::SU3Model<3>::randomSU3(rng);
        EXPECT_LT(maxOffUnitarity(U), 1e-12);
        const Complex det = su3::determinant(U);
        EXPECT_NEAR(det.real(), 1.0, 1e-12);
        EXPECT_NEAR(det.imag(), 0.0, 1e-12);
    }
}

TEST(SU3Math, ProjectionFixesDrift)
{
    Rng rng(3u);
    su3::Element U = su3_model::SU3Model<3>::randomSU3(rng);
    // Inject drift: scale by something other than 1.
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            U.m[i][j] *= Complex(1.0 + 0.02 * rng.normal(),
                                 0.01 * rng.normal());
    const su3::Element P = su3::project_to_su3(U);
    EXPECT_LT(maxOffUnitarity(P), 1e-12);
    const Complex det = su3::determinant(P);
    EXPECT_NEAR(det.real(), 1.0, 1e-12);
    EXPECT_NEAR(det.imag(), 0.0, 1e-12);
}

TEST(SU3Math, EmbedSU2HasUnitDet)
{
    // (s, v) with s² + |v|² = 1 → embedded matrix has det = 1.
    const double s = std::cos(0.7);
    const double a = std::sin(0.7);
    const double v0 = a * 0.3, v1 = a * 0.6, v2 = a * std::sqrt(1 - 0.09 - 0.36);
    for (int sub = 0; sub < 3; ++sub)
    {
        const std::array<std::array<int, 2>, 3> pairs = { {
            {0, 1}, {0, 2}, {1, 2}
        } };
        const su3::Element E = su3::embedSU2(
            pairs[static_cast<std::size_t>(sub)][0],
            pairs[static_cast<std::size_t>(sub)][1],
            s, v0, v1, v2);
        EXPECT_LT(maxOffUnitarity(E), 1e-12);
        const Complex det = su3::determinant(E);
        EXPECT_NEAR(det.real(), 1.0, 1e-12);
        EXPECT_NEAR(det.imag(), 0.0, 1e-12);
    }
}

TEST(SU3Lattice, ColdHasPlaquetteOne)
{
    Lattice<3> L = Lattice<3>::cube(4);
    LinkField<su3::Element, 3> U(L);
    su3_model::SU3Model<3>::cold(U);
    EXPECT_NEAR(su3_model::averagePlaquette(L, U), 1.0, 1e-12);
}

TEST(SU3Lattice, HotHasPlaquetteNearZero)
{
    Lattice<3> L = Lattice<3>::cube(6);
    LinkField<su3::Element, 3> U(L);
    Rng rng(2u);
    su3_model::SU3Model<3>::hot(U, rng);
    const double plaq = su3_model::averagePlaquette(L, U);
    EXPECT_LT(std::abs(plaq), 0.15) << "plaq=" << plaq;
}

TEST(SU3Metropolis, DeltaActionMatchesDirectRecompute)
{
    // ΔS from delta_action should match S[U_new] − S[U_old] for the same
    // single-link change. This is the load-bearing identity for detailed
    // balance, so test it on multiple random configurations.
    Lattice<3> L = Lattice<3>::cube(4);
    LinkField<su3::Element, 3> U(L);
    Rng rng(99u);
    su3_model::SU3Model<3>::hot(U, rng);

    su3_model::SU3Model<3> model(L, /*β*/4.5);
    for (int trial = 0; trial < 32; ++trial)
    {
        const int site = static_cast<int>(rng.uniform() * L.volume());
        const int mu   = static_cast<int>(rng.uniform() * 3);
        const int unit = su3_model::SU3Model<3>::unitOf(site, mu);
        const su3::Element U_new = model.propose(U, unit, rng);

        const double dS_predicted = model.delta_action(U, unit, U_new);

        const double S_before = model.totalAction(U);
        const su3::Element saved = U(site, mu);
        U(site, mu) = U_new;
        const double S_after  = model.totalAction(U);
        U(site, mu) = saved;

        const double dS_actual = S_after - S_before;
        EXPECT_NEAR(dS_predicted, dS_actual, 1e-9)
            << "site=" << site << " mu=" << mu;
    }
}

TEST(SU3Metropolis, ColdLimitDrivesPlaquetteToOne)
{
    // Very large β → action wants U_□ = I everywhere → ⟨P⟩ → 1.
    Lattice<3> L = Lattice<3>::cube(4);
    LinkField<su3::Element, 3> U(L);
    Rng rng(123u);
    su3_model::SU3Model<3>::hot(U, rng);

    // SU(3) Metropolis thermalizes ~3× slower than SU(2) because each
    // Cabibbo-Marinari step only touches one of three sub-groups per link,
    // so we run more sweeps and use a small proposal step to push acceptance.
    su3_model::SU3Model<3> model(L, /*β*/40.0, /*step*/0.08);
    mc::MetropolisSweep<su3_model::SU3Model<3>> sweeper(model);
    sweeper.sweepN(U, rng, 600);

    const double plaq = su3_model::averagePlaquette(L, U);
    EXPECT_GT(plaq, 0.9) << "P=" << plaq << "  acceptance="
                         << sweeper.cumulativeAcceptance();
}

TEST(SU3Metropolis, RandomLimitDrivesPlaquetteToZero)
{
    // β → 0 ⇒ action is flat ⇒ pure-random configuration ⇒ ⟨P⟩ ≈ 0.
    Lattice<3> L = Lattice<3>::cube(4);
    LinkField<su3::Element, 3> U(L);
    Rng rng(7u);
    su3_model::SU3Model<3>::cold(U);

    su3_model::SU3Model<3> model(L, /*β*/0.0, /*step*/1.0);
    mc::MetropolisSweep<su3_model::SU3Model<3>> sweeper(model);
    sweeper.sweepN(U, rng, 200);

    const double plaq = su3_model::averagePlaquette(L, U);
    EXPECT_LT(std::abs(plaq), 0.15) << "P=" << plaq;
}
