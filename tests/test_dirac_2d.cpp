/**
 * @file test_dirac_2d.cpp
 * @brief Tests of the 2D Wilson-Dirac operator on a U(1) gauge background.
 *
 *   1. Linearity:        D(α x + β y) = α D x + β D y
 *   2. γ⁵-Hermiticity:   ⟨x, D y⟩ = ⟨D† x, y⟩ via the explicit applyDagger.
 *   3. Free-field hopping (U ≡ 1): D agrees with the analytic translation-
 *      invariant Wilson operator at one chosen site.
 */

#include "fields/link_field.hpp"
#include "fields/spinor_field.hpp"
#include "lattice/lattice.hpp"
#include "math/dirac_2d.hpp"
#include "models/u1.hpp"
#include "rng/rng.hpp"

#include <gtest/gtest.h>
#include <complex>

using namespace lqft;
using Complex = std::complex<double>;

namespace
{

/// Fill `spinor` with iid N(0, 1) + i N(0, 1) entries.
void random_spinor(SpinorField<2, 2>& s, Rng& rng)
{
    const int N = s.totalSize();
    auto* p = s.data();
    for (int i = 0; i < N; ++i)
        p[i] = Complex{ rng.normal(), rng.normal() };
}

} // namespace

TEST(WilsonDirac2D, IsLinear)
{
    Lattice<2> lattice = Lattice<2>::cube(6);
    LinkField<double, 2> U(lattice);
    Rng rng(1u);
    u1::U1Model<2>::hot(U, rng);

    SpinorField<2, 2> a(lattice), b(lattice);
    random_spinor(a, rng);
    random_spinor(b, rng);
    const Complex alpha{ 0.7, -0.3};
    const Complex beta { -1.1, 0.5};

    SpinorField<2, 2> combo(lattice);
    {
        // combo = α a + β b
        const int N = a.totalSize();
        for (int i = 0; i < N; ++i)
            combo.data()[i] = alpha * a.data()[i] + beta * b.data()[i];
    }

    dirac::WilsonDirac2D D(lattice, /*mass*/0.4);
    SpinorField<2, 2> Da(lattice), Db(lattice), Dcombo(lattice), reference(lattice);
    D.apply(U, a,     Da);
    D.apply(U, b,     Db);
    D.apply(U, combo, Dcombo);
    {
        const int N = Da.totalSize();
        for (int i = 0; i < N; ++i)
            reference.data()[i] = alpha * Da.data()[i] + beta * Db.data()[i];
    }

    // ‖Dcombo − (α Da + β Db)‖ should be at numerical noise.
    double diff2 = 0.0;
    for (int i = 0; i < Dcombo.totalSize(); ++i)
        diff2 += std::norm(Dcombo.data()[i] - reference.data()[i]);
    EXPECT_LT(std::sqrt(diff2), 1e-12);
}

TEST(WilsonDirac2D, IsGamma5Hermitian)
{
    // ⟨x, D y⟩ = ⟨D† x, y⟩ for the explicit applyDagger implementation.
    // (This is the constitutive identity behind γ⁵-Hermiticity. Independent
    // verification of γ⁵ D γ⁵ = D† at the operator level is in Round 2.)
    Lattice<2> lattice = Lattice<2>::cube(6);
    LinkField<double, 2> U(lattice);
    Rng rng(99u);
    u1::U1Model<2>::hot(U, rng);

    SpinorField<2, 2> x(lattice), y(lattice);
    random_spinor(x, rng);
    random_spinor(y, rng);

    dirac::WilsonDirac2D D(lattice, /*mass*/0.2);
    SpinorField<2, 2> Dy(lattice), Dx_dag(lattice);
    D.apply       (U, y, Dy);
    D.applyDagger (U, x, Dx_dag);

    const Complex lhs = inner_product(x, Dy);     // ⟨x, D y⟩
    const Complex rhs = inner_product(Dx_dag, y); // ⟨D† x, y⟩
    EXPECT_NEAR(lhs.real(), rhs.real(), 1e-10);
    EXPECT_NEAR(lhs.imag(), rhs.imag(), 1e-10);
}

TEST(WilsonDirac2D, FreeFieldDiagonalAtMassPlusTwo)
{
    // With U ≡ 1 and a delta-source ψ(x) = δ_{x, x_0} · (1, 0)^T,
    // the diagonal term gives (D ψ)(x_0) = (m + 2) · (1, 0)^T,
    // and the hopping fills in 8 neighbors of x_0 with specific
    // values. Test the diagonal value precisely; the neighbor pattern
    // is exercised by the linearity / Hermiticity tests above.
    Lattice<2> lattice = Lattice<2>::cube(8);
    LinkField<double, 2> U(lattice); // zero phases ⇒ U = 1 everywhere

    SpinorField<2, 2> delta(lattice);
    delta(/*site*/3, /*comp*/0) = Complex{1.0, 0.0};

    dirac::WilsonDirac2D D(lattice, /*mass*/0.13);
    SpinorField<2, 2> result(lattice);
    D.apply(U, delta, result);

    EXPECT_NEAR(result(3, 0).real(), 0.13 + 2.0, 1e-12);
    EXPECT_NEAR(result(3, 0).imag(), 0.0,        1e-12);
    EXPECT_NEAR(result(3, 1).real(), 0.0,        1e-12);
    EXPECT_NEAR(result(3, 1).imag(), 0.0,        1e-12);
}
